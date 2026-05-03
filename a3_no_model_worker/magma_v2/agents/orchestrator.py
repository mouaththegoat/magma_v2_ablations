"""MAGMA v2 orchestrator."""

from __future__ import annotations

import asyncio
import os
import json
import re
from typing import Any

from strands import Agent, tool

from magma_v2.agents.data_worker import data_worker
from magma_v2.agents.data_worker_validator import data_worker_validator
from magma_v2.agents.judge import judge
from magma_v2.agents.literature_worker import literature_worker
from magma_v2.agents.model_worker import build_grounded_model_handoff_from_artifacts, model_worker, single_shot_model_worker, write_model_handoff
from magma_v2.model_config import create_agent_model
from magma_v2.runtime.actions import log_action
from magma_v2.runtime.artifacts import read_json, run_dir_from, write_json
from magma_v2.runtime.errors import append_error, append_repair_attempt, classify_failure, load_recovery_state, save_recovery_state
from magma_v2.runtime.errors.history import state_summary
from magma_v2.runtime.errors.policy import build_repair_attempt, decide_recovery
from magma_v2.runtime.errors.schema import RetryPolicy, StageError
from magma_v2.runtime.human_loop import pending_human_input, write_human_input_request
from magma_v2.runtime.literature import research_is_stale, should_run_literature_stage
from magma_v2.runtime.provider_errors import classify_provider_error


def _run_dir(tool_context: object | None, artifact_dir: str = "./artifacts") -> str:
    if isinstance(tool_context, dict):
        invocation_state = tool_context.get("invocation_state", {})
    else:
        invocation_state = getattr(tool_context, "invocation_state", {}) if tool_context else {}
    return run_dir_from(invocation_state.get("run_dir") or artifact_dir)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        payload = json.loads(raw)
        return payload if isinstance(payload, dict) else None
    except Exception:
        pass
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


async def _run_agent(agent: Agent, prompt: str, tool_context: object | None, action_name: str) -> dict:
    run_dir = _run_dir(tool_context)
    invocation_state = {}
    if isinstance(tool_context, dict):
        invocation_state = tool_context.get("invocation_state", {})
    elif tool_context is not None:
        invocation_state = getattr(tool_context, "invocation_state", {})

    text_parts: list[str] = []
    tool_uses: dict[str, dict[str, Any]] = {}
    tool_errors: list[dict[str, Any]] = []
    try:
        delay_seconds = float(invocation_state.get("agent_delay_seconds") or os.getenv("MAGMA_AGENT_DELAY_SECONDS") or 0)
        if delay_seconds > 0:
            await asyncio.sleep(delay_seconds)
        async for event in agent.stream_async(prompt, invocation_state=invocation_state):
            if event.get("type") == "tool_use_stream":
                current_tool_use = event.get("current_tool_use", {})
                if isinstance(current_tool_use, dict):
                    tool_use_id = (
                        current_tool_use.get("toolUseId")
                        or current_tool_use.get("tool_use_id")
                        or current_tool_use.get("id")
                    )
                    if tool_use_id:
                        tool_uses[str(tool_use_id)] = dict(current_tool_use)
            elif event.get("type") == "tool_result":
                tool_result = event.get("tool_result", {})
                if isinstance(tool_result, dict) and tool_result.get("status") == "error":
                    tool_use_id = str(tool_result.get("toolUseId") or "")
                    content = tool_result.get("content", [])
                    error_text = "\n".join(
                        block.get("text", "")
                        for block in content
                        if isinstance(block, dict) and block.get("text")
                    ).strip() or "Tool returned error status."
                    tool_name = None
                    if tool_use_id and tool_use_id in tool_uses:
                        tool_name = tool_uses[tool_use_id].get("name")
                    error_event = {
                        "tool_name": tool_name or "unknown_tool",
                        "tool_use_id": tool_use_id or None,
                        "error": error_text,
                        "tool_result": tool_result,
                    }
                    tool_errors.append(error_event)
                    _log_tool_error(run_dir, agent.name, error_event)
            if "data" in event:
                text_parts.append(str(event["data"]))
            elif "result" in event:
                result = event["result"]
                message = getattr(result, "message", {})
                if isinstance(message, dict):
                    blocks = message.get("content", [])
                    tool_errors.extend(_extract_tool_errors_from_blocks(agent.name, run_dir, blocks, tool_uses))
                    if not text_parts:
                        text_parts.extend(
                            block.get("text", "")
                            for block in blocks
                            if isinstance(block, dict)
                        )
        text = "".join(text_parts).strip()
        log_action(run_dir, "orchestrator", action_name, success=True, metadata={"tool_error_count": len(tool_errors)})
        return {"status": "success", "text": text, "tool_errors": tool_errors}
    except Exception as exc:
        classified = classify_provider_error(exc)
        log_action(run_dir, "orchestrator", action_name, success=False, error=classified["message"], metadata={"provider_error": classified})
        return {"status": "error", "error": classified["message"], "provider_error": classified, "text": "".join(text_parts), "tool_errors": tool_errors}


@tool(context=True)
async def call_data_worker(instructions: str, tool_context: object | None = None) -> dict:
    """Run the data worker."""
    run_dir = _run_dir(tool_context)
    if pending_human_input(run_dir):
        return {"status": "blocked", "error": "Waiting for human input before data stage can continue."}
    result = await _run_agent(data_worker, instructions, tool_context, "call_data_worker")
    return _ensure_stage_outcome(
        run_dir=run_dir,
        result=result,
        stage="data_worker",
        owner_stage="data",
        handoff_filename="DATA_HANDOFF.json",
        error_filename="DATA_ERROR.json",
    )


@tool(context=True)
async def call_model_worker(instructions: str, tool_context: object | None = None) -> dict:
    """A3 ablation: single-shot model writer (no agentic model_worker, no validator)."""
    run_dir = _run_dir(tool_context)
    result = await single_shot_model_worker(instructions, run_dir)
    return _ensure_stage_outcome(
        run_dir=run_dir,
        result=result,
        stage="model_worker",
        owner_stage="model",
        handoff_filename="MODEL_HANDOFF.json",
        error_filename="MODEL_ERROR.json",
    )


@tool(context=True)
async def call_data_worker_validator(instructions: str = "", tool_context: object | None = None) -> dict:
    """Run the data-worker validator."""
    result = await _run_agent(data_worker_validator, instructions, tool_context, "call_data_worker_validator")
    run_dir = _run_dir(tool_context)
    report_path = os.path.join(run_dir, "DATA_VALIDATION_REPORT.json")
    if os.path.exists(report_path):
        return {**result, "data_validation_report_path": report_path}
    log_action(
        run_dir,
        "orchestrator",
        "data_worker_validator_missing_report",
        success=True,
        error="DATA_VALIDATION_REPORT.json was not produced.",
        metadata={"tool_errors": result.get("tool_errors", [])},
    )
    return {**result, "status": "warning", "warning": "DATA_VALIDATION_REPORT.json was not produced."}


@tool(context=True)
async def call_literature_worker(instructions: str, tool_context: object | None = None) -> dict:
    """Run the optional non-blocking literature worker."""
    result = await _run_agent(literature_worker, instructions, tool_context, "call_literature_worker")
    run_dir = _run_dir(tool_context)
    handoff_path = os.path.join(run_dir, "RESEARCH_HANDOFF.json")
    if os.path.exists(handoff_path):
        return {**result, "research_handoff_path": handoff_path}
    log_action(
        run_dir,
        "orchestrator",
        "literature_worker_missing_handoff",
        success=True,
        error="Optional research stage did not produce RESEARCH_HANDOFF.json.",
        metadata={"tool_errors": result.get("tool_errors", [])},
    )
    return {**result, "status": "warning", "warning": "Optional RESEARCH_HANDOFF.json was not produced."}


@tool(context=True)
async def call_judge(instructions: str, reward_lambda: float = 0.5, tool_context: object | None = None) -> dict:
    """Run the judge."""
    prompt = instructions + f"\n\nUse reward_lambda={reward_lambda} when calling write_judge_verdict."
    return await _run_agent(judge, prompt, tool_context, "call_judge")


async def _plan_human_clarification(
    *,
    run_dir: str,
    user_request: str,
    prior_answers_text: str,
    stage: str,
    issue_context: str,
) -> dict[str, Any] | None:
    prompt = "\n".join(
        [
            "Assess whether MAGMA still needs human clarification before proceeding.",
            "Return JSON only with keys: needs_human_input, message, question, reason.",
            "If enough information exists to proceed safely, return needs_human_input=false and leave question empty.",
            "Enough information means a usable data source plus a clear clinical ML objective/target. Workers can inspect files, infer columns when explicitly requested, and report structured errors later.",
            "Do not ask just because details like split strategy, model family, or preprocessing are unspecified unless they are truly required and cannot be inferred from files/request.",
            "If clarification is needed, ask exactly one concise, natural follow-up question in a ChatGPT-like tone.",
            "Do not use a rigid questionnaire. Ask only the next most useful question.",
            json.dumps(
                {
                    "stage": stage,
                    "issue_context": issue_context,
                    "user_request": user_request,
                    "prior_human_answers_text": prior_answers_text,
                },
                indent=2,
            ),
        ]
    )
    planner_context = {"invocation_state": {"run_dir": run_dir}}
    result = await _run_agent(human_clarification_planner, prompt, planner_context, "plan_human_clarification")
    if result.get("status") != "success":
        return None
    return _extract_json_object(result.get("text", ""))


async def plan_initial_human_clarification(
    *,
    run_dir: str,
    user_request: str,
    prior_answers_text: str = "",
) -> dict[str, Any] | None:
    """Ask the HIL planner whether the run needs clarification before orchestration."""
    if os.path.exists(os.path.join(run_dir, "DATA_HANDOFF.json")):
        return {"needs_human_input": False, "message": "", "question": "", "reason": "DATA_HANDOFF.json already exists."}
    if pending_human_input(run_dir):
        return {"needs_human_input": True, "message": "", "question": "", "reason": "A human input request is already pending."}
    return await _plan_human_clarification(
        run_dir=run_dir,
        user_request=user_request,
        prior_answers_text=prior_answers_text,
        stage="orchestrator",
        issue_context="Before starting the pipeline, decide whether the initial request plus prior human answers contain enough information to begin a defensible clinical ML workflow.",
    )


@tool(context=True)
def ask_human_question(question: Any, tool_context: object | None = None) -> dict:
    """Pause the run and ask the human one direct question."""
    run_dir = _run_dir(tool_context)
    payload = question if isinstance(question, dict) else _extract_json_object(str(question or ""))
    if not isinstance(payload, dict):
        payload = {}
    prompt = str(payload.get("question") or question or "").strip()
    if not prompt:
        prompt = "What clarification do you need from the human to continue?"
    message = str(payload.get("message") or "I need one clarification to continue defensibly.").strip()
    error = classify_failure(
        stage="orchestrator",
        owner_stage="human",
        message=message,
        structured_details={
            "error_class": "ambiguity_error",
            "error_subclass": "ambiguous_user_request",
            "recoverability": "human_blocking",
            "retry_scope": "human_input",
            "repair_instruction": prompt,
            "suggested_next_action": "ask_human_question",
            "required_user_input": [
                {
                    "name": "human_decision",
                    "description": prompt,
                    "prompt": prompt,
                }
            ]
        },
    )
    request = write_human_input_request(run_dir, error)
    log_action(
        run_dir,
        "orchestrator",
        "ask_human_question",
        output_refs=[os.path.join(run_dir, "HUMAN_INPUT_REQUEST.json")],
        metadata={"request_id": request.get("request_id")},
    )
    return {"status": "blocked", "human_input_request": request, "error": "Waiting for human input."}


@tool(context=True)
def inspect_run_contract(artifact_dir: str = "./artifacts", tool_context: object | None = None) -> dict:
    """Inspect canonical handoffs and stage errors for orchestration routing."""
    run_dir = _run_dir(tool_context, artifact_dir)
    payload = {}
    for filename in (
        "DATA_HANDOFF.json",
        "DATA_VALIDATION_REPORT.json",
        "RESEARCH_HANDOFF.json",
        "MODEL_HANDOFF.json",
        "MODEL_VALIDATION_REPORT.json",
        "judge_verdict.json",
        "reward.json",
        "DATA_ERROR.json",
        "MODEL_ERROR.json",
    ):
        path = os.path.join(run_dir, filename)
        payload[filename] = read_json(path) if os.path.exists(path) else None
    task_path = os.path.join(run_dir, "task_input.json")
    task = read_json(task_path) if os.path.exists(task_path) else {}
    research_recommended, research_reasons = should_run_literature_stage(
        str(task.get("raw_prompt") or ""),
        payload.get("DATA_HANDOFF.json"),
    )
    research_stale = research_is_stale(payload.get("RESEARCH_HANDOFF.json"), payload.get("DATA_HANDOFF.json"))
    recovery_state = load_recovery_state(run_dir)
    return {
        "status": "success",
        "run_dir": run_dir,
        "contract": payload,
        "research_stage": {
            "recommended": research_recommended,
            "reasons": research_reasons,
            "handoff_present": payload.get("RESEARCH_HANDOFF.json") is not None,
            "stale": research_stale,
        },
        "recovery_state": recovery_state.model_dump(mode="json"),
        "recovery_summary": state_summary(recovery_state),
    }


@tool(context=True)
def decide_repair_route(
    error_json: Any | None = None,
    orchestration_repair_budget: int = 2,
    artifact_dir: str = "./artifacts",
    tool_context: object | None = None,
) -> dict:
    """Classify an error and decide the next repair route with loop/budget protection."""
    run_dir = _run_dir(tool_context, artifact_dir)
    raw_error = _coerce_error_payload(error_json, run_dir)
    if raw_error is None:
        error = classify_failure(
            stage="orchestrator",
            owner_stage="orchestrator",
            message="No structured stage error was available for routing.",
        )
    else:
        try:
            error = StageError(**raw_error)
        except Exception:
            error = classify_failure(
                stage=str(raw_error.get("stage") or "orchestrator"),
                owner_stage=str(raw_error.get("owner_stage") or "orchestrator"),
                message=str(raw_error.get("message") or raw_error.get("error") or "Stage failed"),
                stderr=str(raw_error.get("structured_details") or ""),
                related_artifacts=raw_error.get("related_artifacts") or [],
                structured_details=raw_error,
            )
    append_error(run_dir, error)
    state = load_recovery_state(run_dir)
    decision = decide_recovery(
        error,
        state,
        RetryPolicy(orchestration_repair_budget=orchestration_repair_budget),
    )
    repair_attempt = None
    if decision.status in {"retry", "degraded", "provisional"}:
        repair_attempt = build_repair_attempt(error, decision)
        append_repair_attempt(run_dir, repair_attempt)
    elif decision.status == "terminal":
        state.final_escalation_reason = decision.reason
        state.run_status = "failed"
        save_recovery_state(run_dir, state)
    human_request = None
    if decision.status == "ask_human":
        human_request = write_human_input_request(run_dir, error)
    log_action(
        run_dir,
        "orchestrator",
        "decide_repair_route",
        success=decision.status != "terminal",
        error=None if decision.status != "terminal" else decision.reason,
        metadata={"decision": decision.model_dump(mode="json"), "error": error.model_dump(mode="json")},
    )
    return {
        "status": "success",
        "error": error.model_dump(mode="json"),
        "decision": decision.model_dump(mode="json"),
        "repair_attempt": repair_attempt.model_dump(mode="json") if repair_attempt else None,
        "human_input_request": human_request,
        "recovery_state": load_recovery_state(run_dir).model_dump(mode="json"),
    }


@tool(context=True)
def write_recovery_summary(artifact_dir: str = "./artifacts", tool_context: object | None = None) -> dict:
    """Write recovery_summary.json for final reporting."""
    run_dir = _run_dir(tool_context, artifact_dir)
    state = load_recovery_state(run_dir)
    path = os.path.join(run_dir, "recovery_summary.json")
    write_json(path, {
        "summary": state_summary(state),
        "state": state.model_dump(mode="json"),
    })
    log_action(run_dir, "orchestrator", "write_recovery_summary", output_refs=[path])
    return {"status": "success", "path": path, "summary": state_summary(state)}


@tool(context=True)
def write_terminal_failure_report(
    error_json: Any | None = None,
    artifact_dir: str = "./artifacts",
    tool_context: object | None = None,
) -> dict:
    """Write a final failure report from structured error and recovery history."""
    run_dir = _run_dir(tool_context, artifact_dir)
    raw_error = _coerce_error_payload(error_json, run_dir) or {}
    try:
        error = StageError(**raw_error)
    except Exception:
        error = classify_failure(
            stage=str(raw_error.get("stage") or "orchestrator"),
            owner_stage=str(raw_error.get("owner_stage") or "orchestrator"),
            message=str(raw_error.get("message") or raw_error.get("error") or "Pipeline failed"),
            structured_details=raw_error,
        )
    append_error(run_dir, error)
    state = load_recovery_state(run_dir)
    state.final_escalation_reason = state.final_escalation_reason or error.suggested_next_action or error.message
    state.run_status = "failed"
    save_recovery_state(run_dir, state)
    lines = [
        "# Final Report",
        "",
        "## Outcome",
        "**Status: Failed after exhausting defensible recovery paths.**",
        "",
        "## Terminal Error",
        f"- Stage: `{error.stage}`",
        f"- Owner stage: `{error.owner_stage}`",
        f"- Error class: `{error.error_class}`",
        f"- Error subclass: `{error.error_subclass}`",
        f"- Recoverability: `{error.recoverability}`",
        f"- Root cause hypothesis: {error.root_cause_hypothesis or error.message}",
        "",
        "## Recovery Attempts",
    ]
    if state.repair_history:
        for attempt in state.repair_history:
            lines.append(
                f"- `{attempt.owner_stage}` `{attempt.retry_scope}` `{attempt.error_class}/{attempt.error_subclass}`: {attempt.repair_instruction} [{attempt.outcome}]"
            )
    else:
        lines.append("- No automated repair attempts were recorded.")
    lines.extend([
        "",
        "## Why Recovery Stopped",
        f"- {state.final_escalation_reason or error.suggested_next_action or 'No further structured recovery path was available.'}",
        "",
        "## Partial Artifacts",
    ])
    partials = []
    for filename in ("DATA_HANDOFF.json", "MODEL_HANDOFF.json", "DATA_ERROR.json", "MODEL_ERROR.json", "recovery_state.json"):
        path = os.path.join(run_dir, filename)
        if os.path.exists(path):
            partials.append(filename)
    lines.extend([f"- `{name}`" for name in partials] or ["- No canonical partial artifacts were found."])
    lines.extend([
        "",
        "## Human Input",
        "- Human input may unblock the run if the terminal error is ambiguity-related or references missing raw assets/labels.",
    ])
    report = "\n".join(lines) + "\n"
    path = os.path.join(run_dir, "final_report.md")
    with open(path, "w") as f:
        f.write(report)
    log_action(run_dir, "orchestrator", "write_terminal_failure_report", success=False, error=error.message, output_refs=[path])
    return {"status": "success", "path": path, "markdown": report}


@tool(context=True)
def write_final_report(markdown: str, artifact_dir: str = "./artifacts", tool_context: object | None = None) -> dict:
    """Write final_report.md at run root."""
    run_dir = _run_dir(tool_context, artifact_dir)
    path = os.path.join(run_dir, "final_report.md")
    with open(path, "w") as f:
        f.write(markdown)
    log_action(run_dir, "orchestrator", "write_final_report", output_refs=[path])
    return {"status": "success", "path": path}


def _coerce_error_payload(error_json: Any | None, run_dir: str) -> dict[str, Any] | None:
    if isinstance(error_json, dict):
        return error_json
    if isinstance(error_json, str) and error_json.strip():
        possible_path = error_json.strip()
        if os.path.exists(possible_path):
            return read_json(possible_path)
        run_local_path = os.path.join(run_dir, os.path.basename(possible_path))
        if os.path.exists(run_local_path):
            return read_json(run_local_path)
        try:
            parsed = json.loads(error_json)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return {"stage": "orchestrator", "owner_stage": "orchestrator", "message": error_json}
    for filename in ("MODEL_ERROR.json", "DATA_ERROR.json"):
        path = os.path.join(run_dir, filename)
        if os.path.exists(path):
            return read_json(path)
    return None


def _ensure_stage_outcome(
    *,
    run_dir: str,
    result: dict[str, Any],
    stage: str,
    owner_stage: str,
    handoff_filename: str,
    error_filename: str,
) -> dict[str, Any]:
    handoff_path = os.path.join(run_dir, handoff_filename)
    error_path = os.path.join(run_dir, error_filename)
    if os.path.exists(handoff_path):
        if os.path.exists(error_path) and os.path.getmtime(error_path) > os.path.getmtime(handoff_path):
            _maybe_write_human_input_request_from_error(run_dir, error_path)
            return result
        if os.path.exists(error_path):
            log_action(
                run_dir,
                "orchestrator",
                f"{stage}_stale_error_ignored",
                input_refs=[error_path, handoff_path],
                metadata={
                    "reason": f"{handoff_filename} is newer than {error_filename}",
                    "handoff_mtime": os.path.getmtime(handoff_path),
                    "error_mtime": os.path.getmtime(error_path),
                },
            )
        return result
    if os.path.exists(error_path):
        _maybe_write_human_input_request_from_error(run_dir, error_path)
        return result

    if stage == "model_worker":
        recovered = _try_write_grounded_model_handoff(run_dir, result)
        if recovered.get("status") == "success":
            result = dict(result)
            result["grounded_recovery"] = recovered
            return result

    message = (
        f"Canonical {handoff_filename} is missing and {error_filename} is absent after {stage} execution. "
        "The worker must either produce the canonical handoff or write a structured stage error."
    )
    error = classify_failure(
        stage=stage,
        owner_stage=owner_stage,
        message=message,
        related_artifacts=[],
        structured_details={
            "error_class": "contract_error",
            "error_subclass": "missing_required_artifact",
            "severity": "error",
            "recoverability": "retryable_with_modified_plan",
            "retry_scope": f"{owner_stage}_repair" if owner_stage in {"data", "model"} else "modified_code",
            "root_cause_hypothesis": f"{handoff_filename} was not produced and {error_filename} was absent.",
            "repair_instruction": "Rerun the owning worker with explicit instruction to produce the canonical handoff or a structured error artifact.",
            "suggested_next_action": "rerun_owner_to_produce_handoff_or_error",
            "worker_status": result.get("status"),
            "tool_errors": result.get("tool_errors", []),
            "text_present": bool(result.get("text")),
            "note": "Worker free text is intentionally excluded from recovery evidence. Use artifacts and tool_errors.",
        },
    )
    append_error(run_dir, error)
    write_json(error_path, error.model_dump(mode="json"))
    result = dict(result)
    result.update({
        "status": "error",
        "error": message,
        "structured_error": error.model_dump(mode="json"),
        "error_path": error_path,
    })
    log_action(run_dir, "orchestrator", f"{stage}_missing_contract", success=False, error=message, output_refs=[error_path])
    return result


def _maybe_write_human_input_request_from_error(run_dir: str, error_path: str) -> None:
    """Create a pending human-input request for human-owned stage errors.

    This is intentionally deterministic. The CLI can only pause if
    HUMAN_INPUT_REQUEST.json exists, so human-blocking DATA_ERROR/MODEL_ERROR
    artifacts must not depend on the orchestrator LLM choosing the right routing
    tool call.
    """
    try:
        error = StageError(**read_json(error_path))
    except Exception as exc:
        log_action(
            run_dir,
            "orchestrator",
            "human_input_request_from_error",
            success=False,
            error=f"Could not parse stage error for human input routing: {exc}",
            input_refs=[error_path],
        )
        return
    if error.owner_stage != "human" and error.recoverability != "human_blocking":
        return
    request = write_human_input_request(run_dir, error)
    log_action(
        run_dir,
        "orchestrator",
        "human_input_request_from_error",
        output_refs=[os.path.join(run_dir, "HUMAN_INPUT_REQUEST.json")],
        metadata={"request_id": request.get("request_id"), "source_error": error_path},
    )


def _try_write_grounded_model_handoff(run_dir: str, result: dict[str, Any]) -> dict[str, Any]:
    handoff, errors = build_grounded_model_handoff_from_artifacts(run_dir)
    if errors:
        log_action(
            run_dir,
            "orchestrator",
            "grounded_model_handoff_recovery",
            success=False,
            error="; ".join(errors),
            metadata={"tool_errors": result.get("tool_errors", [])},
        )
        return {"status": "error", "details": errors}
    write_result = write_model_handoff(handoff, tool_context={"invocation_state": {"run_dir": run_dir}})
    success = write_result.get("status") == "success"
    log_action(
        run_dir,
        "orchestrator",
        "grounded_model_handoff_recovery",
        success=success,
        error=None if success else str(write_result.get("error") or write_result.get("details")),
        output_refs=[write_result.get("handoff_path", "")] if success else [],
        metadata={"tool_errors": result.get("tool_errors", []), "recovery_source": "model_artifacts"},
    )
    return write_result


def _extract_tool_errors_from_blocks(
    agent_name: str,
    run_dir: str,
    blocks: list[Any],
    tool_uses: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        tool_result = block.get("toolResult") or block.get("tool_result")
        if not isinstance(tool_result, dict) or tool_result.get("status") != "error":
            continue
        tool_use_id = str(tool_result.get("toolUseId") or "")
        content = tool_result.get("content", [])
        error_text = "\n".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("text")
        ).strip() or "Tool returned error status."
        tool_name = None
        if tool_use_id and tool_use_id in tool_uses:
            tool_name = tool_uses[tool_use_id].get("name")
        payload = {
            "tool_name": tool_name or "unknown_tool",
            "tool_use_id": tool_use_id or None,
            "error": error_text,
            "tool_result": tool_result,
        }
        errors.append(payload)
        _log_tool_error(run_dir, agent_name, payload)
    return errors


def _log_tool_error(run_dir: str, agent_name: str, payload: dict[str, Any]) -> None:
    log_action(
        run_dir,
        agent_name,
        "tool_error",
        success=False,
        error=str(payload.get("error") or "Tool returned error status."),
        metadata=payload,
    )
    path = os.path.join(run_dir, "tool_errors.jsonl")
    with open(path, "a") as f:
        f.write(json.dumps({"agent": agent_name, **payload}, default=str) + "\n")


ORCHESTRATOR_PROMPT = """You are MAGMA v2 Orchestrator.

You receive natural-language clinical ML requests. You coordinate code-first agents. Multimodal is a core capability, not an add-on.

Human-in-the-loop policy:
- Ask the human only when the pipeline cannot make a defensible decision from the prompt/files.
- Examples: unknown target, ambiguous label source, unsafe leakage decision, impossible modality mapping, or missing required files.
- Do not ask for confirmation at routine step boundaries.
- Use ask_human_question to pause the run and ask one direct clarification question.
- After ask_human_question, stop the current turn. Do not call data/model/judge/final-report tools until the runtime collects the human answer and resumes.

Workflow:
1. Produce a concise visible IMPLEMENTATION_PLAN with stages, expected handoffs, human-in-loop triggers, and success criteria. Do not expose private reasoning.
2. Send the full user request and the implementation plan to call_data_worker.
3. Call inspect_run_contract. If DATA_HANDOFF.json exists, call_data_worker_validator before moving on. If DATA_ERROR.json exists or DATA_HANDOFF.json is missing, call decide_repair_route before deciding whether to rerun data, ask human, degrade, or fail.
4. If inspect_run_contract.research_stage.recommended is true and RESEARCH_HANDOFF.json is absent or stale, call_literature_worker. This stage is optional and non-blocking; if it fails or produces thin evidence, continue and let judge warn rather than failing the run.
5. If DATA_HANDOFF.json exists, call_model_worker. Include any RESEARCH_HANDOFF.json context in the model instructions and tell model_worker to cite method_ids when it uses literature-grounded choices.
6. (A3 ablation: model_worker_validator is disabled — call_model_worker uses single-shot execution with no validator.) After MODEL_HANDOFF.json exists, proceed directly to call_judge.
7. call_model_worker deterministically writes MODEL_ERROR.json if the worker returns without MODEL_HANDOFF.json or MODEL_ERROR.json. Call inspect_run_contract. If MODEL_ERROR.json exists, call decide_repair_route and route by decision, error_class, retry_scope, and repair history:
   - path_resolution_error, split_error, leakage_error, join_error -> call_data_worker with the structured error and repair_instruction, then rerun model if DATA_HANDOFF is repaired.
   - training_error -> call_model_worker with fallback baseline instructions.
   - calibration_error -> call_model_worker for calibration/evaluation partial rerun or degraded no-calibration result.
   - metrics_error -> call_model_worker for evaluate/export partial rerun, not full retraining unless artifacts are unusable.
   - ambiguity_error or owner_stage="human" -> use ask_human_question if you can phrase the missing clarification directly; otherwise call decide_repair_route if the structured stage error already captures the human-blocking issue.
   - capability_gap -> attempt simpler supported baseline or write degraded/provisional report when allowed.
8. If MODEL_HANDOFF.json exists, call_judge.
9. Call write_recovery_summary and write final_report.md using write_final_report. If recovery is terminal, use write_terminal_failure_report.

Rules:
- Preserve DATA_HANDOFF and MODEL_HANDOFF as source of truth.
- If the runtime says this is a reward-improvement retry because reward is below threshold, do not simply finalize from existing valid artifacts. Use judge_verdict improvement_plan and judge_retry_signals as the next-trial critique and exploration plan.
- If improvement_plan.owner_stage="model", preserve DATA_HANDOFF when preserve_data_handoff=true and call model_worker with the required_next_actions, required_new_artifacts, and exploration_strategy. Then call model_worker_validator and judge to refresh reward.json.
- For every model run and model-owned retry, require bounded hyperparameter optimization as part of the model contract. The model stage must compare candidates, tune meaningful hyperparameters within a small local budget, select by validation AUROC with validation AUPRC as tie-breaker, evaluate once on test, and write candidate_results.json, best_model_selection.json, hyperparameter_search.json, plus training_curves.csv/json and convergence_report.json when applicable.
- The judge does not provide a retry/no-retry decision. The main runtime controls retry with reward threshold and max attempts. If the judge has no high-confidence action, run bounded exploration: untried repo-declared model families, feature strategies, convergence instrumentation, calibration, or TRIPOD/evaluation artifact completion.
- Low AUROC/AUPRC after a valid DATA_HANDOFF and valid MODEL_HANDOFF is model-owned by default, not a reason to rebuild data.
- Low TRIPODScore is documentation/evaluation-owned by default: ask model_worker to complete missing evaluation artifacts where possible, then judge again.
- RESEARCH_HANDOFF is advisory evidence, not a hard gate. Use it to ground method-family choices for multimodal/SOTA/literature-backed requests, but do not block ordinary baseline runs when it is absent.
- RESEARCH_HANDOFF depends on the final DATA_HANDOFF. If DATA_HANDOFF changes after research is written, inspect_run_contract marks research_stage.stale=true and you must regenerate research before calling model_worker when research_stage.recommended=true.
- Do not invent artifact paths. Refer to the actual handoff paths.
- If a canonical handoff is missing, report that as a failed contract instead of using substitute files.
- Stage errors must be brought back to orchestration through DATA_ERROR.json or MODEL_ERROR.json.
- Worker tool failure is not automatically terminal. If a deterministic tool fails but the worker can still write and execute a flat stage script that produces validated canonical artifacts, route the worker toward that fallback instead of ending the run.
- For data-owned recoverable errors such as path_resolution_error, split_error, and unsupported folder layout, explicitly instruct data_worker to try a materially different free-code fallback before treating the run as terminal.
- Human-blocking decisions are first-class workflow states, not failures. When decide_repair_route returns ask_human, the runtime will pause, collect HUMAN_INPUT_RESPONSE.json, and rerun orchestration. Use those answers instead of repeating the same ambiguity.
- If any tool returns status="blocked" because it needs human input, stop the current turn and let the runtime collect the answer.
- If a worker returns without either its canonical handoff or stage error, treat that as contract_error/missing_required_artifact and route it through decide_repair_route. Do not jump directly to terminal failure.
- Do not use owner_stage alone. Use error_class, error_subclass, recoverability, retry_scope, suggested_next_action, and recovery_state.
- Separate worker-local retry from orchestration retry. Respect local_repair_budget in stage errors and orchestration_repair_budget in decide_repair_route.
- Do not repeat an identical repair instruction after it already failed. If decide_repair_route marks repeated_strategy or terminal, write a failed final report.
- Some failures may produce degraded/provisional completion. This must be explicit in MODEL_HANDOFF, judge verdict, and final_report.
- Keep summaries concise and clinically meaningful.
- Terminal final reports must include failed stage, error_class/error_subclass, attempted repairs, why recovery stopped, partial useful artifacts, and whether human input could unblock the run.
"""

HUMAN_CLARIFICATION_PLANNER_PROMPT = """You are MAGMA v2 Human Clarification Planner.

Your job is to decide whether the current run still needs human clarification before it can proceed safely.
Ask for human input only when the request plus prior human answers still lacks information that is required to begin a defensible clinical ML workflow.
Do not ask if the request already gives a usable data source and a clear modeling objective/target, even if some implementation details can be inferred or checked by workers.
If clarification is needed, ask exactly one natural, concise follow-up question in a ChatGPT-like tone.
Do not use a rigid questionnaire. Ask only the next most useful question.
Return JSON only with keys: needs_human_input, message, question, reason.
"""


human_clarification_planner = Agent(
    name="human_clarification_planner",
    description="Determines whether MAGMA needs human clarification and drafts one natural follow-up question.",
    model=create_agent_model("orchestrator"),
    system_prompt=HUMAN_CLARIFICATION_PLANNER_PROMPT,
    tools=[],
    callback_handler=None,
)


orchestrator = Agent(
    name="orchestrator",
    description="MAGMA v2 clinical AI orchestrator.",
    model=create_agent_model("orchestrator"),
    system_prompt=ORCHESTRATOR_PROMPT,
    tools=[
        call_data_worker,
        call_data_worker_validator,
        call_literature_worker,
        call_model_worker,
        call_judge,
        ask_human_question,
        inspect_run_contract,
        decide_repair_route,
        write_recovery_summary,
        write_terminal_failure_report,
        write_final_report,
    ],
    callback_handler=None,
)
