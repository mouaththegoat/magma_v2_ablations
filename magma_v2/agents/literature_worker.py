"""Safe literature/context worker for MAGMA v2."""

from __future__ import annotations

import json
import os
from typing import Any

from strands import Agent, tool

from magma_v2.model_config import create_agent_model
from magma_v2.runtime.actions import log_action
from magma_v2.runtime.artifacts import read_json, run_dir_from, write_json
from magma_v2.runtime.literature import (
    compact_research_context,
    handoff_fingerprint,
    normalize_research_handoff,
    research_report_markdown,
    search_literature_providers,
)


def _run_dir(tool_context: object | None, artifact_dir: str = "./artifacts") -> str:
    if isinstance(tool_context, dict):
        invocation_state = tool_context.get("invocation_state", {})
    else:
        invocation_state = getattr(tool_context, "invocation_state", {}) if tool_context else {}
    return run_dir_from(invocation_state.get("run_dir") or artifact_dir)


@tool(context=True)
def read_research_context(artifact_dir: str = "./artifacts", tool_context: object | None = None) -> dict:
    """Read bounded research context from canonical task/data artifacts only."""
    run_dir = _run_dir(tool_context, artifact_dir)
    task_path = os.path.join(run_dir, "task_input.json")
    data_path = os.path.join(run_dir, "DATA_HANDOFF.json")
    task = read_json(task_path) if os.path.exists(task_path) else {}
    data = read_json(data_path) if os.path.exists(data_path) else {}
    context = compact_research_context(task, data)
    context["data_handoff_path"] = data_path if os.path.exists(data_path) else None
    context["data_handoff_mtime"] = os.path.getmtime(data_path) if os.path.exists(data_path) else None
    context["data_handoff_fingerprint"] = handoff_fingerprint(data)
    return {"status": "success", "run_dir": run_dir, "context": context}


@tool
def search_literature(query_plan: dict[str, Any], max_results: int = 8) -> dict:
    """Search fixed allowlisted literature providers from a semantic query plan."""
    return search_literature_providers(query_plan, max_results=max_results)


@tool(context=True)
def write_research_handoff(handoff_json: Any, artifact_dir: str = "./artifacts", tool_context: object | None = None) -> dict:
    """Write canonical RESEARCH_HANDOFF.json and research_report.md."""
    run_dir = _run_dir(tool_context, artifact_dir)
    payload, error = _parse_json_payload(handoff_json)
    if error:
        return {"status": "error", "error": error}
    assert payload is not None
    context_result = read_research_context(artifact_dir=artifact_dir, tool_context=tool_context)
    context = (context_result.get("context") if isinstance(context_result, dict) else {}) or {}
    handoff = normalize_research_handoff(payload, context=context)
    handoff_path = os.path.join(run_dir, "RESEARCH_HANDOFF.json")
    report_path = os.path.join(run_dir, "research_report.md")
    write_json(handoff_path, handoff)
    with open(report_path, "w") as f:
        f.write(research_report_markdown(handoff))
    log_action(run_dir, "literature_worker", "write_research_handoff", output_refs=[handoff_path, report_path])
    return {"status": "success", "handoff_path": handoff_path, "report_path": report_path, "handoff": handoff}


def _parse_json_payload(value: Any) -> tuple[dict[str, Any] | None, str | None]:
    if isinstance(value, dict):
        return value, None
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            return None, f"handoff_json must be valid JSON: {exc}"
        if not isinstance(parsed, dict):
            return None, "handoff_json must decode to an object"
        return parsed, None
    return None, "handoff_json must be an object or JSON string"


LITERATURE_WORKER_PROMPT = """You are MAGMA Literature Worker.

You are a safe, non-blocking evidence layer. You do not have shell, curl,
generic file read/write, browser, or arbitrary filesystem tools. Use only the
provided wrapped tools.

Workflow:
1. Call read_research_context.
2. Produce a concise IMPLEMENTATION_PLAN with the task/modality focus and why research is or is not useful. Do not expose private reasoning.
3. If the context is relevant, call search_literature with a compact query_plan. Prefer multiple targeted queries in query_plan.queries for multimodal work. Keep max_results small.
4. Synthesize method_cards, not a generic paper summary.
5. Call write_research_handoff. Always write RESEARCH_HANDOFF.json, even if provider search fails; record evidence_gaps and quality warnings instead of failing the pipeline.

Rules:
- The handoff is advisory and non-blocking.
- Center method_cards, recommendation, evidence_gaps, and limitations.
- Papers support method cards; they are not the main product.
- Method cards must be complete enough to guide modeling: method_family, modalities, summary, missing_modality_handling, imbalance_strategy, evaluation_protocol, calibration, supporting_paper_ids, and evidence_basis.
- Use evidence_basis="retrieved_paper" only when supporting_paper_ids links the method to retrieved papers. Otherwise use evidence_basis="practice_prior" or "unsupported".
- Do not over-prescribe exact hyperparameters. Recommend method families and constraints.
- Include method_id values that model_worker can cite later.
- If evidence is thin, say so explicitly.
- Do not fabricate papers, DOIs, URLs, citation counts, or years. Use provider results or leave fields null.
- Do not claim SOTA. Use cautious wording: evidence-supported, commonly used, plausible baseline, or evidence gap.
- Output strict JSON through write_research_handoff.
"""


literature_worker = Agent(
    name="literature_worker",
    description="Safe evidence/context worker for clinical ML method grounding.",
    model=create_agent_model("literature_worker"),
    system_prompt=LITERATURE_WORKER_PROMPT,
    tools=[
        read_research_context,
        search_literature,
        write_research_handoff,
    ],
    callback_handler=None,
)
