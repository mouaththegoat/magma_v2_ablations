"""Internal model-worker validator for MAGMA v2."""

from __future__ import annotations

import json
import os
from typing import Any

from strands import Agent, tool

from magma_v2.model_config import create_agent_model
from magma_v2.runtime.actions import log_action
from magma_v2.runtime.artifacts import run_dir_from, write_json
from magma_v2.runtime.validation import (
    build_model_validation_bundle,
    normalize_validation_report,
    safe_read_validation_artifact,
    summarize_validation_bundle,
)


def _run_dir(tool_context: object | None, artifact_dir: str = "./artifacts") -> str:
    if isinstance(tool_context, dict):
        invocation_state = tool_context.get("invocation_state", {})
    else:
        invocation_state = getattr(tool_context, "invocation_state", {}) if tool_context else {}
    return run_dir_from(invocation_state.get("run_dir") or artifact_dir)


@tool(context=True)
def read_model_worker_validation_context(artifact_dir: str = "./artifacts", tool_context: object | None = None) -> dict:
    """Read model-stage validation context and draft audit."""
    run_dir = _run_dir(tool_context, artifact_dir)
    bundle = build_model_validation_bundle(run_dir)
    return {"status": "success", "run_dir": run_dir, "bundle": summarize_validation_bundle(bundle)}


@tool(context=True)
def read_model_worker_validation_artifact(path: str, max_lines: int = 20, artifact_dir: str = "./artifacts", tool_context: object | None = None) -> dict:
    """Read a run-local model artifact safely for validator inspection."""
    run_dir = _run_dir(tool_context, artifact_dir)
    return safe_read_validation_artifact(run_dir, path, default_stage="model", max_lines=max_lines)


@tool(context=True)
def write_model_worker_validation_report(report_json: Any, artifact_dir: str = "./artifacts", tool_context: object | None = None) -> dict:
    """Write MODEL_VALIDATION_REPORT.json."""
    run_dir = _run_dir(tool_context, artifact_dir)
    report, error = _parse_json_payload(report_json, "report_json")
    if error:
        return {"status": "error", "error": error}
    assert report is not None
    normalized = normalize_validation_report(report, stage="model")
    path = os.path.join(run_dir, "MODEL_VALIDATION_REPORT.json")
    write_json(path, normalized)
    log_action(run_dir, "model_worker_validator", "write_model_worker_validation_report", output_refs=[path])
    return {"status": "success", "path": path, "report": normalized}


def _parse_json_payload(payload: Any, name: str) -> tuple[dict[str, Any] | None, str | None]:
    if isinstance(payload, dict):
        return payload, None
    if not isinstance(payload, str):
        return None, f"{name} must be a JSON string or object, got {type(payload).__name__}"
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        return None, f"Invalid {name}: {exc}"
    if not isinstance(parsed, dict):
        return None, f"{name} must decode to an object"
    return parsed, None


MODEL_WORKER_VALIDATOR_PROMPT = """You are MAGMA v2 Model-Worker Validator.

You are an internal audit layer for the model worker. You do not train models.
You inspect what the model worker actually produced, verify that it matches the
claimed contract, suggest concrete ways to improve current metrics, and help
debug failures.

Workflow:
1. Call read_model_worker_validation_context.
2. Inspect additional run-local artifacts with read_model_worker_validation_artifact only when the draft audit is not enough.
3. Produce a concise IMPLEMENTATION_PLAN listing the concrete contradictions, metric-improvement ideas, and debug questions you checked. Do not expose private reasoning.
4. Write MODEL_VALIDATION_REPORT.json with three sections:
   - compliance_check
   - improvement_suggestions
   - debug_assessment
   Also include recommended_repair_target and recommended_next_action.

Rules:
- Compliance is strict about contradictions between metrics, predictions, logs, code, and MODEL_HANDOFF.
- Improvements should be concrete and metric-oriented: overfitting reduction, calibration, imbalance handling, ablations, stronger baselines, and thresholding/evaluation improvements.
- If model performance can likely be improved, set recommended_repair_target="model" and recommended_next_action="rerun_model_worker_for_improvement".
- Improvement suggestions must name the specific experiment or artifact to add, such as candidate_results.json, best_model_selection.json, hyperparameter_search.json, calibration.json, subgroup_metrics.json, or error_examples.json.
- Debugging should localize likely root causes and suggested repairs without mutating any artifacts.
- Do not rewrite MODEL_HANDOFF or MODEL_ERROR.
"""


model_worker_validator = Agent(
    name="model_worker_validator",
    description="Internal compliance/improvement/debug validator for the model worker stage.",
    model=create_agent_model("model_validator"),
    system_prompt=MODEL_WORKER_VALIDATOR_PROMPT,
    tools=[
        read_model_worker_validation_context,
        read_model_worker_validation_artifact,
        write_model_worker_validation_report,
    ],
    callback_handler=None,
)
