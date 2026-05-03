"""Internal data-worker validator for MAGMA v2."""

from __future__ import annotations

import json
import os
from typing import Any

from strands import Agent, tool

from magma_v2.model_config import create_agent_model
from magma_v2.runtime.actions import log_action
from magma_v2.runtime.artifacts import run_dir_from, write_json
from magma_v2.runtime.validation import (
    build_data_validation_bundle,
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
def read_data_worker_validation_context(artifact_dir: str = "./artifacts", tool_context: object | None = None) -> dict:
    """Read data-stage validation context and draft audit."""
    run_dir = _run_dir(tool_context, artifact_dir)
    bundle = build_data_validation_bundle(run_dir)
    return {"status": "success", "run_dir": run_dir, "bundle": summarize_validation_bundle(bundle)}


@tool(context=True)
def read_data_worker_validation_artifact(path: str, max_lines: int = 20, artifact_dir: str = "./artifacts", tool_context: object | None = None) -> dict:
    """Read a run-local data artifact safely for validator inspection."""
    run_dir = _run_dir(tool_context, artifact_dir)
    return safe_read_validation_artifact(run_dir, path, default_stage="data", max_lines=max_lines)


@tool(context=True)
def write_data_worker_validation_report(report_json: Any, artifact_dir: str = "./artifacts", tool_context: object | None = None) -> dict:
    """Write DATA_VALIDATION_REPORT.json."""
    run_dir = _run_dir(tool_context, artifact_dir)
    report, error = _parse_json_payload(report_json, "report_json")
    if error:
        return {"status": "error", "error": error}
    assert report is not None
    normalized = normalize_validation_report(report, stage="data")
    baseline = normalize_validation_report(build_data_validation_bundle(run_dir), stage="data")
    if normalized.get("compliance_status") == "failed" and baseline.get("compliance_status") != "failed":
        normalized["status"] = "warning"
        normalized["compliance_status"] = baseline.get("compliance_status") or "warning"
        normalized.setdefault("debug_assessment", {}).setdefault("suggested_repairs", []).append(
            "Validator hard failure was downgraded because the deterministic artifact-based audit did not confirm a compliance failure."
        )
    path = os.path.join(run_dir, "DATA_VALIDATION_REPORT.json")
    write_json(path, normalized)
    log_action(run_dir, "data_worker_validator", "write_data_worker_validation_report", output_refs=[path])
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


DATA_WORKER_VALIDATOR_PROMPT = """You are MAGMA v2 Data-Worker Validator.

You are an internal audit layer for the data worker. You do not prepare data.
You inspect what the data worker actually produced, verify that it matches the
claimed contract, suggest concrete improvements, and help debug failures.

Workflow:
1. Call read_data_worker_validation_context.
2. Inspect additional run-local artifacts with read_data_worker_validation_artifact only when the draft audit is not enough.
3. Produce a concise IMPLEMENTATION_PLAN listing the concrete contradictions, improvements, and debug questions you checked. Do not expose private reasoning.
4. Write DATA_VALIDATION_REPORT.json with three sections:
   - compliance_check
   - improvement_suggestions
   - debug_assessment

Rules:
- Compliance is strict about contradictions between request, artifacts, and DATA_HANDOFF.
- Improvements are advisory. Include concrete ideas when metrics/splits/modalities could be improved.
- Debugging should localize likely root causes and suggested repairs without mutating any artifacts.
- Do not rewrite DATA_HANDOFF or DATA_ERROR.
"""


data_worker_validator = Agent(
    name="data_worker_validator",
    description="Internal compliance/improvement/debug validator for the data worker stage.",
    model=create_agent_model("data_validator"),
    system_prompt=DATA_WORKER_VALIDATOR_PROMPT,
    tools=[
        read_data_worker_validation_context,
        read_data_worker_validation_artifact,
        write_data_worker_validation_report,
    ],
    callback_handler=None,
)
