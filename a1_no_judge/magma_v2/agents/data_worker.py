"""Code-first multimodal data worker for MAGMA v2.

This version keeps the original public tool surface but hardens the parts that
were causing orchestration loops:
- first-pass normalization failures preserve useful error classes;
- fallback success archives stale DATA_ERROR.json;
- DATA_HANDOFF hygiene is normalized before writing;
- produced_files contains data artifacts, not generated code/log/cache files;
- model_ready_data is not auto-filled with split manifests;
- EHR/text directory-root failures are treated as recoverable fallback cases.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import strands_tools.file_read as strands_file_read
from strands import Agent, tool
from strands_tools.shell import shell

from magma_v2.model_config import create_agent_model
from magma_v2.runtime.actions import log_action
from magma_v2.runtime.artifacts import list_files, run_dir_from, stage_dir, write_json, write_stage_readme
from magma_v2.runtime.code_exec import execute_stage_code
from magma_v2.runtime.errors import append_error, classify_failure
from magma_v2.runtime.errors.schema import StageError
from magma_v2.runtime.inspect import inspect_user_prompt
from magma_v2.runtime.normalization import normalize_inputs
from magma_v2.runtime.normalization.contracts import DataHandoff
from magma_v2.runtime.run_file_tools import write_run_file


# =============================================================================
# Runtime helpers
# =============================================================================


def _run_dir(tool_context: object | None, artifact_dir: str = "./artifacts") -> str:
    if isinstance(tool_context, dict):
        invocation_state = tool_context.get("invocation_state", {})
    else:
        invocation_state = getattr(tool_context, "invocation_state", {}) if tool_context else {}
    return run_dir_from(invocation_state.get("run_dir") or artifact_dir)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# =============================================================================
# Public tools
# =============================================================================


@tool(context=True)
def inspect_inputs(user_prompt: str, artifact_dir: str = "./artifacts", tool_context: object | None = None) -> dict:
    """Inspect natural-language prompt paths without assuming modality or split style."""
    run_dir = _run_dir(tool_context, artifact_dir)
    result = inspect_user_prompt(user_prompt)
    log_action(run_dir, "data_worker", "inspect_inputs", output_refs=[], metadata=result)
    return result


@tool(context=True)
def execute_data_code(filename: str, artifact_dir: str = "./artifacts", tool_context: object | None = None) -> dict:
    """Execute one Python entrypoint with cwd set to data/."""
    run_dir = _run_dir(tool_context, artifact_dir)
    result = execute_stage_code(run_dir, "data", filename)

    if result.get("status") == "error":
        error = classify_failure(
            stage="data_worker",
            owner_stage="data",
            message=result.get("error") or result.get("stderr") or "Data code execution failed",
            stderr=result.get("stderr", ""),
            stdout=result.get("stdout", ""),
            related_artifacts=[result.get("log_path", "")] if result.get("log_path") else [],
            structured_details={"exception_type": result.get("exception_type")} if result.get("exception_type") else {},
        )
        append_error(run_dir, error)
        result["structured_error"] = error.model_dump(mode="json")

    log_action(
        run_dir,
        "data_worker",
        "execute_data_code",
        output_refs=[result.get("log_path", "")],
        success=result.get("status") == "success",
        error=result.get("stderr") or result.get("error"),
    )
    return result


@tool(context=True)
def normalize_data_inputs(
    raw_paths: list[str],
    user_request: str,
    target: str | None = None,
    patient_id_column: str | None = None,
    join_keys: list[str] | None = None,
    artifact_dir: str = "./artifacts",
    tool_context: object | None = None,
) -> dict:
    """Normalize raw inputs into a canonical data contract and write DATA_HANDOFF.json.

    The deterministic normalizer is an advisory first pass. If it cannot handle
    the layout, this tool writes a structured DATA_ERROR.json that explicitly
    encourages free-code fallback when the failure is layout/path related.
    """
    run_dir = _run_dir(tool_context, artifact_dir)
    data_dir = stage_dir(run_dir, "data")

    contract = normalize_inputs(
        raw_paths=raw_paths,
        data_dir=data_dir,
        user_request=user_request,
        target=target,
        patient_id_column=patient_id_column,
        join_keys=join_keys or [],
    )

    if contract.status == "error":
        error_payload = _build_normalization_error_payload(
            contract=contract,
            raw_paths=raw_paths,
            user_request=user_request,
        )
        error_payload = _standardize_stage_error(
            error_payload,
            default_stage="data_worker",
            default_owner=error_payload.get("owner_stage") or "data",
        )

        error_path = os.path.join(run_dir, "DATA_ERROR.json")
        write_json(error_path, {**error_payload, "raw_paths": raw_paths})
        _append_stage_error_safely(run_dir, error_payload)

        log_action(
            run_dir,
            "data_worker",
            "normalize_data_inputs",
            success=False,
            error=error_payload.get("message"),
            output_refs=[error_path],
            metadata={
                "normalization_status": "failed_first_pass",
                "error_class": error_payload.get("error_class"),
                "error_subclass": error_payload.get("error_subclass"),
                "recoverability": error_payload.get("recoverability"),
                "raw_paths": raw_paths,
            },
        )
        return {
            "status": "error",
            "error": error_payload.get("message"),
            "error_path": error_path,
            "contract": contract.model_dump(mode="json"),
            "next_action_hint": error_payload.get("suggested_next_action"),
        }

    handoff = contract.to_data_handoff()
    handoff = _normalize_data_handoff(run_dir, handoff)
    handoff = _attach_resolved_error_archive(run_dir, handoff, reason="normalize_data_inputs succeeded")

    handoff_path = os.path.join(run_dir, "DATA_HANDOFF.json")
    descriptions = _build_readme_descriptions(handoff)
    descriptions["normalization_contract.json"] = "Canonical normalization contract."
    descriptions["validator_reports.json"] = "Machine-readable normalization validator reports."
    readme_path = write_stage_readme(data_dir, "Data Directory", descriptions)
    handoff["readme_path"] = readme_path

    validation_warnings = _validate_handoff_paths(run_dir, handoff)
    if validation_warnings:
        handoff.setdefault("handoff_validation_warnings", []).extend(validation_warnings)
        log_action(
            run_dir,
            "data_worker",
            "data_handoff_validation_warnings",
            success=True,
            error="; ".join(validation_warnings),
            output_refs=[handoff_path],
        )

    write_json(handoff_path, handoff)
    log_action(
        run_dir,
        "data_worker",
        "normalize_data_inputs",
        output_refs=[handoff_path, readme_path],
        metadata={"normalized_layout": handoff.get("normalized_layout")},
    )
    return {"status": "success", "handoff_path": handoff_path, "readme_path": readme_path, "handoff": handoff}


@tool(context=True)
def write_data_handoff(handoff_json: Any, artifact_dir: str = "./artifacts", tool_context: object | None = None) -> dict:
    """Write DATA_HANDOFF.json and data/README.md.

    A successful handoff is the authoritative data-stage outcome. If a previous
    DATA_ERROR.json exists from a failed first pass, it is archived as resolved
    so the orchestrator does not keep routing on a stale error.
    """
    run_dir = _run_dir(tool_context, artifact_dir)
    handoff, error = _parse_json_payload(handoff_json, "handoff_json")
    if error:
        return {"status": "error", "error": error}
    assert handoff is not None

    handoff = _normalize_data_handoff(run_dir, handoff)
    handoff = _attach_resolved_error_archive(run_dir, handoff, reason="write_data_handoff succeeded")

    validation_warnings = _validate_handoff_paths(run_dir, handoff)
    if validation_warnings:
        handoff.setdefault("handoff_validation_warnings", []).extend(validation_warnings)
        log_action(
            run_dir,
            "data_worker",
            "data_handoff_validation_warnings",
            success=True,
            error="; ".join(validation_warnings),
        )

    data_dir = stage_dir(run_dir, "data")
    readme_path = write_stage_readme(data_dir, "Data Directory", _build_readme_descriptions(handoff))
    handoff["readme_path"] = readme_path

    handoff_path = os.path.join(run_dir, "DATA_HANDOFF.json")
    write_json(handoff_path, handoff)
    log_action(run_dir, "data_worker", "write_data_handoff", output_refs=[handoff_path, readme_path])
    return {"status": "success", "handoff_path": handoff_path, "readme_path": readme_path, "handoff": handoff}


@tool(context=True)
def write_data_error(error_json: Any, artifact_dir: str = "./artifacts", tool_context: object | None = None) -> dict:
    """Write DATA_ERROR.json when data preparation cannot produce a valid contract."""
    run_dir = _run_dir(tool_context, artifact_dir)
    payload, error = _parse_json_payload(error_json, "error_json")
    if error:
        return {"status": "error", "error": error}
    assert payload is not None

    payload = _standardize_stage_error(payload, default_stage="data_worker", default_owner="data")
    path = os.path.join(run_dir, "DATA_ERROR.json")
    write_json(path, payload)
    _append_stage_error_safely(run_dir, payload)
    log_action(run_dir, "data_worker", "write_data_error", success=False, error=payload.get("message"), output_refs=[path])
    return {"status": "success", "path": path, "error": payload}


# =============================================================================
# Error construction and parsing
# =============================================================================


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


def _build_normalization_error_payload(contract: Any, raw_paths: list[str], user_request: str) -> dict[str, Any]:
    """Build a useful DATA_ERROR for first-pass normalization failures.

    Many normalization failures are recoverable by free-code fallback. Preserve
    the original normalization error when available instead of collapsing to an
    opaque unexpected_internal_error.
    """
    contract_payload = contract.model_dump(mode="json") if hasattr(contract, "model_dump") else {}
    message = str(getattr(contract, "error", None) or "Normalization failed")
    lowered = message.lower()

    # Promote common first-pass directory-layout errors to a useful class.
    looks_like_modality_scanner_miss = (
        "does not contain supported image" in lowered
        or "does not contain supported" in lowered
        or "unsupported" in lowered
        or "no supported" in lowered
    )
    prompt_mentions_tabular_or_text = any(
        token in user_request.lower()
        for token in ("ehr", "csv", "table", "tabular", "notes", "note", "text", "discharge")
    )

    error_class = getattr(contract, "error_class", None)
    error_subclass = getattr(contract, "error_subclass", None)
    owner_stage = getattr(contract, "owner_stage", None) or "data"
    recoverability = getattr(contract, "recoverability", None) or "retryable_with_modified_plan"
    retry_scope = getattr(contract, "retry_scope", None) or "data_repair"

    if looks_like_modality_scanner_miss and prompt_mentions_tabular_or_text:
        error_class = "normalization_layout_error"
        error_subclass = "unsupported_directory_first_pass"
        recoverability = "retryable_with_modified_plan"
        retry_scope = "data_free_code_fallback"
        repair_instruction = (
            "Inspect the raw directory tree for tabular/text assets such as CSV, parquet, json, txt, "
            "or known clinical filenames. For EHR + notes prompts, look for files such as all_stays.csv, "
            "admissions.csv, stays.csv, patients.csv, discharge.csv, notes.csv, radiology.csv, or noteevents.csv. "
            "Then generate a focused data-stage fallback script that writes train/val/test artifacts and DATA_HANDOFF.json."
        )
        suggested_next_action = "inspect_directory_and_run_free_code_fallback"
        root_cause = (
            "The deterministic first-pass normalizer treated a tabular/text dataset root as an unsupported "
            "modality directory. This is recoverable by direct file inspection and a focused fallback script."
        )
    else:
        error_class = error_class or "normalization_error"
        error_subclass = error_subclass or "first_pass_failed"
        repair_instruction = (
            "Inspect raw files and directories, then either run a focused free-code fallback or request human input "
            "if target, label source, join key, or modality semantics are genuinely ambiguous."
        )
        suggested_next_action = "inspect_and_repair_data_normalization"
        root_cause = message

    required_user_input = getattr(contract, "required_user_input", None)
    if required_user_input:
        owner_stage = "human"
        error_class = "ambiguity_error"
        error_subclass = "required_user_input"
        recoverability = "human_blocking"
        retry_scope = "ask_human"
        repair_instruction = "Ask the human only for the missing required information, then resume from data normalization."
        suggested_next_action = "ask_human"

    return {
        "status": "error",
        "stage": "data_worker",
        "owner_stage": owner_stage,
        "message": message,
        "error_class": error_class,
        "error_subclass": error_subclass,
        "severity": "error",
        "recoverability": recoverability,
        "retry_recommended": recoverability != "terminal",
        "retry_scope": retry_scope,
        "attempted_repairs": [],
        "max_repair_budget": 3,
        "root_cause_hypothesis": root_cause,
        "repair_instruction": repair_instruction,
        "suggested_next_action": suggested_next_action,
        "blocking_artifacts": [],
        "related_artifacts": raw_paths,
        "structured_details": {
            "raw_paths": raw_paths,
            "normalization_contract": contract_payload,
            "required_user_input": required_user_input,
            "first_pass_normalizer_failed": True,
        },
    }


def _standardize_stage_error(payload: dict[str, Any], default_stage: str, default_owner: str) -> dict[str, Any]:
    """Return a StageError-shaped payload while preserving useful fields.

    If the local StageError schema rejects newer/extra fields, fall back to the
    project classifier but copy the original error class/subclass back into the
    JSON payload used for DATA_ERROR.json.
    """
    enriched = dict(payload)
    enriched.setdefault("status", "error")
    enriched.setdefault("stage", default_stage)
    enriched.setdefault("owner_stage", default_owner)
    enriched.setdefault("message", enriched.get("error") or "Stage failed")
    enriched.setdefault("error_class", "data_error")
    enriched.setdefault("error_subclass", "unspecified")
    enriched.setdefault("severity", "error")
    enriched.setdefault("recoverability", "retryable_with_modified_plan")
    enriched.setdefault("retry_scope", "data_repair")
    enriched.setdefault("attempted_repairs", [])
    enriched.setdefault("blocking_artifacts", [])
    enriched.setdefault("related_artifacts", [])
    enriched.setdefault("structured_details", {})

    try:
        return StageError(**enriched).model_dump(mode="json")
    except Exception:
        classified = classify_failure(
            stage=str(enriched.get("stage") or default_stage),
            owner_stage=str(enriched.get("owner_stage") or default_owner),
            message=str(enriched.get("message") or enriched.get("error") or "Stage failed"),
            stderr=json.dumps(enriched.get("structured_details", {}), default=str),
            related_artifacts=enriched.get("related_artifacts") or enriched.get("blocking_artifacts") or [],
            structured_details=enriched,
        ).model_dump(mode="json")

        # Preserve the useful original classification in the written artifact.
        for key in (
            "status",
            "error_class",
            "error_subclass",
            "recoverability",
            "retry_scope",
            "repair_instruction",
            "suggested_next_action",
            "root_cause_hypothesis",
            "attempted_repairs",
            "blocking_artifacts",
            "related_artifacts",
            "structured_details",
        ):
            if enriched.get(key) is not None:
                classified[key] = enriched[key]
        return classified


def _append_stage_error_safely(run_dir: str, payload: dict[str, Any]) -> None:
    try:
        append_error(run_dir, StageError(**payload))
    except Exception:
        fallback = classify_failure(
            stage=str(payload.get("stage") or "data_worker"),
            owner_stage=str(payload.get("owner_stage") or "data"),
            message=str(payload.get("message") or payload.get("error") or "Data stage failed"),
            stderr=json.dumps(payload.get("structured_details", {}), default=str),
            related_artifacts=payload.get("related_artifacts") or payload.get("blocking_artifacts") or [],
            structured_details=payload,
        )
        append_error(run_dir, fallback)


# =============================================================================
# DATA_HANDOFF hygiene
# =============================================================================


def _normalize_data_handoff(run_dir: str, handoff: dict[str, Any]) -> dict[str, Any]:
    """Best-effort coercion for DATA_HANDOFF before writing.

    Handoff emission should not fail because of minor type drift, but the result
    should be clean enough for model_worker and validator consumption.
    """
    normalized = dict(handoff)
    warnings = _as_list(normalized.get("handoff_validation_warnings"))
    normalized["handoff_validation_warnings"] = warnings

    data_dir = Path(stage_dir(run_dir, "data")).resolve()

    _normalize_observed_layout(normalized, warnings)
    _normalize_label_spec(normalized, warnings)
    _normalize_common_list_fields(normalized)
    _normalize_produced_files(run_dir, data_dir, normalized)
    _discover_core_split_artifacts(data_dir, normalized, warnings)
    _fix_model_ready_data(run_dir, normalized, warnings)
    _set_safe_defaults(normalized)

    try:
        # If DataHandoff accepts it, use the canonical model dump. If not, keep
        # the cleaned dictionary with warnings; communication should continue.
        normalized = DataHandoff(**normalized).model_dump(mode="json")
    except Exception as exc:
        warnings = _as_list(normalized.get("handoff_validation_warnings"))
        warnings.append(f"DATA_HANDOFF schema coercion still reported: {exc}")
        normalized["handoff_validation_warnings"] = warnings

    return normalized


def _normalize_observed_layout(normalized: dict[str, Any], warnings: list[str]) -> None:
    observed = normalized.get("observed_layout")
    if isinstance(observed, dict):
        normalized.setdefault("path_resolution", {})
        if isinstance(normalized["path_resolution"], dict):
            normalized["path_resolution"].setdefault("observed_layout_details", observed)
        normalized["observed_layout"] = _infer_observed_layout_string(observed)
        warnings.append("observed_layout was converted from object to string; details moved to path_resolution.observed_layout_details.")
    elif observed is None:
        normalized["observed_layout"] = "unknown"
    elif not isinstance(observed, str):
        normalized["observed_layout"] = str(observed)
        warnings.append("observed_layout was coerced to string.")


def _infer_observed_layout_string(observed: dict[str, Any]) -> str:
    keys = {str(k).lower() for k in observed.keys()}
    values = " ".join(str(v).lower() for v in observed.values())
    if {"ehr_source", "notes_source"}.issubset(keys) or ("all_stays" in values and "discharge" in values):
        return "ehr_csv_root_plus_clinical_notes_csv_root"
    if "image" in values or "cxr" in values:
        return "image_manifest_or_directory"
    if "ecg" in values or "signal" in values:
        return "ecg_manifest_or_directory"
    return "structured_observed_layout"


def _normalize_label_spec(normalized: dict[str, Any], warnings: list[str]) -> None:
    label_spec = normalized.get("label_spec")
    if not isinstance(label_spec, dict):
        label_spec = {"source_type": "unknown", "notes": [f"label_spec was coerced from {type(label_spec).__name__}."]}
        normalized["label_spec"] = label_spec
        warnings.append("label_spec was coerced to object.")

    label_spec.setdefault("target_name", normalized.get("target"))
    label_spec.setdefault("task_type", normalized.get("task_type"))

    evidence = label_spec.get("evidence")
    if evidence is None:
        label_spec["evidence"] = []
    elif isinstance(evidence, list):
        cleaned = []
        for item in evidence:
            if isinstance(item, dict):
                cleaned.append(item)
            else:
                cleaned.append({"type": "unspecified", "description": str(item)})
        label_spec["evidence"] = cleaned
    elif isinstance(evidence, dict):
        label_spec["evidence"] = [evidence]
    else:
        label_spec["evidence"] = [{"type": "unspecified", "description": str(evidence)}]
    normalized["label_spec"] = label_spec


def _normalize_common_list_fields(normalized: dict[str, Any]) -> None:
    for key in (
        "modalities",
        "join_keys",
        "leakage_controls",
        "assumptions",
        "limitations",
        "feature_exclusion_columns",
        "normalization_artifacts",
        "validator_reports",
        "unresolved_questions",
    ):
        normalized[key] = _as_list(normalized.get(key))


def _normalize_produced_files(run_dir: str, data_dir: Path, normalized: dict[str, Any]) -> None:
    produced_files = normalized.get("produced_files")
    if not isinstance(produced_files, list):
        produced_files = []

    cleaned: list[dict[str, Any]] = []
    seen: set[str] = set()

    for item in produced_files:
        if isinstance(item, dict):
            path_text = str(item.get("path") or "").strip()
            role = str(item.get("role") or _infer_data_role(path_text))
            purpose = str(item.get("purpose") or "Data-worker output.")
        else:
            path_text = str(item).strip()
            role = _infer_data_role(path_text)
            purpose = "Coerced data artifact."

        if not path_text or not _is_data_artifact_path(path_text):
            continue
        if path_text in seen:
            continue
        seen.add(path_text)
        cleaned.append({"path": path_text, "role": role, "purpose": purpose})

    # Discover data files created under data/, excluding generated code/log/cache.
    existing = list_files(str(data_dir))
    for file_item in existing:
        rel = str(file_item.get("relative_path") or "")
        if not rel or not _is_data_artifact_path(rel):
            continue
        name = Path(rel).name
        if name == "README.md" or rel in seen or name in seen:
            continue
        seen.add(rel)
        cleaned.append({
            "path": rel,
            "role": _infer_data_role(rel),
            "purpose": "Discovered data-stage artifact.",
        })

    normalized["produced_files"] = cleaned


def _discover_core_split_artifacts(data_dir: Path, normalized: dict[str, Any], warnings: list[str]) -> None:
    for field, hints in {
        "train_data": ("train",),
        "val_data": ("val", "valid", "validation"),
        "test_data": ("test",),
    }.items():
        if _stage_reference_exists(data_dir, normalized.get(field)):
            continue
        discovered = _discover_data_file(data_dir, hints, exclude_manifest=True)
        if discovered:
            normalized[field] = discovered
            warnings.append(f"{field} was discovered as {discovered}.")

    # model_ready_data is optional. If absent, prefer a true combined modeling
    # table, not a subject/split manifest.
    if not _stage_reference_exists(data_dir, normalized.get("model_ready_data")):
        discovered = _discover_data_file(data_dir, ("model_ready", "cohort_with_splits", "cohort_joined", "cohort"), exclude_manifest=True)
        if discovered:
            normalized["model_ready_data"] = discovered
            warnings.append(f"model_ready_data was discovered as {discovered}.")
        else:
            normalized.pop("model_ready_data", None)


def _fix_model_ready_data(run_dir: str, normalized: dict[str, Any], warnings: list[str]) -> None:
    value = normalized.get("model_ready_data")
    if not value:
        return
    lower = str(value).lower()
    if "subject" in lower and "manifest" in lower:
        warnings.append(f"model_ready_data={value} looked like a split/audit manifest and was removed.")
        normalized.pop("model_ready_data", None)
        return
    if "split_manifest" in lower or lower.endswith("manifest.csv"):
        warnings.append(f"model_ready_data={value} looked like a manifest rather than model-ready table and was removed.")
        normalized.pop("model_ready_data", None)


def _set_safe_defaults(normalized: dict[str, Any]) -> None:
    normalized.setdefault("observed_layout", "unknown")
    normalized.setdefault("normalized_layout", normalized.get("data_layout") or "unknown")
    normalized.setdefault("prediction_unit", "unknown")
    normalized.setdefault("split_source", "unknown")
    normalized.setdefault("task_type", (normalized.get("label_spec") or {}).get("task_type") if isinstance(normalized.get("label_spec"), dict) else "unknown")
    if not normalized.get("target") and isinstance(normalized.get("label_spec"), dict):
        normalized["target"] = normalized["label_spec"].get("target_name") or normalized["label_spec"].get("column")


def _attach_resolved_error_archive(run_dir: str, handoff: dict[str, Any], reason: str) -> dict[str, Any]:
    error_path = Path(run_dir) / "DATA_ERROR.json"
    if not error_path.exists():
        return handoff

    archive_name = f"DATA_ERROR.resolved.{_utc_stamp()}.json"
    archive_path = Path(run_dir) / archive_name
    try:
        os.replace(error_path, archive_path)
        handoff.setdefault("resolved_errors", [])
        if not isinstance(handoff["resolved_errors"], list):
            handoff["resolved_errors"] = [str(handoff["resolved_errors"])]
        handoff["resolved_errors"].append({
            "source": "DATA_ERROR.json",
            "archived_as": archive_name,
            "resolution": reason,
        })
        log_action(
            run_dir,
            "data_worker",
            "archive_resolved_data_error",
            output_refs=[str(archive_path)],
            metadata={"reason": reason},
        )
    except OSError as exc:
        handoff.setdefault("handoff_validation_warnings", [])
        handoff["handoff_validation_warnings"].append(f"Could not archive stale DATA_ERROR.json after successful handoff: {exc}")
    return handoff


# =============================================================================
# Artifact/path helpers
# =============================================================================


def _build_readme_descriptions(handoff: dict[str, Any]) -> dict[str, str]:
    descriptions: dict[str, str] = {}
    for item in handoff.get("produced_files", []):
        if not isinstance(item, dict):
            continue
        path = str(item.get("path", "")).strip()
        if path and _is_data_artifact_path(path):
            descriptions[path] = str(item.get("purpose") or "Data-worker output.")
    return descriptions


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _is_data_artifact_path(value: str) -> bool:
    if not value:
        return False
    path = Path(value)
    parts = {part.lower() for part in path.parts}
    name = path.name
    suffix = path.suffix.lower()

    if "__pycache__" in parts:
        return False
    if name in {"README.md", "DATA_HANDOFF_PAYLOAD.json"}:
        return False
    if suffix in {".py", ".pyc", ".log"}:
        return False
    return suffix in {".csv", ".parquet", ".json", ".jsonl", ".tsv", ".txt", ".npy", ".npz", ".pkl", ".joblib"}


def _stage_reference_exists(data_dir: Path, value: Any) -> bool:
    if not value:
        return False
    text = str(value)
    if "://" in text:
        return False
    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        return candidate.exists()
    return (data_dir / candidate).exists()


def _discover_data_file(data_dir: Path, hints: tuple[str, ...], exclude_manifest: bool = False) -> str | None:
    if not data_dir.exists():
        return None
    candidates = [path for path in data_dir.iterdir() if path.is_file() and _is_data_artifact_path(path.name)]
    preferred_suffixes = {".csv", ".parquet", ".json", ".jsonl", ".tsv"}
    for hint in hints:
        for path in candidates:
            lower = path.stem.lower()
            if exclude_manifest and "manifest" in lower:
                continue
            if hint in lower and path.suffix.lower() in preferred_suffixes:
                return path.name
    return None


def _infer_data_role(name: str) -> str:
    lower = Path(name).name.lower()
    if lower.startswith("train") or "train" in lower:
        return "train_data"
    if lower.startswith("val") or "valid" in lower or "validation" in lower:
        return "validation_data"
    if lower.startswith("test") or "test" in lower:
        return "test_data"
    if "cohort" in lower:
        return "cohort"
    if "split" in lower and "manifest" in lower:
        return "split_manifest"
    if "manifest" in lower:
        return "manifest"
    if "report" in lower or "validation" in lower:
        return "validation_report"
    if "metric" in lower or "summary" in lower:
        return "summary"
    return "data_artifact"


def _validate_handoff_paths(run_dir: str, handoff: dict[str, Any]) -> list[str]:
    """Communication-only handoff check.

    v2 intentionally does not make schema/path validation a hard gate. Return
    warnings only. The orchestrator requires DATA_HANDOFF.json or DATA_ERROR.json
    to exist, while validators/judge can reason over warnings.
    """
    warnings: list[str] = []
    data_dir = Path(stage_dir(run_dir, "data")).resolve()

    for field in ("train_data", "val_data", "test_data"):
        value = handoff.get(field)
        if value and not _stage_reference_exists(data_dir, value):
            warnings.append(f"{field} path does not exist relative to data/: {value}")

    model_ready = handoff.get("model_ready_data")
    if model_ready and "manifest" in str(model_ready).lower() and "model_ready" not in str(model_ready).lower():
        warnings.append(f"model_ready_data looks like a manifest rather than a modeling table: {model_ready}")

    return warnings


# =============================================================================
# Agent prompt
# =============================================================================


DATA_WORKER_PROMPT = """You are MAGMA v2 Data Worker.

You are code-first and multimodal-native. Clinical inputs may be:
- one table
- train/val/test files
- train/val/test directories
- image folders
- text files or notes tables
- ECG/time-series files
- multiple modalities that need alignment

Core objective:
Produce a canonical root-level DATA_HANDOFF.json or a structured DATA_ERROR.json. A first-pass normalization failure is not terminal when observed files support a defensible free-code fallback.

Private reasoning policy:
Use private step-by-step reasoning internally, but do not expose chain-of-thought. Before acting, privately work through the following checklist:
1. What raw paths did the user provide, and what modality does each path likely contain?
2. What files actually exist under those paths?
3. Which task semantics are explicit: target, label source, prediction unit, join keys, patient/group key, split rule?
4. Which semantics are inferred from strong evidence, and which would require human input?
5. Can deterministic normalization handle this layout, or is focused free-code fallback more appropriate?
6. What artifacts must be produced for model_worker to train safely?
7. What leakage risks must be controlled before writing DATA_HANDOFF?
8. Is any previous DATA_ERROR stale because a later fallback succeeded?

Visible reasoning policy:
Do not print private reasoning. Instead, emit only a concise IMPLEMENTATION_PLAN with:
- raw_paths_observed
- observed_file_evidence
- inferred_modalities
- explicit_task_semantics
- inferred_task_semantics
- unresolved_questions, if any
- normalization_strategy
- fallback_strategy, if needed
- leakage_controls
- expected_outputs

Evidence discipline:
Every inferred target, join key, patient/group identifier, split decision, and modality must be backed by one of:
- explicit user prompt text;
- observed filename/schema evidence;
- existing DATA_HANDOFF / repair request / human response;
- deterministic validator evidence.
If none applies, ask for human input rather than guessing.

Workflow:
1. Call inspect_inputs on the full user prompt.
2. Produce a concise visible IMPLEMENTATION_PLAN explaining which raw paths will be normalized and which hints, if any, are evidence-backed. Do not expose private reasoning.
3. Call normalize_data_inputs with the raw paths from inspect_inputs as an advisory first pass. Only pass target, patient_id_column, or join_keys if the user explicitly supplied them or the evidence is strong.
4. If normalize_data_inputs succeeds, inspect files only if needed using Strands file/shell tools, then stop.
5. If normalize_data_inputs fails, do not immediately stop unless the failure is truly human-blocking. Inspect the error and use Strands file tools to write a focused free-code fallback under data/, then execute it with execute_data_code when a defensible repair can be implemented from observed files.
6. After a free-code fallback succeeds, call write_data_handoff to overwrite/replace any earlier DATA_ERROR.json with the canonical DATA_HANDOFF.json. If the fallback cannot produce a valid handoff, then call write_data_error.

First-pass normalization failure policy:
- Deterministic normalization is the first pass, not the only pass.
- If normalize_data_inputs fails because a directory does not match a built-in image/ECG/modality scanner, do not treat that as terminal.
- First inspect the directory tree for tabular/text assets such as CSV, parquet, json, txt, or known clinical filenames.
- For EHR + notes prompts, look for files named like all_stays.csv, admissions.csv, stays.csv, patients.csv, discharge.csv, notes.csv, radiology.csv, or noteevents.csv before classifying the directory as unsupported.
- If files contain enough information to implement the prompt, write a focused fallback script and continue.

Free-code fallback policy:
- Use free-code fallback for recoverable layout/path/split/format issues, including unfamiliar folder structures, nested image roots, filename-only manifests, unusual split files, unsupported tabular/text roots, and partially supported modalities.
- The fallback may create multiple flat helper modules under data/ plus one entrypoint. Use file_read and write_run_file for ordinary file inspection/creation/rewrites, then execute the declared entrypoint with execute_data_code so MAGMA captures logs and structured errors.
- Use Strands shell for exploratory commands only when direct file tools are insufficient. Keep generated pipeline execution through execute_data_code whenever possible so failures are recoverable and logged.
- Fallback code may inspect raw files/directories directly and write new flat artifacts under data/.
- The fallback script must not create a substitute handoff. It should write a payload only if needed, then call write_data_handoff with that payload. If no defensible contract can be produced, call write_data_error.
- If normalize_data_inputs already wrote DATA_ERROR.json and fallback later succeeds, success wins: write DATA_HANDOFF.json. The tool will archive the earlier DATA_ERROR as resolved.
- Do not use free-code fallback to guess missing clinical semantics. Unknown target, label source, or join key still requires human input.

Recovery policy:
- Distinguish local recoverable failures from upstream/human-blocking failures.
- Local repair budget is 3 targeted attempts per data-stage issue.
- Do not retry blindly. Each repair must change the strategy and be described in attempted_repairs.
- path_resolution_error: first inspect the folder tree and try free-code path resolution against observed roots; ask human only if raw asset location is genuinely unknown.
- split_error: retry split generation with a fallback defensible strategy, preserving patient/group leakage controls when available.
- leakage_error: repair prepared data/feature_exclusion_columns and regenerate DATA_HANDOFF; do not continue with leaked model-ready data.
- join_error: repair only if join-key evidence is present; otherwise emit ambiguity_error with owner_stage="human".
- ambiguity_error/missing_target: do not brute force. Emit structured DATA_ERROR requiring human input.
- generated code failure in a fallback/repair script: patch/regenerate locally before escalation, but avoid repeating the same failed strategy.
- DATA_ERROR.json must use structured fields: error_class, error_subclass, severity, recoverability, retry_scope, attempted_repairs, root_cause_hypothesis, repair_instruction, suggested_next_action, blocking_artifacts, related_artifacts, structured_details.

Rules:
- All generated code files and data outputs must be flat under data/. Multiple code files are allowed and preferred for non-trivial work; nested generated code directories are not.
- If multiple code files are generated, declare one entrypoint such as run_data.py and import helper modules from the same directory.
- Generated code runs with cwd=data/. Use Path.cwd() for outputs.
- When using Strands generic file tools, write only inside the active run directory unless reading user-provided raw input paths. Do not write to the project root or global data folders.
- Never hardcode artifact paths unless the user provided the raw path explicitly or it is discovered from inspected files. Generated outputs should still use Path.cwd().
- Never invent a target, label source, patient/group identifier, modality, or split strategy. Use the prompt plus observed files/directories. If the evidence is not strong enough, ask the human.
- If resumed request includes explicit human answers for a previously ambiguous target, label source, join key, or split decision, treat those answers as authoritative for this run and do not re-ask the same question.
- Core parsing/normalization helpers live in magma_v2.runtime.normalization, but they are helpers, not a cage. If they do not fit a multimodal layout, write focused code that handles the observed data directly.
- If no split exists, create train/val/test. If patient/group ID exists, use group-aware splitting.
- If train/val/test directories or files exist, preserve them.
- For multimodal data, create an alignment manifest and document join keys.
- Do not create substitute handoffs like DATA_HANDOFF_PREP.json. The canonical root-level DATA_HANDOFF.json is required.
- Before calling write_data_handoff, list/check files if practical, but missing optional artifacts should be documented rather than blocking communication.
- If orchestrator sends a repair request, update the generated preparation script and regenerate the affected data files and DATA_HANDOFF.json.
- Data outputs must not contain duplicate target helper columns as usable features. If helper label columns are needed internally, drop them from model-ready outputs or explicitly list them in feature_exclusion_columns.
- produced_files must be discovered from the actual generated data outputs. Do not include generated code, logs, __pycache__, pyc files, or temporary handoff payload files in produced_files.

DATA_HANDOFF schema hygiene:
- observed_layout must be a string.
- Detailed source path objects belong in path_resolution or schema_summary, not observed_layout.
- label_spec.evidence must be a list of objects, not strings.
- produced_files should contain data artifacts only, not generated code, logs, __pycache__, or temporary handoff payload files.
- model_ready_data must not point to a split manifest. Use train_data/val_data/test_data for split-based modeling, or use cohort_with_splits/model_ready only when that file is actually the modeling table.
- Prefer paths relative to data/ for generated data artifacts. If an artifact references raw external assets, make the path absolute and document it in path_resolution or schema_summary.
- The handoff must be valid JSON. Prefer these fields when known, but do not force fake values:
  modality/modalities, task_type, data_layout/observed_layout/normalized_layout, split_strategy/split_source,
  train_data, val_data, test_data, model_ready_data, target/label_spec, patient_id_column, join_keys,
  leakage_controls, assumptions, limitations, feature_exclusion_columns, produced_files, path_resolution,
  schema_summary, normalization_artifacts, validator_reports, unresolved_questions, confidence_summary.

Stop and ask for human input only when the target, label source, join key, split decision, or modality cannot be inferred from prompt/files.
"""


data_worker = Agent(
    name="data_worker",
    description="Code-first multimodal clinical data worker.",
    model=create_agent_model("data_worker"),
    system_prompt=DATA_WORKER_PROMPT,
    tools=[
        strands_file_read,
        write_run_file,
        shell,
        inspect_inputs,
        normalize_data_inputs,
        execute_data_code,
        write_data_handoff,
        write_data_error,
    ],
    callback_handler=None,
)