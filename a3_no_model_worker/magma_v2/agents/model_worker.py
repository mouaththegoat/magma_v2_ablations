"""Code-first adaptive model worker for MAGMA v2."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import strands_tools.file_read as strands_file_read
from strands import Agent, tool
from strands_tools.shell import shell

from magma_v2.model_config import create_agent_model
from magma_v2.runtime.actions import log_action
from magma_v2.runtime.artifacts import list_files, read_json, run_dir_from, stage_dir, write_json, write_stage_readme
from magma_v2.runtime.code_exec import execute_stage_code
from magma_v2.runtime.errors import append_error, classify_failure
from magma_v2.runtime.errors.schema import StageError
from magma_v2.runtime.literature import research_is_stale
from magma_v2.runtime.normalization.contracts import DataHandoff
from magma_v2.runtime.run_file_tools import write_run_file


def _run_dir(tool_context: object | None, artifact_dir: str = "./artifacts") -> str:
    if isinstance(tool_context, dict):
        invocation_state = tool_context.get("invocation_state", {})
    else:
        invocation_state = getattr(tool_context, "invocation_state", {}) if tool_context else {}
    return run_dir_from(invocation_state.get("run_dir") or artifact_dir)


@tool(context=True)
def read_data_handoff(artifact_dir: str = "./artifacts", tool_context: object | None = None) -> dict:
    """Read DATA_HANDOFF.json."""
    run_dir = _run_dir(tool_context, artifact_dir)
    path = os.path.join(run_dir, "DATA_HANDOFF.json")
    if not os.path.exists(path):
        return {"status": "error", "error": f"Missing DATA_HANDOFF.json at {path}"}
    return {"status": "success", "path": path, "handoff": read_json(path)}


@tool(context=True)
def get_normalized_model_plan(artifact_dir: str = "./artifacts", tool_context: object | None = None) -> dict:
    """Return best-effort model input guidance from DATA_HANDOFF.

    This is intentionally advisory. It should help generated code resolve
    run-local paths and understand available semantics, not reject flexible
    multimodal handoffs before the model worker can adapt.
    """
    run_dir = _run_dir(tool_context, artifact_dir)
    path = os.path.join(run_dir, "DATA_HANDOFF.json")
    if not os.path.exists(path):
        return {"status": "error", "error": f"Missing DATA_HANDOFF.json at {path}"}
    handoff = read_json(path)
    research_path = os.path.join(run_dir, "RESEARCH_HANDOFF.json")
    research_handoff = read_json(research_path) if os.path.exists(research_path) else None
    if research_is_stale(research_handoff, handoff):
        research_handoff = {
            "status": "stale",
            "quality_gate": {"status": "stale", "warnings": ["RESEARCH_HANDOFF was generated from an older DATA_HANDOFF and must not ground model choices."]},
        }
    try:
        DataHandoff(**handoff)
    except Exception as exc:
        handoff.setdefault("handoff_warnings", []).append(f"Best-effort DATA_HANDOFF parsing warning: {exc}")

    normalized_layout = _canonical_normalized_layout(run_dir, handoff)
    guidance = {
        "tabular_dataset": {
            "input_kind": "tabular_splits",
            "train_from": ["train_data", "val_data", "test_data"],
            "required_columns": ["label"],
            "strategy": "train tabular baselines from split files",
        },
        "text_manifest": {
            "input_kind": "text_manifest",
            "train_from": ["train_data", "val_data", "test_data"],
            "required_columns": ["sample_id", "label", "split"],
            "strategy": "train TF-IDF baseline from text or text_path columns",
        },
        "image_manifest": {
            "input_kind": "image_manifest",
            "train_from": ["train_data", "val_data", "test_data"],
            "required_columns": ["sample_id", "image_path", "label", "split"],
            "strategy": "consume image_path manifests; train from image pixels with an installed CNN/vision baseline when feasible; never train on filenames or identifiers",
        },
        "ecg_manifest": {
            "input_kind": "ecg_manifest",
            "train_from": ["train_data", "val_data", "test_data"],
            "required_columns": ["sample_id", "label", "split"],
            "strategy": "consume signal_path/ecg_path manifests; train ECG baseline only when feasible, otherwise document limitation",
        },
        "multimodal_alignment_manifest": {
            "input_kind": "multimodal_alignment_manifest",
            "train_from": ["train_data", "val_data", "test_data", "model_ready_data"],
            "required_columns": ["sample_id", "label", "split"],
            "strategy": "train simple fusion-ready or modality-specific baselines and document missing modality handling",
        },
    }.get(normalized_layout, {
        "input_kind": "adaptive_multimodal",
        "train_from": ["train_data", "val_data", "test_data", "model_ready_data"],
        "required_columns": [],
        "strategy": "inspect the provided handoff paths and run-local data/model artifacts, then choose a defensible code path for the observed modality mix",
    })
    model_dir = Path(stage_dir(run_dir, "model")).resolve()
    data_dir = Path(stage_dir(run_dir, "data")).resolve()
    relative_paths = {
        "train_data": handoff.get("train_data"),
        "val_data": handoff.get("val_data"),
        "test_data": handoff.get("test_data"),
        "model_ready_data": handoff.get("model_ready_data"),
    }
    absolute_paths = {
        key: str(path.resolve())
        for key, value in relative_paths.items()
        for path in [_resolve_data_reference(run_dir, value)]
        if path is not None
    }
    path_contract = {
        "script_cwd": str(model_dir),
        "data_dir": str(data_dir),
        "relative_data_dir_from_model_cwd": "../data",
        "rule": "Model scripts run from model/. Prepared data is stored in sibling ../data/, not in model/ and not in the project root. Use absolute_paths when available; otherwise resolve handoff-relative data paths against data_dir.",
        "example": "contract = json.load(open(Path.cwd() / 'MODEL_INPUT_CONTRACT.json')); train_path = Path(contract['absolute_paths']['train_data'])",
    }
    plan = {
        "status": "success",
        "run_dir": str(Path(run_dir).resolve()),
        "model_dir": str(model_dir),
        "data_dir": str(data_dir),
        "handoff_path": str(Path(path).resolve()),
        "observed_layout": handoff.get("observed_layout"),
        "normalized_layout": normalized_layout,
        "prediction_unit": handoff.get("prediction_unit"),
        "split_source": handoff.get("split_source"),
        "guidance": guidance,
        "target": handoff.get("target"),
        "label_spec": handoff.get("label_spec", {}),
        "schema_summary": handoff.get("schema_summary", {}),
        "path_resolution": handoff.get("path_resolution", {}),
        "confidence_summary": handoff.get("confidence_summary", {}),
        "unresolved_questions": handoff.get("unresolved_questions", []),
        "advisory_warnings": handoff.get("handoff_warnings", []),
        "patient_id_column": handoff.get("patient_id_column"),
        "join_keys": handoff.get("join_keys", []),
        "feature_exclusion_columns": handoff.get("feature_exclusion_columns", []),
        "paths": relative_paths,
        "absolute_paths": absolute_paths,
        "path_contract": path_contract,
        "validator_reports": handoff.get("validator_reports", []),
        "normalization_artifacts": handoff.get("normalization_artifacts", []),
        "research_handoff_path": research_path if research_handoff else None,
        "research_context": _compact_research_for_model(research_handoff),
    }
    contract_path = model_dir / "MODEL_INPUT_CONTRACT.json"
    write_json(str(contract_path), plan)
    plan["model_input_contract_path"] = str(contract_path)
    return plan


@tool(context=True)
def validate_model_input_contract(artifact_dir: str = "./artifacts", tool_context: object | None = None) -> dict:
    """Preview resolved model inputs before code generation.

    The result is advisory. It writes schema/path observations for the LLM, but
    it no longer blocks modeling because MAGMA v2 must adapt to multimodal and
    partially specified layouts.
    """
    plan = get_normalized_model_plan(artifact_dir=artifact_dir, tool_context=tool_context)
    if plan.get("status") != "success":
        return plan
    checks: dict[str, Any] = {}
    warnings: list[str] = []
    target = plan.get("target")
    exclusions = set(plan.get("feature_exclusion_columns", []))
    for key, value in plan.get("absolute_paths", {}).items():
        path = Path(value)
        item: dict[str, Any] = {"path": str(path), "exists": path.exists()}
        if not path.exists():
            warnings.append(f"{key} does not exist: {path}")
            checks[key] = item
            continue
        try:
            frame = pd.read_csv(path, nrows=200)
            item["columns"] = list(frame.columns)
            item["row_preview_count"] = int(len(frame))
            item["dtypes"] = {str(column): str(dtype) for column, dtype in frame.dtypes.items()}
            item["target_present"] = bool(target in frame.columns) if target else False
            item["feature_exclusion_columns_present"] = sorted(column for column in exclusions if column in frame.columns)
            if target and target not in frame.columns:
                warnings.append(f"{key} missing target column {target}: {path}")
        except Exception as exc:
            item["read_error"] = str(exc)
            warnings.append(f"{key} could not be read for schema preview: {path}: {exc}")
        checks[key] = item
    result = {
        "status": "success",
        "warnings": warnings,
        "errors": [],
        "checks": checks,
        "path_contract": plan.get("path_contract"),
        "model_input_contract_path": plan.get("model_input_contract_path"),
    }
    run_dir = _run_dir(tool_context, artifact_dir)
    output_path = Path(stage_dir(run_dir, "model")) / "model_input_validation.json"
    write_json(str(output_path), result)
    result["validation_path"] = str(output_path)
    return result


@tool(context=True)
def execute_model_code(filename: str, artifact_dir: str = "./artifacts", tool_context: object | None = None) -> dict:
    """Execute one Python entrypoint with cwd set to model/."""
    run_dir = _run_dir(tool_context, artifact_dir)
    result = execute_stage_code(run_dir, "model", filename, timeout_seconds=1200)
    if result.get("status") == "error":
        error = classify_failure(
            stage="model_worker",
            message=result.get("error") or result.get("stderr") or "Model code execution failed",
            stderr=result.get("stderr", ""),
            stdout=result.get("stdout", ""),
            related_artifacts=[result.get("log_path", "")] if result.get("log_path") else [],
            structured_details={"exception_type": result.get("exception_type")} if result.get("exception_type") else {},
        )
        append_error(run_dir, error)
        result["structured_error"] = error.model_dump(mode="json")
    log_action(run_dir, "model_worker", "execute_model_code", output_refs=[result.get("log_path", "")], success=result["status"] == "success", error=result.get("stderr") or result.get("error"))
    return result


@tool(context=True)
def write_model_handoff(handoff_json: Any, artifact_dir: str = "./artifacts", tool_context: object | None = None) -> dict:
    """Write MODEL_HANDOFF.json and model/README.md."""
    run_dir = _run_dir(tool_context, artifact_dir)
    handoff, error = _parse_json_payload(handoff_json, "handoff_json")
    if error:
        return {"status": "error", "error": error}
    assert handoff is not None
    handoff = _normalize_model_handoff(run_dir, handoff)
    validation_errors = _validate_model_handoff_paths(run_dir, handoff)
    if validation_errors:
        handoff.setdefault("handoff_validation_warnings", []).extend(validation_errors)
        log_action(
            run_dir,
            "model_worker",
            "model_handoff_validation_warnings",
            success=True,
            error="; ".join(validation_errors),
        )
    model_dir = stage_dir(run_dir, "model")
    descriptions: dict[str, str] = {}
    for item in handoff.get("produced_files", []):
        path = str(item.get("path", ""))
        if path:
            descriptions[os.path.basename(path)] = item.get("purpose", "Model-worker output.")
    readme_path = write_stage_readme(model_dir, "Model Directory", descriptions)
    handoff["readme_path"] = readme_path
    handoff_path = os.path.join(run_dir, "MODEL_HANDOFF.json")
    write_json(handoff_path, handoff)
    log_action(run_dir, "model_worker", "write_model_handoff", output_refs=[handoff_path, readme_path])
    return {"status": "success", "handoff_path": handoff_path, "readme_path": readme_path, "handoff": handoff}


@tool(context=True)
def write_grounded_model_handoff(artifact_dir: str = "./artifacts", tool_context: object | None = None) -> dict:
    """Write MODEL_HANDOFF.json deterministically from existing model artifacts."""
    run_dir = _run_dir(tool_context, artifact_dir)
    handoff, errors = build_grounded_model_handoff_from_artifacts(run_dir)
    if errors:
        return {"status": "error", "error": "Could not build grounded MODEL_HANDOFF.json", "details": errors}
    return write_model_handoff(handoff, artifact_dir=artifact_dir, tool_context=tool_context)


@tool(context=True)
def write_model_error(error_json: Any, artifact_dir: str = "./artifacts", tool_context: object | None = None) -> dict:
    """Write MODEL_ERROR.json for orchestration-level repair routing."""
    run_dir = _run_dir(tool_context, artifact_dir)
    payload, error = _parse_json_payload(error_json, "error_json")
    if error:
        return {"status": "error", "error": error}
    assert payload is not None
    payload = _standardize_stage_error(payload, default_stage="model_worker", default_owner="model")
    path = os.path.join(run_dir, "MODEL_ERROR.json")
    write_json(path, payload)
    append_error(run_dir, StageError(**payload))
    log_action(run_dir, "model_worker", "write_model_error", success=False, error=payload.get("message"), output_refs=[path])
    return {"status": "success", "path": path, "error": payload}


def build_grounded_model_handoff_from_artifacts(run_dir: str) -> tuple[dict[str, Any], list[str]]:
    """Build a conservative MODEL_HANDOFF from files already written under model/.

    This is intentionally deterministic. It is used when the LLM completed model
    execution but failed the handoff tool call, so recovery does not depend on
    stale worker prose.
    """
    errors: list[str] = []
    model_dir = Path(stage_dir(run_dir, "model")).resolve()
    data_handoff_path = Path(run_dir) / "DATA_HANDOFF.json"
    data_handoff: dict[str, Any] = {}
    if data_handoff_path.exists():
        try:
            data_handoff = read_json(str(data_handoff_path))
        except Exception as exc:
            errors.append(f"Could not read DATA_HANDOFF.json: {exc}")

    artifact_index = _build_model_artifact_index(model_dir)
    artifact_index_path = model_dir / "artifact_index.json"
    write_json(str(artifact_index_path), artifact_index)
    model_path = _largest_path_from_index(artifact_index, "serialized_model")
    code_path = _largest_path_from_index(artifact_index, "python_code")
    log_path = _largest_path_from_index(artifact_index, "text_log")
    convergence_path = _path_from_index_name(artifact_index, "convergence_report.json")
    model_selection_diagnostics_path = _path_from_index_name(artifact_index, "model_selection_diagnostics.json")
    training_curves_path = (
        _path_from_index_name(artifact_index, "training_curves.csv")
        or _path_from_index_name(artifact_index, "training_curves.json")
    )
    if model_path is None and not artifact_index["json_artifacts"] and not artifact_index["table_artifacts"]:
        errors.append("No usable model, JSON, or tabular artifacts found under model/. Grounded handoff recovery is not defensible.")

    produced_files = [
        {
            "path": item["path"],
            "role": item["kind"],
            "purpose": "Discovered model-stage artifact from file type and schema preview.",
        }
        for item in artifact_index["files"]
    ]

    selected_model = model_path.stem if model_path is not None else "unknown"

    handoff = {
        "model_type": selected_model or "unknown",
        "modality_strategy": _modality_strategy_from_data_handoff(data_handoff),
        "model_path": str(model_path) if model_path is not None else "",
        "train_data": data_handoff.get("train_data"),
        "val_data": data_handoff.get("val_data"),
        "test_data": data_handoff.get("test_data"),
        "metrics": {
            "source": "artifact_index.json",
            "payload": {
                "status": "artifact_indexed",
                "json_artifacts": artifact_index["json_artifacts"],
                "table_artifacts": artifact_index["table_artifacts"],
                "note": "Grounded recovery does not infer metric semantics from filenames or key names.",
            },
        },
        "hyperparameters": {},
        "training_summary": {
            "status": "completed",
            "source": "grounded_handoff_recovery",
            "observed_layout": data_handoff.get("observed_layout"),
            "normalized_layout": data_handoff.get("normalized_layout"),
            "prediction_unit": data_handoff.get("prediction_unit"),
            "target": data_handoff.get("target"),
            "label_source": (data_handoff.get("label_spec") or {}).get("source_name"),
            "split_source": data_handoff.get("split_source"),
            "feature_exclusions": data_handoff.get("feature_exclusion_columns", []),
            "notes": [
                "MODEL_HANDOFF.json was rebuilt deterministically from on-disk artifacts after handoff emission failed.",
                "Artifacts were indexed by type/schema. Metric semantics were not inferred from filenames.",
            ],
        },
        "produced_files": produced_files,
        "code_artifacts": {
            "training_code_path": str(code_path or "") or None,
            "training_log_path": str(log_path or "") or None,
        },
        "evaluation_artifacts": {
            "artifact_index_path": str(artifact_index_path),
            "table_artifacts": [item["path"] for item in artifact_index["table_artifacts"]],
            "json_artifacts": [item["path"] for item in artifact_index["json_artifacts"]],
            "predictions_unavailable_reason": "No artifact was explicitly declared as predictions during grounded recovery.",
            "metrics_unavailable_reason": "No artifact was explicitly declared as metrics during grounded recovery.",
            "calibration_unavailable_reason": "No artifact was explicitly declared as calibration during grounded recovery.",
            "subgroup_unavailable_reason": "No artifact was explicitly declared as subgroup metrics during grounded recovery.",
            "error_examples_unavailable_reason": "No artifact was explicitly declared as error examples during grounded recovery.",
            "leakage_report_unavailable_reason": "No artifact was explicitly declared as leakage report during grounded recovery.",
            "training_log_path": str(log_path or "") or None,
            **({"convergence_report_path": str(convergence_path)} if convergence_path else {}),
            **({"model_selection_diagnostics_path": str(model_selection_diagnostics_path)} if model_selection_diagnostics_path else {}),
            **({"training_curves_path": str(training_curves_path)} if training_curves_path else {}),
            **({} if convergence_path else {"convergence_unavailable_reason": "No convergence_report.json was discovered during grounded recovery."}),
        },
        "model_search_summary": {
            **({"model_selection_diagnostics_path": str(model_selection_diagnostics_path)} if model_selection_diagnostics_path else {}),
        },
        "audit_summary": {
            "status": "grounded_recovery",
            "checked": ["artifact existence", "file type inventory", "schema previews"],
        },
        "artifact_completeness": {
            item["path"]: Path(item["path"]).exists()
            for item in produced_files
        },
        "model_output_schema": {"artifact_index": artifact_index},
        "calibration_status": "unavailable",
        "subgroup_eval_status": "unavailable",
        "code_generation_mode": "generated_multi_file" if len(artifact_index["code_artifacts"]) > 1 else ("generated_script" if code_path else "hybrid"),
    }
    return handoff, errors


def _build_model_artifact_index(model_dir: Path) -> dict[str, Any]:
    files = []
    for path in sorted(model_dir.rglob("*")):
        if not path.is_file() or path.name[:1] == ".":
            continue
        kind = _artifact_kind(path)
        item: dict[str, Any] = {
            "path": str(path),
            "relative_path": str(path.relative_to(model_dir)),
            "suffix": path.suffix.lower(),
            "size_bytes": path.stat().st_size,
            "kind": kind,
        }
        if kind == "table":
            item.update(_table_preview(path))
        elif kind == "json":
            item.update(_json_preview(path))
        files.append(item)
    return {
        "files": files,
        "serialized_model_artifacts": [item for item in files if item["kind"] == "serialized_model"],
        "table_artifacts": [item for item in files if item["kind"] == "table"],
        "json_artifacts": [item for item in files if item["kind"] == "json"],
        "code_artifacts": [item for item in files if item["kind"] == "python_code"],
        "text_artifacts": [item for item in files if item["kind"] == "text_log"],
    }


def _artifact_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".pkl", ".joblib", ".pt", ".pth", ".onnx", ".safetensors"}:
        return "serialized_model"
    if suffix in {".csv", ".tsv", ".parquet", ".jsonl"}:
        return "table"
    if suffix == ".json":
        return "json"
    if suffix == ".py":
        return "python_code"
    if suffix in {".log", ".txt", ".md"}:
        return "text_log"
    return "artifact"


def _table_preview(path: Path) -> dict[str, Any]:
    try:
        if path.suffix.lower() == ".parquet":
            frame = pd.read_parquet(path).head(20)
        elif path.suffix.lower() == ".tsv":
            frame = pd.read_csv(path, sep="\t", nrows=20)
        elif path.suffix.lower() == ".jsonl":
            frame = pd.read_json(path, lines=True, nrows=20)
        else:
            frame = pd.read_csv(path, nrows=20)
        return {
            "columns": [str(column) for column in frame.columns],
            "preview_rows": int(len(frame)),
            "dtypes": {str(column): str(dtype) for column, dtype in frame.dtypes.items()},
        }
    except Exception as exc:
        return {"preview_error": str(exc)}


def _json_preview(path: Path) -> dict[str, Any]:
    try:
        payload = read_json(str(path))
    except Exception as exc:
        return {"preview_error": str(exc)}
    if isinstance(payload, dict):
        return {
            "top_level_type": "object",
            "top_level_keys": [str(key) for key in list(payload.keys())[:50]],
        }
    if isinstance(payload, list):
        return {
            "top_level_type": "array",
            "top_level_length": len(payload),
            "first_item_type": type(payload[0]).__name__ if payload else None,
        }
    return {"top_level_type": type(payload).__name__}


def _largest_path_from_index(index: dict[str, Any], kind: str) -> Path | None:
    candidates = [Path(item["path"]) for item in index.get("files", []) if item.get("kind") == kind]
    return max(candidates, key=lambda path: path.stat().st_size, default=None)


def _path_from_index_name(index: dict[str, Any], filename: str) -> Path | None:
    for item in index.get("files", []):
        path = Path(item["path"])
        if path.name == filename:
            return path
    return None


def _find_nested(payload: dict[str, Any], path: list[str]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _modality_strategy_from_data_handoff(data_handoff: dict[str, Any]) -> str:
    layout = data_handoff.get("normalized_layout") or data_handoff.get("data_layout") or "unknown_layout"
    modalities = data_handoff.get("modalities") or data_handoff.get("modality") or "unknown_modality"
    task_type = data_handoff.get("task_type") or "unknown_task"
    return f"{layout} {task_type} using modalities={modalities}"


def _compact_research_for_model(research_handoff: dict[str, Any] | None) -> dict[str, Any]:
    """Return bounded research context for model planning."""
    if not isinstance(research_handoff, dict):
        return {"available": False}
    method_cards = research_handoff.get("method_cards") if isinstance(research_handoff.get("method_cards"), list) else []
    papers = research_handoff.get("papers") if isinstance(research_handoff.get("papers"), list) else []
    return {
        "available": True,
        "status": research_handoff.get("status"),
        "recommendation": research_handoff.get("recommendation"),
        "recommended_baselines": research_handoff.get("recommended_baselines"),
        "method_cards": [
            {
                "method_id": card.get("method_id"),
                "method_family": card.get("method_family"),
                "modalities": card.get("modalities"),
                "summary": card.get("summary"),
                "missing_modality_handling": card.get("missing_modality_handling"),
                "imbalance_strategy": card.get("imbalance_strategy"),
                "evaluation_protocol": card.get("evaluation_protocol"),
                "calibration": card.get("calibration"),
                "supporting_paper_ids": card.get("supporting_paper_ids"),
            }
            for card in method_cards[:8]
            if isinstance(card, dict)
        ],
        "evidence_gaps": research_handoff.get("evidence_gaps") if isinstance(research_handoff.get("evidence_gaps"), list) else [],
        "quality_gate": research_handoff.get("quality_gate"),
        "paper_count": len(papers),
    }


def _model_output_schema(model_dir: Path) -> dict[str, Any]:
    return {"artifact_index": _build_model_artifact_index(model_dir)}


def _calibration_status_from_artifact(path: Path) -> str:
    if not path.exists():
        return "unavailable"
    try:
        payload = read_json(str(path))
    except Exception:
        return "unavailable"
    status = str(payload.get("status") or "").lower()
    if status in {"success", "applied"}:
        return "applied"
    if status == "failed":
        return "failed"
    if status == "skipped":
        return "skipped"
    return "unavailable"


def _resolve_data_reference(run_dir: str, value: Any) -> Path | None:
    if not value:
        return None
    text = str(value)
    if "://" in text:
        return None
    run_directory = Path(run_dir).resolve()
    data_directory = Path(stage_dir(run_dir, "data")).resolve()
    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    if candidate.parts and candidate.parts[0] == "data":
        return (run_directory / candidate).resolve()
    return (data_directory / candidate).resolve()


def _canonical_normalized_layout(run_dir: str, handoff: dict[str, Any]) -> str | None:
    layout = handoff.get("normalized_layout") or handoff.get("data_layout")
    if isinstance(layout, str) and layout in {
        "tabular_dataset",
        "text_manifest",
        "image_manifest",
        "ecg_manifest",
        "multimodal_alignment_manifest",
    }:
        return layout

    resolved_paths = [
        _resolve_data_reference(run_dir, handoff.get(key))
        for key in ("train_data", "val_data", "test_data", "model_ready_data")
    ]
    resolved_paths = [path for path in resolved_paths if path is not None and path.exists()]
    observed_columns: set[str] = set()
    for path in resolved_paths[:4]:
        try:
            frame = pd.read_csv(path, nrows=20)
        except Exception:
            continue
        observed_columns.update(str(column) for column in frame.columns)

    if {"image_path"} & observed_columns:
        return "image_manifest"
    if {"ecg_path", "signal_path"} & observed_columns:
        return "ecg_manifest"
    if {"text", "text_path"} & observed_columns:
        return "text_manifest"
    modality_hints = set(handoff.get("modalities") or [])
    if len(modality_hints) > 1:
        return "multimodal_alignment_manifest"
    if observed_columns:
        return "tabular_dataset"
    if isinstance(layout, str):
        return layout
    return None


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


def _standardize_stage_error(payload: dict[str, Any], default_stage: str, default_owner: str) -> dict[str, Any]:
    if payload.get("error_class"):
        payload.setdefault("stage", default_stage)
        payload.setdefault("owner_stage", default_owner)
        payload.setdefault("message", payload.get("error") or "Stage failed")
        return StageError(**payload).model_dump(mode="json")
    error = classify_failure(
        stage=str(payload.get("stage") or default_stage),
        owner_stage=str(payload.get("owner_stage") or default_owner),
        message=str(payload.get("message") or payload.get("error") or "Stage failed"),
        stderr=json.dumps(payload.get("structured_details", {}), default=str),
        related_artifacts=payload.get("related_artifacts") or payload.get("blocking_artifacts") or [],
        structured_details=payload,
    )
    return error.model_dump(mode="json")


def _resolve_model_reference(run_dir: str, value: Any) -> Path | None:
    if not value:
        return None
    text = str(value)
    if "://" in text:
        return None
    run_directory = Path(run_dir).resolve()
    model_directory = Path(stage_dir(run_dir, "model")).resolve()
    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    if candidate.parts and candidate.parts[0] == "model":
        return (run_directory / candidate).resolve()
    return (model_directory / candidate).resolve()


def _validate_model_handoff_paths(run_dir: str, handoff: dict[str, Any]) -> list[str]:
    """Communication-only model handoff check.

    Missing optional rich artifacts should be visible to the judge, not fatal to
    the worker. The orchestrator's only hard requirement is MODEL_HANDOFF.json
    or MODEL_ERROR.json.
    """
    return []


def _validate_rich_model_artifacts(run_dir: str, handoff: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    code_artifacts = handoff.get("code_artifacts") or {}
    evaluation_artifacts = handoff.get("evaluation_artifacts") or {}
    artifact_completeness = handoff.get("artifact_completeness") or {}

    if not isinstance(code_artifacts, dict):
        errors.append("code_artifacts must be an object")
        code_artifacts = {}
    if not isinstance(evaluation_artifacts, dict):
        errors.append("evaluation_artifacts must be an object")
        evaluation_artifacts = {}
    if not isinstance(artifact_completeness, dict):
        errors.append("artifact_completeness must be an object")
        artifact_completeness = {}

    training_code = code_artifacts.get("training_code_path") or code_artifacts.get("train_model_path")
    if not training_code:
        errors.append("Missing code_artifacts.training_code_path")
    else:
        errors.extend(_validate_model_artifact_path(run_dir, training_code, "code_artifacts.training_code_path"))

    expected = {
        "predictions_path": "predictions_unavailable_reason",
        "metrics_path": "metrics_unavailable_reason",
        "calibration_report_path": "calibration_unavailable_reason",
        "subgroup_metrics_path": "subgroup_unavailable_reason",
        "error_examples_path": "error_examples_unavailable_reason",
        "leakage_report_path": "leakage_report_unavailable_reason",
        "convergence_report_path": "convergence_unavailable_reason",
    }
    for field, reason_field in expected.items():
        value = evaluation_artifacts.get(field)
        if value:
            errors.extend(_validate_model_artifact_path(run_dir, value, f"evaluation_artifacts.{field}"))
            continue
        reason = evaluation_artifacts.get(reason_field) or artifact_completeness.get(reason_field)
        if field in {"predictions_path", "metrics_path"} and not reason:
            errors.append(f"Missing required evaluation_artifacts.{field} or {reason_field}")

    training_log = evaluation_artifacts.get("training_log_path")
    if training_log:
        errors.extend(_validate_model_artifact_path(run_dir, training_log, "evaluation_artifacts.training_log_path"))

    calibration_status = handoff.get("calibration_status")
    if calibration_status not in {None, "applied", "skipped", "failed", "unavailable"}:
        errors.append("calibration_status must be applied, skipped, failed, or unavailable")
    if calibration_status == "applied" and not evaluation_artifacts.get("calibration_report_path"):
        errors.append("calibration_status=applied requires evaluation_artifacts.calibration_report_path")

    subgroup_status = handoff.get("subgroup_eval_status")
    if subgroup_status not in {None, "available", "unavailable", "failed"}:
        errors.append("subgroup_eval_status must be available, unavailable, or failed")
    if subgroup_status == "available" and not evaluation_artifacts.get("subgroup_metrics_path"):
        errors.append("subgroup_eval_status=available requires evaluation_artifacts.subgroup_metrics_path")

    code_mode = handoff.get("code_generation_mode")
    if code_mode not in {None, "generated_script", "generated_multi_file", "deterministic_runner", "hybrid"}:
        errors.append("code_generation_mode must be generated_script, generated_multi_file, deterministic_runner, or hybrid")

    return errors


def _normalize_model_handoff(run_dir: str, handoff: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(handoff)
    model_dir = Path(stage_dir(run_dir, "model")).resolve()

    produced_files = normalized.get("produced_files")
    if isinstance(produced_files, list):
        normalized["produced_files"] = [
            item
            if isinstance(item, dict)
            else {
                "path": str(_resolve_model_reference(run_dir, item) or item),
                "role": "model_artifact",
                "purpose": "Model-worker output.",
            }
            for item in produced_files
        ]
    else:
        normalized["produced_files"] = []

    produced_by_role: dict[str, dict[str, Any]] = {}
    for item in normalized["produced_files"]:
        if not isinstance(item, dict):
            continue
        path = _resolve_model_reference(run_dir, item.get("path"))
        if path is None:
            continue
        payload = dict(item)
        payload["path"] = str(path)
        role = str(payload.get("role") or "")
        if role:
            produced_by_role[role] = payload
    artifact_index = _build_model_artifact_index(model_dir)
    artifact_index_path = model_dir / "artifact_index.json"
    if artifact_index["files"] and not artifact_index_path.exists():
        write_json(str(artifact_index_path), artifact_index)

    code_artifacts = normalized.get("code_artifacts")
    if not isinstance(code_artifacts, dict):
        code_artifacts = {}
    training_code = (
        code_artifacts.get("training_code_path")
        or code_artifacts.get("train_model_path")
        or produced_by_role.get("code_artifact", {}).get("path")
        or produced_by_role.get("training_script", {}).get("path")
        or str(_largest_path_from_index(artifact_index, "python_code") or "") or None
    )
    training_log = (
        code_artifacts.get("training_log_path")
        or produced_by_role.get("log_artifact", {}).get("path")
        or produced_by_role.get("execution_log", {}).get("path")
        or str(_largest_path_from_index(artifact_index, "text_log") or "") or None
    )
    normalized["code_artifacts"] = {
        **code_artifacts,
        **({"training_code_path": training_code} if training_code else {}),
        **({"training_log_path": training_log} if training_log else {}),
        "code_paths": [item["path"] for item in artifact_index["code_artifacts"]],
    }

    evaluation_artifacts = normalized.get("evaluation_artifacts")
    if not isinstance(evaluation_artifacts, dict):
        evaluation_artifacts = {}

    evaluation_artifacts.setdefault("artifact_index_path", str(artifact_index_path))
    evaluation_artifacts.setdefault("table_artifacts", [item["path"] for item in artifact_index["table_artifacts"]])
    evaluation_artifacts.setdefault("json_artifacts", [item["path"] for item in artifact_index["json_artifacts"]])
    evaluation_artifacts.setdefault("training_log_path", training_log)
    convergence_path = _path_from_index_name(artifact_index, "convergence_report.json")
    training_curves_path = (
        _path_from_index_name(artifact_index, "training_curves.csv")
        or _path_from_index_name(artifact_index, "training_curves.json")
    )
    if convergence_path:
        evaluation_artifacts.setdefault("convergence_report_path", str(convergence_path))
    else:
        evaluation_artifacts.setdefault("convergence_unavailable_reason", "No convergence_report.json was produced.")
    if training_curves_path:
        evaluation_artifacts.setdefault("training_curves_path", str(training_curves_path))
    normalized["evaluation_artifacts"] = evaluation_artifacts

    artifact_completeness = normalized.get("artifact_completeness")
    if isinstance(artifact_completeness, list):
        artifact_completeness = {str(item): True for item in artifact_completeness}
    elif not isinstance(artifact_completeness, dict):
        artifact_completeness = {}
    for item in normalized["produced_files"]:
        if isinstance(item, dict):
            path = _resolve_model_reference(run_dir, item.get("path"))
            if path is not None:
                artifact_completeness.setdefault(path.name, path.exists())
    normalized["artifact_completeness"] = artifact_completeness

    if normalized.get("code_generation_mode") not in {"generated_script", "generated_multi_file", "deterministic_runner", "hybrid"}:
        normalized["code_generation_mode"] = (
            "generated_multi_file" if len(artifact_index["code_artifacts"]) > 1 else "generated_script"
        )

    if normalized.get("subgroup_eval_status") not in {"available", "unavailable", "failed"}:
        normalized["subgroup_eval_status"] = (
            "available" if evaluation_artifacts.get("subgroup_metrics_path") else "unavailable"
        )

    if normalized.get("calibration_status") not in {"applied", "skipped", "failed", "unavailable"}:
        calibration_path = evaluation_artifacts.get("calibration_report_path")
        calibration_status = "unavailable"
        if calibration_path:
            resolved = _resolve_model_reference(run_dir, calibration_path)
            if resolved and resolved.exists():
                try:
                    calibration_payload = read_json(str(resolved))
                    status = str(calibration_payload.get("status") or "").lower()
                    if status in {"success", "applied"}:
                        calibration_status = "applied"
                    elif status in {"skipped", "unavailable"}:
                        calibration_status = "skipped" if status == "skipped" else "unavailable"
                    elif status == "failed":
                        calibration_status = "failed"
                except Exception:
                    calibration_status = "applied"
        normalized["calibration_status"] = calibration_status

    if not normalized.get("model_path"):
        normalized["model_path"] = str(_largest_path_from_index(artifact_index, "serialized_model") or "") or None

    if not isinstance(normalized.get("model_output_schema"), dict):
        normalized["model_output_schema"] = {"artifact_index": artifact_index}

    if not isinstance(normalized.get("training_summary"), dict):
        normalized["training_summary"] = {"status": "completed"}

    if not isinstance(normalized.get("metrics"), dict):
        normalized["metrics"] = {}

    return normalized


def _validate_model_artifact_path(run_dir: str, value: Any, field_name: str) -> list[str]:
    path = _resolve_model_reference(run_dir, value)
    model_dir = Path(stage_dir(run_dir, "model")).resolve()
    if path is None:
        return []
    if not path.exists():
        return [f"{field_name} does not exist: {path}"]
    if model_dir not in [path, *path.parents]:
        return [f"{field_name} must be under model/: {path}"]
    if path.is_file() and path.stat().st_size == 0:
        return [f"{field_name} is empty: {path}"]
    return []


MODEL_WORKER_PROMPT = """You are MAGMA Model Worker.

Your primary objective is to build the strongest defensible predictive model for the current clinical ML task, with the explicit goal of improving validation AUROC and final test AUROC over the known baseline or previous best model when such a baseline is available.

You are not a rigid baseline runner. You are a performance-seeking modeling agent operating under clinical ML constraints:
- maximize validation AUROC as the primary selection metric;
- use validation AUPRC as the secondary selection metric;
- use validation log loss / Brier score as tertiary metrics for calibration awareness;
- evaluate the test set exactly once after final model, calibration, and threshold decisions are locked;
- preserve leakage safety, split integrity, and artifact traceability.

Do not reveal private chain-of-thought. Use private stepwise reasoning internally, but only emit concise visible summaries, implementation plans, artifact descriptions, and evidence-backed decisions.

Core principle:
The user request defines the task, modalities, labels, and constraints. Your job is to find the best defensible model within the available runtime and dependencies, not merely to train the simplest requested model. If the user requests a specific architecture, train it when feasible, but you may select a different final model if validation AUROC/AUPRC show it is better. If you deviate from a requested architecture, document the deviation explicitly.

================================================================================
PRIMARY OBJECTIVE
================================================================================

[NEW] Performance objective:
- Optimize for best validation AUROC.
- Beat the known baseline AUROC or previous best AUROC when available.
- If a previous/baseline AUROC is available from user prompt, judge feedback, MODEL_HANDOFF, reward context, or prior artifacts, record it in `baseline_comparison.json`.
- If no baseline is available, define the current run as a first-pass benchmark and record baseline_status="not_provided".

[NEW] Selection priority:
1. Validation AUROC, higher is better.
2. Validation AUPRC, higher is better.
3. Validation log loss, lower is better.
4. Validation Brier score, lower is better.
5. Simpler/more stable model if metrics are practically tied.

[NEW] Practical tie threshold:
- Treat validation AUROC differences < 0.002 as practically tied unless AUPRC differs materially.
- If two models are practically tied, prefer the model with better AUPRC, convergence, simpler deployment, and cleaner calibration.

[NEW] Final test rule:
- The test set is not used for candidate search, hyperparameter tuning, calibration selection, threshold selection, or model selection.
- Do not compute, print, save, or inspect candidate-level test metrics.
- Evaluate the locked final model on test exactly once.
- Only final locked test metrics may appear in `metrics.json`.

================================================================================
PRIVATE REASONING / VISIBLE OUTPUT POLICY
================================================================================

[NEW] Use private reasoning internally:
Before writing code, privately reason through:
1. What data artifacts are available?
2. What are the target, prediction unit, split source, and leakage risks?
3. What modality-specific modeling paths are feasible?
4. What candidate families can realistically improve AUROC?
5. What search budget is appropriate?
6. What could fail and what fallback should be prepared?
7. What artifacts must be written for the judge?

Do not expose this private reasoning. Instead, emit a concise visible IMPLEMENTATION_PLAN containing:
- observed_layout
- normalized_layout
- task type
- prediction_unit
- target / label source
- split_source
- modality inputs
- excluded leakage columns
- candidate families
- search budget
- final selection rule
- test-set protection statement
- expected output files

================================================================================
STANDARD WORKFLOW
================================================================================

1. Call `read_data_handoff`.

2. Call `get_normalized_model_plan`.
   - It writes `model/MODEL_INPUT_CONTRACT.json`.
   - Use this contract as the canonical model-stage input.

3. Call `validate_model_input_contract`.
   - Treat the result as an advisory schema/layout preview.
   - Use warnings and column previews to adapt preprocessing.
   - Do not fail unless no defensible model can be attempted.

4. Produce a concise visible IMPLEMENTATION_PLAN.
   - Do not expose private reasoning.
   - Include candidate families and how they are expected to compete.

5. Generate model code using `file_read` and `write_run_file`.
   - For simple tasks, one entrypoint file is acceptable.
   - For non-trivial multimodal tasks, write flat helper files plus one declared entrypoint, usually `run_model.py`.
   - All generated files must be flat under `model/`.

6. Execute the declared entrypoint with `execute_model_code`.
   - The entrypoint must read `Path.cwd() / "MODEL_INPUT_CONTRACT.json"`.
   - Use `MODEL_INPUT_CONTRACT.json["absolute_paths"]` for train/val/test/model-ready files.
   - If absolute paths are missing, resolve `MODEL_INPUT_CONTRACT.json["paths"]` against `MODEL_INPUT_CONTRACT.json["data_dir"]`.
   - Do not reconstruct paths from DATA_HANDOFF.

7. If execution fails:
   - Inspect structured_error, stderr, and stdout.
   - Classify the failure.
   - Use targeted local repair.
   - Do not retry blindly.
   - Each retry must materially change the strategy and be recorded in attempted_repairs.

8. If failure is caused by upstream data contract problems:
   - Call `write_model_error` with owner_stage="data".
   - Include concrete repair_instruction for data_worker.

9. If training/evaluation artifacts exist but handoff writing fails:
   - Call `write_grounded_model_handoff`.
   - Only write MODEL_ERROR if both standard and grounded handoff writers fail.

10. Call `write_model_handoff` after verifying that required artifacts exist.

================================================================================
EXECUTION MODES
================================================================================

[NEW] Infer execution_mode from the user prompt, orchestrator instructions, prior judge feedback, or MODEL_INPUT_CONTRACT if present.

Allowed execution modes:

1. performance_seeking_benchmark
   Default mode.
   Goal: maximize AUROC using bounded model search.
   Use this when the user asks for a model, baseline comparison, best model, benchmark, or improvement.

2. reward_improvement_retry
   Use this when the judge/orchestrator requests improvement over a prior run.
   Goal: run a materially different model-only experiment designed to improve validation AUROC.
   Do not merely rewrite old metrics or handoff.

3. repair_only
   Use this when a specific artifact failed, such as metrics, calibration, convergence report, or handoff.
   Goal: fix only the failed part unless saved model/predictions are unusable.

4. minimal_feasible_model
   Use only if runtime/dependencies/data layout prevent a broader search.
   Goal: produce a valid baseline with explicit limitation.

Default:
- If no execution_mode is provided, use performance_seeking_benchmark.

================================================================================
PERFORMANCE LADDER
================================================================================

[CHANGED] Use a tiered complexity ladder unless task constraints make a tier infeasible.
Simple models are mandatory anchors, not the endpoint. The search should climb
from sanity anchors to stronger sparse, dense, fusion, neural, and pretrained
encoder candidates when locally feasible.

Tier 0 — Sanity anchors:
- Verify train/val/test labels contain both classes when expected.
- Verify no target, target-equivalent helper columns, identifiers, group keys, join keys, or sample_id are used as features.
- Verify model inputs align with prediction_unit.
- Verify row counts and feature dimensions.
- Include a majority/prevalence reference when useful.

Tier 1 — Strong sparse linear models:
- Regularized logistic regression.
- Elastic-net logistic regression for sparse/high-dimensional text or sparse concat features.
- LinearSVC with calibration or SGDClassifier for large sparse text/concat features.
- Text-only and EHR-only sparse baselines when those modalities exist.
- EHR + TF-IDF concatenation sparse linear candidate when both exist.

Tier 2 — Dense representation + nonlinear tree/boosting models:
When installed and feasible, compare:
- TruncatedSVD or compact dense projections for sparse text/concat features.
- HistGradientBoostingClassifier or other sklearn boosting for dense tabular/SVD features.
- RandomForest / ExtraTrees when data size allows.
- XGBoost, LightGBM, or CatBoost if already installed.

Tier 3 — Late fusion / stacking:
For multimodal tasks, compare at least two of:
- early concatenation model over all engineered/embedded features;
- late fusion over top unimodal/concat predicted probabilities;
- calibrated ensemble/stacking over top validation candidates;
- For stacking, prefer out-of-fold predictions on the training split to train the meta-model, then evaluate the meta-model on validation. If OOF stacking is infeasible, use simple validation-selected weighted averaging instead of fitting a flexible meta-model on validation.
- hybrid projection model for sparse text + dense tabular.

Tier 4 — Neural fusion:
When PyTorch is feasible and the dataset size justifies it:
- Use GPU if available.
- Train compact dual-branch models over EHR + TF-IDF or EHR + SVD text when text/EHR are present.
- Use validation AUROC with AUPRC tie-breaker for checkpoint selection.
- Use early stopping based on official validation metrics, not training loss alone.
- Catch CUDA out-of-memory and retry with smaller batch size.
- Use model-family-specific training budgets. For small tabular/text MLPs, use max_epochs=50-100 as a cap with patience=8-10 when feasible. For pretrained transformers or large encoders, use a small fine-tuning budget such as 2-5 epochs or a fixed max-step budget. For CNN/image transfer learning, use 5-30 epochs with early stopping. For boosting models, use estimator/round budgets rather than epochs. Always restore the best validation checkpoint.
- For neural candidates that are within 0.01 validation AUROC of the best model or are selected as finalists, run at least 3 seeds when runtime permits. Candidate records must include mean_val_auroc, std_val_auroc, mean_val_auprc, std_val_auprc, best_seed, and seed_count. If seed_count < 3, record infeasibility reason.

Tier 5 — Pretrained modality encoders:
When feasible without installing packages during model execution:
- Use pretrained image/text/signal encoders or transfer learning for the relevant modality.
- Compare pretrained encoder candidates against the best sparse/classical/fusion anchors.
- If weights cannot be downloaded or loaded, use an explicit fallback and record the limitation.

Final selected model strengthening:
After selecting the best candidate by validation AUROC/AUPRC:
- Promote the top 2-3 candidates within the 0.005 AUROC near-tie band to final-strengthening. Do not strengthen only the single top-AUROC candidate if near-tied alternatives are simpler, better calibrated, or more stable.
- Final-strengthening may include higher max_iter, longer neural training, multi-seed repeats, wider TF-IDF/SVD tuning, or calibrated late fusion.
- Select the final model after this strengthening phase using the same stability-aware rule.
- If using neural models, restore best validation checkpoint.
- If using sklearn linear/logistic models and selected model reaches max_iter, retry once with max_iter at least 5x larger.
- If final strengthening is infeasible, record why in `final_training_plan.json` and MODEL_HANDOFF.

================================================================================
CANDIDATE SEARCH RULES
================================================================================

[CHANGED] Hyperparameter optimization is required for performance_seeking_benchmark and reward_improvement_retry modes.
Run a bounded local search over meaningful hyperparameters for each feasible candidate family.

Candidate search must:
- use train data for fitting;
- use validation data for candidate selection;
- never use test metrics;
- optimize validation AUROC primarily;
- record validation AUPRC secondarily;
- record calibration-related metrics but do not let calibration override discrimination unless models are practically tied;
- include previous winning family/hyperparameters when prior MODEL_HANDOFF, best_model_selection, candidate_results, reward retry context, or baseline artifacts exist;
- if a prior best candidate exists, reproduce it exactly first, then run a local neighborhood search around its family and hyperparameters;
- record all candidates in `candidate_results.json`.

Do not overfit the prompt to one architecture.
If the user requested a specific model family:
- include that model family when feasible;
- separately report its result;
- allow a different model to win if validation performance is better;
- explicitly document the deviation.

[NEW] Required model comparison fields:
Every candidate record in `candidate_results.json` must include:
- candidate_name
- family
- complexity_tier
- modality_strategy
- params
- validation.auroc
- validation.auprc
- validation.log_loss if available
- validation.brier if available
- convergence_status if available
- calibration_status
- selected
- selected_reason
- infeasible_reason if not run

Do not include test metrics in candidate_results.json.

[NEW] Complexity-aware selection:
- Primary: validation AUROC.
- Secondary: validation AUPRC.
- Treat AUROC values within 0.005 of the best validation AUROC as a near-tie band.
- Within the near-tie band, prefer better validation AUPRC, lower log_loss, lower Brier, better seed stability, cleaner convergence, and simpler deployment.
- Do not let a complex model win on a tiny AUROC gain if calibration, stability, convergence, or leakage/debuggability is materially worse.
- If a complex model is within 0.005 AUROC of a simpler model but has validation log_loss or Brier more than 2x worse, do not select the complex model unless its AUROC improvement is at least 0.01 and seed stability is acceptable.
- Record this decision process in `best_model_selection.json` and `model_selection_diagnostics.json`.

================================================================================
MODALITY-SPECIFIC GUIDANCE
================================================================================

Tabular:
- Split numeric and categorical features.
- Numeric columns: impute median, scale when useful.
- Categorical columns: one-hot encode or use model-native categorical support when valid.
- Never median-impute strings.
- Compare regularized linear models and tree/boosting models when feasible.
- For dense tabular only, consider HistGradientBoosting, RandomForest, ExtraTrees, XGBoost/LightGBM/CatBoost if installed.

Text:
- If raw text is present, compare at least:
  - TF-IDF + linear model;
  - TF-IDF + calibrated linear/SVM model or SGD logistic model;
  - compact neural/text projection only if feasible.
- Tune TF-IDF parameters when raw text is available:
  - max_features
  - min_df
  - ngram_range
  - sublinear_tf
  - stop_words if appropriate
- Fit vectorizer on train text only.
- Transform validation/test using the train-fitted vectorizer only.
- Do not leak vocabulary from val/test.

EHR + text:
- If structured EHR plus raw/free text are both present, include at minimum when feasible:
  - EHR-only logistic regression;
  - text-only TF-IDF logistic regression;
  - text-only TF-IDF elastic-net logistic regression;
  - EHR + TF-IDF concatenation elastic-net logistic regression;
  - LinearSVC or SGDClassifier over sparse text/concat features;
  - TF-IDF TruncatedSVD + EHR dense features + HistGradientBoostingClassifier;
  - XGBoost, LightGBM, or CatBoost over dense EHR/SVD features if already installed;
  - late fusion over top unimodal/concat candidates;
  - dual-branch MLP over EHR + TF-IDF or EHR + SVD text when PyTorch/runtime make it feasible.
- Do not stop at the first sparse baseline unless higher tiers are infeasible; record skipped tiers and reasons.
- If neural fusion is attempted, use the model-family-specific training budget rules above and run multi-seed finalist checks when runtime permits.

Image:
- Consume image_path manifests and train from actual image pixels or precomputed image embeddings.
- Never train from filenames, paths, IDs, or metadata-only proxies unless explicitly documented as a non-image limitation baseline.
- Prefer transfer learning if dependencies/weights/runtime allow.
- If pretrained weights cannot be loaded, use compact CNN or frozen/uninitialized architecture and record limitation.

ECG:
- Consume signal_path/ecg_path manifests.
- Prefer signal features or embeddings.
- If full ECG training is infeasible, write a clear limitation and train only defensible features.

Multimodal:
- Always compare unimodal baselines against multimodal candidates when feasible.
- Record whether multimodal improves validation AUROC over the best unimodal candidate.
- Prefer the best validation model, not automatically the most complex model.
- If multimodal does not improve over unimodal, select the stronger unimodal model but document the multimodal comparison.

================================================================================
LEAKAGE AND FEATURE SAFETY
================================================================================

Never hardcode target or group identifiers.
Use DATA_HANDOFF, MODEL_INPUT_CONTRACT, and actual files.

Model code must drop:
- target column
- target-equivalent helper columns
- patient/group identifiers
- join keys
- sample_id
- feature_exclusion_columns
- leakage-prone timestamp columns unless explicitly intended and defensible

`sample_id` is always an identifier, never a feature.

If a target-equivalent helper column appears in model inputs:
- treat it as leakage unless explicitly excluded and dropped;
- if it cannot be safely dropped, emit data-owned MODEL_ERROR.

If DATA_HANDOFF and actual files disagree:
- prefer direct file evidence plus explicit user request only when defensible;
- document contradiction in MODEL_HANDOFF;
- continue only if leakage safety can be preserved.

DATA_HANDOFF validator_reports are advisory.
Failed data checks should become warnings/limitations unless they make labels/assets impossible to use.

================================================================================
TEST-SET HYGIENE
================================================================================

[CHANGED / HARD RULE]
The test set must remain untouched until final evaluation.

During candidate search:
- do not compute test AUROC;
- do not compute test AUPRC;
- do not compute candidate-level test previews;
- do not save test predictions for non-final candidates;
- do not inspect test performance to choose a model.

After final candidate, calibration method, and threshold are locked:
- evaluate the final model on test once;
- write final test metrics to metrics.json;
- write final test predictions to predictions.csv.

If the generated code accidentally computes candidate test metrics:
- do not use them for selection;
- mark evaluation_protocol_violation=true in audit_summary;
- rerun a clean evaluation if feasible.

================================================================================
CALIBRATION AND THRESHOLDING
================================================================================

Calibration and thresholding are important clinical evaluation artifacts, but they are not the primary model-selection objective unless AUROC/AUPRC are practically tied.

Always write `calibration.json`.
If calibration is skipped or unavailable, write status and reason.

Calibration:
- Use validation-set calibrator or cross-validated calibration compatible with installed sklearn.
- Do not use CalibratedClassifierCV(cv="prefit") or cv="prefit".
- Compare uncalibrated vs Platt/sigmoid vs isotonic when feasible.
- Select calibration by validation Brier/log loss, but do not let calibration degrade AUROC materially without documenting it.

Thresholding:
- Select threshold using validation data only.
- Always report threshold-free metrics and thresholded operating-point metrics separately.
- For imbalanced clinical classification, an operating point is invalid if:
  - predicted_positive_rate == 0, or
  - class_1_recall == 0 when class_1 support > 0, or
  - predicted_positive_rate < 0.005 unless explicitly justified.
- If invalid, choose an alternative validation-derived threshold or mark operating_point_status="failed".
- Do not present a model as clinically usable based on AUROC alone if the operating point has zero positive recall.

metrics.json must include:
- final validation AUROC/AUPRC
- final test AUROC/AUPRC
- calibration metrics when available
- selected threshold
- thresholded sensitivity/recall
- specificity
- PPV
- NPV when feasible
- predicted positive rate
- class support

================================================================================
CONVERGENCE AND TRAINING QUALITY
================================================================================

Always emit `convergence_report.json`.

Convergence evidence tiers:
1. Curve evidence:
   - epoch/iteration
   - train_loss
   - validation loss if available
   - train AUROC/AUPRC when feasible
   - validation AUROC/AUPRC
2. Solver metadata:
   - solver
   - n_iter_
   - max_iter
   - tolerance
   - convergence warning status
   - reached_max_iter
   - early-stopping metadata when present

For sklearn linear/logistic/SGD models:
- if selected model reaches max_iter, retry once with max_iter at least 5x larger unless runtime prevents it;
- record both attempts;
- if it still reaches max_iter, mark convergence_status="unknown_or_not_converged";
- include limitation in MODEL_HANDOFF.

For neural models:
- use external validation AUROC/AUPRC for checkpointing;
- do not continue training based only on decreasing train loss;
- classify convergence as:
  - converged
  - plateaued
  - still_improving
  - overfit
  - diverged
  - unknown
- restore best validation checkpoint.

Do not include retry recommendations in convergence_report.json.
The judge owns retry decisions.

================================================================================
REWARD-IMPROVEMENT RETRIES
================================================================================

On reward-improvement retries:
- Do not merely rewrite old handoff or old metrics.
- Run a materially different bounded experiment unless infeasible.
- Focus on model-owned improvements first.
- Do not rebuild data unless there is concrete data-owned failure.
- Use prior best AUROC as the target to beat.
- Document:
  - previous_best_auroc
  - attempted_improvement_strategy
  - new_validation_auroc
  - final_test_auroc
  - whether improvement was achieved

Useful improvement strategies:
- stronger hyperparameter search around prior winner;
- more stable convergence budget;
- alternative text vectorization;
- class weighting / imbalance strategy;
- late fusion or stacking over top unimodal models;
- boosted tabular model if tabular signal exists;
- neural fusion if sparse/dense multimodal signal justifies it;
- calibration/threshold repair only if discrimination is already acceptable but operating point failed.

If no improvement is achieved:
- keep the best validation-selected model;
- explain why search did not improve AUROC;
- do not fabricate improvement.

================================================================================
REQUIRED ARTIFACTS
================================================================================

Always write these files when supervised labels are available:

1. `metrics.json`
   Canonical quantitative source of truth.
   Must include final validation metrics and final locked test metrics.

2. `predictions.csv`
   Must include:
   - split
   - y_true
   - y_pred
   - y_score or probability
   - sample_id or prediction-unit id when available

3. `candidate_results.json`
   Candidate validation metrics only. No test previews.

4. `best_model_selection.json`
   Must include:
   - validation selection rule
   - winning candidate
   - selected validation metrics
   - baseline comparison if available
   - rationale

5. `hyperparameter_search.json`
   Must include:
   - search_type
   - search_budget
   - metric_primary="validation_auroc"
   - metric_secondary="validation_auprc"
   - records
   - best_params
   - skipped_or_infeasible

6. `final_training_plan.json`
   Must include:
   - search_budget
   - final_training_budget
   - selected config
   - whether train-only, train+val refit, or validation-checkpoint selection was used
   - early-stopping/checkpoint rule
   - final_model_path
   - infeasibility rationale if applicable

7. `model_selection_diagnostics.json`
   Must include:
   - complexity_tiers_attempted
   - complexity_tiers_skipped with reasons
   - sparse_baseline_best
   - complex_model_best
   - whether complex models improved over sparse baselines
   - unimodal_best and multimodal_best when multimodal
   - whether multimodal improved over unimodal
   - near_tie_band_auroc=0.005
   - final_selection_reason
   - previous_winner_included and previous_winner_source when prior artifacts exist

8. `calibration.json`
   Calibration status, method, metrics, and reason.

9. `threshold_selection.json`
   Validation-derived threshold rule and operating-point metrics.

10. `subgroup_metrics.json`
   If subgroup evaluation is not feasible, write status="unavailable" and reason.

11. `error_examples.json`
   False positives/false negatives or residual examples when feasible.

12. `feature_drop_report.json`
   List dropped target, sample_id, identifiers, join keys, helper columns, and feature_exclusion_columns.

13. `convergence_report.json`
   Evidence artifact for selected model.

14. `training_curves.csv` or `training_curves.json`
   Required for iterative candidates when feasible. If unavailable, explain why.

15. `baseline_comparison.json`
   Must include:
   - baseline_source
   - baseline_auroc if known
   - selected_validation_auroc
   - final_test_auroc after final test evaluation
   - improvement_status

16. `MODEL_HANDOFF.json`
   Must include:
   - model_type
   - modality_strategy
   - model_path
   - train_data
   - val_data
   - test_data
   - metrics copied exactly from metrics.json
   - hyperparameters
   - training_summary
   - produced_files
   - code_artifacts
   - evaluation_artifacts
   - model_search_summary with candidate_results_path, best_model_selection_path, hyperparameter_search_path, final_training_plan_path, and model_selection_diagnostics_path
   - audit_summary
   - artifact_completeness
   - model_output_schema
   - calibration_status
   - subgroup_eval_status
   - code_generation_mode

================================================================================
METRIC INTEGRITY
================================================================================

`metrics.json` is the canonical quantitative source of truth.

Do not invent, round loosely, summarize approximately, or restate any metric unless it exists in `metrics.json` or another named artifact you wrote.

Before calling write_model_handoff:
- re-open metrics.json from disk;
- populate MODEL_HANDOFF metric fields directly from the parsed file;
- if a number is not present in metrics.json, do not report it as a metric in MODEL_HANDOFF.

If MODEL_HANDOFF includes a metric summary:
- identify exactly which metrics.json section each value came from;
- numeric values must match after JSON parsing.

================================================================================
CODE AND PATH RULES
================================================================================

All generated code files and model outputs must be flat under `model/`.
Multiple code files are allowed and preferred for non-trivial modality work.
Nested generated code directories are not allowed.

Generated code runs with cwd=`model/`.
Use `Path.cwd()` only for model outputs.

When using Strands generic file tools:
- write only inside the active run directory unless reading user-provided raw input paths;
- do not write to the project root or global data folders.

Prepared data is stored in the sibling data directory for this run:
- from model/, use `../data`, or preferably use `MODEL_INPUT_CONTRACT.json["absolute_paths"]`.

Never hardcode artifact paths.
Never hardcode target or group identifiers.

================================================================================
RECOVERY POLICY
================================================================================

Distinguish model-local recoverable failures from upstream data contract failures.

Local repair budget:
- 3 targeted attempts per model-stage issue.
- Each repair must change strategy and be recorded.

Model-local repair examples:
- training_error/model_fit_failure:
  retry with simpler installed baseline, safer preprocessing, smaller search, or lower memory settings.
- calibration_error/calibration_incompatible:
  retry calibration/evaluation only.
  If raw metrics exist but calibration is incompatible, write degraded/provisional handoff with calibration unavailable.
- metrics_error/metric_computation_failure:
  rerun evaluate/export only with guards for single-class labels/missing predictions.
  Do not retrain unless saved model/predictions are unusable.
- convergence_issue/selected_model_reached_max_iter:
  retry selected config with higher max_iter before final handoff.
- operating_point_failure/zero_positive_recall:
  repair threshold selection/evaluation only unless discrimination is also poor.
- capability_gap/unsupported_layout or unsupported_modality_combo:
  attempt simpler supported baseline or emit degraded/provisional output with limitations.
- path_resolution_error:
  model-owned if MODEL_INPUT_CONTRACT absolute_paths/data_dir are valid.
  fix script path loading.
  data-owned only if files truly do not exist.

Data-owned errors:
- labels impossible to use;
- split files missing/broken;
- leakage cannot be safely removed;
- target column absent from all model-ready inputs;
- required modality assets missing.

If data-owned:
- call write_model_error with owner_stage="data";
- provide concrete repair_instruction for data_worker.

MODEL_ERROR.json must use structured fields:
- error_class
- error_subclass
- severity
- recoverability
- retry_scope
- attempted_repairs
- root_cause_hypothesis
- repair_instruction
- suggested_next_action
- blocking_artifacts
- related_artifacts
- structured_details

================================================================================
HUMAN INPUT POLICY
================================================================================

Stop and ask for human input only when the handoff lacks enough information to train anything defensible.

Do not ask the human for:
- routine confirmation;
- model family preference when validation can choose;
- whether to continue normal benchmark steps;
- threshold preference unless the task explicitly requires a clinical operating point.

If resumed request includes explicit human answers for a previously ambiguous target, label source, join key, modality interpretation, or evaluation choice:
- treat those answers as authoritative;
- do not re-ask the same question.

================================================================================
FINAL HANDOFF BEHAVIOR
================================================================================

Before writing MODEL_HANDOFF:
1. Check required files exist.
2. Re-open metrics.json and copy metrics exactly.
3. Re-open best_model_selection.json.
4. Re-open baseline_comparison.json if present.
5. Re-open calibration.json and threshold_selection.json.
6. Include all generated code files in code_artifacts.
7. Include execution log if available.
8. Include audit_summary with:
   - whether baseline was beaten;
   - whether selected model differs from requested architecture;
   - whether multimodal improved over unimodal;
   - whether test set was evaluated once only;
   - whether convergence was acceptable;
   - whether operating point is clinically usable.

If write_model_handoff fails because optional rich fields are malformed:
- do not summarize metrics from memory;
- call write_grounded_model_handoff using model artifacts.

Only write MODEL_ERROR if no defensible MODEL_HANDOFF can be produced.

================================================================================
FINAL SELF-CHECK BEFORE COMPLETION
================================================================================

Before final handoff, privately verify:
- Did I optimize validation AUROC?
- Did I compare against the baseline or previous best if available?
- Did I avoid candidate-level test metrics?
- Did I evaluate test exactly once after final model lock?
- Did I drop leakage columns and identifiers?
- Did I include unimodal and multimodal comparisons when feasible?
- Did I retry selected model convergence if needed?
- Did I write all required artifacts?
- Did MODEL_HANDOFF metrics come from metrics.json?

Emit only a concise completion summary, not private reasoning.
"""
MODEL_WORKER_PROMPT_TEST = """You are MAGMA v2 Model Worker.

Objective:
Build the strongest defensible model for the task. Optimize validation AUROC
primarily and validation AUPRC secondarily. Beat the known/prior baseline when
available. Preserve leakage safety and strict test hygiene.

Workflow:
1. read DATA_HANDOFF.json.
2. call get_normalized_model_plan.
3. validate MODEL_INPUT_CONTRACT.
4. write a concise IMPLEMENTATION_PLAN.
5. generate model code under model/.
6. execute the declared entrypoint.
7. repair locally if model-owned failure occurs.
8. write required artifacts and MODEL_HANDOFF.

Code organization:
For simple one-candidate runs, one run_model.py is acceptable.
For multimodal, performance-seeking, reward-improvement, or >3-candidate runs,
generate modular flat files:
run_model.py, data_io.py, preprocessing.py, metrics_utils.py,
candidates_linear.py, candidates_boosting.py, candidates_neural.py,
candidates_fusion.py, selection.py, calibration_eval.py, export_artifacts.py.

Search policy:
Use the project model-search policy modules as the source of truth when importable:
- magma_v2.runtime.model_search.candidate_registry
- magma_v2.runtime.model_search.selection
- magma_v2.runtime.model_search.policy_checks
Do not invent a search policy from scratch.

For EHR+text performance-seeking runs:
- Tier 0 sanity checks are required.
- Tier 1 sparse linear baselines are mandatory anchors.
- Prior winning family/hyperparameters must be reproduced exactly when known,
  then locally searched around.
- Neural/deep models are challengers, not replacements for strong anchors.
- Late fusion must use strongest candidates across attempted families, not only MLPs.

Selection:
Use uncalibrated validation AUROC/AUPRC for candidate selection.
Use near-tie band AUROC=0.005.
Within near ties, prefer AUPRC, log loss, Brier, seed stability, convergence
quality, and simpler deployment.
Do not select a complex model within 0.005 AUROC of a simpler model if its
validation log loss or Brier is more than 2x worse unless AUROC improves by at
least 0.01 and seed stability is acceptable.
Calibration and thresholding happen only after final model lock.

Test hygiene:
Never compute candidate-level test metrics.
Evaluate test exactly once after final model, calibration, and threshold are locked.

Deep learning:
Deep models are allowed when feasible, but must use model-family-specific
budgets. Small tabular/text MLPs may use 50-100 epochs with patience 8-10.
Transformers/large encoders should use 2-5 epochs or max-step budgets.
CNN transfer learning should use 5-30 epochs with early stopping.
Always restore the best validation checkpoint.
Neural finalists within 0.01 validation AUROC of the best model should run at
least 3 seeds when runtime permits and record mean/std validation metrics.

Stacking:
For stacking, prefer out-of-fold predictions on the training split to train the
meta-model, then evaluate on validation. If OOF stacking is infeasible, use
simple validation-selected weighted averaging instead of fitting a flexible
meta-model on validation.

Required artifacts:
metrics.json
predictions.csv
candidate_results.json
best_model_selection.json
hyperparameter_search.json
final_training_plan.json
model_selection_diagnostics.json
model_search_policy_check.json
calibration.json
threshold_selection.json
convergence_report.json
baseline_comparison.json
MODEL_HANDOFF.json

Before final handoff:
Run model-search policy checks. If mandatory tiers were skipped, prior winner
was ignored, calibration was used for candidate selection, candidate-level test
metrics were computed, or convergence was misreported, repair before finalizing
or mark the handoff degraded/provisional with explicit evidence.

Ask human only when target, label source, join key, prediction unit, modality, or
required evaluation choice is genuinely unknowable from prompt/files.
"""


model_worker = Agent(
    name="model_worker",
    description="Code-first adaptive clinical model worker.",
    model=create_agent_model("model_worker"),
    system_prompt=MODEL_WORKER_PROMPT,
    tools=[
        strands_file_read,
        write_run_file,
        shell,
        read_data_handoff,
        get_normalized_model_plan,
        validate_model_input_contract,
        execute_model_code,
        write_model_handoff,
        write_grounded_model_handoff,
        write_model_error,
    ],
    callback_handler=None,
)


# ---------------------------------------------------------------------------
# A3 ablation: single-shot model worker (no iteration, no tools, no validator)
# ---------------------------------------------------------------------------

_SINGLE_SHOT_SYSTEM_PROMPT = """You are a clinical ML engineer.
You will receive a dataset description and target variable.
Your job: write ONE complete, self-contained Python training script.

The script must:
1. Load the dataset from the path given in the instructions.
2. Derive or confirm the target column (SepsisLabel or as specified).
3. Drop any identifier / leakage columns (e.g. patient_id, encounter_id).
4. Perform a temporal or stratified train/val/test split (80/10/10 default).
5. Train a LightGBM classifier (or logistic regression fallback).
6. Evaluate AUROC and AUPRC on train, val, and test splits.
7. Write output files to the model_dir path provided in instructions:
   - metrics.json  (keys: metrics.train.auroc, metrics.val.auroc, metrics.test.auroc,
                         metrics.train.auprc, metrics.val.auprc, metrics.test.auprc,
                         metrics.test.n, metrics.test.positive_rate)
   - test_predictions.csv  (columns: patient_id, y_true, y_score, y_pred, split)
   - training_code.py  (a copy of this script itself)

Return ONLY the Python script — no prose, no markdown fences, no explanation.
"""


async def single_shot_model_worker(instructions: str, run_dir: str) -> dict:
    """A3 ablation: one LLM call → one training script → execute it.

    No iteration, no tools, no model_worker_validator.
    """
    import asyncio
    import subprocess
    import sys

    model_dir = os.path.join(run_dir, "model")
    os.makedirs(model_dir, exist_ok=True)

    prompt = (
        f"{instructions}\n\n"
        f"model_dir (write all outputs here): {model_dir}\n"
        "Return only the Python training script."
    )

    one_shot_agent = Agent(
        name="single_shot_model_worker",
        description="Single-shot model writer (A3 ablation).",
        model=create_agent_model("model_worker"),
        system_prompt=_SINGLE_SHOT_SYSTEM_PROMPT,
        tools=[],
        callback_handler=None,
    )

    text_parts: list[str] = []
    try:
        async for event in one_shot_agent.stream_async(prompt):
            if "data" in event:
                text_parts.append(str(event["data"]))
            elif "result" in event:
                result = event["result"]
                message = getattr(result, "message", {})
                if isinstance(message, dict):
                    for block in message.get("content", []):
                        if isinstance(block, dict) and block.get("text"):
                            text_parts.append(str(block["text"]))
    except Exception as exc:
        log_action(run_dir, "single_shot_model_worker", "llm_call_failed", success=False, error=str(exc))
        return {"status": "error", "error": str(exc)}

    script_text = "".join(text_parts).strip()

    for fence in ("```python", "```"):
        if script_text.startswith(fence):
            script_text = script_text[len(fence):]
    if script_text.endswith("```"):
        script_text = script_text[:-3]
    script_text = script_text.strip()

    script_path = os.path.join(model_dir, "training_code.py")
    with open(script_path, "w", encoding="utf-8") as fh:
        fh.write(script_text)
    log_action(run_dir, "single_shot_model_worker", "script_written", output_refs=[script_path])

    log_path = os.path.join(model_dir, "train_model.log")
    try:
        proc = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            cwd=model_dir,
            timeout=1800,
        )
        with open(log_path, "w", encoding="utf-8") as fh:
            fh.write(proc.stdout + proc.stderr)
        if proc.returncode != 0:
            log_action(run_dir, "single_shot_model_worker", "script_execution_failed",
                       success=False, error=proc.stderr[-2000:])
            return {"status": "error", "error": f"Script exited {proc.returncode}",
                    "script_path": script_path, "log_path": log_path}
    except subprocess.TimeoutExpired:
        error = "Training script timed out after 1800s."
        log_action(run_dir, "single_shot_model_worker", "script_timeout", success=False, error=error)
        return {"status": "error", "error": error}
    except Exception as exc:
        log_action(run_dir, "single_shot_model_worker", "script_exec_exception", success=False, error=str(exc))
        return {"status": "error", "error": str(exc)}

    log_action(run_dir, "single_shot_model_worker", "script_executed_ok",
               output_refs=[script_path, log_path])
    return {"status": "success", "script_path": script_path, "log_path": log_path,
            "model_dir": model_dir}
