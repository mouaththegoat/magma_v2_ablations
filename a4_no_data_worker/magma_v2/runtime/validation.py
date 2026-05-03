"""Validation helpers for MAGMA v2 internal validator agents."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any

import pandas as pd

from magma_v2.runtime.artifacts import list_files, read_json, run_dir_from, stage_dir


MODALITY_TERMS = {
    "text": ("text", "note", "notes", "nlp"),
    "image": ("image", "images", "cxr", "xray", "mri", "png", "jpg", "jpeg"),
    "ecg": ("ecg", "ekg", "waveform", "wfdb", "lead"),
    "tabular": ("ehr", "table", "tabular", "structured"),
}

VALIDATION_CONTEXT_KEYS = {
    "status",
    "stage",
    "compliance_status",
    "improvement_priority",
    "debug_value",
    "blocking_findings",
    "non_blocking_findings",
    "recommended_repair_target",
    "recommended_next_action",
    "compliance_check",
    "improvement_suggestions",
    "debug_assessment",
}


def build_data_validation_bundle(run_dir: str) -> dict[str, Any]:
    task = _safe_read_json(Path(run_dir) / "task_input.json")
    handoff = _safe_read_json(Path(run_dir) / "DATA_HANDOFF.json")
    error = _safe_read_json(Path(run_dir) / "DATA_ERROR.json")
    data_dir = Path(stage_dir(run_dir, "data"))

    compliance = {
        "status": "passed",
        "contradictions": [],
        "missing_evidence": [],
        "checks": {},
    }
    improvements: list[dict[str, Any]] = []
    debug = {
        "status": "none",
        "suspected_root_causes": [],
        "suggested_repairs": [],
        "related_artifacts": [],
    }

    if not handoff:
        compliance["status"] = "failed"
        compliance["missing_evidence"].append("DATA_HANDOFF.json is missing")
        if error:
            debug["status"] = "critical"
            debug["suspected_root_causes"].append(str(error.get("message") or "data stage failed"))
        return {
            "status": "failed",
            "task_input": task,
            "data_handoff": handoff,
            "data_error": error,
            "artifact_index": list_files(str(data_dir)),
            "compliance_check": compliance,
            "improvement_suggestions": improvements,
            "debug_assessment": debug,
        }

    request = str((task or {}).get("raw_prompt") or "").lower()
    requested_modalities = _requested_modalities(request)
    actual_modalities = {str(item) for item in (handoff.get("modalities") or []) if item}
    if not actual_modalities and handoff.get("modality"):
        actual_modalities = {str(handoff.get("modality"))}
    explicit_multimodal = _explicit_multimodal_request(request)
    non_tabular_requested = requested_modalities - {"tabular"}
    if (
        (explicit_multimodal or len(non_tabular_requested) > 1)
        and len(actual_modalities - {"tabular"}) < len(non_tabular_requested)
    ):
        compliance["status"] = "failed"
        compliance["contradictions"].append(
            f"Request implies multimodal data ({sorted(requested_modalities)}), but DATA_HANDOFF modalities={sorted(actual_modalities) or ['unknown']}."
        )
        improvements.append({
            "priority": "high",
            "title": "Preserve all requested modalities in normalization",
            "rationale": "The current handoff appears to have dropped at least one requested modality.",
            "suggested_actions": ["Rebuild manifests/alignment so the normalized layout retains all requested modalities.", "Re-check notes/text directories and join keys."],
        })

    produced_paths = []
    for item in handoff.get("produced_files", []):
        if isinstance(item, dict) and item.get("path"):
            produced_paths.append(str(item["path"]))
    missing_produced = [path for path in produced_paths if not _resolve_path(run_dir, path, default_stage="data").exists()]
    if missing_produced:
        compliance["status"] = "warning" if compliance["status"] == "passed" else compliance["status"]
        compliance["missing_evidence"].extend([f"produced_files missing: {path}" for path in missing_produced[:10]])

    for field in ("train_data", "val_data", "test_data", "model_ready_data"):
        path_value = handoff.get(field)
        if not path_value:
            continue
        resolved = _resolve_path(run_dir, str(path_value), default_stage="data")
        check = {"path": str(resolved), "exists": resolved.exists()}
        if resolved.exists() and resolved.suffix.lower() in {".csv", ".tsv"}:
            try:
                frame = pd.read_csv(resolved, nrows=200)
                check["columns"] = list(frame.columns)
                check["preview_rows"] = int(len(frame))
            except Exception as exc:
                check["read_error"] = str(exc)
        compliance["checks"][field] = check
        if not resolved.exists():
            compliance["status"] = "failed"
            compliance["missing_evidence"].append(f"{field} missing: {resolved}")

    if handoff.get("patient_id_column") is None and any(term in request for term in ("patient", "group", "leakage")):
        improvements.append({
            "priority": "medium",
            "title": "Strengthen patient/group leakage controls",
            "rationale": "The request references patient- or leakage-aware splitting, but patient_id_column is absent.",
            "suggested_actions": ["Inspect candidate identifier columns.", "Document patient/group split enforcement explicitly in DATA_HANDOFF."],
        })

    if handoff.get("unresolved_questions"):
        improvements.append({
            "priority": "medium",
            "title": "Resolve outstanding data ambiguities",
            "rationale": "Unresolved questions remain in the final handoff.",
            "suggested_actions": [str(item) for item in handoff.get("unresolved_questions", [])[:5]],
        })

    if error:
        debug["status"] = "useful"
        debug["suspected_root_causes"].append(str(error.get("message") or "data stage error present"))
        debug["related_artifacts"].append(str(Path(run_dir) / "DATA_ERROR.json"))

    if _canonical_target_alias_matches_request(task, handoff):
        compliance["contradictions"] = [
            item
            for item in compliance["contradictions"]
            if not _is_target_name_mismatch(item)
        ]
        if not compliance["contradictions"] and compliance["status"] == "failed" and not compliance["missing_evidence"]:
            compliance["status"] = "passed"

    overall = "failed" if compliance["status"] == "failed" else "warning" if compliance["status"] == "warning" or improvements else "passed"
    return {
        "status": overall,
        "task_input": task,
        "data_handoff": handoff,
        "data_error": error,
        "artifact_index": list_files(str(data_dir)),
        "compliance_check": compliance,
        "improvement_suggestions": improvements,
        "debug_assessment": debug,
    }


def build_model_validation_bundle(run_dir: str) -> dict[str, Any]:
    task = _safe_read_json(Path(run_dir) / "task_input.json")
    data_handoff = _safe_read_json(Path(run_dir) / "DATA_HANDOFF.json")
    research_handoff = _safe_read_json(Path(run_dir) / "RESEARCH_HANDOFF.json")
    model_handoff = _safe_read_json(Path(run_dir) / "MODEL_HANDOFF.json")
    model_error = _safe_read_json(Path(run_dir) / "MODEL_ERROR.json")
    model_dir = Path(stage_dir(run_dir, "model"))

    compliance = {
        "status": "passed",
        "contradictions": [],
        "missing_evidence": [],
        "checks": {},
    }
    improvements: list[dict[str, Any]] = []
    debug = {
        "status": "none",
        "suspected_root_causes": [],
        "suggested_repairs": [],
        "related_artifacts": [],
    }

    if not model_handoff:
        compliance["status"] = "failed"
        compliance["missing_evidence"].append("MODEL_HANDOFF.json is missing")
        if model_error:
            debug["status"] = "critical"
            debug["suspected_root_causes"].append(str(model_error.get("message") or "model stage failed"))
            debug["related_artifacts"].append(str(Path(run_dir) / "MODEL_ERROR.json"))
        return {
            "status": "failed",
            "task_input": task,
            "data_handoff": data_handoff,
            "research_handoff": research_handoff,
            "model_handoff": model_handoff,
            "model_error": model_error,
            "artifact_index": list_files(str(model_dir)),
            "compliance_check": compliance,
            "improvement_suggestions": improvements,
            "debug_assessment": debug,
        }

    evaluation = model_handoff.get("evaluation_artifacts") or {}
    metrics_path = _resolve_path(run_dir, evaluation.get("metrics_path") or _find_produced_file(model_handoff, "metrics"), default_stage="model")
    predictions_path = _resolve_path(run_dir, evaluation.get("predictions_path") or _find_produced_file(model_handoff, "prediction"), default_stage="model")
    log_path = _resolve_path(run_dir, evaluation.get("training_log_path") or _find_produced_file(model_handoff, "log"), default_stage="model")

    metrics_payload = _safe_read_json(metrics_path) if metrics_path and metrics_path.exists() else None
    prediction_summary = _prediction_summary(predictions_path) if predictions_path and predictions_path.exists() else None
    compliance["checks"]["metrics_path"] = {"path": str(metrics_path) if metrics_path else None, "exists": bool(metrics_path and metrics_path.exists())}
    compliance["checks"]["predictions_path"] = {"path": str(predictions_path) if predictions_path else None, "exists": bool(predictions_path and predictions_path.exists())}
    if not metrics_path or not metrics_path.exists():
        compliance["status"] = "failed"
        compliance["missing_evidence"].append("metrics artifact missing")
    if not predictions_path or not predictions_path.exists():
        compliance["status"] = "failed"
        compliance["missing_evidence"].append("predictions artifact missing")

    split_metrics = _extract_split_metrics(metrics_payload)
    train_auroc = _metric_value(split_metrics.get("train"), "auroc")
    val_auroc = _metric_value(split_metrics.get("val"), "auroc") or _metric_value(split_metrics.get("validation"), "auroc")
    test_auroc = _metric_value(split_metrics.get("test"), "auroc")
    val_ap = _metric_value(split_metrics.get("val"), "average_precision") or _metric_value(split_metrics.get("validation"), "average_precision")
    test_ap = _metric_value(split_metrics.get("test"), "average_precision")
    prevalence = _metric_value(split_metrics.get("test"), "prevalence") or _metric_value(split_metrics.get("val"), "prevalence")

    if train_auroc is not None and test_auroc is not None and train_auroc - test_auroc > 0.1:
        improvements.append({
            "priority": "high",
            "title": "Reduce overfitting",
            "rationale": f"Train AUROC ({train_auroc:.3f}) is much higher than test AUROC ({test_auroc:.3f}).",
            "suggested_actions": ["Try a simpler baseline or stronger regularization.", "Reduce model capacity or tune with stricter validation.", "Add unimodal baselines/ablations before complex fusion."],
        })

    if prevalence is not None and test_ap is not None and test_ap < max(prevalence * 2, 0.1):
        improvements.append({
            "priority": "high",
            "title": "Strengthen imbalance handling",
            "rationale": f"Test AUPRC ({test_ap:.3f}) remains low relative to class prevalence ({prevalence:.3f}).",
            "suggested_actions": ["Try class weighting or focal/weighted loss.", "Tune threshold-independent training objectives.", "Inspect positive-class coverage and hard false negatives."],
        })
    elif val_ap is not None and val_ap < 0.2:
        improvements.append({
            "priority": "medium",
            "title": "Improve discriminative signal for the positive class",
            "rationale": f"Validation AUPRC is low ({val_ap:.3f}).",
            "suggested_actions": ["Try feature engineering or modality-specific encoders.", "Review class imbalance strategy.", "Inspect error examples for systematic misses."],
        })

    calibration_status = str(model_handoff.get("calibration_status") or "").lower()
    if calibration_status in {"failed", "unavailable", ""}:
        improvements.append({
            "priority": "medium",
            "title": "Improve calibration",
            "rationale": "Calibration is unavailable or failed.",
            "suggested_actions": ["Add post-hoc validation-set calibration.", "Track Brier score and calibration curve explicitly.", "Check whether probability outputs are too extreme."],
        })

    subgroup_status = str(model_handoff.get("subgroup_eval_status") or "").lower()
    if subgroup_status in {"unavailable", ""}:
        improvements.append({
            "priority": "low",
            "title": "Add subgroup evaluation",
            "rationale": "Subgroup metrics are unavailable.",
            "suggested_actions": ["Evaluate performance across available demographic or acquisition strata.", "Document unavailable subgroup keys if none exist."],
        })

    if data_handoff and str(data_handoff.get("normalized_layout")) == "multimodal_alignment_manifest":
        improvements.append({
            "priority": "medium",
            "title": "Add modality ablations and fusion comparisons",
            "rationale": "Multimodal runs should compare unimodal baselines with at least one fusion baseline.",
            "suggested_actions": ["Run image-only and text-only baselines.", "Compare late fusion vs compact early fusion.", "Document whether multimodality improves AUROC/AUPRC."],
        })

    search_summary = model_handoff.get("model_search_summary")
    if not isinstance(search_summary, dict):
        search_summary = {}
    search_mode = str(search_summary.get("search_mode") or "")
    candidate_results_path = search_summary.get("candidate_results_path")
    best_selection_path = search_summary.get("best_model_selection_path")
    tuning_path = search_summary.get("hyperparameter_search_path")
    final_training_path = search_summary.get("final_training_plan_path") or search_summary.get("final_training_plan")
    layout = str((data_handoff or {}).get("normalized_layout") or "")
    modalities = {str(item) for item in ((data_handoff or {}).get("modalities") or [])}
    is_tabular = layout in {"tabular_dataset", "patient_grouped_row_level_tabular_splits"} or (data_handoff or {}).get("modality") == "tabular" or "tabular" in modalities
    if is_tabular:
        if not candidate_results_path or not _resolve_path(run_dir, candidate_results_path, default_stage="model").exists():
            improvements.append({
                "priority": "high",
                "title": "Compare multiple tabular model families",
                "rationale": "Tabular runs should compare multiple cheap candidates before selecting the winner.",
                "suggested_actions": ["Evaluate at least logistic regression, random forest, and a boosting baseline.", "Write model/candidate_results.json and select the best validation model explicitly."],
            })
        if not best_selection_path or not _resolve_path(run_dir, best_selection_path, default_stage="model").exists():
            improvements.append({
                "priority": "medium",
                "title": "Write an explicit best-model selection artifact",
                "rationale": "The run does not record how the winning model was chosen.",
                "suggested_actions": ["Write model/best_model_selection.json with ranking, winner, and selection metric.", "Summarize selected hyperparameters in MODEL_HANDOFF."],
            })
        if not tuning_path or not _resolve_path(run_dir, tuning_path, default_stage="model").exists():
            improvements.append({
                "priority": "high",
                "title": "Run bounded hyperparameter optimization",
                "rationale": "Hyperparameter optimization is part of the required model-stage contract for tabular runs.",
                "suggested_actions": ["Write model/hyperparameter_search.json with the search budget, tried parameters, validation AUROC/AUPRC, and best parameters.", "If tuning is infeasible, write an explicit infeasibility record in hyperparameter_search.json."],
            })

    if search_mode in {"targeted_improvement", "heavy_exploration"}:
        if not tuning_path or not _resolve_path(run_dir, tuning_path, default_stage="model").exists():
            improvements.append({
                "priority": "medium",
                "title": "Record bounded hyperparameter tuning results",
                "rationale": "The run claims a stronger search mode but does not expose tuning artifacts.",
                "suggested_actions": ["Write model/hyperparameter_search.json with the tuning records or infeasibility rationale.", "Record the chosen search budget and best parameters."],
            })

    if candidate_results_path and best_selection_path:
        final_training_exists = bool(final_training_path and _resolve_path(run_dir, final_training_path, default_stage="model").exists())
        final_infeasible = any(
            str(search_summary.get(key) or "").lower() in {"infeasible", "skipped", "not_feasible", "unavailable"}
            for key in ("final_training_status", "final_retraining_status", "final_refit_status")
        )
        if not final_training_exists and not final_infeasible:
            improvements.append({
                "priority": "medium",
                "title": "Finalize selected model after candidate search",
                "rationale": "The run records candidate search and best-model selection, but does not expose a final training plan showing whether the selected configuration was retrained/continued with a stronger budget before test evaluation.",
                "suggested_actions": [
                    "Write model/final_training_plan.json with search_budget versus final_training_budget, selected config, checkpoint rule, and final_model_path.",
                    "If final retraining is infeasible, record final_training_status='infeasible' with the runtime or environment reason.",
                    "Keep the test set untouched until the final selected checkpoint is evaluated once.",
                ],
            })

    if prediction_summary and split_metrics:
        if prediction_summary.get("splits") and not {"test", "val"} & set(prediction_summary["splits"]):
            compliance["status"] = "warning" if compliance["status"] == "passed" else compliance["status"]
            compliance["contradictions"].append("Predictions artifact does not expose expected validation/test split names.")

    if model_error:
        debug["status"] = "useful"
        debug["suspected_root_causes"].append(str(model_error.get("message") or "model stage error present"))
        debug["related_artifacts"].append(str(Path(run_dir) / "MODEL_ERROR.json"))
    if log_path and log_path.exists():
        tail = _tail_text(log_path, 80)
        if "Traceback" in tail or "Error" in tail or "Exception" in tail:
            debug["status"] = "critical" if debug["status"] == "none" else debug["status"]
            debug["suspected_root_causes"].append("Execution log contains traceback/error output.")
            debug["suggested_repairs"].append("Inspect the final traceback in the model log and patch the failing subphase instead of retraining blindly.")
            debug["related_artifacts"].append(str(log_path))

    overall = "failed" if compliance["status"] == "failed" else "warning" if compliance["status"] == "warning" or improvements or debug["status"] != "none" else "passed"
    return {
        "status": overall,
        "task_input": task,
        "data_handoff": data_handoff,
        "research_handoff": research_handoff,
        "model_handoff": model_handoff,
        "model_error": model_error,
        "artifact_index": list_files(str(model_dir)),
        "prediction_summary": prediction_summary,
        "metrics_summary": split_metrics,
        "compliance_check": compliance,
        "improvement_suggestions": improvements,
        "debug_assessment": debug,
    }


def normalize_validation_report(payload: dict[str, Any], stage: str) -> dict[str, Any]:
    compliance = payload.get("compliance_check") if isinstance(payload.get("compliance_check"), dict) else {}
    improvements = payload.get("improvement_suggestions") if isinstance(payload.get("improvement_suggestions"), list) else []
    debug = payload.get("debug_assessment") if isinstance(payload.get("debug_assessment"), dict) else {}
    compliance_status = str(compliance.get("status") or "warning").strip().lower()
    blocking_findings = []
    blocking_findings.extend(str(item) for item in compliance.get("contradictions", []) if item)
    blocking_findings.extend(str(item) for item in compliance.get("missing_evidence", []) if item)
    if compliance_status not in {"passed", "warning", "failed"}:
        compliance_status = "failed" if blocking_findings else "warning"
    non_blocking_findings = []
    non_blocking_findings.extend(str(item.get("title")) for item in improvements if isinstance(item, dict) and item.get("title"))
    has_high_priority_improvement = any(
        isinstance(item, dict) and str(item.get("priority") or "").strip().lower() == "high"
        for item in improvements
    )
    recommended_repair_target = payload.get("recommended_repair_target") or stage
    recommended_next_action = payload.get("recommended_next_action")
    if not recommended_next_action:
        if compliance_status == "failed":
            recommended_next_action = "rerun_data_worker" if stage == "data" else "rerun_model_worker"
        elif stage == "model" and has_high_priority_improvement:
            recommended_next_action = "rerun_model_worker_for_improvement"
        elif improvements:
            recommended_next_action = "continue_with_warnings"
        else:
            recommended_next_action = "proceed"
    elif compliance_status == "failed" and str(recommended_next_action).strip().lower() in {"continue_with_warnings", "proceed"}:
        recommended_next_action = "rerun_data_worker" if stage == "data" else "rerun_model_worker"
    return {
        "stage": stage,
        "status": "failed" if compliance_status == "failed" else (payload.get("status") or compliance_status),
        "compliance_status": compliance_status,
        "improvement_priority": _improvement_priority(improvements),
        "debug_value": debug.get("status") or "none",
        "blocking_findings": blocking_findings,
        "non_blocking_findings": non_blocking_findings,
        "recommended_repair_target": recommended_repair_target,
        "recommended_next_action": recommended_next_action,
        "compliance_check": compliance,
        "improvement_suggestions": improvements,
        "debug_assessment": debug,
    }


def summarize_validation_bundle(payload: dict[str, Any]) -> dict[str, Any]:
    summary = {key: payload.get(key) for key in VALIDATION_CONTEXT_KEYS if key in payload}
    compliance = summary.get("compliance_check")
    if isinstance(compliance, dict):
        checks = compliance.get("checks")
        if isinstance(checks, dict):
            compact_checks: dict[str, Any] = {}
            for name, item in checks.items():
                if not isinstance(item, dict):
                    compact_checks[name] = item
                    continue
                compact_checks[name] = {
                    "path": item.get("path"),
                    "exists": item.get("exists"),
                    "columns": item.get("columns", [])[:12] if isinstance(item.get("columns"), list) else item.get("columns"),
                    "preview_rows": item.get("preview_rows"),
                    "read_error": item.get("read_error"),
                }
            compliance["checks"] = compact_checks
    improvements = summary.get("improvement_suggestions")
    if isinstance(improvements, list):
        summary["improvement_suggestions"] = improvements[:5]
    debug = summary.get("debug_assessment")
    if isinstance(debug, dict):
        debug["related_artifacts"] = list(debug.get("related_artifacts", []))[:8]
        debug["suggested_repairs"] = list(debug.get("suggested_repairs", []))[:8]
    return summary


def safe_read_validation_artifact(run_dir: str, path: str, default_stage: str, max_lines: int = 20) -> dict[str, Any]:
    resolved = _resolve_path(run_dir, path, default_stage=default_stage)
    run_root = Path(run_dir).resolve()
    if run_root not in [resolved, *resolved.parents]:
        return {"status": "error", "error": f"Artifact path is outside run directory: {resolved}"}
    if not resolved.exists():
        return {"status": "error", "error": f"Artifact not found: {resolved}"}
    if resolved.suffix.lower() in {".json"}:
        try:
            return {"status": "success", "path": str(resolved), "content": read_json(str(resolved))}
        except Exception as exc:
            return {"status": "error", "error": str(exc), "path": str(resolved)}
    if resolved.suffix.lower() in {".csv", ".tsv"}:
        try:
            sep = "\t" if resolved.suffix.lower() == ".tsv" else ","
            frame = pd.read_csv(resolved, sep=sep, nrows=max_lines)
            return {
                "status": "success",
                "path": str(resolved),
                "preview": frame.to_dict(orient="records"),
                "columns": list(frame.columns),
            }
        except Exception as exc:
            return {"status": "error", "error": str(exc), "path": str(resolved)}
    return {"status": "success", "path": str(resolved), "text": _tail_text(resolved, max_lines)}


def _canonical_target_alias_matches_request(task: dict[str, Any] | None, handoff: dict[str, Any]) -> bool:
    request = str((task or {}).get("raw_prompt") or "")
    requested_target = _requested_target_name(request)
    if not requested_target:
        return False
    canonical_target = str(handoff.get("target") or (handoff.get("label_spec") or {}).get("target_name") or "").strip()
    source_target = str((handoff.get("label_spec") or {}).get("source_name") or "").strip()
    return canonical_target == "label" and source_target.lower() == requested_target.lower()


def _requested_target_name(request: str) -> str | None:
    match = re.search(r"target\s+is\s+([A-Za-z0-9_]+)", request, re.IGNORECASE)
    return match.group(1) if match else None


def _is_target_name_mismatch(item: Any) -> bool:
    if isinstance(item, dict):
        return str(item.get("type") or "").strip().lower() == "target_name_mismatch"
    text = str(item).lower()
    return "target_name_mismatch" in text


def _safe_read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = read_json(str(path))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _requested_modalities(request: str) -> set[str]:
    normalized = request.lower()
    tokens = set(_prompt_tokens(normalized))
    found: set[str] = set()
    for modality, terms in MODALITY_TERMS.items():
        if any(term in tokens for term in terms):
            found.add(modality)
    if "x ray" in normalized or "x-ray" in normalized:
        found.add("image")
    return found


def _explicit_multimodal_request(request: str) -> bool:
    normalized = request.lower()
    return any(
        phrase in normalized
        for phrase in (
            "multimodal",
            "multi modal",
            "image plus text",
            "text plus image",
            "image and text",
            "text and image",
            "fusion",
            "cross-modal",
            "cross modal",
        )
    )


def _prompt_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def _resolve_path(run_dir: str, path: str | None, default_stage: str | None = None) -> Path:
    if not path:
        return Path(run_dir)
    candidate = Path(str(path)).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    run_root = Path(run_dir).resolve()
    if candidate.parts and candidate.parts[0] in {"data", "model", "scratch"}:
        return (run_root / candidate).resolve()
    if default_stage:
        return (Path(stage_dir(run_dir, default_stage)).resolve() / candidate).resolve()
    return (run_root / candidate).resolve()


def _find_produced_file(model_handoff: dict[str, Any], needle: str) -> str | None:
    for item in model_handoff.get("produced_files", []):
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or "")
        role = str(item.get("role") or "")
        purpose = str(item.get("purpose") or "")
        if needle in path.lower() or needle in role.lower() or needle in purpose.lower():
            return path
    return None


def _prediction_summary(path: Path) -> dict[str, Any] | None:
    try:
        frame = pd.read_csv(path)
    except Exception:
        return None
    splits = sorted({str(value) for value in frame.get("split", pd.Series(dtype=str)).dropna().unique()}) if "split" in frame.columns else []
    return {"rows": int(len(frame)), "columns": list(frame.columns), "splits": splits}


def _extract_split_metrics(metrics_payload: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not isinstance(metrics_payload, dict):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for key in ("train", "val", "validation", "test"):
        value = metrics_payload.get(key)
        if isinstance(value, dict):
            result[key] = value
    metrics_nested = metrics_payload.get("metrics")
    if isinstance(metrics_nested, dict):
        for key in ("train", "val", "validation", "test"):
            value = metrics_nested.get(key)
            if isinstance(value, dict):
                result[key] = value
    return result


def _metric_value(section: dict[str, Any] | None, key: str) -> float | None:
    if not isinstance(section, dict):
        return None
    value = section.get(key)
    try:
        return None if value is None else float(value)
    except Exception:
        return None


def _tail_text(path: Path, max_lines: int) -> str:
    try:
        text = path.read_text(errors="replace")
    except Exception:
        return ""
    lines = text.splitlines()
    return "\n".join(lines[-max_lines:])


def _improvement_priority(improvements: list[dict[str, Any]]) -> str:
    order = {"high": 3, "medium": 2, "low": 1}
    score = max((order.get(str(item.get("priority")).lower(), 0) for item in improvements if isinstance(item, dict)), default=0)
    return {3: "high", 2: "medium", 1: "low"}.get(score, "none")
