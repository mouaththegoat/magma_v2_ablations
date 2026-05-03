"""Clinical judge and reward writer for MAGMA v2."""

from __future__ import annotations

import json
import os
import ast
from pathlib import Path
from typing import Any

import pandas as pd
from strands import Agent, tool

from magma_v2.model_config import create_agent_model
from magma_v2.runtime.actions import compute_reward, log_action
from magma_v2.runtime.artifacts import read_json, run_dir_from, stage_dir, write_json
from magma_v2.runtime.literature import research_is_stale
from magma_v2.runtime.validation import build_data_validation_bundle, normalize_validation_report


TRIPOD_ITEMS = [
    "title_identifies_prediction_model",
    "background_and_objectives",
    "objectives_specified",
    "data_sources_described",
    "eligibility_criteria_defined",
    "data_preparation_described",
    "outcome_clearly_defined",
    "sample_size_reported",
    "missing_data_handled",
    "analytical_methods_described",
    "class_imbalance_addressed",
    "fairness_addressed",
    "model_output_described",
    "train_eval_differences_described",
    "model_specification_provided",
    "model_performance_reported",
    "interpretation_provided",
    "limitations_discussed",
    "usability_described",
]

TRUTH_COLUMN_ALIASES = ("y_true", "label", "target", "ground_truth")
PREDICTION_COLUMN_ALIASES = ("y_pred", "prediction", "predicted_label", "predicted_value")
SCORE_COLUMN_ALIASES = ("y_score", "probability", "selected_probability", "raw_probability", "score", "logit")
SPLIT_COLUMN_ALIASES = ("split", "dataset_split", "fold", "partition")
ID_COLUMN_ALIASES = (
    "sample_id",
    "patient_id",
    "encounter_id",
    "study_id",
    "image_id",
    "note_id",
    "document_id",
    "ecg_id",
    "record_id",
)
BINARY_ARTIFACT_SUFFIXES = {
    ".pkl",
    ".pickle",
    ".joblib",
    ".pt",
    ".pth",
    ".onnx",
    ".npy",
    ".npz",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".tif",
    ".tiff",
    ".wav",
    ".mp3",
    ".mp4",
    ".avi",
}


def _run_dir(tool_context: object | None, artifact_dir: str = "./artifacts") -> str:
    if isinstance(tool_context, dict):
        invocation_state = tool_context.get("invocation_state", {})
    else:
        invocation_state = getattr(tool_context, "invocation_state", {}) if tool_context else {}
    return run_dir_from(invocation_state.get("run_dir") or artifact_dir)


@tool(context=True)
def read_pipeline_handoffs(artifact_dir: str = "./artifacts", tool_context: object | None = None) -> dict:
    """Read task, data, optional research, and model handoffs."""
    run_dir = _run_dir(tool_context, artifact_dir)
    payload = {}
    for filename in ("task_input.json", "DATA_HANDOFF.json", "RESEARCH_HANDOFF.json", "MODEL_HANDOFF.json"):
        path = os.path.join(run_dir, filename)
        payload[filename] = read_json(path) if os.path.exists(path) else None
    return {"status": "success", "run_dir": run_dir, "payload": payload, "tripod_items": TRIPOD_ITEMS}


@tool(context=True)
def read_pipeline_bundle(artifact_dir: str = "./artifacts", tool_context: object | None = None) -> dict:
    """Read canonical handoffs plus rich model artifacts, code, logs, and audit checks."""
    run_dir = _run_dir(tool_context, artifact_dir)
    payload = _read_core_payload(run_dir)
    audit = _build_judge_audit(run_dir, payload)
    return {"status": "success", "run_dir": run_dir, "payload": payload, "audit": audit, "tripod_items": TRIPOD_ITEMS}


@tool(context=True)
def read_judge_retry_signals(artifact_dir: str = "./artifacts", tool_context: object | None = None) -> dict:
    """Read compact retry-routing signals recomputed from validation helpers and first-party artifacts."""
    run_dir = _run_dir(tool_context, artifact_dir)
    payload = _read_core_payload(run_dir)
    audit = _build_judge_audit(run_dir, payload)
    return {"status": "success", "run_dir": run_dir, "judge_retry_signals": audit["judge_retry_signals"]}


@tool(context=True)
def read_model_artifact(path: str, artifact_dir: str = "./artifacts", tool_context: object | None = None) -> dict:
    """Read a model artifact as JSON, CSV summary, or text snippet."""
    run_dir = _run_dir(tool_context, artifact_dir)
    resolved = _resolve_model_reference(run_dir, path)
    if resolved is None or not resolved.exists():
        return {"status": "error", "error": f"Artifact not found: {path}", "resolved_path": str(resolved) if resolved else None}
    return {"status": "success", "artifact": _read_artifact_summary(resolved)}


@tool(context=True)
def inspect_predictions_consistency(artifact_dir: str = "./artifacts", tool_context: object | None = None) -> dict:
    """Check predictions and metrics consistency from MODEL_HANDOFF evaluation artifacts."""
    run_dir = _run_dir(tool_context, artifact_dir)
    payload = _read_core_payload(run_dir)
    return {"status": "success", "run_dir": run_dir, "consistency": _inspect_predictions_consistency(run_dir, payload)}


@tool(context=True)
def inspect_code_artifact(artifact_dir: str = "./artifacts", tool_context: object | None = None) -> dict:
    """Inspect generated model code for contract, leakage, and calibration risks."""
    run_dir = _run_dir(tool_context, artifact_dir)
    payload = _read_core_payload(run_dir)
    return {"status": "success", "run_dir": run_dir, "code_audit": _inspect_code_artifact(run_dir, payload)}


@tool(context=True)
def inspect_logs(artifact_dir: str = "./artifacts", tool_context: object | None = None) -> dict:
    """Summarize model execution logs and suspicious log patterns."""
    run_dir = _run_dir(tool_context, artifact_dir)
    payload = _read_core_payload(run_dir)
    return {"status": "success", "run_dir": run_dir, "logs": _inspect_logs(run_dir, payload)}


@tool(context=True)
def write_judge_verdict(
    verdict_json: object,
    reward_lambda: float = 0.5,
    artifact_dir: str = "./artifacts",
    tool_context: object | None = None,
) -> dict:
    """Write judge_verdict.json and reward.json using RT = lambda AUROC + (1-lambda) TRIPODScore."""
    run_dir = _run_dir(tool_context, artifact_dir)
    verdict, error = _parse_json_payload(verdict_json, "verdict_json")
    if error:
        return {"status": "error", "error": error}
    assert verdict is not None
    if "judge_retry_signals" not in verdict or "artifact_consistency_summary" not in verdict or "improvement_plan" not in verdict:
        payload = _read_core_payload(run_dir)
        audit = _build_judge_audit(run_dir, payload)
        verdict.setdefault("judge_retry_signals", audit["judge_retry_signals"])
        verdict.setdefault("artifact_consistency_summary", _default_artifact_consistency_summary(audit))
        verdict.setdefault("improvement_plan", _default_improvement_plan(audit["judge_retry_signals"]))

    auroc = verdict.get("auroc")
    tripod_score = float(verdict.get("tripod_score", 0.0))
    reward = compute_reward(auroc, tripod_score, reward_lambda)
    verdict["reward_lambda"] = reward_lambda
    verdict["reward"] = reward
    verdict_path = os.path.join(run_dir, "judge_verdict.json")
    reward_path = os.path.join(run_dir, "reward.json")
    write_json(verdict_path, verdict)
    write_json(reward_path, {
        "equation": "RT = lambda * AUROC + (1 - lambda) * TRIPODScore",
        "lambda": reward_lambda,
        "auroc": auroc,
        "tripod_score": tripod_score,
        "reward": reward,
    })
    log_action(run_dir, "judge", "write_judge_verdict", output_refs=[verdict_path, reward_path])
    return {"status": "success", "verdict_path": verdict_path, "reward_path": reward_path, "reward": reward}


def _read_core_payload(run_dir: str) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for filename in ("task_input.json", "DATA_HANDOFF.json", "RESEARCH_HANDOFF.json", "MODEL_HANDOFF.json"):
        path = os.path.join(run_dir, filename)
        payload[filename] = read_json(path) if os.path.exists(path) else None
    return payload


def _build_judge_audit(run_dir: str, payload: dict[str, Any]) -> dict[str, Any]:
    artifact_audit = _inspect_artifact_completeness(run_dir, payload)
    prediction_audit = _inspect_predictions_consistency(run_dir, payload)
    code_audit = _inspect_code_artifact(run_dir, payload)
    log_audit = _inspect_logs(run_dir, payload)
    research_audit = _inspect_research_grounding(payload)
    retry_signals = _build_judge_retry_signals(
        run_dir=run_dir,
        payload=payload,
        artifact_audit=artifact_audit,
        prediction_audit=prediction_audit,
        code_audit=code_audit,
        log_audit=log_audit,
    )

    missing_artifacts = artifact_audit["missing_artifacts"]
    hard_failures = []
    hard_failures.extend(missing_artifacts)
    hard_failures.extend(prediction_audit["critical_issues"])
    hard_failures.extend(code_audit["critical_issues"])
    hard_failures.extend(log_audit["critical_issues"])

    warnings = []
    warnings.extend(artifact_audit["warnings"])
    warnings.extend(prediction_audit["warnings"])
    warnings.extend(code_audit["warnings"])
    warnings.extend(log_audit["warnings"])
    warnings.extend(research_audit["warnings"])

    return {
        "technical_correctness_status": "failed" if hard_failures else "warning" if warnings else "passed",
        "code_audit_status": code_audit["status"],
        "artifact_consistency_status": artifact_audit["status"] if artifact_audit["status"] == "failed" else prediction_audit["status"],
        "leakage_audit_status": code_audit["leakage_status"],
        "calibration_audit_status": artifact_audit["calibration_status"],
        "subgroup_eval_status": artifact_audit["subgroup_status"],
        "code_issues": code_audit["issues"],
        "artifact_issues": artifact_audit["issues"],
        "leakage_issues": code_audit["leakage_issues"],
        "metric_consistency_issues": prediction_audit["issues"],
        "missing_artifacts": missing_artifacts,
        "suspicious_patterns": code_audit["suspicious_patterns"] + log_audit["suspicious_patterns"],
        "warnings": warnings,
        "critical_failures": hard_failures,
        "research_grounding_status": research_audit["status"],
        "research_issues": research_audit["issues"],
        "research_audit": research_audit,
        "artifact_audit": artifact_audit,
        "prediction_audit": prediction_audit,
        "code_audit": code_audit,
        "log_audit": log_audit,
        "judge_retry_signals": retry_signals,
    }


def _build_judge_retry_signals(
    *,
    run_dir: str,
    payload: dict[str, Any],
    artifact_audit: dict[str, Any],
    prediction_audit: dict[str, Any],
    code_audit: dict[str, Any],
    log_audit: dict[str, Any],
) -> dict[str, Any]:
    metrics = _load_metrics_artifact(run_dir, payload)
    reward = _load_reward_artifact(run_dir)
    return {
        "data_gate": _build_stage_gate(run_dir, "data"),
        "artifact_gate": _build_artifact_gate(artifact_audit, prediction_audit, code_audit, log_audit),
        "generalization_gap": _build_generalization_gap(metrics),
        "auprc_vs_prevalence": _build_auprc_vs_prevalence(metrics),
        "calibration_status": _build_calibration_signal(run_dir, payload, metrics),
        "model_search_completeness": _build_model_search_signal(run_dir, payload, metrics),
        "loss_convergence": _build_loss_convergence_signal(run_dir, payload),
        "reward_components": {
            "auroc": _number_or_none(reward.get("auroc")),
            "tripod_score": _number_or_none(reward.get("tripod_score")),
            "reward": _number_or_none(reward.get("reward")),
            "reward_lambda": _number_or_none(reward.get("lambda")),
        },
    }


def _build_stage_gate(run_dir: str, stage: str) -> dict[str, Any]:
    try:
        if stage == "data":
            report = normalize_validation_report(build_data_validation_bundle(run_dir), stage="data")
            owner = "data"
        else:
            return {
                "status": "unknown",
                "blocking": False,
                "owner_if_failed": stage,
                "reasons": [f"No judge retry gate is defined for stage={stage}."],
            }
    except Exception as exc:
        return {
            "status": "failed",
            "blocking": True,
            "owner_if_failed": owner if "owner" in locals() else stage,
            "reasons": [f"{stage} validation signal could not be computed: {exc}"],
        }
    blocking = [str(item) for item in report.get("blocking_findings", [])]
    reasons = blocking or [str(item) for item in report.get("non_blocking_findings", [])]
    status = str(report.get("status") or report.get("compliance_status") or "unknown")
    compliance_status = str(report.get("compliance_status") or status)
    return {
        "status": status,
        "blocking": bool(blocking or compliance_status == "failed"),
        "owner_if_failed": owner,
        "reasons": reasons,
    }


def _build_artifact_gate(
    artifact_audit: dict[str, Any],
    prediction_audit: dict[str, Any],
    code_audit: dict[str, Any],
    log_audit: dict[str, Any],
) -> dict[str, Any]:
    missing = list(artifact_audit.get("missing_artifacts", []))
    contradictions: list[str] = []
    contradictions.extend(str(item) for item in artifact_audit.get("issues", []))
    contradictions.extend(str(item) for item in prediction_audit.get("issues", []))
    contradictions.extend(str(item) for item in log_audit.get("critical_issues", []))
    status = "failed" if missing or contradictions else "warning" if artifact_audit.get("warnings") or prediction_audit.get("warnings") else "passed"
    return {
        "status": status,
        "missing_core_artifacts": missing,
        "contradictions": contradictions,
    }


def _load_metrics_artifact(run_dir: str, payload: dict[str, Any]) -> dict[str, Any]:
    model_handoff = payload.get("MODEL_HANDOFF.json") or {}
    evaluation = model_handoff.get("evaluation_artifacts") or {}
    candidates = [
        evaluation.get("metrics_path"),
        _find_model_artifact_path(run_dir, "metrics.json"),
    ]
    for candidate in candidates:
        path = _resolve_model_reference(run_dir, candidate)
        if path is not None and path.exists():
            try:
                loaded = read_json(str(path))
            except Exception:
                continue
            return loaded if isinstance(loaded, dict) else {}
    return {}


def _load_reward_artifact(run_dir: str) -> dict[str, Any]:
    path = Path(run_dir) / "reward.json"
    if not path.exists():
        return {}
    try:
        loaded = read_json(str(path))
    except Exception:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _build_generalization_gap(metrics: dict[str, Any]) -> dict[str, Any]:
    train_auroc = _metric_from_split(metrics, "train", "auroc")
    val_auroc = _metric_from_split(metrics, "val", "auroc") or _metric_from_split(metrics, "validation", "auroc")
    test_auroc = _metric_from_split(metrics, "test", "auroc")
    train_auprc = _metric_from_split(metrics, "train", "auprc") or _metric_from_split(metrics, "train", "average_precision")
    val_auprc = _metric_from_split(metrics, "val", "auprc") or _metric_from_split(metrics, "val", "average_precision") or _metric_from_split(metrics, "validation", "auprc") or _metric_from_split(metrics, "validation", "average_precision")
    test_auprc = _metric_from_split(metrics, "test", "auprc") or _metric_from_split(metrics, "test", "average_precision")
    train_test_gap = train_auroc - test_auroc if train_auroc is not None and test_auroc is not None else None
    train_val_gap = train_auroc - val_auroc if train_auroc is not None and val_auroc is not None else None
    val_test_gap = val_auroc - test_auroc if val_auroc is not None and test_auroc is not None else None
    assessment = "unknown"
    if train_test_gap is not None and train_test_gap >= 0.10:
        assessment = "overfitting"
    elif train_auroc is not None and val_auroc is not None and train_auroc < 0.65 and val_auroc < 0.65:
        assessment = "underfitting"
    elif val_test_gap is not None and val_test_gap >= 0.08:
        assessment = "split_shift"
    elif train_auroc is not None and val_auroc is not None and test_auroc is not None:
        assessment = "acceptable"
    return {
        "train_auroc": train_auroc,
        "val_auroc": val_auroc,
        "test_auroc": test_auroc,
        "train_test_auroc_gap": train_test_gap,
        "train_val_auroc_gap": train_val_gap,
        "val_test_auroc_gap": val_test_gap,
        "train_auprc": train_auprc,
        "val_auprc": val_auprc,
        "test_auprc": test_auprc,
        "assessment": assessment,
    }


def _build_auprc_vs_prevalence(metrics: dict[str, Any]) -> dict[str, Any]:
    test_auprc = _metric_from_split(metrics, "test", "auprc") or _metric_from_split(metrics, "test", "average_precision")
    prevalence = _metric_from_split(metrics, "test", "positive_rate") or _metric_from_split(metrics, "test", "prevalence")
    lift = test_auprc / prevalence if test_auprc is not None and prevalence not in (None, 0) else None
    assessment = "unknown"
    if lift is not None:
        assessment = "weak_signal" if lift < 1.5 else "useful_signal"
    return {
        "test_auprc": test_auprc,
        "test_prevalence": prevalence,
        "lift": lift,
        "assessment": assessment,
    }


def _build_calibration_signal(run_dir: str, payload: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    model_handoff = payload.get("MODEL_HANDOFF.json") or {}
    evaluation = model_handoff.get("evaluation_artifacts") or {}
    handoff_status = str(model_handoff.get("calibration_status") or "").lower() or "unknown"
    path = _resolve_model_reference(run_dir, evaluation.get("calibration_report_path")) or _find_model_artifact_path(run_dir, "calibration.json")
    artifact_status = None
    if path is not None and path.exists():
        try:
            payload_json = read_json(str(path))
            artifact_status = str((payload_json or {}).get("status") or "available").lower()
        except Exception:
            artifact_status = "failed"
    status = "unknown"
    if artifact_status in {"available", "applied", "success", "passed"}:
        status = "available"
    elif artifact_status == "failed":
        status = "failed"
    elif handoff_status in {"available", "applied", "passed"}:
        status = "available"
    elif handoff_status in {"failed"}:
        status = "failed"
    elif handoff_status in {"unavailable"}:
        status = "unavailable"
    if path is not None and path.exists() and handoff_status in {"unavailable", "failed"} and status == "available":
        status = "contradictory"
    return {
        "status": status,
        "test_brier": _metric_from_split(metrics, "test", "brier_score"),
        "test_ece": _metric_from_split(metrics, "test", "ece_10"),
        "evidence_path": str(path) if path is not None and path.exists() else None,
    }


def _build_model_search_signal(run_dir: str, payload: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    model_handoff = payload.get("MODEL_HANDOFF.json") or {}
    search_summary = model_handoff.get("model_search_summary") if isinstance(model_handoff.get("model_search_summary"), dict) else {}
    candidate_models = _nested_get(metrics, ["selection", "candidate_models"])
    candidate_count = len(candidate_models) if isinstance(candidate_models, dict) else 0
    candidate_path = search_summary.get("candidate_results_path") or "candidate_results.json"
    best_path = search_summary.get("best_model_selection_path") or "best_model_selection.json"
    tuning_path = search_summary.get("hyperparameter_search_path") or "hyperparameter_search.json"
    has_candidate = _model_path_exists(run_dir, candidate_path)
    has_best = _model_path_exists(run_dir, best_path)
    has_tuning = _model_path_exists(run_dir, tuning_path)
    if candidate_count == 0 and has_candidate:
        candidate_payload = _read_model_json_if_present(run_dir, candidate_path)
        if isinstance(candidate_payload, list):
            candidate_count = len(candidate_payload)
        elif isinstance(candidate_payload, dict) and isinstance(candidate_payload.get("candidates"), list):
            candidate_count = len(candidate_payload["candidates"])
    if candidate_count >= 2 and has_candidate and has_best and has_tuning:
        assessment = "complete"
    elif candidate_count >= 2 or has_candidate or has_best or has_tuning:
        assessment = "partial"
    else:
        assessment = "missing"
    return {
        "candidate_count": candidate_count,
        "has_candidate_results_artifact": has_candidate,
        "has_best_selection_artifact": has_best,
        "has_hyperparameter_search_artifact": has_tuning,
        "assessment": assessment,
    }


def _read_model_json_if_present(run_dir: str, value: Any) -> Any:
    path = _resolve_model_reference(run_dir, value)
    if path is None or not path.exists():
        return None
    try:
        return read_json(str(path))
    except Exception:
        return None


def _build_loss_convergence_signal(run_dir: str, payload: dict[str, Any]) -> dict[str, Any]:
    model_handoff = payload.get("MODEL_HANDOFF.json") or {}
    evaluation = model_handoff.get("evaluation_artifacts") or {}
    path = (
        _resolve_model_reference(run_dir, evaluation.get("convergence_report_path"))
        or _find_model_artifact_path(run_dir, "convergence_report.json")
        or _find_model_artifact_path(run_dir, "training_curves.json")
        or _find_model_artifact_path(run_dir, "training_curves.csv")
    )
    if path is None or not path.exists():
        return {
            "status": "missing",
            "convergence_status": "unknown",
            "judge_recommended_action": "unknown",
            "evidence_path": None,
        }
    if path.suffix.lower() == ".json":
        try:
            payload_json = read_json(str(path))
        except Exception:
            payload_json = {}
        convergence_status = str(payload_json.get("convergence_status") or payload_json.get("status") or "unknown")
        raw_status = str(payload_json.get("status") or "").lower()
        curve_path = _resolve_model_reference(run_dir, payload_json.get("evidence_path"))
        curve_assessment = _infer_curve_convergence(curve_path)
        if curve_assessment.get("convergence_status"):
            convergence_status = str(curve_assessment["convergence_status"])
        judge_action = _recommended_action_for_convergence(convergence_status)
        if raw_status in {"solver_metadata_only", "curve_available", "curve_evidence", "not_instrumented", "not_applicable"}:
            status = raw_status
        elif convergence_status == "not_applicable":
            status = "not_applicable"
        else:
            status = "available"
        return {
            "status": status,
            "convergence_status": convergence_status,
            "judge_recommended_action": judge_action,
            "evidence_path": str(path),
            "evidence": payload_json.get("evidence") if isinstance(payload_json.get("evidence"), dict) else {},
            **({"curve_assessment": curve_assessment} if curve_assessment else {}),
        }
    curve_assessment = _infer_curve_convergence(path)
    convergence_status = str(curve_assessment.get("convergence_status") or "unknown")
    return {
        "status": "curve_evidence" if curve_assessment else "available",
        "convergence_status": convergence_status,
        "judge_recommended_action": _recommended_action_for_convergence(convergence_status),
        "evidence_path": str(path),
        **({"curve_assessment": curve_assessment} if curve_assessment else {}),
    }


def _infer_curve_convergence(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists() or path.suffix.lower() not in {".csv", ".json"}:
        return {}
    try:
        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path)
        else:
            payload = read_json(str(path))
            if isinstance(payload, list):
                frame = pd.DataFrame(payload)
            elif isinstance(payload, dict):
                rows = payload.get("rows") or payload.get("records") or payload.get("curves")
                frame = pd.DataFrame(rows) if isinstance(rows, list) else pd.DataFrame()
            else:
                frame = pd.DataFrame()
    except Exception:
        return {}
    if frame.empty:
        return {}

    loss_col = _first_present_column(frame, ("validation_loss", "val_loss", "valid_loss", "eval_loss", "train_loss"))
    score_col = _first_present_column(frame, ("validation_auroc", "val_auroc", "validation_auprc", "val_auprc"))
    internal_score_col = _first_present_column(frame, ("internal_validation_score", "sklearn_internal_validation_score"))
    train_loss_col = _first_present_column(frame, ("train_loss", "training_loss", "loss"))
    if score_col is None and loss_col is None and internal_score_col is None:
        return {"points": int(len(frame)), "convergence_status": "unknown", "reason": "No external validation score/loss, internal score, or train loss columns were found."}

    status = "unknown"
    reason = "Insufficient curve evidence."
    best_index: int | None = None
    final_index = int(len(frame) - 1)
    used_internal_score = False

    active_score_col = score_col or internal_score_col
    if active_score_col is not None:
        used_internal_score = score_col is None
        scores = pd.to_numeric(frame[active_score_col], errors="coerce").dropna()
        if len(scores) >= 3:
            best_pos = int(scores.to_numpy().argmax())
            best_index = int(scores.index[best_pos])
            final_score = float(scores.iloc[-1])
            best_score = float(scores.iloc[best_pos])
            recent = scores.tail(min(3, len(scores))).to_list()
            if best_index >= final_index - 1:
                status = "unknown" if used_internal_score else "still_improving"
                reason = (
                    f"{active_score_col} is internal metadata and cannot alone prove still-improving."
                    if used_internal_score
                    else f"{active_score_col} best point is at or near the final iteration."
                )
            elif final_score < best_score:
                status = "overfit" if _train_loss_improved_after_best(frame, train_loss_col, best_index) else "plateaued"
                reason = f"{active_score_col} peaked before the final iteration and ended lower than the best value."
            elif len(recent) >= 3 and recent[-1] <= recent[-2] <= recent[-3]:
                status = "plateaued"
                reason = f"{active_score_col} did not improve over the final iterations."

    if status == "unknown" and loss_col is not None and score_col is not None:
        losses = pd.to_numeric(frame[loss_col], errors="coerce").dropna()
        if len(losses) >= 3:
            recent = losses.tail(min(3, len(losses))).to_list()
            if recent[-1] < recent[-2] < recent[-3]:
                status = "still_improving"
                reason = f"{loss_col} is still decreasing at the final iterations."
            elif recent[-1] >= recent[-2] >= recent[-3]:
                status = "diverged"
                reason = f"{loss_col} increased over the final iterations."
            else:
                status = "plateaued"
                reason = f"{loss_col} no longer shows clear final improvement."

    return {
        "points": int(len(frame)),
        "score_column": score_col,
        "internal_score_column": internal_score_col,
        "loss_column": loss_col,
        "best_iteration_index": best_index,
        "final_iteration_index": final_index,
        "convergence_status": status,
        "used_internal_score_only": used_internal_score,
        "reason": reason,
    }


def _first_present_column(frame: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    lower_to_actual = {str(column).lower(): str(column) for column in frame.columns}
    for name in names:
        if name.lower() in lower_to_actual:
            return lower_to_actual[name.lower()]
    return None


def _train_loss_improved_after_best(frame: pd.DataFrame, train_loss_col: str | None, best_index: int | None) -> bool:
    if train_loss_col is None or best_index is None or best_index >= len(frame) - 1:
        return False
    losses = pd.to_numeric(frame[train_loss_col], errors="coerce")
    best_loss = losses.iloc[best_index]
    final_loss = losses.iloc[-1]
    if pd.isna(best_loss) or pd.isna(final_loss):
        return False
    return bool(final_loss < best_loss)


def _default_artifact_consistency_summary(audit: dict[str, Any]) -> dict[str, Any]:
    retry_signals = audit.get("judge_retry_signals", {})
    artifact_gate = retry_signals.get("artifact_gate", {}) if isinstance(retry_signals, dict) else {}
    return {
        "status": artifact_gate.get("status") or audit.get("artifact_consistency_status") or "unknown",
        "contradictions": artifact_gate.get("contradictions") or [],
        "trusted_sources": ["metrics.json", "predictions.csv", "calibration.json", "feature_drop_report.json", "generated code/logs"],
    }


def _default_improvement_plan(signals: dict[str, Any]) -> dict[str, Any]:
    data_gate = signals.get("data_gate", {}) if isinstance(signals, dict) else {}
    artifact_gate = signals.get("artifact_gate", {}) if isinstance(signals, dict) else {}
    gap = signals.get("generalization_gap", {}) if isinstance(signals, dict) else {}
    search = signals.get("model_search_completeness", {}) if isinstance(signals, dict) else {}
    convergence = signals.get("loss_convergence", {}) if isinstance(signals, dict) else {}
    calibration = signals.get("calibration_status", {}) if isinstance(signals, dict) else {}

    if data_gate.get("blocking") or data_gate.get("status") == "failed":
        return {
            "owner_stage": "data",
            "reason": "data_gate_failed",
            "preserve_data_handoff": False,
            "preserve_model_handoff": False,
            "required_next_actions": data_gate.get("reasons", []),
            "required_new_artifacts": ["DATA_HANDOFF.json"],
            "exploration_strategy": {
                "try_model_families": [],
                "try_feature_strategies": [],
                "try_training_changes": [],
                "try_selection_changes": [],
                "try_calibration_changes": [],
            },
        }

    actions: list[str] = []
    artifacts: list[str] = []
    reason = None
    if artifact_gate.get("status") == "failed":
        reason = "missing_or_contradictory_model_artifacts"
        actions.extend(artifact_gate.get("missing_core_artifacts", []) or artifact_gate.get("contradictions", []))
    if gap.get("assessment") == "overfitting":
        reason = reason or "overfitting"
        actions.extend(["try stronger regularization", "reduce model capacity", "use early stopping when applicable"])
    elif gap.get("assessment") == "underfitting":
        reason = reason or "underfitting"
        actions.extend(["try richer model families", "run bounded feature/model search"])
    elif gap.get("assessment") == "split_shift":
        reason = reason or "validation_test_shift"
        actions.extend(["diagnose validation/test shift", "prefer robust model selection over test-chasing"])
    if search.get("assessment") in {"missing", "partial"}:
        reason = reason or "model_search_incomplete"
        actions.append("run bounded hyperparameter optimization and compare candidates")
        artifacts.extend(["candidate_results.json", "best_model_selection.json", "hyperparameter_search.json"])
    if calibration.get("status") in {"failed", "contradictory"}:
        reason = reason or "calibration_issue"
        actions.append("try calibrated model variants only if calibration is expected to improve validation log loss/Brier score without hurting AUROC/AUPRC")
        artifacts.append("calibration.json")
    if convergence.get("convergence_status") == "still_improving":
        reason = reason or "training_still_improving"
        actions.append("train longer only for model families with meaningful epoch/iteration curves")
        artifacts.extend(["training_curves.csv", "convergence_report.json"])
    elif convergence.get("convergence_status") == "overfit":
        reason = reason or "training_curve_overfit"
        actions.append("early stop sooner or regularize based on convergence report")
        artifacts.append("convergence_report.json")
    elif convergence.get("status") == "not_instrumented":
        reason = reason or "convergence_not_instrumented"
        actions.append("emit convergence evidence with curves or solver metadata")
        artifacts.append("convergence_report.json")
    elif convergence.get("status") in {"missing", "solver_metadata_only"} or convergence.get("convergence_status") == "unknown":
        reason = reason or "convergence_evidence_weak"
        if _convergence_hit_iteration_cap_without_plateau(convergence):
            actions.append("try a larger iteration/epoch budget with external validation convergence curves because the previous run hit max_iter without plateau evidence")
            actions.append("train longer only while external validation AUROC/AUPRC improves or validation loss decreases")
        else:
            actions.append("add external validation convergence curves or stronger solver/staged-iteration evidence")
        artifacts.extend(["training_curves.csv", "training_curves.json", "convergence_report.json"])

    if not actions:
        return {
            "owner_stage": "model",
            "reason": "exploratory_reward_improvement",
            "preserve_data_handoff": True,
            "preserve_model_handoff": False,
            "required_next_actions": [
                "run bounded exploratory model improvement because reward threshold is the retry controller",
                "try untried repo-declared model families such as XGBoost, LightGBM, and CatBoost when applicable",
                "try clinically safe temporal feature strategies such as lags, deltas, rolling summaries, and missingness patterns when the data layout supports them",
                "compare AUROC-primary selection with AUPRC/log-loss-sensitive validation selection for rare-event classification",
                "try imbalance-aware objectives, class weights, scale_pos_weight, or threshold-independent ranking improvements to lift validation AUROC/AUPRC",
            ],
            "required_new_artifacts": [
                "candidate_results.json",
                "best_model_selection.json",
                "hyperparameter_search.json",
                "training_curves.csv",
                "convergence_report.json",
            ],
            "exploration_strategy": {
                "try_model_families": ["XGBoost", "LightGBM", "CatBoost", "SGDClassifier_log_loss", "MLPClassifier"],
                "try_feature_strategies": ["temporal_lags", "deltas", "rolling_summaries", "missingness_patterns"],
                "try_training_changes": ["external_validation_early_stopping", "class_weight_or_scale_pos_weight", "bounded_optuna_or_grid_search"],
                "try_selection_changes": ["AUROC_primary", "AUPRC_primary", "log_loss_sensitive_selection"],
                "try_calibration_changes": ["calibrated_variant_only_if_it_improves_validation_metrics"],
            },
        }
    return {
        "owner_stage": "model",
        "reason": reason or "model_retry_signal",
        "preserve_data_handoff": True,
        "preserve_model_handoff": False,
        "required_next_actions": list(dict.fromkeys(str(item) for item in actions if item)),
        "required_new_artifacts": list(dict.fromkeys(str(item) for item in artifacts if item)),
        "exploration_strategy": {
            "try_model_families": ["XGBoost", "LightGBM", "CatBoost"],
            "try_feature_strategies": ["temporal_lags", "deltas", "rolling_summaries"],
            "try_training_changes": ["larger_iteration_budget_when_max_iter_was_hit", "external_validation_early_stopping", "stronger_convergence_evidence"],
            "try_selection_changes": ["AUROC_primary", "AUPRC_primary"],
            "try_calibration_changes": ["validation_set_calibration"] if calibration.get("status") in {"failed", "contradictory", "unavailable"} else [],
        },
    }


def _convergence_hit_iteration_cap_without_plateau(convergence: dict[str, Any]) -> bool:
    evidence = convergence.get("evidence") if isinstance(convergence.get("evidence"), dict) else {}
    curve = convergence.get("curve_assessment") if isinstance(convergence.get("curve_assessment"), dict) else {}
    reached_max_iter = evidence.get("reached_max_iter")
    if reached_max_iter is None:
        n_iter = _number_or_none(evidence.get("n_iter"))
        max_iter = _number_or_none(evidence.get("max_iter"))
        reached_max_iter = bool(n_iter is not None and max_iter is not None and n_iter >= max_iter)
    no_plateau_signal = str(curve.get("convergence_status") or convergence.get("convergence_status") or "unknown") == "unknown"
    return bool(reached_max_iter and no_plateau_signal)


def _inspect_research_grounding(payload: dict[str, Any]) -> dict[str, Any]:
    """Check literature claims against RESEARCH_HANDOFF without making it a hard gate."""
    task = payload.get("task_input.json") or {}
    research = payload.get("RESEARCH_HANDOFF.json")
    model_handoff = payload.get("MODEL_HANDOFF.json") or {}
    text_blob = json.dumps(
        {
            "request": task.get("raw_prompt") or task.get("request"),
            "modality_strategy": model_handoff.get("modality_strategy"),
            "training_summary": model_handoff.get("training_summary"),
            "audit_summary": model_handoff.get("audit_summary"),
        },
        default=str,
    ).lower()
    claim_terms = ("sota", "state of the art", "literature", "paper", "evidence", "research", "method_id")
    has_claim = any(term in text_blob for term in claim_terms)
    warnings: list[str] = []
    issues: list[str] = []

    if not research:
        if has_claim:
            issues.append("Model/request makes literature or SOTA-style claims but RESEARCH_HANDOFF.json is absent.")
            return {"status": "warning", "issues": issues, "warnings": warnings, "method_ids": []}
        return {"status": "unavailable", "issues": [], "warnings": [], "method_ids": []}

    if research_is_stale(research, payload.get("DATA_HANDOFF.json")):
        issues.append("RESEARCH_HANDOFF was generated from an older DATA_HANDOFF and is stale.")

    method_cards = research.get("method_cards") if isinstance(research.get("method_cards"), list) else []
    method_ids = [str(card.get("method_id")) for card in method_cards if isinstance(card, dict) and card.get("method_id")]
    papers = research.get("papers") if isinstance(research.get("papers"), list) else []
    if not method_cards:
        warnings.append("RESEARCH_HANDOFF has no method_cards; model planning evidence may be too generic.")
    if not papers:
        warnings.append("RESEARCH_HANDOFF has no supporting papers; evidence should be treated as thin.")
    if has_claim and method_ids and not any(method_id.lower() in text_blob for method_id in method_ids):
        warnings.append("Literature-grounded language appears in model artifacts, but no RESEARCH_HANDOFF method_id was cited.")
    if "sota" in text_blob or "state of the art" in text_blob:
        warnings.append("SOTA wording detected; judge should verify this is framed cautiously and supported by method cards.")

    return {
        "status": "warning" if issues or warnings else "passed",
        "issues": issues,
        "warnings": warnings,
        "method_ids": method_ids,
        "paper_count": len(papers),
        "quality_gate": research.get("quality_gate"),
    }


def _inspect_artifact_completeness(run_dir: str, payload: dict[str, Any]) -> dict[str, Any]:
    model_handoff = payload.get("MODEL_HANDOFF.json") or {}
    evaluation = model_handoff.get("evaluation_artifacts") or {}
    code = model_handoff.get("code_artifacts") or {}
    issues: list[str] = []
    warnings: list[str] = []
    missing: list[str] = []
    prediction_paths = _collect_prediction_paths(run_dir, evaluation)

    required = {
        "code_artifacts.training_code_path": (code.get("training_code_path") or code.get("train_model_path"), None),
        "evaluation_artifacts.metrics_path": (evaluation.get("metrics_path"), evaluation.get("metrics_unavailable_reason")),
    }
    for label, (value, unavailable_reason) in required.items():
        if not value:
            if unavailable_reason:
                warnings.append(f"{label} unavailable: {unavailable_reason}")
            else:
                missing.append(label)
            continue
        path = _resolve_model_reference(run_dir, value)
        if path is None or not path.exists():
            missing.append(label)
        elif path.is_file() and path.stat().st_size == 0:
            issues.append(f"{label} is empty: {path}")

    if not prediction_paths:
        reason = evaluation.get("predictions_unavailable_reason")
        if reason:
            warnings.append(f"evaluation_artifacts.predictions_path unavailable: {reason}")
        else:
            missing.append("evaluation_artifacts.predictions_path")
    else:
        for label, path in prediction_paths.items():
            if path is None or not path.exists():
                missing.append(label)
            elif path.is_file() and path.stat().st_size == 0:
                issues.append(f"{label} is empty: {path}")

    optional_with_status = {
        "evaluation_artifacts.calibration_report_path": (evaluation.get("calibration_report_path"), model_handoff.get("calibration_status")),
        "evaluation_artifacts.subgroup_metrics_path": (evaluation.get("subgroup_metrics_path"), model_handoff.get("subgroup_eval_status")),
        "evaluation_artifacts.error_examples_path": (evaluation.get("error_examples_path"), None),
        "evaluation_artifacts.leakage_report_path": (evaluation.get("leakage_report_path"), None),
    }
    for label, (value, status) in optional_with_status.items():
        if value:
            path = _resolve_model_reference(run_dir, value)
            if path is None or not path.exists():
                issues.append(f"{label} declared but missing: {value}")
        elif status in {"applied", "available"}:
            issues.append(f"{label} missing despite status={status}")
        else:
            warnings.append(f"{label} unavailable or not declared")

    calibration_status = model_handoff.get("calibration_status")
    if calibration_status == "applied" and not evaluation.get("calibration_report_path"):
        issues.append("calibration_status=applied but calibration report is missing")
    subgroup_status = model_handoff.get("subgroup_eval_status")
    if subgroup_status == "available" and not evaluation.get("subgroup_metrics_path"):
        issues.append("subgroup_eval_status=available but subgroup metrics are missing")

    return {
        "status": "failed" if missing or issues else "warning" if warnings else "passed",
        "issues": issues,
        "warnings": warnings,
        "missing_artifacts": missing,
        "calibration_status": "failed" if any("calibration" in item for item in issues) else "warning" if calibration_status in {None, "failed", "unavailable"} else "passed",
        "subgroup_status": "failed" if any("subgroup" in item for item in issues) else "unavailable" if subgroup_status in {None, "unavailable"} else "passed",
    }


def _inspect_predictions_consistency(run_dir: str, payload: dict[str, Any]) -> dict[str, Any]:
    data_handoff = payload.get("DATA_HANDOFF.json") or {}
    model_handoff = payload.get("MODEL_HANDOFF.json") or {}
    evaluation = model_handoff.get("evaluation_artifacts") or {}
    metrics_path = _resolve_model_reference(run_dir, evaluation.get("metrics_path"))
    issues: list[str] = []
    critical: list[str] = []
    warnings: list[str] = []
    summary: dict[str, Any] = {
        "normalized_layout": data_handoff.get("normalized_layout"),
        "prediction_unit": data_handoff.get("prediction_unit"),
        "task_type": data_handoff.get("task_type"),
        "artifacts": {},
    }
    prediction_paths = _collect_prediction_paths(run_dir, evaluation)

    if not prediction_paths:
        reason = evaluation.get("predictions_unavailable_reason")
        if reason:
            warnings.append(f"predictions artifact unavailable: {reason}")
            return {"status": "warning", "issues": [], "critical_issues": [], "warnings": warnings, "summary": summary}
        return {"status": "failed", "issues": ["predictions artifact missing"], "critical_issues": ["predictions artifact missing"], "warnings": warnings, "summary": summary}

    combined_split_counts: dict[str, int] = {}
    saw_score_column = False
    for label, pred_path in prediction_paths.items():
        if pred_path is None or not pred_path.exists():
            critical.append(f"{label} missing")
            continue
        try:
            predictions = _read_structured_table(pred_path)
        except Exception as exc:
            critical.append(f"{label} unreadable: {exc}")
            continue

        if predictions.empty:
            critical.append(f"{label} has no rows")
            continue

        truth_col = _find_alias(predictions.columns, TRUTH_COLUMN_ALIASES)
        pred_col = _find_alias(predictions.columns, PREDICTION_COLUMN_ALIASES)
        score_col = _find_alias(predictions.columns, SCORE_COLUMN_ALIASES)
        split_col = _find_alias(predictions.columns, SPLIT_COLUMN_ALIASES)
        id_col = _find_alias(predictions.columns, ID_COLUMN_ALIASES)

        task_type = str(data_handoff.get("task_type") or "").lower()
        if task_type in {"binary_classification", "multiclass_classification", "classification", ""}:
            if truth_col is None:
                critical.append(f"{label} missing truth/label column")
            if pred_col is None:
                critical.append(f"{label} missing prediction column")
            if score_col is None:
                warnings.append(f"{label} missing score/probability column")
            else:
                saw_score_column = True
        elif task_type in {"regression"}:
            if truth_col is None:
                critical.append(f"{label} missing truth/target column")
            if pred_col is None and score_col is None:
                critical.append(f"{label} missing prediction value column")
        else:
            if truth_col is None:
                warnings.append(f"{label} missing recognizable truth column for task_type={task_type or 'unknown'}")

        split_counts = _extract_split_counts(predictions, split_col, label)
        for split_name, count in split_counts.items():
            combined_split_counts[split_name] = combined_split_counts.get(split_name, 0) + count

        if id_col is None:
            warnings.append(f"{label} missing recognizable prediction-unit identifier column")

        summary["artifacts"][label] = {
            "path": str(pred_path),
            "rows": int(len(predictions)),
            "columns": list(predictions.columns),
            "truth_column": truth_col,
            "prediction_column": pred_col,
            "score_column": score_col,
            "split_column": split_col,
            "id_column": id_col,
            "split_counts": split_counts,
        }

    summary["prediction_rows"] = int(sum(item["rows"] for item in summary["artifacts"].values()))
    summary["split_counts"] = combined_split_counts
    summary["has_score_column"] = saw_score_column

    metrics = {}
    if metrics_path is None or not metrics_path.exists():
        critical.append("metrics artifact missing")
    else:
        try:
            metrics = read_json(str(metrics_path))
        except Exception as exc:
            critical.append(f"metrics artifact unreadable: {exc}")

    if metrics:
        handoff_metrics = model_handoff.get("metrics") or {}
        metrics_payload = metrics.get("metrics") if isinstance(metrics.get("metrics"), dict) else metrics
        for split, count in combined_split_counts.items():
            metric_n = _nested_get(metrics_payload, [split, "n"])
            if metric_n is not None and int(metric_n) != int(count):
                issues.append(f"metrics n for split {split}={metric_n} does not match predictions count={count}")
        if handoff_metrics and metrics_payload:
            handoff_auroc = _find_metric_value(handoff_metrics, "auroc")
            artifact_auroc = _find_metric_value(metrics_payload, "auroc")
            if handoff_auroc is not None and artifact_auroc is not None and abs(float(handoff_auroc) - float(artifact_auroc)) > 1e-6:
                issues.append(f"MODEL_HANDOFF auroc={handoff_auroc} differs from metrics artifact auroc={artifact_auroc}")

    return {
        "status": "failed" if critical else "warning" if issues or warnings else "passed",
        "issues": issues,
        "critical_issues": critical,
        "warnings": warnings,
        "summary": summary,
    }


def _inspect_code_artifact(run_dir: str, payload: dict[str, Any]) -> dict[str, Any]:
    data_handoff = payload.get("DATA_HANDOFF.json") or {}
    model_handoff = payload.get("MODEL_HANDOFF.json") or {}
    code_artifacts = model_handoff.get("code_artifacts") or {}
    code_path = _resolve_model_reference(run_dir, code_artifacts.get("training_code_path") or code_artifacts.get("train_model_path"))
    code_paths = [
        path
        for path in (_resolve_model_reference(run_dir, item) for item in code_artifacts.get("code_paths", []) if item)
        if path is not None and path.exists()
    ]
    if code_path is not None and code_path.exists() and code_path not in code_paths:
        code_paths.append(code_path)
    issues: list[str] = []
    critical: list[str] = []
    leakage_issues: list[str] = []
    warnings: list[str] = []
    suspicious: list[str] = []
    if code_path is None or not code_path.exists():
        return {
            "status": "failed",
            "issues": ["training code artifact missing"],
            "critical_issues": ["training code artifact missing"],
            "leakage_issues": [],
            "leakage_status": "failed",
            "warnings": [],
            "suspicious_patterns": [],
        }
    code = code_path.read_text(errors="replace")
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return {
            "status": "failed",
            "issues": [f"training code is not parseable Python: {exc}"],
            "critical_issues": [f"training code is not parseable Python: {exc}"],
            "leakage_issues": [],
            "leakage_status": "failed",
            "warnings": [],
            "suspicious_patterns": [],
        }
    target = data_handoff.get("target")
    label_spec = data_handoff.get("label_spec") or {}
    allowed_target_literals = {str(value) for value in (target, label_spec.get("source_name"), label_spec.get("target_name")) if value}
    exclusion_columns = {str(item) for item in data_handoff.get("feature_exclusion_columns", [])}
    join_keys = {str(item) for item in data_handoff.get("join_keys", [])}
    patient_id = data_handoff.get("patient_id_column")
    if patient_id:
        exclusion_columns.add(str(patient_id))
    exclusion_columns |= join_keys

    all_code_text = code + "\n" + "\n".join(path.read_text(errors="replace") for path in code_paths if path != code_path)
    if "MODEL_INPUT_CONTRACT.json" not in all_code_text:
        critical.append("training code does not read MODEL_INPUT_CONTRACT.json")
    if _ast_has_string_literal(tree, "DATA_HANDOFF_PREP"):
        critical.append("training code references substitute DATA_HANDOFF_PREP")
    if _ast_has_string_literal_containing(tree, "artifacts/runs") or _ast_has_string_literal_containing(tree, "./artifacts"):
        critical.append("training code contains hardcoded artifact path")
    if _ast_has_calibration_prefit(tree):
        critical.append("training code uses disallowed CalibratedClassifierCV prefit mode")

    target_misuse = _find_target_contract_misuse(tree, allowed_target_literals)
    critical.extend(target_misuse["critical"])
    warnings.extend(target_misuse["warnings"])

    if exclusion_columns and "feature_exclusion_columns" not in all_code_text and not _ast_loads_contract_key(tree, "feature_exclusion_columns"):
        leakage_issues.append("training code does not clearly read feature_exclusion_columns from MODEL_INPUT_CONTRACT.json")
    if leakage_issues:
        critical.extend(leakage_issues)

    if not _ast_calls_any(tree, {"to_csv", "to_parquet", "to_json"}) and not _ast_has_string_literal_containing(tree, "pred"):
        warnings.append("training code does not obviously write predictions")
    if not _ast_has_string_literal(tree, "metrics.json") and not _ast_has_string_literal_containing(tree, "metric"):
        warnings.append("training code does not obviously write metrics")
    if _ast_name_or_attr_used(tree, "roc_auc_score") and not (_ast_name_or_attr_used(tree, "unique") or _ast_name_or_attr_used(tree, "nunique")):
        suspicious.append("AUROC appears to be computed without an obvious single-class guard")

    return {
        "status": "failed" if critical else "warning" if warnings or suspicious else "passed",
        "issues": critical + warnings,
        "critical_issues": critical,
        "leakage_issues": leakage_issues,
        "leakage_status": "failed" if leakage_issues else "passed",
        "warnings": warnings,
        "suspicious_patterns": suspicious,
        "code_path": str(code_path),
    }


def _inspect_logs(run_dir: str, payload: dict[str, Any]) -> dict[str, Any]:
    model_handoff = payload.get("MODEL_HANDOFF.json") or {}
    evaluation = model_handoff.get("evaluation_artifacts") or {}
    log_path = _resolve_model_reference(run_dir, evaluation.get("training_log_path") or evaluation.get("evaluation_log_path") or "train_model.log")
    warnings: list[str] = []
    critical: list[str] = []
    suspicious: list[str] = []
    if log_path is None or not log_path.exists():
        warnings.append("model execution log missing")
        return {"status": "warning", "critical_issues": critical, "warnings": warnings, "suspicious_patterns": suspicious}
    text = log_path.read_text(errors="replace")[-20000:]
    lowered = text.lower()
    for pattern in ("traceback", "error", "exception", "failed"):
        if pattern in lowered:
            suspicious.append(f"log contains {pattern!r}")
    if "traceback" in lowered:
        critical.append("model log contains traceback")
    return {"status": "failed" if critical else "warning" if suspicious or warnings else "passed", "critical_issues": critical, "warnings": warnings, "suspicious_patterns": suspicious, "log_path": str(log_path)}


def _number_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric_from_split(metrics: dict[str, Any], split: str, metric_name: str) -> float | None:
    payload = metrics.get("metrics") if isinstance(metrics.get("metrics"), dict) else metrics
    section = payload.get(split) if isinstance(payload, dict) else None
    if not isinstance(section, dict):
        return None
    aliases = {
        "auprc": ("auprc", "average_precision", "ap"),
        "average_precision": ("average_precision", "auprc", "ap"),
        "prevalence": ("prevalence", "positive_rate"),
        "positive_rate": ("positive_rate", "prevalence"),
    }.get(metric_name, (metric_name,))
    for alias in aliases:
        value = _number_or_none(section.get(alias))
        if value is not None:
            return value
    return None


def _find_model_artifact_path(run_dir: str, filename: str) -> Path | None:
    candidate = Path(stage_dir(run_dir, "model")) / filename
    return candidate.resolve() if candidate.exists() else None


def _model_path_exists(run_dir: str, value: Any) -> bool:
    path = _resolve_model_reference(run_dir, value)
    return bool(path is not None and path.exists())


def _recommended_action_for_convergence(convergence_status: str) -> str:
    return {
        "converged": "stop",
        "plateaued": "switch_model_or_features",
        "still_improving": "train_longer",
        "overfit": "regularize",
        "diverged": "tune_learning_rate",
        "not_applicable": "not_applicable",
    }.get(str(convergence_status), "unknown")


def _resolve_model_reference(run_dir: str, value: Any) -> Path | None:
    if not value:
        return None
    text = str(value)
    if "://" in text:
        return None
    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    run_directory = Path(run_dir).resolve()
    model_directory = Path(stage_dir(run_dir, "model")).resolve()
    if candidate.parts and candidate.parts[0] == "model":
        return (run_directory / candidate).resolve()
    return (model_directory / candidate).resolve()


def _read_artifact_summary(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return {"path": str(path), "kind": "json", "content": read_json(str(path))}
    if suffix in {".csv", ".tsv", ".parquet", ".jsonl"}:
        frame = _read_structured_table(path, nrows=200)
        return {"path": str(path), "kind": "table", "columns": list(frame.columns), "preview_rows": frame.head(20).to_dict(orient="records"), "row_preview_count": int(len(frame))}
    if suffix in BINARY_ARTIFACT_SUFFIXES:
        return {"path": str(path), "kind": "binary", "size_bytes": path.stat().st_size}
    text = path.read_text(errors="replace")
    return {"path": str(path), "kind": "text", "text_preview": text[:12000], "size_bytes": path.stat().st_size}


def _read_structured_table(path: Path, nrows: int | None = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, nrows=nrows)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t", nrows=nrows)
    if suffix == ".parquet":
        return pd.read_parquet(path).head(nrows) if nrows is not None else pd.read_parquet(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True).head(nrows) if nrows is not None else pd.read_json(path, lines=True)
    if suffix == ".json":
        payload = read_json(str(path))
        frame = _payload_to_frame(payload)
        if nrows is not None:
            return frame.head(nrows)
        return frame
    raise ValueError(f"Unsupported structured artifact type: {suffix}")


def _payload_to_frame(payload: Any) -> pd.DataFrame:
    if isinstance(payload, list):
        return pd.DataFrame(payload)
    if isinstance(payload, dict):
        for key in ("predictions", "rows", "records", "items", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return pd.DataFrame(value)
    raise ValueError("JSON artifact does not contain a table-like payload")


def _collect_prediction_paths(run_dir: str, evaluation: dict[str, Any]) -> dict[str, Path]:
    candidates: dict[str, Path] = {}
    single = _resolve_model_reference(run_dir, evaluation.get("predictions_path"))
    if single is not None:
        candidates["evaluation_artifacts.predictions_path"] = single
    for field in ("train_predictions_path", "validation_predictions_path", "val_predictions_path", "test_predictions_path"):
        path = _resolve_model_reference(run_dir, evaluation.get(field))
        if path is not None:
            candidates[f"evaluation_artifacts.{field}"] = path
    return candidates


def _find_alias(columns: Any, aliases: tuple[str, ...]) -> str | None:
    lowered = {str(column).lower(): str(column) for column in columns}
    for alias in aliases:
        if alias.lower() in lowered:
            return lowered[alias.lower()]
    return None


def _extract_split_counts(predictions: pd.DataFrame, split_col: str | None, label: str) -> dict[str, int]:
    if split_col and split_col in predictions.columns:
        counts = predictions[split_col].astype(str).str.lower().replace({"valid": "val", "validation": "val"}).value_counts().to_dict()
        return {str(key): int(value) for key, value in counts.items()}

    lowered_label = label.lower()
    for split_name in ("train", "val", "test"):
        if split_name in lowered_label or (split_name == "val" and "validation" in lowered_label):
            return {split_name: int(len(predictions))}
    return {"all": int(len(predictions))}


def _nested_get(payload: dict[str, Any], path: list[str]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _find_metric_value(payload: Any, metric_name: str) -> Any:
    if isinstance(payload, dict):
        if metric_name in payload:
            return payload[metric_name]
        for value in payload.values():
            found = _find_metric_value(value, metric_name)
            if found is not None:
                return found
    return None


def _find_target_contract_misuse(tree: ast.AST, allowed_target_literals: set[str]) -> dict[str, list[str]]:
    critical: list[str] = []
    warnings: list[str] = []

    uses_contract_target = _ast_loads_contract_key(tree, "target")
    hardcoded_target_assignments = _ast_string_assignments_to_name(tree, "target") - allowed_target_literals
    if hardcoded_target_assignments:
        critical.append(
            f"training code assigns target from hardcoded literals not declared by DATA_HANDOFF: {sorted(set(hardcoded_target_assignments))}"
        )

    hardcoded_y_sources = _ast_label_sources_for_y_variables(tree) - allowed_target_literals
    if hardcoded_y_sources:
        critical.append(
            f"training labels appear to be sourced from hardcoded outcome columns not declared by DATA_HANDOFF: {sorted(set(hardcoded_y_sources))}"
        )

    if not uses_contract_target and not hardcoded_target_assignments:
        warnings.append("training code does not clearly load target from MODEL_INPUT_CONTRACT.json")

    return {"critical": critical, "warnings": warnings}


def _ast_has_string_literal(tree: ast.AST, literal: str) -> bool:
    return any(isinstance(node, ast.Constant) and node.value == literal for node in ast.walk(tree))


def _ast_has_string_literal_containing(tree: ast.AST, fragment: str) -> bool:
    return any(isinstance(node, ast.Constant) and isinstance(node.value, str) and fragment in node.value for node in ast.walk(tree))


def _ast_calls_any(tree: ast.AST, function_names: set[str]) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id in function_names:
            return True
        if isinstance(func, ast.Attribute) and func.attr in function_names:
            return True
    return False


def _ast_name_or_attr_used(tree: ast.AST, name: str) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == name:
            return True
        if isinstance(node, ast.Attribute) and node.attr == name:
            return True
    return False


def _ast_has_calibration_prefit(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg == "cv" and isinstance(keyword.value, ast.Constant) and keyword.value.value == "prefit":
                return True
    return False


def _ast_loads_contract_key(tree: ast.AST, key: str) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and _ast_name_or_attr_node(node.value, "contract") and _slice_literal(node.slice) == key:
            return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get":
            if _ast_name_or_attr_node(node.func.value, "contract") and node.args and _constant_value(node.args[0]) == key:
                return True
    return False


def _ast_name_or_attr_node(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Name) and node.id == name


def _slice_literal(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    return None


def _constant_value(node: ast.AST) -> Any:
    return node.value if isinstance(node, ast.Constant) else None


def _ast_string_assignments_to_name(tree: ast.AST, variable_name: str) -> set[str]:
    values: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == variable_name for target in node.targets):
            continue
        value = _constant_value(node.value)
        if isinstance(value, str):
            values.add(value)
    return values


def _ast_label_sources_for_y_variables(tree: ast.AST) -> set[str]:
    sources: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        target_names = {target.id for target in node.targets if isinstance(target, ast.Name)}
        if not target_names or "y" not in target_names:
            continue
        literal = _subscript_string_literal(node.value) or _pop_string_literal(node.value)
        if literal:
            sources.add(literal)
    return sources


def _subscript_string_literal(node: ast.AST) -> str | None:
    if isinstance(node, ast.Subscript):
        value = _slice_literal(node.slice)
        return value if isinstance(value, str) else None
    return None


def _pop_string_literal(node: ast.AST) -> str | None:
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "pop":
        if node.args:
            value = _constant_value(node.args[0])
            return value if isinstance(value, str) else None
    return None


def _parse_json_payload(payload: object, name: str) -> tuple[dict | None, str | None]:
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


JUDGE_PROMPT = """You are MAGMA v2 Judge.

You are the clinical AI quality gate. Your job is not just model scoring; it is clinical documentation and TRIPOD-aware evaluation.

Workflow:
1. Call read_pipeline_bundle.
2. Review audit.judge_retry_signals. These compact signals are audit input for retry routing, not the final verdict.
3. Call inspect_predictions_consistency, inspect_code_artifact, inspect_logs, read_judge_retry_signals, or read_model_artifact only if read_pipeline_bundle reports missing, contradictory, suspicious, or insufficient evidence.
4. Produce a concise visible IMPLEMENTATION_PLAN with the concrete files you checked: canonical handoffs, optional RESEARCH_HANDOFF, predictions, metrics, calibration, subgroup metrics, error examples, feature-drop/leakage report, generated code, logs, convergence/loss artifacts when present, and TRIPOD. Do not expose private reasoning.
5. Evaluate whether DATA_HANDOFF, optional RESEARCH_HANDOFF, MODEL_HANDOFF, generated code, logs, predictions, metrics, calibration, subgroup outputs, error-analysis outputs, and compact retry signals agree.
6. Fail hard for demonstrated leakage, target/contract violation, missing core artifacts, unreadable predictions/metrics, disallowed calibration, code/handoff disagreement, or artifact contradictions.
7. Allow provisional status for partial but defensible results only when limitations and unavailable artifacts are explicitly documented.
8. Score the 19 TRIPOD items as true/false using available handoffs, artifact bundle, README files, metrics, and limitations.
9. Compute tripod_score as passed_items / 19.
10. Extract AUROC from metrics artifact when available, falling back to MODEL_HANDOFF only if the artifact is absent with a documented reason.
11. Call write_judge_verdict with strict JSON.

Grounding rule:
- Treat `metrics.json` as the canonical source of quantitative truth when it exists.
- If MODEL_HANDOFF, judge prose, reward inputs, or any other artifact reports a quantitative result, it must match `metrics.json` or another explicitly written artifact on disk.
- Do not trust free-form metric summaries in MODEL_HANDOFF when `metrics.json` is present.
- Any reported metric value not grounded in `metrics.json` or another named artifact should be treated as suspicious and surfaced as an artifact-consistency issue.
- Judge retry signals are audit input, not the verdict. If handoffs and direct artifacts disagree, prefer direct artifact evidence and report the disagreement as an artifact-consistency issue.

Verdict JSON fields:
- status: passed | failed | provisional
- auroc: number or null
- tripod_score: 0-1
- tripod_items: object with all 19 item booleans
- technical_correctness_status: passed | warning | failed
- code_audit_status: passed | warning | failed
- artifact_consistency_status: passed | warning | failed
- leakage_audit_status: passed | warning | failed
- calibration_audit_status: passed | warning | failed
- subgroup_eval_status: passed | unavailable | warning | failed
- research_grounding_status: passed | unavailable | warning | failed
- code_issues: array
- artifact_issues: array
- research_issues: array
- leakage_issues: array
- metric_consistency_issues: array
- missing_artifacts: array
- suspicious_patterns: array
- critical_failures: array
- warnings: array
- judge_retry_signals: object with data_gate, artifact_gate, generalization_gap, auprc_vs_prevalence, calibration_status, model_search_completeness, loss_convergence, and reward_components
- artifact_consistency_summary: object with status, contradictions, and trusted_sources
- improvement_plan: object with owner_stage, reason, preserve_data_handoff, preserve_model_handoff, required_next_actions, required_new_artifacts, and exploration_strategy. Do not include retry/no-retry fields such as should_retry or stop_reason.
- feedback: concise clinical explanation

Rules:
- TRIPOD is essential. A high AUROC does not compensate for poor clinical reporting.
- Missing DATA_HANDOFF.json or MODEL_HANDOFF.json is a critical failure.
- Missing RESEARCH_HANDOFF.json is not a failure for ordinary baseline runs. It is a warning only when the request/model claims SOTA, literature grounding, evidence-backed architecture choice, or multimodal method recommendations.
- Missing predictions or metrics artifacts are critical failures for supervised runs unless MODEL_HANDOFF explicitly documents why unavailable.
- Leakage, missing outcome definition, missing data split, absent performance metrics, hardcoded wrong target, target/helper feature usage, or disallowed calibration are critical failures.
- Do not fail code audit based on broad keywords such as label, outcome, sepsis, mortality, or clinical reporting text. Those terms are allowed in report JSON, TRIPOD fields, comments, logs, and interpretation strings.
- Focus code audit on bad behavior: hardcoded target assignment instead of MODEL_INPUT_CONTRACT, training labels sourced from undeclared columns, hardcoded run/data paths, substitute handoffs, feature leakage, patient/group identifiers used as features, split misuse, disallowed calibration, and generated claims that contradict actual artifacts.
- Treat contradictions as evidence problems: if code, DATA_HANDOFF, MODEL_HANDOFF, metrics, predictions, logs, or reports disagree, surface the concrete contradiction and severity.
- Treat false or unsupported literature claims as evidence problems. If MODEL_HANDOFF claims literature/SOTA grounding, it should cite RESEARCH_HANDOFF method_ids or clearly state that no literature pass was available.
- Prefer warnings over failures for suspicious wording unless it changes training/evaluation behavior or contradicts artifacts.
- Handoff summaries are not sufficient evidence when richer artifacts exist. Compare the handoff claims to actual files.
- Do not recommend rerunning data when data_gate passed unless there is concrete data-owned evidence.
- The judge does not decide whether reward-improvement retries stop. The runtime reward threshold and max-attempt budget decide that. Even for passed runs, provide the most useful next improvement or exploration plan when reward is below target.
- improvement_plan.required_next_actions must be about improving model/data performance, not about reporting, documentation, or collecting more descriptive information about the current run. Use whatever evidence is available to recommend the next experiment.
- Do not put TRIPOD prose fixes, MODEL_HANDOFF cleanup, threshold-reporting-only work, subgroup-description-only work, or calibration-reporting-only work in required_next_actions. Those can remain warnings or feedback, but the improvement plan should propose actions expected to improve validation/test AUROC, AUPRC, log loss, calibration quality, or generalization.
- If calibration is suggested in improvement_plan, phrase it as a model experiment, such as trying calibrated variants and selecting only if validation log loss/Brier score improves without hurting AUROC/AUPRC.
- Do not recommend longer training unless loss_convergence has curve evidence that training is still improving.
- Treat external validation AUROC/AUPRC or validation loss curves as primary convergence evidence. Treat sklearn internal early-stopping scores as weak metadata only, especially for imbalanced clinical classification.
- Do not recommend longer training from decreasing train loss alone. If train loss improves while external validation metrics are flat or worse, classify the run as plateaued or overfit.
- If convergence evidence is weak but the selected model reached max_iter/max epochs without early stopping or plateau evidence, include a longer-training exploration with external validation curves in improvement_plan.
- If loss_convergence is plateaued, prefer switching model/features or accepting limitations over longer training.
- If loss_convergence status is not_instrumented, require convergence_report.json repair with curve evidence or solver metadata before treating convergence as known.
- If loss_convergence status is solver_metadata_only, treat it as valid convergence evidence for non-curve selected estimators but do not infer plateau/still-improving from it.
- Treat convergence_report.json as model-produced evidence only. The judge owns retry recommendations even if a convergence artifact contains decision-like fields.
- If model_search_completeness is partial or missing on a low-performing model-owned run, require candidate_results.json, best_model_selection.json, and hyperparameter_search.json in improvement_plan.required_new_artifacts.
- Do not perform cosmetic code review. Focus on correctness, leakage, contract usage, metrics, calibration, split usage, and artifact consistency.
- Verdict feedback should cite concrete artifact evidence where practical.
- Ask for human input only if a clinical decision cannot be made from available artifacts.
"""


judge = Agent(
    name="judge",
    description="Clinical TRIPOD-aware judge and reward writer.",
    model=create_agent_model("judge"),
    system_prompt=JUDGE_PROMPT,
    tools=[
        read_pipeline_handoffs,
        read_pipeline_bundle,
        read_judge_retry_signals,
        read_model_artifact,
        inspect_predictions_consistency,
        inspect_code_artifact,
        inspect_logs,
        write_judge_verdict,
    ],
    callback_handler=None,
)
