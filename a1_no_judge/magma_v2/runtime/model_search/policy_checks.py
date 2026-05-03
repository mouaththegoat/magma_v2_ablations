"""Executable checks for MAGMA v2 model-search policy."""

from __future__ import annotations

from typing import Any

from .candidate_registry import EHR_TEXT_REQUIRED_CANDIDATES


STRICT_TEST_KEYS = {
    "test_auroc",
    "test_auc",
    "test_auprc",
    "test_ap",
    "test_metrics",
    "test_cache",
}


def _candidate_name(candidate: dict[str, Any]) -> str:
    return str(candidate.get("candidate_name") or candidate.get("name") or candidate.get("id") or "")


def _contains_test_metrics(value: Any) -> bool:
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key).lower() in STRICT_TEST_KEYS:
                return True
            if _contains_test_metrics(nested):
                return True
    elif isinstance(value, list):
        return any(_contains_test_metrics(item) for item in value)
    return False


def _has_candidate(candidates: list[dict[str, Any]], candidate_id: str) -> bool:
    target = candidate_id.lower()
    for candidate in candidates:
        name = _candidate_name(candidate).lower()
        family = str(candidate.get("family") or "").lower()
        strategy = str(candidate.get("modality_strategy") or "").lower()
        if target in name or target in family or target in strategy:
            return True
    return False


def check_model_search_policy(run: dict[str, Any]) -> dict[str, Any]:
    """Return policy violations and warnings for model-search artifacts.

    Expected input is a lightweight dict with keys such as execution_mode,
    task_family, candidates, previous_winner_known, previous_winner_included,
    calibration_used_for_candidate_selection, and final_test_evaluated_once.
    """
    violations: list[str] = []
    warnings: list[str] = []

    candidates = run.get("candidates") if isinstance(run.get("candidates"), list) else []
    candidate_dicts = [item for item in candidates if isinstance(item, dict)]
    execution_mode = str(run.get("execution_mode") or "")
    task_family = str(run.get("task_family") or run.get("task") or "").lower()

    performance_mode = execution_mode in {"performance_seeking_benchmark", "reward_improvement_retry", ""}
    if performance_mode and task_family in {"ehr_text", "ehr+text", "tabular_text", "multimodal_ehr_text"}:
        missing = [
            candidate_id
            for candidate_id in EHR_TEXT_REQUIRED_CANDIDATES
            if not _has_candidate(candidate_dicts, candidate_id)
        ]
        if missing:
            violations.append("missing_required_ehr_text_candidates:" + ",".join(missing))

    if run.get("previous_winner_known") and not run.get("previous_winner_included"):
        violations.append("previous_winner_not_reproduced")

    if run.get("calibration_used_for_candidate_selection"):
        violations.append("calibration_used_for_model_selection")

    if any(_contains_test_metrics(candidate) for candidate in candidate_dicts):
        violations.append("candidate_test_metrics_violation")

    if run.get("final_test_evaluated_once") is False:
        violations.append("final_test_not_evaluated_exactly_once")

    if run.get("mandatory_tiers_skipped_without_reason"):
        violations.append("mandatory_complexity_tiers_skipped_without_reason")

    if not run.get("model_selection_diagnostics_present"):
        warnings.append("missing_model_selection_diagnostics")

    return {
        "status": "passed" if not violations else "failed",
        "violations": violations,
        "warnings": warnings,
    }
