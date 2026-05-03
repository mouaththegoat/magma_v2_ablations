"""Validation-only model selection helpers."""

from __future__ import annotations

from math import inf
from typing import Any


NEAR_TIE_AUROC = 0.005
COMPLEX_MIN_AUROC_GAIN_FOR_BAD_PROBABILITY = 0.01
BAD_PROBABILITY_MULTIPLIER = 2.0


def _metric(candidate: dict[str, Any], *keys: str) -> float | None:
    current: Any = candidate
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return float(current) if isinstance(current, (int, float)) else None


def _complexity_penalty(candidate: dict[str, Any]) -> float:
    tier = candidate.get("complexity_tier")
    if isinstance(tier, int):
        return float(tier)
    family = str(candidate.get("family") or "").lower()
    if any(token in family for token in ("neural", "mlp", "transformer", "cnn", "fusion")):
        return 4.0
    if any(token in family for token in ("boost", "forest", "tree")):
        return 2.0
    return 1.0


def _is_complex(candidate: dict[str, Any]) -> bool:
    return _complexity_penalty(candidate) >= 3.0


def _bad_probability_quality(complex_candidate: dict[str, Any], simple_candidate: dict[str, Any]) -> bool:
    for key in ("log_loss", "brier"):
        complex_value = _metric(complex_candidate, "validation", key) or _metric(complex_candidate, f"validation_{key}")
        simple_value = _metric(simple_candidate, "validation", key) or _metric(simple_candidate, f"validation_{key}")
        if complex_value is not None and simple_value is not None and simple_value > 0:
            if complex_value > BAD_PROBABILITY_MULTIPLIER * simple_value:
                return True
    return False


def select_candidate(candidates: list[dict[str, Any]], near_tie_auroc: float = NEAR_TIE_AUROC) -> dict[str, Any]:
    """Select a validation-locked candidate with stability-aware tie-breaking.

    The function intentionally ignores all test metrics. Candidate records may
    use nested `validation.{metric}` or flat `validation_{metric}` fields.
    """
    valid = [
        candidate
        for candidate in candidates
        if candidate.get("valid_for_selection", True)
        and (_metric(candidate, "validation", "auroc") is not None or _metric(candidate, "validation_auroc") is not None)
    ]
    if not valid:
        raise ValueError("No candidates with validation AUROC are valid for selection.")

    def auroc(candidate: dict[str, Any]) -> float:
        return _metric(candidate, "validation", "auroc") or _metric(candidate, "validation_auroc") or -inf

    best_auc = max(auroc(candidate) for candidate in valid)
    near_ties = [candidate for candidate in valid if auroc(candidate) >= best_auc - near_tie_auroc]
    simple_near_ties = [candidate for candidate in near_ties if not _is_complex(candidate)]

    filtered: list[dict[str, Any]] = []
    for candidate in near_ties:
        if not _is_complex(candidate) or not simple_near_ties:
            filtered.append(candidate)
            continue
        best_simple_auc = max(auroc(item) for item in simple_near_ties)
        stability_ok = bool(candidate.get("seed_stability_acceptable", candidate.get("seed_count", 0) >= 3))
        if (
            auroc(candidate) < best_simple_auc + COMPLEX_MIN_AUROC_GAIN_FOR_BAD_PROBABILITY
            and any(_bad_probability_quality(candidate, simple) for simple in simple_near_ties)
            and not stability_ok
        ):
            continue
        filtered.append(candidate)

    pool = filtered or near_ties

    def score(candidate: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
        val_auprc = _metric(candidate, "validation", "auprc") or _metric(candidate, "validation_auprc") or -inf
        val_log_loss = _metric(candidate, "validation", "log_loss") or _metric(candidate, "validation_log_loss") or inf
        val_brier = _metric(candidate, "validation", "brier") or _metric(candidate, "validation_brier") or inf
        seed_std = (
            candidate.get("std_val_auroc")
            if isinstance(candidate.get("std_val_auroc"), (int, float))
            else candidate.get("seed_std_auroc", inf)
        )
        seed_std_float = float(seed_std) if isinstance(seed_std, (int, float)) else inf
        return (
            auroc(candidate),
            val_auprc,
            -val_log_loss,
            -val_brier,
            -seed_std_float,
            -_complexity_penalty(candidate),
        )

    return sorted(pool, key=score, reverse=True)[0]
