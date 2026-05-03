"""Candidate registry for MAGMA v2 model search.

The registry defines policy expectations, not implementation details. Generated
model code can map these candidate ids to local sklearn/PyTorch implementations
that are feasible for the current run.
"""

from __future__ import annotations


EHR_TEXT_REQUIRED_CANDIDATES = [
    "prevalence_reference",
    "ehr_logreg",
    "text_tfidf_logreg",
    "text_tfidf_elasticnet",
    "concat_tfidf_ehr_logreg",
    "concat_tfidf_ehr_elasticnet",
    "sparse_sgd_logloss_or_linear_svc",
    "svd_ehr_histgb",
    "installed_boosting_if_available",
    "late_fusion_best_family_sources",
    "dual_branch_mlp_challenger",
]


COMPLEXITY_LADDER = [
    {
        "tier": 0,
        "name": "sanity_anchors",
        "purpose": "prevalence/reference checks and feature/label/split sanity checks",
    },
    {
        "tier": 1,
        "name": "strong_sparse_linear",
        "purpose": "mandatory sparse/text/tabular linear anchors",
    },
    {
        "tier": 2,
        "name": "dense_projection_boosting",
        "purpose": "SVD/dense representations with tree or boosting models",
    },
    {
        "tier": 3,
        "name": "late_fusion_stacking",
        "purpose": "leakage-safe weighted averaging or OOF stacking over top candidates",
    },
    {
        "tier": 4,
        "name": "neural_fusion",
        "purpose": "compact neural challengers with validation checkpointing and seed checks",
    },
    {
        "tier": 5,
        "name": "pretrained_modality_encoders",
        "purpose": "pretrained image/text/signal encoders when available without new installs",
    },
]


def complexity_ladder() -> list[dict[str, object]]:
    """Return a copy of the current complexity ladder."""
    return [dict(item) for item in COMPLEXITY_LADDER]
