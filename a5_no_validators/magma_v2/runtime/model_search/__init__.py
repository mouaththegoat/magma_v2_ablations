"""Model-search policy helpers for MAGMA v2.

These modules are intentionally small and dependency-light so generated model
code can import them when available, and validators can reason about the same
policy without relying on long prompt text.
"""

from .candidate_registry import EHR_TEXT_REQUIRED_CANDIDATES, complexity_ladder
from .policy_checks import check_model_search_policy
from .selection import select_candidate

__all__ = [
    "EHR_TEXT_REQUIRED_CANDIDATES",
    "check_model_search_policy",
    "complexity_ladder",
    "select_candidate",
]
