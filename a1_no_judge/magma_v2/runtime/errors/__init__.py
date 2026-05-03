"""Structured error and repair helpers for MAGMA v2."""

from magma_v2.runtime.errors.classify import classify_failure
from magma_v2.runtime.errors.history import (
    append_error,
    append_repair_attempt,
    load_recovery_state,
    repair_signature,
    save_recovery_state,
)
from magma_v2.runtime.errors.policy import decide_recovery
from magma_v2.runtime.errors.schema import (
    EscalationDecision,
    RecoveryState,
    RepairAttempt,
    RetryPolicy,
    StageError,
)

__all__ = [
    "EscalationDecision",
    "RecoveryState",
    "RepairAttempt",
    "RetryPolicy",
    "StageError",
    "append_error",
    "append_repair_attempt",
    "classify_failure",
    "decide_recovery",
    "load_recovery_state",
    "repair_signature",
    "save_recovery_state",
]
