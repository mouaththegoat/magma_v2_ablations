"""Persistent repair/error history for MAGMA v2 runs."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from magma_v2.runtime.artifacts import read_json, write_json
from magma_v2.runtime.errors.schema import RecoveryState, RepairAttempt, StageError


RECOVERY_STATE_FILENAME = "recovery_state.json"


def recovery_state_path(run_dir: str) -> str:
    return str(Path(run_dir) / RECOVERY_STATE_FILENAME)


def load_recovery_state(run_dir: str) -> RecoveryState:
    path = recovery_state_path(run_dir)
    if not os.path.exists(path):
        return RecoveryState()
    try:
        return RecoveryState(**read_json(path))
    except (ValidationError, TypeError, ValueError):
        return RecoveryState()


def save_recovery_state(run_dir: str, state: RecoveryState) -> str:
    path = recovery_state_path(run_dir)
    write_json(path, state.model_dump(mode="json"))
    return path


def repair_signature(error_class: str, error_subclass: str, retry_scope: str, repair_instruction: str | None) -> str:
    payload = "|".join([
        error_class or "",
        error_subclass or "",
        retry_scope or "",
        (repair_instruction or "").strip().lower(),
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def append_error(run_dir: str, error: StageError) -> RecoveryState:
    state = load_recovery_state(run_dir)
    state.error_history.append(error)
    stage_count_key = f"{error.stage}:{error.error_class}:{error.error_subclass}"
    state.stage_attempt_counts[stage_count_key] = state.stage_attempt_counts.get(stage_count_key, 0) + 1
    for artifact in error.blocking_artifacts:
        if artifact not in state.failed_artifacts:
            state.failed_artifacts.append(artifact)
    save_recovery_state(run_dir, state)
    return state


def append_repair_attempt(run_dir: str, attempt: RepairAttempt) -> RecoveryState:
    state = load_recovery_state(run_dir)
    state.repair_history.append(attempt)
    state.orchestration_attempt_count += 1
    save_recovery_state(run_dir, state)
    return state


def has_repeated_strategy(state: RecoveryState, signature: str) -> bool:
    return any(attempt.strategy_signature == signature for attempt in state.repair_history)


def state_summary(state: RecoveryState) -> dict[str, Any]:
    return {
        "stage_attempt_counts": state.stage_attempt_counts,
        "orchestration_attempt_count": state.orchestration_attempt_count,
        "repair_history_count": len(state.repair_history),
        "error_history_count": len(state.error_history),
        "failed_artifacts": state.failed_artifacts,
        "fallback_modes_used": state.fallback_modes_used,
        "final_escalation_reason": state.final_escalation_reason,
        "run_status": state.run_status,
    }
