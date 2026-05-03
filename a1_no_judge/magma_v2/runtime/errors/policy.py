"""Retry and routing policy for structured MAGMA v2 failures."""

from __future__ import annotations

from magma_v2.runtime.errors.history import has_repeated_strategy, repair_signature
from magma_v2.runtime.errors.schema import (
    EscalationDecision,
    RecoveryState,
    RepairAttempt,
    RetryPolicy,
    StageError,
)


def decide_recovery(
    error: StageError,
    state: RecoveryState,
    policy: RetryPolicy | None = None,
) -> EscalationDecision:
    """Return the next routing decision for a structured stage error."""
    policy = policy or RetryPolicy()
    default_repair_instruction = error.repair_instruction or "Retry with a materially different strategy."
    signature = repair_signature(
        error.error_class,
        error.error_subclass,
        error.retry_scope,
        default_repair_instruction,
    )
    repeated = has_repeated_strategy(state, signature)
    budget_remaining = max(0, policy.orchestration_repair_budget - state.orchestration_attempt_count)

    if error.recoverability == "human_blocking" or error.owner_stage == "human":
        return EscalationDecision(
            status="ask_human",
            target_stage="human",
            retry_scope="human_input",
            reason=error.root_cause_hypothesis or error.message,
            repair_instruction=error.repair_instruction or "Ask the user for the missing clinical/data decision.",
            suggested_next_action=error.suggested_next_action or "ask_human",
            budget_remaining=budget_remaining,
            repeated_strategy=repeated,
            should_ask_human=True,
        )

    if error.recoverability == "terminal":
        return _terminal(error, budget_remaining, repeated, "Error is marked terminal.")

    if repeated:
        return _terminal(error, budget_remaining, repeated, "Repeated identical repair strategy was blocked.")

    if budget_remaining <= 0:
        return _terminal(error, budget_remaining, repeated, "Orchestration repair budget exhausted.")

    if error.recoverability == "degradable" and policy.allow_degraded_completion:
        return EscalationDecision(
            status="degraded",
            target_stage=_target_stage(error),
            retry_scope=error.retry_scope,
            reason=error.root_cause_hypothesis or error.message,
            repair_instruction=error.repair_instruction or "Use a degraded/provisional fallback and document limitations.",
            suggested_next_action=error.suggested_next_action or "degrade_or_retry_fallback",
            budget_remaining=budget_remaining,
            repeated_strategy=False,
            should_call_data_worker=error.owner_stage == "data",
            should_call_model_worker=error.owner_stage == "model",
            degraded_allowed=True,
        )

    target = _target_stage(error)
    return EscalationDecision(
        status="retry",
        target_stage=target,
        retry_scope=error.retry_scope,
        reason=error.root_cause_hypothesis or error.message,
        repair_instruction=default_repair_instruction,
        suggested_next_action=error.suggested_next_action or "retry_owner_stage",
        budget_remaining=budget_remaining,
        repeated_strategy=False,
        should_call_data_worker=target == "data",
        should_call_model_worker=target == "model",
    )


def build_repair_attempt(error: StageError, decision: EscalationDecision) -> RepairAttempt:
    return RepairAttempt(
        stage=error.stage,
        owner_stage=decision.target_stage,
        error_class=error.error_class,
        error_subclass=error.error_subclass,
        retry_scope=decision.retry_scope,
        repair_instruction=decision.repair_instruction,
        strategy_signature=repair_signature(
            error.error_class,
            error.error_subclass,
            decision.retry_scope,
            decision.repair_instruction,
        ),
        outcome="planned",
        details={"decision_status": decision.status, "reason": decision.reason},
    )


def _target_stage(error: StageError) -> str:
    if error.error_class in {"path_resolution_error", "split_error", "leakage_error", "join_error"}:
        return "data"
    if error.error_class in {"training_error", "metrics_error", "calibration_error", "capability_gap"}:
        return "model"
    if error.owner_stage in {"data", "model", "judge", "orchestrator"}:
        return error.owner_stage
    return "orchestrator"


def _terminal(error: StageError, budget_remaining: int, repeated: bool, reason: str) -> EscalationDecision:
    return EscalationDecision(
        status="terminal",
        target_stage=_target_stage(error),
        retry_scope="none",
        reason=reason,
        repair_instruction=error.repair_instruction or "No defensible automated repair remains.",
        suggested_next_action="write_failed_final_report",
        budget_remaining=budget_remaining,
        repeated_strategy=repeated,
        terminal=True,
    )
