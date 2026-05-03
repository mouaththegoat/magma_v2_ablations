"""Shared structured error schemas for MAGMA v2 recovery."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


StageName = Literal["orchestrator", "data_worker", "model_worker", "judge", "main"]
OwnerStage = Literal["data", "model", "judge", "orchestrator", "human", "system"]
Severity = Literal["info", "warning", "error", "critical"]
Recoverability = Literal["retryable", "retryable_with_modified_plan", "human_blocking", "terminal", "degradable"]
RetryScope = Literal[
    "same_phase",
    "modified_code",
    "data_repair",
    "model_repair",
    "human_input",
    "fallback_mode",
    "partial_rerun",
    "none",
]
RunStatus = Literal["success", "provisional", "degraded", "failed"]

ERROR_CLASSES = {
    "contract_error",
    "input_error",
    "validation_error",
    "execution_error",
    "codegen_error",
    "training_error",
    "metrics_error",
    "calibration_error",
    "split_error",
    "leakage_error",
    "path_resolution_error",
    "join_error",
    "ambiguity_error",
    "capability_gap",
    "resource_error",
    "unexpected_internal_error",
}

ERROR_SUBCLASSES = {
    "missing_target",
    "missing_label_source",
    "invalid_path_column",
    "unresolved_relative_paths",
    "bad_join_keys",
    "split_integrity_failure",
    "patient_leakage_risk",
    "missing_required_artifact",
    "generated_code_syntax_error",
    "dependency_import_failure",
    "model_fit_failure",
    "metric_computation_failure",
    "calibration_incompatible",
    "unsupported_layout",
    "unsupported_modality_combo",
    "timeout",
    "memory_limit",
    "ambiguous_user_request",
    "unknown",
}


class RetryPolicy(BaseModel):
    local_repair_budget: int = 3
    orchestration_repair_budget: int = 2
    allow_degraded_completion: bool = True


class RepairAttempt(BaseModel):
    attempt_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat().replace("+00:00", "Z"))
    stage: str
    owner_stage: str
    error_class: str
    error_subclass: str = "unknown"
    retry_scope: RetryScope = "none"
    repair_instruction: str
    strategy_signature: str
    outcome: Literal["planned", "attempted", "succeeded", "failed", "skipped"] = "planned"
    details: dict[str, Any] = Field(default_factory=dict)


class StageError(BaseModel):
    status: Literal["error"] = "error"
    stage: str
    owner_stage: str
    message: str
    error_class: str = "unexpected_internal_error"
    error_subclass: str = "unknown"
    severity: Severity = "error"
    recoverability: Recoverability = "retryable_with_modified_plan"
    retry_recommended: bool = True
    retry_scope: RetryScope = "modified_code"
    attempted_repairs: list[RepairAttempt] = Field(default_factory=list)
    max_repair_budget: int = 3
    root_cause_hypothesis: str | None = None
    repair_instruction: str | None = None
    suggested_next_action: str | None = None
    blocking_artifacts: list[str] = Field(default_factory=list)
    related_artifacts: list[str] = Field(default_factory=list)
    structured_details: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat().replace("+00:00", "Z"))
    schema_warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def coerce_error_payload(cls, value: Any) -> dict[str, Any]:
        payload = dict(value) if isinstance(value, dict) else {"message": str(value)}
        warnings = payload.setdefault("schema_warnings", [])
        if not isinstance(warnings, list):
            warnings = [str(warnings)]
            payload["schema_warnings"] = warnings
        payload.setdefault("stage", "orchestrator")
        payload.setdefault("owner_stage", "orchestrator")
        payload.setdefault("message", payload.get("error") or payload.get("detail") or "Stage failed")
        payload.setdefault("error_class", "unexpected_internal_error")
        payload.setdefault("error_subclass", "unknown")
        payload.setdefault("severity", "error")
        payload.setdefault("recoverability", "retryable_with_modified_plan")
        payload.setdefault("retry_scope", "modified_code")
        for key in ("stage", "owner_stage", "message", "error_class", "error_subclass", "severity", "recoverability", "retry_scope"):
            if payload.get(key) is not None:
                payload[key] = str(payload[key])
        if payload.get("severity") not in {"info", "warning", "error", "critical"}:
            warnings.append(f"unsupported severity {payload.get('severity')!r}; coerced to error")
            payload["severity"] = "error"
        if payload.get("recoverability") not in {"retryable", "retryable_with_modified_plan", "human_blocking", "terminal", "degradable"}:
            warnings.append(f"unsupported recoverability {payload.get('recoverability')!r}; coerced to retryable_with_modified_plan")
            payload["recoverability"] = "retryable_with_modified_plan"
        if payload.get("retry_scope") not in {"same_phase", "modified_code", "data_repair", "model_repair", "human_input", "fallback_mode", "partial_rerun", "none"}:
            warnings.append(f"unsupported retry_scope {payload.get('retry_scope')!r}; coerced to modified_code")
            payload["retry_scope"] = "modified_code"
        for key in ("blocking_artifacts", "related_artifacts", "attempted_repairs"):
            if not isinstance(payload.get(key), list):
                payload[key] = [] if payload.get(key) is None else [payload.get(key)]
        if not isinstance(payload.get("structured_details"), dict):
            payload["structured_details"] = {"raw_structured_details": payload.get("structured_details")}
        return payload

    @model_validator(mode="after")
    def validate_error_terms(self) -> "StageError":
        if self.error_class not in ERROR_CLASSES:
            self.schema_warnings.append(f"unsupported error_class {self.error_class!r}; coerced to unexpected_internal_error")
            self.error_class = "unexpected_internal_error"
        if self.error_subclass not in ERROR_SUBCLASSES:
            self.schema_warnings.append(f"unsupported error_subclass {self.error_subclass!r}; coerced to unknown")
            self.error_subclass = "unknown"
        if self.owner_stage not in {"data", "model", "judge", "orchestrator", "human", "system"}:
            self.schema_warnings.append(f"unsupported owner_stage {self.owner_stage!r}; coerced to orchestrator")
            self.owner_stage = "orchestrator"
        if self.retry_scope not in {"same_phase", "modified_code", "data_repair", "model_repair", "human_input", "fallback_mode", "partial_rerun", "none"}:
            self.schema_warnings.append(f"unsupported retry_scope {self.retry_scope!r}; coerced to modified_code")
            self.retry_scope = "modified_code"
        if self.recoverability in {"human_blocking", "terminal"}:
            self.retry_recommended = False
        if self.recoverability == "human_blocking":
            self.owner_stage = "human"
            self.retry_scope = "human_input"
        return self


class RecoveryState(BaseModel):
    stage_attempt_counts: dict[str, int] = Field(default_factory=dict)
    orchestration_attempt_count: int = 0
    repair_history: list[RepairAttempt] = Field(default_factory=list)
    error_history: list[StageError] = Field(default_factory=list)
    last_successful_handoff: dict[str, str] = Field(default_factory=dict)
    failed_artifacts: list[str] = Field(default_factory=list)
    fallback_modes_used: list[str] = Field(default_factory=list)
    final_escalation_reason: str | None = None
    run_status: RunStatus | None = None


class EscalationDecision(BaseModel):
    status: Literal["retry", "ask_human", "terminal", "degraded", "provisional"]
    target_stage: OwnerStage
    retry_scope: RetryScope
    reason: str
    repair_instruction: str
    suggested_next_action: str
    budget_remaining: int
    repeated_strategy: bool = False
    should_call_data_worker: bool = False
    should_call_model_worker: bool = False
    should_ask_human: bool = False
    terminal: bool = False
    degraded_allowed: bool = False
