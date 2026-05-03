"""Structured failure classification for MAGMA v2.

This module intentionally avoids broad keyword routing. Workers should emit
structured stage errors whenever they know the failure class. When only raw
stdout/stderr is available, classification stays conservative and recoverable
instead of guessing from terms in prose.
"""

from __future__ import annotations

from typing import Any

from magma_v2.runtime.errors.schema import ERROR_CLASSES, ERROR_SUBCLASSES, StageError


def classify_failure(
    *,
    stage: str,
    message: str = "",
    stderr: str = "",
    stdout: str = "",
    owner_stage: str | None = None,
    related_artifacts: list[str] | None = None,
    structured_details: dict[str, Any] | None = None,
) -> StageError:
    """Create a structured StageError without semantic keyword guessing."""
    details = structured_details or {}
    explicit = _explicit_error_payload(details)
    if explicit:
        explicit.setdefault("stage", stage)
        explicit.setdefault("owner_stage", owner_stage or _default_owner(stage))
        explicit.setdefault("message", message or explicit.get("message") or "Stage failed")
        explicit.setdefault("related_artifacts", related_artifacts or [])
        return StageError(**explicit)

    exception_type = str(details.get("exception_type") or details.get("exception") or "")
    mapped = _classify_exception_type(exception_type)
    owner = owner_stage or _default_owner(stage)
    if mapped:
        error_class, error_subclass, retry_scope = mapped
        return StageError(
            stage=stage,
            owner_stage=owner,
            message=message or exception_type or "Stage failed",
            error_class=error_class,
            error_subclass=error_subclass,
            retry_scope=retry_scope,
            recoverability="retryable_with_modified_plan",
            root_cause_hypothesis=message or exception_type or "Structured exception reported by stage.",
            repair_instruction="Use a materially different local repair strategy, then re-emit the canonical handoff or structured error.",
            suggested_next_action="retry_with_modified_strategy",
            related_artifacts=related_artifacts or [],
            structured_details={
                **details,
                "stdout_tail": stdout[-2000:] if stdout else "",
                "stderr_tail": stderr[-2000:] if stderr else "",
            },
        )

    return StageError(
        stage=stage,
        owner_stage=owner,
        message=message or "Stage failed without structured error metadata.",
        error_class="unexpected_internal_error",
        error_subclass="unknown",
        retry_scope="modified_code",
        recoverability="retryable_with_modified_plan",
        root_cause_hypothesis="The stage failed but did not provide structured error metadata.",
        repair_instruction="Inspect artifacts/logs and retry with a materially different free-code or fallback strategy. Do not infer clinical semantics from stderr text.",
        suggested_next_action="retry_with_structured_error_or_fallback",
        related_artifacts=related_artifacts or [],
        structured_details={
            **details,
            "stdout_tail": stdout[-2000:] if stdout else "",
            "stderr_tail": stderr[-2000:] if stderr else "",
        },
    )


def _explicit_error_payload(details: dict[str, Any]) -> dict[str, Any] | None:
    payload = details.get("stage_error") or details.get("error_payload") or details.get("structured_error")
    if isinstance(payload, dict):
        return dict(payload)
    if details.get("required_user_input"):
        return {
            "owner_stage": "human",
            "error_class": "ambiguity_error",
            "error_subclass": "ambiguous_user_request",
            "recoverability": "human_blocking",
            "retry_scope": "human_input",
            "structured_details": {"required_user_input": details.get("required_user_input")},
        }
    if details.get("error_class") in ERROR_CLASSES:
        return {
            key: value
            for key, value in details.items()
            if key in {
                "stage",
                "owner_stage",
                "message",
                "error_class",
                "error_subclass",
                "severity",
                "recoverability",
                "retry_scope",
                "root_cause_hypothesis",
                "repair_instruction",
                "suggested_next_action",
                "blocking_artifacts",
                "related_artifacts",
                "structured_details",
            }
        }
    return None


def _classify_exception_type(exception_type: str) -> tuple[str, str, str] | None:
    mapping = {
        "SyntaxError": ("codegen_error", "generated_code_syntax_error", "modified_code"),
        "IndentationError": ("codegen_error", "generated_code_syntax_error", "modified_code"),
        "ModuleNotFoundError": ("execution_error", "dependency_import_failure", "fallback_mode"),
        "ImportError": ("execution_error", "dependency_import_failure", "fallback_mode"),
        "FileNotFoundError": ("path_resolution_error", "unresolved_relative_paths", "data_repair"),
        "TimeoutError": ("resource_error", "timeout", "fallback_mode"),
        "MemoryError": ("resource_error", "memory_limit", "fallback_mode"),
    }
    result = mapping.get(exception_type)
    if result and result[1] in ERROR_SUBCLASSES:
        return result
    return None


def _default_owner(stage: str) -> str:
    stage_family = stage.split("_", 1)[0]
    if stage_family == "data":
        return "data"
    if stage_family == "model":
        return "model"
    if stage_family == "judge":
        return "judge"
    return "orchestrator"
