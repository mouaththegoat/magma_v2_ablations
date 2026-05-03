"""Typed contracts for MAGMA v2 normalization."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


ObservedLayout = str
SplitSource = str
NormalizedLayout = str


class ProducedFileEntry(BaseModel):
    path: str
    role: str
    purpose: str = ""

    @model_validator(mode="before")
    @classmethod
    def coerce_entry(cls, value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            payload = dict(value)
            payload["path"] = "" if payload.get("path") is None else str(payload.get("path"))
            payload["role"] = str(payload.get("role") or "artifact")
            payload["purpose"] = str(payload.get("purpose") or "")
            return payload
        return {"path": "" if value is None else str(value), "role": "artifact", "purpose": "Coerced artifact entry."}


ProducedFile = ProducedFileEntry


class LabelSpec(BaseModel):
    source_type: str = "unknown"
    source_name: str | None = None
    target_name: str | None = None
    task_type: str | None = None
    class_labels: list[Any] = Field(default_factory=list)
    positive_class: Any | None = None
    confidence: float | None = None
    notes: list[str] = Field(default_factory=list)
    # Backward-compatible aliases used by older prompts/tools.
    column: str | None = None
    source: str | None = None
    evidence: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def coerce_label_spec(cls, value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            return {"source_type": "unknown", "notes": [f"Coerced non-object label_spec of type {type(value).__name__}."]}
        payload = dict(value)
        for key in ("source_name", "target_name", "task_type", "column", "source"):
            if payload.get(key) is not None:
                payload[key] = str(payload[key])
        payload["source_type"] = str(payload.get("source_type") or "unknown")
        if not isinstance(payload.get("class_labels"), list):
            payload["class_labels"] = [] if payload.get("class_labels") is None else [payload.get("class_labels")]
        if not isinstance(payload.get("notes"), list):
            payload["notes"] = [] if payload.get("notes") is None else [str(payload.get("notes"))]
        if not isinstance(payload.get("evidence"), list):
            payload["evidence"] = []
        return payload


class PathResolutionSummary(BaseModel):
    path_columns: list[str] = Field(default_factory=list)
    base_dirs: list[str] = Field(default_factory=list)
    total_paths_checked: int = 0
    existing_paths: int = 0
    missing_paths: int = 0
    exists_rate: float | None = None
    file_types_detected: dict[str, int] = Field(default_factory=dict)
    # Backward-compatible counters retained for older readers.
    total_references: int = 0
    resolved_references: int = 0
    missing_references: int = 0
    missing_examples: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def coerce_summary(cls, value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            return {}
        payload = dict(value)
        for key in ("path_columns", "base_dirs"):
            if not isinstance(payload.get(key), list):
                payload[key] = [] if payload.get(key) is None else [str(payload.get(key))]
        for key in ("total_paths_checked", "existing_paths", "missing_paths", "total_references", "resolved_references", "missing_references"):
            try:
                payload[key] = int(payload.get(key) or 0)
            except (TypeError, ValueError):
                payload[key] = 0
        try:
            payload["exists_rate"] = None if payload.get("exists_rate") is None else float(payload.get("exists_rate"))
        except (TypeError, ValueError):
            payload["exists_rate"] = None
        if not isinstance(payload.get("file_types_detected"), dict):
            payload["file_types_detected"] = {}
        if not isinstance(payload.get("missing_examples"), list):
            payload["missing_examples"] = []
        return payload

    @model_validator(mode="after")
    def sync_legacy_counters(self) -> "PathResolutionSummary":
        if self.total_paths_checked == 0 and self.total_references:
            self.total_paths_checked = self.total_references
        if self.existing_paths == 0 and self.resolved_references:
            self.existing_paths = self.resolved_references
        if self.missing_paths == 0 and self.missing_references:
            self.missing_paths = self.missing_references
        if self.total_references == 0 and self.total_paths_checked:
            self.total_references = self.total_paths_checked
        if self.resolved_references == 0 and self.existing_paths:
            self.resolved_references = self.existing_paths
        if self.missing_references == 0 and self.missing_paths:
            self.missing_references = self.missing_paths
        if self.total_paths_checked and self.exists_rate is None:
            self.exists_rate = self.existing_paths / self.total_paths_checked
        return self


class SchemaSummary(BaseModel):
    source_paths: list[str] = Field(default_factory=list)
    columns: list[str] = Field(default_factory=list)
    row_count: int | None = None
    numeric_column_count: int = 0
    categorical_column_count: int = 0
    modality_evidence: dict[str, Any] = Field(default_factory=dict)
    columns_by_source: dict[str, list[str]] = Field(default_factory=dict)
    dtypes_by_source: dict[str, dict[str, str]] = Field(default_factory=dict)
    row_counts_by_source: dict[str, int] = Field(default_factory=dict)
    candidate_targets: list[dict[str, Any]] = Field(default_factory=list)
    candidate_group_ids: list[dict[str, Any]] = Field(default_factory=list)
    candidate_path_columns: list[dict[str, Any]] = Field(default_factory=list)
    candidate_text_columns: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def coerce_schema_summary(cls, value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            return {}
        payload = dict(value)
        for key in ("source_paths", "columns"):
            if not isinstance(payload.get(key), list):
                payload[key] = [] if payload.get(key) is None else [str(payload.get(key))]
        for key in ("columns_by_source", "dtypes_by_source", "row_counts_by_source", "modality_evidence"):
            if not isinstance(payload.get(key), dict):
                payload[key] = {}
        for key in ("candidate_targets", "candidate_group_ids", "candidate_path_columns", "candidate_text_columns"):
            if not isinstance(payload.get(key), list):
                payload[key] = []
        return payload


class ValidatorReport(BaseModel):
    name: str
    status: str = "provisional"
    path: str | None = None
    critical_failures: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def coerce_report(cls, value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            return {"name": "coerced_validator_report", "status": "provisional", "warnings": [f"Coerced non-object validator report of type {type(value).__name__}."]}
        payload = dict(value)
        payload["name"] = str(payload.get("name") or "validator_report")
        status = str(payload.get("status") or "provisional").lower()
        payload["status"] = status if status in {"passed", "failed", "provisional", "warning"} else "provisional"
        for key in ("critical_failures", "warnings"):
            if not isinstance(payload.get(key), list):
                payload[key] = [] if payload.get(key) is None else [str(payload.get(key))]
        if not isinstance(payload.get("evidence"), dict):
            payload["evidence"] = {}
        if payload.get("path") is not None:
            payload["path"] = str(payload["path"])
        return payload


class ConfidenceSummary(BaseModel):
    layout_confidence: float | None = None
    label_confidence: float | None = None
    join_key_confidence: float | None = None
    split_confidence: float | None = None
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def coerce_confidence(cls, value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            return {}
        payload = dict(value)
        for key in ("layout_confidence", "label_confidence", "join_key_confidence", "split_confidence"):
            try:
                payload[key] = None if payload.get(key) is None else float(payload.get(key))
            except (TypeError, ValueError):
                payload[key] = None
        if not isinstance(payload.get("notes"), list):
            payload["notes"] = [] if payload.get("notes") is None else [str(payload.get("notes"))]
        return payload


class DataHandoff(BaseModel):
    """Root DATA_HANDOFF.json schema.

    This is intentionally additive: old fields remain valid, but downstream
    stages should prefer semantic fields like normalized_layout and label_spec.
    """

    model_config = ConfigDict(extra="allow")

    status: Literal["success", "error"] = "success"
    observed_layout: ObservedLayout
    normalized_layout: NormalizedLayout
    prediction_unit: str
    label_spec: LabelSpec
    split_source: SplitSource
    schema_summary: SchemaSummary = Field(default_factory=SchemaSummary)
    path_resolution: PathResolutionSummary = Field(default_factory=PathResolutionSummary)
    normalization_artifacts: list[ProducedFileEntry] = Field(default_factory=list)
    validator_reports: list[ValidatorReport] = Field(default_factory=list)
    unresolved_questions: list[dict[str, Any]] = Field(default_factory=list)
    confidence_summary: ConfidenceSummary = Field(default_factory=ConfidenceSummary)

    modality: str | None = None
    modalities: list[str] = Field(default_factory=list)
    task_type: str
    data_layout: str | None = None
    split_strategy: str | None = None
    train_data: str | None = None
    val_data: str | None = None
    test_data: str | None = None
    model_ready_data: str | None = None
    target: str
    patient_id_column: str | None = None
    join_keys: list[str] = Field(default_factory=list)
    leakage_controls: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    feature_exclusion_columns: list[str] = Field(default_factory=list)
    produced_files: list[ProducedFileEntry] = Field(default_factory=list)
    readme_path: str | None = None
    contract_warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def coerce_handoff(cls, value: Any) -> dict[str, Any]:
        payload = dict(value) if isinstance(value, dict) else {}
        warnings = payload.setdefault("contract_warnings", [])
        if not isinstance(warnings, list):
            warnings = [str(warnings)]
            payload["contract_warnings"] = warnings
        if not payload.get("observed_layout"):
            payload["observed_layout"] = "unknown"
            warnings.append("Missing observed_layout was coerced to 'unknown'.")
        if not payload.get("normalized_layout"):
            payload["normalized_layout"] = payload.get("data_layout") or "unknown"
            warnings.append("Missing normalized_layout was coerced from data_layout or 'unknown'.")
        if not payload.get("prediction_unit"):
            payload["prediction_unit"] = "unknown"
            warnings.append("Missing prediction_unit was coerced to 'unknown'.")
        if not payload.get("split_source"):
            payload["split_source"] = "unknown"
        if not payload.get("task_type"):
            label_task = (payload.get("label_spec") or {}).get("task_type") if isinstance(payload.get("label_spec"), dict) else None
            payload["task_type"] = label_task or "unknown"
            warnings.append("Missing task_type was coerced from label_spec or 'unknown'.")
        if not payload.get("target"):
            payload["target"] = (payload.get("label_spec") or {}).get("target_name") if isinstance(payload.get("label_spec"), dict) else None
        if not payload.get("target"):
            payload["target"] = "label"
            warnings.append("Missing target was coerced to 'label'.")
        payload.setdefault("label_spec", {})
        if not isinstance(payload["label_spec"], dict):
            payload["label_spec"] = {"source_type": "unknown", "notes": ["label_spec was not an object and was coerced."]}
        payload["label_spec"].setdefault("target_name", payload.get("target"))
        payload["label_spec"].setdefault("task_type", payload.get("task_type"))
        for key in ("modalities", "join_keys", "leakage_controls", "assumptions", "limitations", "feature_exclusion_columns", "produced_files", "normalization_artifacts", "validator_reports", "unresolved_questions"):
            if not isinstance(payload.get(key), list):
                payload[key] = [] if payload.get(key) is None else [payload.get(key)]
        payload["join_keys"] = [str(key) for key in payload.get("join_keys", []) if key is not None]
        for key in ("modality", "data_layout", "split_strategy", "train_data", "val_data", "test_data", "model_ready_data", "patient_id_column", "readme_path"):
            if payload.get(key) is not None:
                payload[key] = str(payload[key])
        return payload

    @model_validator(mode="after")
    def validate_semantic_contract(self) -> "DataHandoff":
        warnings = self.contract_warnings
        if self.label_spec.target_name and self.label_spec.target_name != self.target:
            warnings.append(f"label_spec.target_name ({self.label_spec.target_name}) did not match target ({self.target}); target was treated as canonical.")
            self.label_spec.target_name = self.target
        if self.label_spec.task_type and self.label_spec.task_type != self.task_type:
            warnings.append(f"label_spec.task_type ({self.label_spec.task_type}) did not match task_type ({self.task_type}); task_type was treated as canonical.")
            self.label_spec.task_type = self.task_type
        if self.path_resolution.total_paths_checked:
            total = self.path_resolution.total_paths_checked
            observed_total = self.path_resolution.existing_paths + self.path_resolution.missing_paths
            if observed_total != total:
                self.path_resolution.missing_paths = max(0, total - self.path_resolution.existing_paths)
                self.path_resolution.missing_references = self.path_resolution.missing_paths
                warnings.append("path_resolution counters were inconsistent and missing_paths was recalculated.")
            if self.path_resolution.exists_rate is not None and not 0 <= self.path_resolution.exists_rate <= 1:
                self.path_resolution.exists_rate = max(0.0, min(1.0, self.path_resolution.exists_rate))
                warnings.append("path_resolution.exists_rate was clamped to [0, 1].")
        return self


class NormalizedContract(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: Literal["success", "error"]
    observed_layout: ObservedLayout
    normalized_layout: NormalizedLayout | None = None
    task_type: str | None = None
    modalities: list[str] = Field(default_factory=list)
    prediction_unit: str | None = None
    label_spec: LabelSpec = Field(default_factory=LabelSpec)
    split_strategy: str | None = None
    split_source: SplitSource = "none"
    join_keys: list[str] = Field(default_factory=list)
    patient_id_column: str | None = None
    feature_exclusion_columns: list[str] = Field(default_factory=list)
    leakage_controls: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    path_resolution: PathResolutionSummary = Field(default_factory=PathResolutionSummary)
    schema_summary: SchemaSummary = Field(default_factory=SchemaSummary)
    confidence_summary: ConfidenceSummary = Field(default_factory=ConfidenceSummary)
    normalization_artifacts: list[ProducedFile] = Field(default_factory=list)
    produced_files: list[ProducedFile] = Field(default_factory=list)
    validator_reports: list[ValidatorReport] = Field(default_factory=list)
    unresolved_questions: list[dict[str, Any]] = Field(default_factory=list)
    train_data: str | None = None
    val_data: str | None = None
    test_data: str | None = None
    model_ready_data: str | None = None
    target: str | None = None
    modality: str | None = None
    data_layout: str | None = None
    error: str | None = None
    error_class: str | None = None
    error_subclass: str | None = None
    owner_stage: str | None = None
    recoverability: str | None = None
    retry_scope: str | None = None
    required_user_input: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_success_fields(self) -> "NormalizedContract":
        if self.status == "error":
            return self
        missing = []
        for field in ("normalized_layout", "task_type", "train_data", "val_data", "test_data", "model_ready_data", "target"):
            if not getattr(self, field):
                missing.append(field)
        if missing:
            self.limitations.append(f"Normalization contract is incomplete but non-blocking: missing {missing}.")
        return self

    def to_data_handoff(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.update({
            "modality": self.modality or (self.modalities[0] if self.modalities else None),
            "data_layout": self.data_layout or self.normalized_layout,
            "target": self.target or self.label_spec.column,
        })
        return DataHandoff(**payload).model_dump(mode="json")
