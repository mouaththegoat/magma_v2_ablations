"""Shared schemas for the MAGMA v2 pipeline."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class ModalitySpec(BaseModel):
    name: str
    paths: list[str] = Field(default_factory=list)
    role: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TargetSpec(BaseModel):
    type: str = "unknown"
    name: str | None = None
    source: str | None = None
    positive_label: str | int | None = None


class SplitSpec(BaseModel):
    strategy: str = "unknown"
    train: str | None = None
    val: str | None = None
    test: str | None = None
    column: str | None = None
    group_column: str | None = None
    notes: list[str] = Field(default_factory=list)


class ClinicalTaskSpec(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid4()))
    raw_prompt: str
    objective: str | None = None
    task_type: str = "binary_classification"
    modalities: list[ModalitySpec] = Field(default_factory=list)
    target: TargetSpec = Field(default_factory=TargetSpec)
    split: SplitSpec = Field(default_factory=SplitSpec)
    join_keys: list[str] = Field(default_factory=list)
    constraints: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class StageFile(BaseModel):
    path: str
    purpose: str
    stage: str
    consumed_by_next_step: bool = False


class DataHandoff(BaseModel):
    modality: str = "multimodal"
    modalities: list[ModalitySpec] = Field(default_factory=list)
    task_type: str = "binary_classification"
    data_layout: str = "unknown"
    split_strategy: str = "unknown"
    train_data: str | None = None
    val_data: str | None = None
    test_data: str | None = None
    model_ready_data: str | None = None
    target: TargetSpec = Field(default_factory=TargetSpec)
    patient_id_column: str | None = None
    join_keys: list[str] = Field(default_factory=list)
    leakage_controls: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    readme_path: str | None = None
    produced_files: list[StageFile] = Field(default_factory=list)


class ModelHandoff(BaseModel):
    model_type: str = "unknown"
    modality_strategy: str = "unknown"
    model_path: str | None = None
    train_data: str | None = None
    val_data: str | None = None
    test_data: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    training_summary: str | None = None
    readme_path: str | None = None
    produced_files: list[StageFile] = Field(default_factory=list)


class JudgeVerdict(BaseModel):
    status: Literal["passed", "failed", "provisional"] = "provisional"
    auroc: float | None = None
    tripod_score: float
    reward_lambda: float = 0.5
    reward: float | None = None
    critical_failures: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    tripod_items: dict[str, bool] = Field(default_factory=dict)
    feedback: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

