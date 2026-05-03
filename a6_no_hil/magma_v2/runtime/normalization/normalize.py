"""Canonical normalization engine for MAGMA v2."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from magma_v2.runtime.artifacts import write_json, write_stage_readme
from magma_v2.runtime.normalization.contracts import (
    ConfidenceSummary,
    LabelSpec,
    NormalizedContract,
    PathResolutionSummary,
    ProducedFile,
    SchemaSummary,
    ValidatorReport,
)
from magma_v2.runtime.normalization.inspect import (
    ECG_SUFFIXES,
    IMAGE_SUFFIXES,
    PATH_SUFFIXES,
    SPLIT_NAMES,
    TABLE_SUFFIXES,
    infer_table_columns,
    inspect_path,
    read_table,
    resolve_path,
)
from magma_v2.runtime.normalization.validators import (
    validate_helper_exclusions,
    validate_join_keys,
    validate_label_source,
    validate_path_resolution,
    validate_produced_files,
    validate_split_integrity,
)


class NormalizationError(ValueError):
    """Structured normalization failure used instead of prose parsing."""

    def __init__(
        self,
        message: str,
        *,
        error_class: str,
        error_subclass: str,
        owner_stage: str = "data",
        recoverability: str = "retryable_with_modified_plan",
        retry_scope: str = "data_repair",
        required_user_input: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(message)
        self.error_class = error_class
        self.error_subclass = error_subclass
        self.owner_stage = owner_stage
        self.recoverability = recoverability
        self.retry_scope = retry_scope
        self.required_user_input = required_user_input or []


def normalize_inputs(
    raw_paths: list[str],
    data_dir: str,
    user_request: str = "",
    target: str | None = None,
    patient_id_column: str | None = None,
    join_keys: list[str] | None = None,
) -> NormalizedContract:
    """Normalize raw inputs into a canonical representation under data_dir."""
    data_path = Path(data_dir).resolve()
    data_path.mkdir(parents=True, exist_ok=True)
    try:
        return _normalize_inputs(raw_paths, data_path, user_request, target, patient_id_column, join_keys or [])
    except NormalizationError as exc:
        return NormalizedContract(
            status="error",
            observed_layout="unknown",
            error=str(exc),
            error_class=exc.error_class,
            error_subclass=exc.error_subclass,
            owner_stage=exc.owner_stage,
            recoverability=exc.recoverability,
            retry_scope=exc.retry_scope,
            required_user_input=exc.required_user_input,
            unresolved_questions=[
                {
                    "blocking": exc.recoverability == "human_blocking",
                    "error_class": exc.error_class,
                    "error_subclass": exc.error_subclass,
                    "required_user_input": exc.required_user_input,
                }
            ],
        )
    except Exception as exc:
        return NormalizedContract(
            status="error",
            observed_layout="unknown",
            error=str(exc),
            error_class="unexpected_internal_error",
            error_subclass="unknown",
            owner_stage="data",
            recoverability="retryable_with_modified_plan",
            retry_scope="data_repair",
        )


def _normalize_inputs(
    raw_paths: list[str],
    data_dir: Path,
    user_request: str,
    target: str | None,
    patient_id_column: str | None,
    join_keys: list[str],
) -> NormalizedContract:
    resolved = [resolve_path(path) for path in raw_paths]
    if not resolved:
        raise ValueError("No existing raw input paths were provided for normalization")
    inspections = [inspect_path(path) for path in resolved]
    table_paths = [path for path in resolved if path.is_file() and path.suffix.lower() in TABLE_SUFFIXES]
    dir_paths = [path for path in resolved if path.is_dir()]
    observed_layout = _observed_layout(table_paths, dir_paths)

    if _are_preexisting_split_tables(table_paths):
        frame, split_source = _combine_preexisting_splits(table_paths)
        source_path = table_paths[0]
        return _normalize_table(
            frame,
            source_path,
            data_dir,
            user_request,
            target,
            patient_id_column,
            observed_layout,
            split_source,
            reference_roots=dir_paths,
        )

    if len(table_paths) >= 2:
        multimodal = _try_multimodal_join(table_paths, data_dir, user_request, target, patient_id_column, join_keys, observed_layout)
        if multimodal is not None:
            return multimodal

    if table_paths:
        source_path = table_paths[0]
        frame = read_table(source_path)
        split_source = "preexisting" if "split" in {str(column).lower() for column in frame.columns} else "generated"
        return _normalize_table(
            frame,
            source_path,
            data_dir,
            user_request,
            target,
            patient_id_column,
            observed_layout,
            split_source,
            reference_roots=dir_paths,
        )

    if dir_paths:
        return _normalize_directory(dir_paths[0], data_dir, user_request, target, patient_id_column, observed_layout)

    raise ValueError(f"No supported files/directories found: {[str(path) for path in resolved]}")


def _observed_layout(table_paths: list[Path], dir_paths: list[Path]) -> str:
    if len(table_paths) > 1:
        return "split_tables" if _are_preexisting_split_tables(table_paths) else "multi_input"
    if table_paths and dir_paths:
        return "multi_input"
    if table_paths:
        return "single_table"
    if dir_paths:
        return "directory_collection"
    return "unknown"


def _are_preexisting_split_tables(paths: list[Path]) -> bool:
    names = {path.stem.lower() for path in paths}
    return bool({"train", "test"} <= names and ({"val", "valid", "validation"} & names))


def _combine_preexisting_splits(paths: list[Path]) -> tuple[pd.DataFrame, str]:
    frames = []
    for path in paths:
        frame = read_table(path)
        name = path.stem.lower()
        split = "val" if name in {"valid", "validation"} else name
        if split in {"train", "val", "test"}:
            frame = frame.copy()
            frame["split"] = split
            frames.append(frame)
    if not frames:
        raise ValueError("Could not identify train/val/test split tables")
    return pd.concat(frames, ignore_index=True), "preexisting"


def _normalize_table(
    frame: pd.DataFrame,
    source_path: Path,
    data_dir: Path,
    user_request: str,
    target: str | None,
    patient_id_column: str | None,
    observed_layout: str,
    split_source: str,
    reference_roots: list[Path] | None = None,
) -> NormalizedContract:
    special_contract = _try_normalize_ecg_metadata_table(
        frame=frame,
        source_path=source_path,
        data_dir=data_dir,
        user_request=user_request,
        target=target,
        patient_id_column=patient_id_column,
        reference_roots=reference_roots or [],
        observed_layout=observed_layout,
    )
    if special_contract is not None:
        return special_contract

    evidence = infer_table_columns(frame, user_request)
    label = target or _choose_candidate(evidence["candidate_targets"], "target")
    group_candidates = [
        candidate for candidate in evidence["candidate_group_ids"]
        if candidate.get("column") != label
    ]
    patient_id = patient_id_column or _choose_candidate(group_candidates, "patient/group id", required=False)
    if not label:
        raise NormalizationError(
            "Could not infer label source. Provide target explicitly.",
            error_class="ambiguity_error",
            error_subclass="missing_target",
            owner_stage="human",
            recoverability="human_blocking",
            retry_scope="human_input",
            required_user_input=[
                {
                    "name": "target",
                    "description": "Specify the outcome/label column or label source.",
                    "prompt": "What target or label source should MAGMA predict?",
                }
            ],
        )
    if label not in frame.columns:
        raise NormalizationError(
            f"Label column not found: {label}",
            error_class="contract_error",
            error_subclass="missing_target",
            owner_stage="human" if target else "data",
            recoverability="human_blocking" if target else "retryable_with_modified_plan",
            retry_scope="human_input" if target else "data_repair",
            required_user_input=[
                {
                    "name": "target",
                    "description": "Specify an existing outcome/label column.",
                    "prompt": f"Target {label!r} was requested but was not found. Which column should be predicted?",
                }
            ] if target else [],
        )

    path_columns = evidence["candidate_path_columns"]
    text_columns = evidence["candidate_text_columns"]
    normalized_layout, modalities = _choose_normalized_layout(path_columns, text_columns)
    semantic_observed_layout = _semantic_observed_layout(observed_layout, normalized_layout, path_columns, text_columns)
    label_values, derived_label = _derive_label_values(frame[label], user_request)
    if normalized_layout == "tabular_dataset":
        canonical = _tabular_frame(frame, label, patient_id, label_values=label_values)
    else:
        canonical = _manifest_frame(
            frame,
            label,
            patient_id,
            path_columns,
            text_columns,
            source_path,
            label_values=label_values,
            reference_roots=reference_roots,
        )
    canonical_patient_id = "patient_id" if patient_id and "patient_id" in canonical.columns else None
    canonical, split_source = _apply_external_split_lists(canonical, source_path, split_source)
    canonical = _ensure_split(canonical, label="label", patient_id_column=canonical_patient_id, split_source=split_source)

    files = _write_normalized_outputs(canonical, data_dir, normalized_layout)
    validators, path_summary = _run_validators(canonical, data_dir, files, label="label", patient_id_column=canonical_patient_id, path_columns=_manifest_path_columns(normalized_layout))
    schema = _schema_summary([source_path], {str(source_path): frame}, evidence)
    task_type = "binary_classification" if derived_label else _infer_task_type(canonical["label"])
    label_spec = _build_label_spec(
        source_label=label,
        canonical_target="label",
        source_path=source_path,
        task_type=task_type,
        label_values=canonical["label"],
        evidence=evidence["candidate_targets"],
        derived_label=derived_label,
    )
    produced = [ProducedFile(path=path, role=role, purpose=purpose) for path, role, purpose in files]
    contract = NormalizedContract(
        status="success",
        observed_layout=semantic_observed_layout,
        normalized_layout=normalized_layout,
        task_type=task_type,
        modalities=modalities,
        prediction_unit=_prediction_unit(normalized_layout, patient_id),
        label_spec=label_spec,
        split_strategy=_split_strategy_summary(split_source, patient_id),
        split_source=split_source,
        join_keys=[canonical_patient_id] if canonical_patient_id else [],
        patient_id_column=canonical_patient_id,
        feature_exclusion_columns=["label", "sample_id"] + ([canonical_patient_id] if canonical_patient_id else []),
        leakage_controls=["label excluded from features", "sample_id excluded from features", "patient/group leakage checked when patient_id exists"],
        assumptions=[],
        limitations=_normalization_limitations(normalized_layout, split_source),
        path_resolution=PathResolutionSummary(**path_summary),
        schema_summary=schema,
        confidence_summary=_confidence_summary(semantic_observed_layout, label_spec, patient_id, split_source),
        normalization_artifacts=produced,
        produced_files=produced,
        validator_reports=validators,
        train_data="train.csv",
        val_data="val.csv",
        test_data="test.csv",
        model_ready_data="manifest.csv" if normalized_layout != "tabular_dataset" else "train.csv",
        target="label",
        modality=modalities[0] if len(modalities) == 1 else "multimodal",
        data_layout=normalized_layout,
    )
    _write_contract_artifacts(data_dir, contract)
    _raise_if_failed(contract)
    return contract


def _try_normalize_ecg_metadata_table(
    *,
    frame: pd.DataFrame,
    source_path: Path,
    data_dir: Path,
    user_request: str,
    target: str | None,
    patient_id_column: str | None,
    reference_roots: list[Path],
    observed_layout: str,
) -> NormalizedContract | None:
    columns = {str(column) for column in frame.columns}
    if "scp_codes" not in columns:
        return None
    waveform_columns = [column for column in ("filename_hr", "filename_lr") if column in columns]
    if not waveform_columns:
        return None

    requested_target = str(target or "").strip()
    if requested_target and requested_target in frame.columns:
        return None

    parsed_codes = frame["scp_codes"].map(_parse_scp_codes)
    observed_codes = sorted({code for codes in parsed_codes for code in codes})
    positive_codes = _positive_codes_from_request(user_request, observed_codes)
    if requested_target and not positive_codes:
        return None

    root_candidates = [source_path.parent, *reference_roots]
    working = frame.copy()
    working["label"] = parsed_codes.map(lambda codes: int(bool(set(codes) & set(positive_codes)))) if positive_codes else 0
    working["ecg_path"] = working.apply(
        lambda row: _resolve_ecg_header_path(row, root_candidates, waveform_columns),
        axis=1,
    )
    working = working[working["ecg_path"].notna()].copy()
    if working.empty:
        raise NormalizationError(
            "No ECG waveform headers could be resolved from the metadata table.",
            error_class="path_resolution_error",
            error_subclass="unresolved_relative_paths",
            owner_stage="data",
            recoverability="retryable_with_modified_plan",
            retry_scope="data_repair",
        )

    if "strat_fold" in working.columns:
        working["split"] = working["strat_fold"].map(_ptbxl_strat_fold_to_split)
        working = working[working["split"].isin({"train", "val", "test"})].copy()
        split_source = "preexisting"
    else:
        split_source = "generated"

    canonical = pd.DataFrame({
        "sample_id": working["ecg_id"].astype(str) if "ecg_id" in working.columns else [f"sample_{i}" for i in range(len(working))],
        "label": working["label"].astype(int),
        "ecg_path": working["ecg_path"].astype(str),
    })
    canonical_patient_id = None
    patient_column = patient_id_column if patient_id_column in working.columns else ("patient_id" if "patient_id" in working.columns else None)
    if patient_column:
        canonical["patient_id"] = working[patient_column]
        canonical_patient_id = "patient_id"
    if "split" in working.columns:
        canonical["split"] = working["split"].astype(str)
    if "scp_codes" in working.columns:
        canonical["scp_codes"] = working["scp_codes"]
    if "strat_fold" in working.columns:
        canonical["strat_fold"] = working["strat_fold"]
    for column in waveform_columns:
        canonical[column] = working[column]

    canonical = _ensure_split(canonical, label="label", patient_id_column=canonical_patient_id, split_source=split_source)
    files = _write_normalized_outputs(canonical, data_dir, "ecg_manifest")
    validators, path_summary = _run_validators(
        canonical,
        data_dir,
        files,
        label="label",
        patient_id_column=canonical_patient_id,
        path_columns=["ecg_path"],
    )

    evidence = infer_table_columns(frame, user_request)
    label_spec = LabelSpec(
        source_type="derived_from_column",
        source_name="scp_codes",
        target_name="label",
        task_type="binary_classification",
        class_labels=[0, 1],
        positive_class=1,
        confidence=0.9 if positive_codes else 0.5,
        notes=[
            f"Binary ECG outcome derived from scp_codes using positive codes: {sorted(positive_codes)}."
            if positive_codes else
            "Binary ECG outcome derived from scp_codes."
        ],
        column="label",
        source=str(source_path),
        evidence=evidence.get("candidate_targets", []),
    )
    produced = [ProducedFile(path=path, role=role, purpose=purpose) for path, role, purpose in files]
    contract = NormalizedContract(
        status="success",
        observed_layout="ecg_manifest_table",
        normalized_layout="ecg_manifest",
        task_type="binary_classification",
        modalities=["ecg"],
        prediction_unit="ecg_recording",
        label_spec=label_spec,
        split_strategy="preexisting stratified fold mapping" if split_source == "preexisting" else _split_strategy_summary(split_source, canonical_patient_id),
        split_source=split_source,
        join_keys=[canonical_patient_id] if canonical_patient_id else [],
        patient_id_column=canonical_patient_id,
        feature_exclusion_columns=["label", "sample_id", "scp_codes", "strat_fold", *waveform_columns] + ([canonical_patient_id] if canonical_patient_id else []),
        leakage_controls=["label excluded from features", "sample_id excluded from features", "patient/group leakage checked when patient_id exists"],
        assumptions=[f"Positive ECG labels derived from scp_codes using {sorted(positive_codes)}."] if positive_codes else [],
        limitations=["ECG manifest points to WFDB header files under the original dataset root."],
        path_resolution=PathResolutionSummary(**path_summary),
        schema_summary=_schema_summary([source_path], {str(source_path): frame}, evidence),
        confidence_summary=_confidence_summary("ecg_manifest_table", label_spec, canonical_patient_id, split_source),
        normalization_artifacts=produced,
        produced_files=produced,
        validator_reports=validators,
        train_data="train.csv",
        val_data="val.csv",
        test_data="test.csv",
        model_ready_data="manifest.csv",
        target="label",
        modality="ecg",
        data_layout="ecg_manifest",
    )
    _write_contract_artifacts(data_dir, contract)
    _raise_if_failed(contract)
    return contract


def _semantic_observed_layout(
    observed_layout: str,
    normalized_layout: str,
    path_columns: list[dict[str, Any]],
    text_columns: list[dict[str, Any]],
) -> str:
    if observed_layout == "directory_collection":
        return "image_folder_plus_labels_file" if normalized_layout == "image_manifest" else observed_layout
    if observed_layout in {"multi_input", "split_tables"} and normalized_layout == "multimodal_alignment_manifest":
        return "multimodal_joined_inputs"
    modalities = {item.get("modality") for item in path_columns}
    if "image" in modalities:
        return "csv_with_image_paths"
    if "ecg" in modalities:
        return "ecg_manifest_table"
    if text_columns:
        return "text_table"
    return "tabular_table"


def _prediction_unit(normalized_layout: str, patient_id: str | None) -> str:
    if normalized_layout == "image_manifest":
        return "image"
    if normalized_layout == "text_manifest":
        return "note"
    if normalized_layout == "ecg_manifest":
        return "ecg_recording"
    if normalized_layout == "multimodal_alignment_manifest":
        return "patient" if patient_id else "sample"
    return "patient" if patient_id else "sample"


def _jsonable_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _build_label_spec(
    source_label: str,
    canonical_target: str,
    source_path: Path,
    task_type: str,
    label_values: pd.Series,
    evidence: list[dict[str, Any]],
    derived_label: dict[str, Any] | None = None,
) -> LabelSpec:
    observed = [_jsonable_value(value) for value in sorted(label_values.dropna().unique().tolist(), key=lambda item: str(item))]
    class_labels = observed if "classification" in task_type else []
    positive_class = None
    if task_type == "binary_classification" and class_labels:
        positive_class = 1 if 1 in class_labels else class_labels[-1]
    confidence = evidence[0].get("score") if evidence else None
    notes = [f"Raw label column {source_label!r} was normalized to canonical target {canonical_target!r}."]
    if derived_label:
        positive_class = derived_label.get("positive_value", positive_class)
        confidence = max(confidence or 0.0, float(derived_label.get("confidence") or 0.0))
        notes.append(str(derived_label.get("note")))
    return LabelSpec(
        source_type="column",
        source_name=source_label,
        target_name=canonical_target,
        task_type=task_type,
        class_labels=class_labels,
        positive_class=positive_class,
        confidence=confidence,
        notes=notes,
        column=canonical_target,
        source=str(source_path),
        evidence=evidence,
    )


def _derive_label_values(label_values: pd.Series, user_request: str) -> tuple[pd.Series, dict[str, Any] | None]:
    """Derive a requested binary label from observed multi-label tokens."""
    lower_request = user_request.lower()
    if not (pd.api.types.is_object_dtype(label_values) or pd.api.types.is_string_dtype(label_values)):
        return label_values, None

    tokens: set[str] = set()
    for value in label_values.dropna().astype(str):
        for token in _split_label_tokens(value):
            cleaned = token.strip()
            if cleaned:
                tokens.add(cleaned)

    requested_tokens = [token for token in sorted(tokens, key=len, reverse=True) if token.lower() in lower_request]
    if len(requested_tokens) != 1:
        return label_values, None

    positive_token = requested_tokens[0]
    derived = label_values.fillna("").astype(str).map(
        lambda value: int(positive_token.lower() in {part.strip().lower() for part in _split_label_tokens(value)})
    )
    return derived, {
        "positive_token": positive_token,
        "positive_value": 1,
        "confidence": 0.9,
        "note": f"Derived binary target: {positive_token!r} present maps to 1, absent maps to 0.",
    }


def _split_label_tokens(value: str) -> list[str]:
    tokens = [value]
    for delimiter in ("|", ",", ";", "/"):
        tokens = [piece for token in tokens for piece in token.split(delimiter)]
    return tokens


def _confidence_summary(
    observed_layout: str,
    label_spec: LabelSpec,
    patient_id: str | None,
    split_source: str,
) -> ConfidenceSummary:
    return ConfidenceSummary(
        layout_confidence=0.9 if observed_layout not in {"unknown", "single_table"} else 0.6,
        label_confidence=label_spec.confidence,
        join_key_confidence=0.9 if patient_id else None,
        split_confidence=0.95 if split_source == "preexisting" else 0.85 if split_source == "generated" else None,
        notes=["Confidence values summarize deterministic evidence strength, not model performance."],
    )


def _choose_candidate(candidates: list[dict[str, Any]], kind: str, required: bool = True) -> str | None:
    if not candidates:
        if required:
            raise NormalizationError(
                f"Could not infer {kind}",
                error_class="ambiguity_error",
                error_subclass="missing_target" if kind == "target" else "ambiguous_user_request",
                owner_stage="human",
                recoverability="human_blocking",
                retry_scope="human_input",
                required_user_input=[
                    {
                        "name": "target" if kind == "target" else kind,
                        "description": f"Specify the {kind}.",
                        "prompt": f"What {kind} should MAGMA use?",
                    }
                ],
            )
        return None
    top = candidates[0]
    if top.get("score", 0) < 0.5 and required:
        raise NormalizationError(
            f"Ambiguous {kind}; top candidate has weak evidence.",
            error_class="ambiguity_error",
            error_subclass="missing_target" if kind == "target" else "ambiguous_user_request",
            owner_stage="human",
            recoverability="human_blocking",
            retry_scope="human_input",
            required_user_input=[
                {
                    "name": "target" if kind == "target" else kind,
                    "description": f"Specify the {kind}.",
                    "prompt": f"What {kind} should MAGMA use?",
                }
            ],
        )
    return top["column"]


def _choose_normalized_layout(path_columns: list[dict[str, Any]], text_columns: list[dict[str, Any]]) -> tuple[str, list[str]]:
    modalities = {item.get("modality") for item in path_columns}
    if "image" in modalities:
        return "image_manifest", ["image"]
    if "ecg" in modalities:
        return "ecg_manifest", ["ecg"]
    if text_columns:
        return "text_manifest", ["text"]
    return "tabular_dataset", ["tabular"]


def _tabular_frame(frame: pd.DataFrame, label: str, patient_id: str | None, label_values: pd.Series | None = None) -> pd.DataFrame:
    canonical = frame.copy()
    canonical = canonical.rename(columns={label: "label"})
    if label_values is not None:
        canonical["label"] = label_values.values
    if patient_id and patient_id in canonical.columns:
        canonical = canonical.rename(columns={patient_id: "patient_id"})
    canonical.insert(0, "sample_id", [f"sample_{i}" for i in range(len(canonical))])
    return canonical


def _manifest_frame(
    frame: pd.DataFrame,
    label: str,
    patient_id: str | None,
    path_columns: list[dict[str, Any]],
    text_columns: list[dict[str, Any]],
    source_path: Path,
    label_values: pd.Series | None = None,
    reference_roots: list[Path] | None = None,
) -> pd.DataFrame:
    canonical = pd.DataFrame({"sample_id": [f"sample_{i}" for i in range(len(frame))], "label": label_values if label_values is not None else frame[label]})
    if patient_id and patient_id in frame.columns:
        canonical["patient_id"] = frame[patient_id]
    reference_index = _build_reference_index(source_path.parent, reference_roots or [])
    for item in path_columns:
        column = item["column"]
        out_col = f"{item.get('modality') or 'file'}_path"
        canonical[out_col] = frame[column].apply(lambda value: _resolve_reference(value, source_path.parent, reference_index))
    for item in text_columns:
        canonical["text"] = frame[item["column"]]
        break
    return canonical


def _build_reference_index(base_dir: Path, reference_roots: list[Path]) -> dict[str, Path]:
    roots = []
    for root in [base_dir, *reference_roots]:
        resolved = root.resolve()
        if resolved.exists() and resolved.is_dir() and resolved not in roots:
            roots.append(resolved)
    index: dict[str, Path] = {}
    for root in roots:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in PATH_SUFFIXES:
                index.setdefault(path.name, path.resolve())
    return index


def _parse_scp_codes(value: Any) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    normalized = text.replace("np.str_(", "").replace(")", "")
    try:
        parsed = ast.literal_eval(normalized)
    except Exception:
        parsed = None
    if isinstance(parsed, dict):
        return [str(key) for key in parsed.keys()]
    if isinstance(parsed, (list, tuple, set)):
        return [str(item) for item in parsed]
    return _quoted_tokens(text)


def _quoted_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    current: list[str] = []
    quote: str | None = None
    for char in text:
        if quote is None:
            if char in {"'", '"'}:
                quote = char
                current = []
        elif char == quote:
            token = "".join(current).strip()
            if token:
                tokens.append(token)
            quote = None
            current = []
        else:
            current.append(char)
    return tokens


def _request_tokens(user_request: str) -> set[str]:
    tokens: set[str] = set()
    current: list[str] = []
    for char in user_request:
        if char.isalnum() or char in {"_", "-"}:
            current.append(char)
            continue
        if current:
            token = "".join(current).strip("-_")
            if token:
                tokens.add(token.upper())
            current = []
    if current:
        token = "".join(current).strip("-_")
        if token:
            tokens.add(token.upper())
    return tokens


def _positive_codes_from_request(user_request: str, observed_codes: list[str]) -> list[str]:
    request_tokens = _request_tokens(user_request)
    observed = {str(code).upper() for code in observed_codes}
    return sorted(token for token in request_tokens if token in observed)


def _resolve_ecg_header_path(row: pd.Series, root_candidates: list[Path], waveform_columns: list[str]) -> str | None:
    for column in waveform_columns:
        rel_value = row.get(column)
        if pd.isna(rel_value) or not str(rel_value).strip():
            continue
        rel_path = Path(str(rel_value))
        for root in root_candidates:
            candidate = (root / rel_path).resolve()
            header_path = candidate.with_suffix(".hea")
            data_path = candidate.with_suffix(".dat")
            if header_path.exists() and data_path.exists():
                return str(header_path)
    return None


def _ptbxl_strat_fold_to_split(value: Any) -> str | None:
    try:
        fold = int(value)
    except (TypeError, ValueError):
        return None
    if 1 <= fold <= 8:
        return "train"
    if fold == 9:
        return "val"
    if fold == 10:
        return "test"
    return None


def _resolve_reference(value: Any, base_dir: Path, reference_index: dict[str, Path] | None = None) -> str:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return str(path.resolve())
    base_candidate = (base_dir / path).resolve()
    if base_candidate.exists():
        return str(base_candidate)
    if reference_index and path.name in reference_index:
        return str(reference_index[path.name])
    return str(base_candidate)


def _apply_external_split_lists(frame: pd.DataFrame, source_path: Path, split_source: str) -> tuple[pd.DataFrame, str]:
    if "image_path" not in frame.columns:
        return frame, split_source

    row_names = set(frame["image_path"].astype(str).map(lambda value: Path(value).name))
    split_lists = _discover_split_lists(source_path.parent, row_names)
    if len(split_lists) < 2:
        return frame, split_source

    assigned = _assign_split_lists(split_lists)
    train_val_names = assigned.get("train_pool") or assigned.get("train")
    test_names = assigned.get("test")
    if not train_val_names or not test_names:
        return frame, split_source

    out = frame.copy()
    names = out["image_path"].astype(str).map(lambda value: Path(value).name)
    out["split"] = None
    out.loc[names.isin(train_val_names), "split"] = "train_pool"
    out.loc[names.isin(test_names), "split"] = "test"
    return out[out["split"].notna()].copy(), "partially_preserved"


def _discover_split_lists(base_dir: Path, row_names: set[str]) -> list[dict[str, Any]]:
    discovered = []
    for path in base_dir.glob("*.txt"):
        values = {line.strip() for line in path.read_text().splitlines() if line.strip()}
        overlap = values & row_names
        if overlap:
            discovered.append({"path": path, "values": overlap, "coverage": len(overlap)})
    return discovered


def _assign_split_lists(split_lists: list[dict[str, Any]]) -> dict[str, set[str]]:
    assigned: dict[str, set[str]] = {}
    unassigned = []
    for item in split_lists:
        name_parts = {part for part in item["path"].stem.lower().replace("-", "_").split("_") if part}
        matched = name_parts & SPLIT_NAMES
        if "test" in matched:
            assigned["test"] = item["values"]
        elif matched:
            assigned["train_pool"] = item["values"]
        else:
            unassigned.append(item)
    if "test" not in assigned and len(split_lists) == 2:
        ordered = sorted(split_lists, key=lambda item: item["coverage"])
        assigned["test"] = ordered[0]["values"]
        assigned["train_pool"] = ordered[1]["values"]
    if "train_pool" not in assigned and unassigned:
        assigned["train_pool"] = max(unassigned, key=lambda item: item["coverage"])["values"]
    return assigned


def _ensure_split(frame: pd.DataFrame, label: str, patient_id_column: str | None, split_source: str) -> pd.DataFrame:
    if "split" in frame.columns:
        frame = frame.copy()
        frame["split"] = frame["split"].astype(str).str.lower().replace({"valid": "val", "validation": "val"})
        if {"train", "val", "test"} <= set(frame["split"].dropna().unique()):
            return frame
        if {"train_pool", "test"} <= set(frame["split"].dropna().unique()):
            return _split_train_pool_preserving_test(frame, patient_id_column)
        frame = frame.drop(columns=["split"])
    if patient_id_column and patient_id_column in frame.columns:
        groups = frame[patient_id_column].astype(str)
        splitter = GroupShuffleSplit(n_splits=1, train_size=0.6, random_state=42)
        train_idx, holdout_idx = next(splitter.split(frame, groups=groups))
        out = frame.copy()
        out["split"] = "train"
        holdout = out.iloc[holdout_idx].copy()
        holdout_groups = holdout[patient_id_column].astype(str)
        splitter2 = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=43)
        val_local, test_local = next(splitter2.split(holdout, groups=holdout_groups))
        out.loc[holdout.index[val_local], "split"] = "val"
        out.loc[holdout.index[test_local], "split"] = "test"
        return out
    stratify = frame[label] if frame[label].nunique(dropna=True) > 1 and frame[label].nunique(dropna=True) <= 20 else None
    train, holdout = _safe_train_test_split(frame, train_size=0.6, random_state=42, stratify=stratify)
    holdout_stratify = holdout[label] if stratify is not None and holdout[label].nunique(dropna=True) > 1 else None
    val, test = _safe_train_test_split(holdout, train_size=0.5, random_state=43, stratify=holdout_stratify)
    out = frame.copy()
    out["split"] = "train"
    out.loc[val.index, "split"] = "val"
    out.loc[test.index, "split"] = "test"
    return out


def _split_train_pool_preserving_test(frame: pd.DataFrame, patient_id_column: str | None) -> pd.DataFrame:
    out = frame.copy()
    pool = out[out["split"] == "train_pool"].copy()
    if pool.empty:
        raise ValueError("External train/val split list produced an empty train pool")
    if patient_id_column and patient_id_column in pool.columns and pool[patient_id_column].nunique(dropna=True) > 1:
        groups = pool[patient_id_column].astype(str)
        splitter = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
        train_idx, val_idx = next(splitter.split(pool, groups=groups))
        out.loc[pool.iloc[train_idx].index, "split"] = "train"
        out.loc[pool.iloc[val_idx].index, "split"] = "val"
    else:
        stratify = pool["label"] if pool["label"].nunique(dropna=True) > 1 and pool["label"].nunique(dropna=True) <= 20 else None
        train, val = _safe_train_test_split(pool, train_size=0.8, random_state=42, stratify=stratify)
        out.loc[train.index, "split"] = "train"
        out.loc[val.index, "split"] = "val"
    return out


def _safe_train_test_split(
    frame: pd.DataFrame,
    *,
    train_size: float,
    random_state: int,
    stratify: pd.Series | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        return train_test_split(frame, train_size=train_size, random_state=random_state, stratify=stratify)
    except ValueError:
        return train_test_split(frame, train_size=train_size, random_state=random_state, stratify=None)


def _split_strategy_summary(split_source: str, patient_id: str | None) -> str:
    if split_source == "preexisting":
        return "preexisting split column"
    if split_source == "partially_preserved":
        return "preexisting test split with generated train/val split" + (" grouped by patient" if patient_id else "")
    return "group-aware generated split" if patient_id else "row-level generated split"


def _normalization_limitations(normalized_layout: str, split_source: str) -> list[str]:
    limitations = [] if normalized_layout == "tabular_dataset" else ["manifest paths point to original raw assets"]
    if split_source == "partially_preserved":
        limitations.append("External train/test list was preserved; validation split was generated from the train list.")
    return limitations


def _write_normalized_outputs(frame: pd.DataFrame, data_dir: Path, normalized_layout: str) -> list[tuple[str, str, str]]:
    files = []
    if normalized_layout == "tabular_dataset":
        for split in ("train", "val", "test"):
            path = data_dir / f"{split}.csv"
            frame[frame["split"] == split].to_csv(path, index=False)
            files.append((path.name, f"{split}_tabular", f"Canonical {split} split for tabular model input."))
    else:
        manifest = data_dir / "manifest.csv"
        frame.to_csv(manifest, index=False)
        files.append((manifest.name, normalized_layout, f"Canonical {normalized_layout}."))
        for split in ("train", "val", "test"):
            path = data_dir / f"{split}.csv"
            frame[frame["split"] == split].to_csv(path, index=False)
            files.append((path.name, f"{split}_manifest", f"Canonical {split} manifest split."))
    return files


def _manifest_path_columns(normalized_layout: str) -> list[str]:
    if normalized_layout == "image_manifest":
        return ["image_path"]
    if normalized_layout == "ecg_manifest":
        return ["ecg_path"]
    if normalized_layout == "text_manifest":
        return ["text_path"]
    if normalized_layout == "multimodal_alignment_manifest":
        return ["image_path", "ecg_path", "text_path"]
    return []


def _run_validators(
    frame: pd.DataFrame,
    data_dir: Path,
    files: list[tuple[str, str, str]],
    label: str,
    patient_id_column: str | None,
    path_columns: list[str],
) -> tuple[list[ValidatorReport], dict[str, Any]]:
    frames = {split: frame[frame["split"] == split].copy() for split in ("train", "val", "test")}
    path_report, path_summary = validate_path_resolution(frame, [column for column in path_columns if column in frame.columns])
    reports = [
        validate_produced_files(data_dir, [{"path": item[0]} for item in files]),
        validate_label_source(data_dir / files[0][0], label),
        validate_split_integrity(frames, split_column="split", group_column=patient_id_column),
        validate_join_keys(frame, [patient_id_column] if patient_id_column else []),
        validate_helper_exclusions(frame, label, ["label", "sample_id"] + (["patient_id"] if patient_id_column else [])),
        path_report,
    ]
    return reports, path_summary


def _schema_summary(paths: list[Path], frames: dict[str, pd.DataFrame], evidence: dict[str, Any]) -> SchemaSummary:
    all_columns = sorted({str(column) for frame in frames.values() for column in frame.columns})
    combined_rows = sum(len(frame) for frame in frames.values())
    numeric_columns = sorted({
        str(column)
        for frame in frames.values()
        for column in frame.select_dtypes(include="number").columns
    })
    categorical_columns = sorted(set(all_columns) - set(numeric_columns))
    return SchemaSummary(
        source_paths=[str(path) for path in paths],
        columns=all_columns,
        row_count=int(combined_rows),
        numeric_column_count=len(numeric_columns),
        categorical_column_count=len(categorical_columns),
        modality_evidence={
            "path_column_count": len(evidence.get("candidate_path_columns", [])),
            "text_column_count": len(evidence.get("candidate_text_columns", [])),
            "path_modalities": sorted({
                str(item.get("modality"))
                for item in evidence.get("candidate_path_columns", [])
                if item.get("modality")
            }),
        },
        columns_by_source={path: list(frame.columns) for path, frame in frames.items()},
        dtypes_by_source={path: {str(column): str(dtype) for column, dtype in frame.dtypes.items()} for path, frame in frames.items()},
        row_counts_by_source={path: int(len(frame)) for path, frame in frames.items()},
        candidate_targets=evidence.get("candidate_targets", []),
        candidate_group_ids=evidence.get("candidate_group_ids", []),
        candidate_path_columns=evidence.get("candidate_path_columns", []),
        candidate_text_columns=evidence.get("candidate_text_columns", []),
    )


def _infer_task_type(label: pd.Series) -> str:
    unique = label.dropna().nunique()
    if unique <= 2:
        return "binary_classification"
    if unique <= 20:
        return "multiclass_classification"
    if pd.api.types.is_numeric_dtype(label):
        return "regression"
    return "classification"


def _write_contract_artifacts(data_dir: Path, contract: NormalizedContract) -> None:
    for report in contract.validator_reports:
        report.path = _validator_report_filename(report.name)
        write_json(str(data_dir / report.path), report.model_dump(mode="json"))
        _append_file(
            contract.normalization_artifacts,
            ProducedFile(path=report.path, role="validator_report", purpose=f"{report.name} validation report."),
        )
    write_json(str(data_dir / "validator_reports.json"), [report.model_dump(mode="json") for report in contract.validator_reports])
    _append_file(
        contract.normalization_artifacts,
        ProducedFile(path="validator_reports.json", role="validator_reports_index", purpose="Index of normalization validator reports."),
    )
    _append_file(
        contract.normalization_artifacts,
        ProducedFile(path="normalization_contract.json", role="normalization_contract", purpose="Machine-readable normalization contract."),
    )
    for item in contract.normalization_artifacts:
        _append_file(contract.produced_files, item)
    write_json(str(data_dir / "normalization_contract.json"), contract.model_dump(mode="json"))
    descriptions = {item.path: item.purpose for item in contract.produced_files}
    descriptions["normalization_contract.json"] = "Machine-readable normalization contract."
    descriptions["validator_reports.json"] = "Machine-readable validation reports for normalized outputs."
    write_stage_readme(str(data_dir), "Data Directory", descriptions)


def _validator_report_filename(name: str) -> str:
    mapping = {
        "path_resolution": "path_validation_report.json",
        "split_integrity": "split_validation_report.json",
        "helper_label_exclusions": "leakage_report.json",
        "join_key_integrity": "join_validation_report.json",
        "label_source": "label_validation_report.json",
        "produced_files": "produced_files_validation_report.json",
    }
    return mapping.get(name, f"{name}_validation_report.json")


def _append_file(files: list[ProducedFile], item: ProducedFile) -> None:
    if not any(existing.path == item.path for existing in files):
        files.append(item)


def _raise_if_failed(contract: NormalizedContract) -> None:
    """Record validator failures as evidence, not runtime blockers.

    MAGMA v2 intentionally keeps normalization advisory. Validators should help
    the LLM/model stages understand risks, but a failed path/split/report check
    must not prevent the worker from writing a communicable handoff. The only
    hard orchestration contract is that a stage writes either HANDOFF or ERROR.
    """
    failures = []
    for report in contract.validator_reports:
        failures.extend(report.critical_failures)
    if failures:
        contract.limitations.append("Non-blocking normalization validator failures: " + "; ".join(failures))


def _normalize_directory(
    directory: Path,
    data_dir: Path,
    user_request: str,
    target: str | None,
    patient_id_column: str | None,
    observed_layout: str,
) -> NormalizedContract:
    files = [path for path in directory.rglob("*") if path.is_file() and path.name[:1] != "."]
    image_files = [path for path in files if path.suffix.lower() in IMAGE_SUFFIXES]
    ecg_files = [path for path in files if path.suffix.lower() in ECG_SUFFIXES]
    if image_files:
        frame = _frame_from_file_collection(image_files, modality="image")
        return _normalize_table(frame, directory, data_dir, user_request, target or "label", patient_id_column, observed_layout, "preexisting" if "split" in frame.columns else "generated")
    if ecg_files:
        frame = _frame_from_file_collection(ecg_files, modality="ecg")
        return _normalize_table(frame, directory, data_dir, user_request, target or "label", patient_id_column, observed_layout, "preexisting" if "split" in frame.columns else "generated")
    raise ValueError(f"Directory does not contain supported image or ECG files: {directory}")


def _frame_from_file_collection(files: list[Path], modality: str) -> pd.DataFrame:
    rows = []
    for i, path in enumerate(files):
        split = path.parent.name.lower()
        rows.append({
            "sample_id": path.stem,
            f"{modality}_path": str(path.resolve()),
            "label": path.parent.parent.name if split in {"train", "val", "valid", "validation", "test"} else path.parent.name,
            "split": "val" if split in {"valid", "validation"} else split if split in {"train", "val", "test"} else None,
        })
    return pd.DataFrame(rows)


def _try_multimodal_join(
    table_paths: list[Path],
    data_dir: Path,
    user_request: str,
    target: str | None,
    patient_id_column: str | None,
    join_keys: list[str],
    observed_layout: str,
) -> NormalizedContract | None:
    frames = {str(path): read_table(path) for path in table_paths}
    evidences = {path: infer_table_columns(frame, user_request) for path, frame in frames.items()}
    manifest_sources = [
        (path, frame, evidence)
        for path, frame in frames.items()
        for evidence in [evidences[path]]
        if evidence["candidate_path_columns"]
    ]
    if not manifest_sources:
        return None
    base_path, base_frame, base_evidence = manifest_sources[0]
    label = target or _choose_candidate(base_evidence["candidate_targets"], "target")
    candidate_join_keys = join_keys or _shared_columns(list(frames.values()))
    if patient_id_column and patient_id_column not in candidate_join_keys:
        candidate_join_keys.insert(0, patient_id_column)
    if not candidate_join_keys:
        raise ValueError("Multiple inputs require explicit or observed join keys")
    join_key = candidate_join_keys[0]
    merged = None
    for path, frame in frames.items():
        if join_key not in frame.columns:
            raise ValueError(f"Join key {join_key} not present in {path}")
        prefix = Path(path).stem
        renamed = frame.rename(columns={column: f"{prefix}__{column}" for column in frame.columns if column != join_key})
        merged = renamed if merged is None else merged.merge(renamed, on=join_key, how="outer")
    if merged is None:
        return None
    label_column = None
    for path in frames:
        candidate = f"{Path(path).stem}__{label}"
        if candidate in merged.columns:
            label_column = candidate
            break
    if not label_column:
        raise ValueError(f"Label {label} not found after multimodal join")
    path_columns = []
    for path, evidence in evidences.items():
        prefix = Path(path).stem
        for item in evidence["candidate_path_columns"]:
            path_columns.append({"column": f"{prefix}__{item['column']}", "modality": item.get("modality")})
    text_columns = []
    for path, evidence in evidences.items():
        prefix = Path(path).stem
        for item in evidence["candidate_text_columns"]:
            text_columns.append({"column": f"{prefix}__{item['column']}"})
    contract = _normalize_table(merged, Path(base_path), data_dir, user_request, label_column, join_key, observed_layout, "generated")
    contract.observed_layout = "multimodal_joined_inputs"
    contract.normalized_layout = "multimodal_alignment_manifest"
    contract.prediction_unit = "patient" if join_key else "sample"
    contract.modalities = sorted({item.get("modality") for item in path_columns if item.get("modality")} | ({"tabular"} if len(table_paths) > 1 else set()))
    contract.modality = "multimodal"
    contract.data_layout = "multimodal_alignment_manifest"
    contract.join_keys = [join_key]
    contract.limitations.append("Multimodal alignment uses an outer join and documents missing modality coverage.")
    _write_contract_artifacts(data_dir, contract)
    return contract


def _shared_columns(frames: list[pd.DataFrame]) -> list[str]:
    if not frames:
        return []
    shared = set(frames[0].columns)
    for frame in frames[1:]:
        shared &= set(frame.columns)
    scored = []
    for column in shared:
        ratios = []
        coverage = []
        for frame in frames:
            series = frame[column]
            non_null = max(int(series.notna().sum()), 1)
            ratios.append(series.nunique(dropna=True) / non_null)
            coverage.append(non_null / max(len(series), 1))
        scored.append((sum(coverage) / len(coverage), sum(ratios) / len(ratios), str(column), column))
    scored.sort(reverse=True)
    return [item[-1] for item in scored]
