"""Validators for normalized MAGMA v2 contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from magma_v2.runtime.normalization.contracts import ValidatorReport
from magma_v2.runtime.normalization.inspect import read_table, resolve_path


def resolve_data_path(data_dir: str | Path, value: str | None) -> Path | None:
    if not value:
        return None
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (Path(data_dir).resolve() / candidate).resolve()


def validate_produced_files(data_dir: str | Path, produced_files: list[dict[str, Any]]) -> ValidatorReport:
    base = Path(data_dir).resolve()
    failures = []
    for item in produced_files:
        path = resolve_data_path(base, item.get("path"))
        if path is None:
            failures.append("produced file missing path")
            continue
        if not path.exists():
            failures.append(f"produced file does not exist: {path}")
        if base not in [path, *path.parents]:
            failures.append(f"produced file is not under data/: {path}")
    return ValidatorReport(name="produced_files", status="failed" if failures else "passed", critical_failures=failures)


def validate_label_source(path: str | Path, label: str | None) -> ValidatorReport:
    failures = []
    if not label:
        failures.append("label source is missing")
    else:
        frame = read_table(path, n_rows=1000)
        if label not in frame.columns:
            failures.append(f"label column not found: {label}")
        elif frame[label].dropna().empty:
            failures.append(f"label column has no observed labels: {label}")
    return ValidatorReport(name="label_source", status="failed" if failures else "passed", critical_failures=failures)


def validate_path_resolution(frame: pd.DataFrame, path_columns: list[str]) -> tuple[ValidatorReport, dict[str, Any]]:
    failures = []
    total = 0
    resolved = 0
    missing_examples = []
    base_dirs: set[str] = set()
    file_types: dict[str, int] = {}
    for column in path_columns:
        if column not in frame.columns:
            failures.append(f"path column not found: {column}")
            continue
        for _, value in frame[column].dropna().items():
            total += 1
            path = Path(str(value)).expanduser()
            if path.exists():
                resolved += 1
                base_dirs.add(str(path.resolve().parent))
                suffix = path.suffix.lower() or "<none>"
                file_types[suffix] = file_types.get(suffix, 0) + 1
            else:
                missing_examples.append({"column": column, "value": str(value)})
                if len(missing_examples) > 20:
                    break
    if missing_examples:
        failures.append(f"{len(missing_examples)} unresolved path examples")
    report = ValidatorReport(
        name="path_resolution",
        status="failed" if failures else "passed",
        critical_failures=failures,
        evidence={"total": total, "resolved": resolved, "missing_examples": missing_examples},
    )
    missing = max(total - resolved, len(missing_examples))
    return report, {
        "path_columns": path_columns,
        "base_dirs": sorted(base_dirs),
        "total_paths_checked": total,
        "existing_paths": resolved,
        "missing_paths": missing,
        "exists_rate": resolved / total if total else None,
        "file_types_detected": file_types,
        "total_references": total,
        "resolved_references": resolved,
        "missing_references": missing,
        "missing_examples": missing_examples,
    }


def validate_split_integrity(frames: dict[str, pd.DataFrame], split_column: str, group_column: str | None) -> ValidatorReport:
    failures = []
    warnings = []
    for name, frame in frames.items():
        if frame.empty:
            failures.append(f"{name} split is empty")
        if split_column not in frame.columns:
            failures.append(f"{name} split missing split column {split_column}")
    if group_column and all(group_column in frame.columns for frame in frames.values()):
        groups = {name: set(frame[group_column].dropna().astype(str)) for name, frame in frames.items()}
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = groups[left] & groups[right]
            if overlap:
                failures.append(f"group leakage between {left} and {right}: {len(overlap)} overlapping groups")
    elif group_column:
        warnings.append(f"group column {group_column} not found in every split")
    return ValidatorReport(name="split_integrity", status="failed" if failures else "passed", critical_failures=failures, warnings=warnings)


def validate_join_keys(frame: pd.DataFrame, join_keys: list[str]) -> ValidatorReport:
    failures = [f"join key not found: {key}" for key in join_keys if key not in frame.columns]
    return ValidatorReport(name="join_key_integrity", status="failed" if failures else "passed", critical_failures=failures)


def validate_helper_exclusions(frame: pd.DataFrame, label: str | None, exclusions: list[str]) -> ValidatorReport:
    failures = []
    warnings = []
    exclusion_set = set(exclusions)
    if label and label in frame.columns and label not in exclusion_set:
        warnings.append(f"label column {label} should be excluded from model features")
    if label and label in frame.columns:
        target = frame[label]
        for column in frame.columns:
            if column == label:
                continue
            if frame[column].equals(target):
                if column not in exclusion_set:
                    failures.append(f"helper/duplicate label column not excluded: {column}")
    return ValidatorReport(name="helper_label_exclusions", status="failed" if failures else "passed", critical_failures=failures, warnings=warnings)
