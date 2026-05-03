"""Deterministic evidence collection for normalization."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


TABLE_SUFFIXES = {".csv", ".tsv", ".parquet", ".json", ".jsonl"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".dcm"}
ECG_SUFFIXES = {".hea", ".mat", ".npy", ".npz", ".wfdb", ".dat"}
TEXT_SUFFIXES = {".txt", ".md"}
PATH_SUFFIXES = IMAGE_SUFFIXES | ECG_SUFFIXES | TEXT_SUFFIXES
SPLIT_NAMES = {"train", "val", "valid", "validation", "test"}


def resolve_path(path: str | Path) -> Path:
    target = Path(path).expanduser()
    if not target.is_absolute():
        target = Path.cwd() / target
    return target.resolve()


def read_table(path: str | Path, n_rows: int | None = None) -> pd.DataFrame:
    target = resolve_path(path)
    suffix = target.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(target, nrows=n_rows)
    if suffix == ".tsv":
        return pd.read_csv(target, sep="\t", nrows=n_rows)
    if suffix == ".parquet":
        frame = pd.read_parquet(target)
        return frame.head(n_rows) if n_rows else frame
    if suffix == ".jsonl":
        return pd.read_json(target, lines=True, nrows=n_rows)
    if suffix == ".json":
        frame = pd.read_json(target)
        return frame.head(n_rows) if n_rows else frame
    raise ValueError(f"Unsupported table suffix: {suffix}")


def inspect_path(path: str | Path) -> dict[str, Any]:
    target = resolve_path(path)
    info: dict[str, Any] = {"path": str(target), "exists": target.exists(), "kind": "missing"}
    if not target.exists():
        return info
    if target.is_dir():
        files = [child for child in target.rglob("*") if child.is_file() and child.name[:1] != "."]
        suffix_counts = Counter(child.suffix.lower() or "<none>" for child in files)
        split_dirs = sorted({child.parent.name.lower() for child in files if child.parent.name.lower() in SPLIT_NAMES})
        info.update({
            "kind": "directory",
            "file_count": len(files),
            "suffix_counts": dict(suffix_counts),
            "split_directories": split_dirs,
            "files_sample": [str(child) for child in files[:100]],
        })
        return info
    info.update({"kind": "file", "suffix": target.suffix.lower(), "size_bytes": target.stat().st_size})
    if target.suffix.lower() in TABLE_SUFFIXES:
        frame = read_table(target, n_rows=1000)
        info["table"] = profile_table(frame, target)
    return info


def profile_table(frame: pd.DataFrame, source: str | Path) -> dict[str, Any]:
    columns = []
    for column in frame.columns:
        series = frame[column]
        examples = [str(value) for value in series.dropna().head(10).tolist()]
        columns.append({
            "name": str(column),
            "dtype": str(series.dtype),
            "non_null": int(series.notna().sum()),
            "null": int(series.isna().sum()),
            "unique": int(series.nunique(dropna=True)),
            "examples": examples,
        })
    return {
        "source": str(resolve_path(source)),
        "rows_sampled": int(len(frame)),
        "columns": [str(column) for column in frame.columns],
        "dtypes": {str(column): str(dtype) for column, dtype in frame.dtypes.items()},
        "column_profiles": columns,
    }


def looks_like_path(value: str) -> bool:
    text = str(value)
    suffix = Path(text.lower()).suffix
    return suffix in PATH_SUFFIXES or "/" in text or "\\" in text or "://" in text


def classify_path_column(values: list[str]) -> str | None:
    suffixes = [Path(str(value).lower()).suffix for value in values if value]
    if not suffixes:
        return None
    counts = Counter(suffixes)
    image_hits = sum(counts[suffix] for suffix in IMAGE_SUFFIXES)
    ecg_hits = sum(counts[suffix] for suffix in ECG_SUFFIXES)
    text_hits = sum(counts[suffix] for suffix in TEXT_SUFFIXES)
    total = max(len(suffixes), 1)
    if image_hits / total >= 0.5:
        return "image"
    if ecg_hits / total >= 0.5:
        return "ecg"
    if text_hits / total >= 0.5:
        return "text"
    return "file"


def infer_table_columns(frame: pd.DataFrame, request: str = "") -> dict[str, Any]:
    lower_request = request.lower()
    target_candidates = []
    group_candidates = []
    path_candidates = []
    text_candidates = []
    profiled_columns: list[dict[str, Any]] = []
    for column in frame.columns:
        name = str(column)
        lower = name.lower()
        series = frame[column]
        examples = [str(value) for value in series.dropna().head(50).tolist()]
        unique = int(series.nunique(dropna=True))
        non_null = int(series.notna().sum())
        row_count = max(int(len(series)), 1)
        unique_ratio = unique / max(non_null, 1)
        path_examples = [value for value in examples if looks_like_path(value)]
        is_path_like = bool(path_examples and len(path_examples) / max(len(examples), 1) >= 0.5)

        evidence: list[dict[str, Any]] = []
        target_score = 0.0
        if lower in lower_request:
            target_score += 0.75
            evidence.append({"type": "column_name_mentioned_in_request"})
        if not is_path_like and unique == 2:
            target_score += 0.7
            evidence.append({"type": "binary_label_compatible", "unique": unique, "unique_ratio": unique_ratio})
        elif not is_path_like and row_count >= 30 and 2 < unique <= max(20, int(row_count * 0.05)):
            target_score += 0.45
            evidence.append({"type": "low_cardinality_observed", "unique": unique, "unique_ratio": unique_ratio})
        if target_score >= 0.5:
            target_candidates.append({"column": name, "score": round(target_score, 3), "evidence": evidence})

        group_score = 0.0
        group_evidence = []
        if lower in lower_request:
            group_score += 0.65
            group_evidence.append({"type": "column_name_mentioned_in_request"})
        if not is_path_like and unique > 1 and unique < non_null:
            group_score += 0.4
            group_evidence.append({"type": "repeated_values_observed", "unique": unique, "non_null": non_null, "unique_ratio": unique_ratio})
        if not is_path_like and unique >= 3:
            group_score += 0.15
            group_evidence.append({"type": "more_than_binary_values", "unique": unique})
        if not is_path_like and 0.05 <= unique_ratio <= 0.98:
            group_score += 0.15
            group_evidence.append({"type": "identifier_cardinality_compatible", "unique_ratio": unique_ratio})
        if not is_path_like and pd.api.types.is_integer_dtype(series):
            group_score += 0.05
            group_evidence.append({"type": "integer_identifier_compatible"})
        if group_score >= 0.4:
            group_candidates.append({"column": name, "score": round(group_score, 3), "evidence": group_evidence})

        if is_path_like:
            path_candidates.append({
                "column": name,
                "modality": classify_path_column(path_examples),
                "evidence": {"examples": path_examples[:5], "fraction": len(path_examples) / max(len(examples), 1)},
            })

        avg_len = sum(len(value) for value in examples) / max(len(examples), 1)
        if examples and not path_examples and avg_len >= 40:
            text_candidates.append({"column": name, "evidence": {"average_example_length": avg_len}})
        profiled_columns.append({
            "column": name,
            "unique": unique,
            "non_null": non_null,
            "unique_ratio": unique_ratio,
            "is_path_like": is_path_like,
        })

    if not target_candidates:
        statistical_targets = [
            item for item in profiled_columns
            if not item["is_path_like"] and item["unique"] == 2
        ]
        if len(statistical_targets) == 1:
            item = statistical_targets[0]
            target_candidates.append({
                "column": item["column"],
                "score": 0.55,
                "evidence": [{"type": "only_low_cardinality_column", "unique": item["unique"], "unique_ratio": item["unique_ratio"]}],
            })

    return {
        "candidate_targets": sorted(target_candidates, key=lambda item: item["score"], reverse=True),
        "candidate_group_ids": sorted(group_candidates, key=lambda item: item["score"], reverse=True),
        "candidate_path_columns": path_candidates,
        "candidate_text_columns": text_candidates,
    }
