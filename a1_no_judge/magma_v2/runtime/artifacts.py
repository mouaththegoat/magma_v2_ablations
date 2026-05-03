"""Artifact directory helpers for MAGMA v2."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


STAGE_DIRS = {"data": "data", "model": "model", "scratch": "scratch"}


def slugify(value: str) -> str:
    chars = [char.lower() if char.isalnum() else "_" for char in value]
    slug = "_".join(part for part in "".join(chars).split("_") if part)
    return slug[:60] or "task"


def create_run_dir(prompt: str, artifact_root: str = "artifacts") -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{slugify(prompt[:80])}"
    runs_dir = Path(artifact_root) / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    candidate = runs_dir / run_name
    suffix = 2
    while candidate.exists():
        candidate = runs_dir / f"{run_name}_{suffix:02d}"
        suffix += 1
    candidate.mkdir()
    for dirname in STAGE_DIRS.values():
        (candidate / dirname).mkdir(exist_ok=True)
    return str(candidate)


def run_dir_from(path: str) -> str:
    current = Path(path).resolve()
    if current.name in STAGE_DIRS.values():
        return str(current.parent)
    return str(current)


def stage_dir(run_or_stage_dir: str, stage: str) -> str:
    run_dir = Path(run_dir_from(run_or_stage_dir))
    dirname = STAGE_DIRS.get(stage, stage)
    target = run_dir / dirname
    target.mkdir(parents=True, exist_ok=True)
    return str(target)


def write_json(path: str, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w") as f:
        json.dump(payload, f, indent=2, default=str)


def read_json(path: str) -> Any:
    with Path(path).open() as f:
        return json.load(f)


def list_files(directory: str) -> list[dict[str, Any]]:
    base = Path(directory).resolve()
    if not base.exists():
        return []
    files: list[dict[str, Any]] = []
    for path in sorted(base.rglob("*")):
        if not path.is_file() or path.name[:1] == ".":
            continue
        stat = path.stat()
        files.append({
            "path": str(path),
            "relative_path": str(path.relative_to(base)),
            "size_bytes": stat.st_size,
        })
    return files


def write_stage_readme(stage_directory: str, title: str, descriptions: dict[str, str]) -> str:
    files = list_files(stage_directory)
    lines = [
        f"# {title}",
        "",
        "| File | Size (bytes) | Description |",
        "|---|---:|---|",
    ]
    for item in files:
        rel = item["relative_path"]
        if rel == "README.md":
            continue
        lines.append(
            f"| `{rel}` | {item['size_bytes']} | {descriptions.get(rel, 'Stage output file.')} |"
        )
    if len(lines) == 4:
        lines.append("| None | 0 | No stage files were present when this README was written. |")
    readme_path = Path(stage_directory) / "README.md"
    readme_path.write_text("\n".join(lines) + "\n")
    return str(readme_path)
