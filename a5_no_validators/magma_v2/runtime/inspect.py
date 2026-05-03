"""Input inspection utilities for code-first agents."""

from __future__ import annotations

import os
import shlex
from pathlib import Path
from typing import Any


TRAILING_PUNCTUATION = ".,;:)]}"


def extract_paths(text: str) -> list[str]:
    """Extract only paths that actually exist, without regex-based guessing."""
    tokens = _prompt_tokens(text)
    paths: list[str] = []
    index = 0
    while index < len(tokens):
        raw = tokens[index]
        candidate = raw.strip().rstrip(TRAILING_PUNCTUATION)
        if not candidate:
            index += 1
            continue
        combined = _combine_with_following_path_tokens(candidate, tokens[index + 1:])
        if combined:
            combined_path, consumed = combined
            paths.append(os.path.abspath(os.path.expanduser(combined_path)))
            index += consumed + 1
            continue
        expanded = os.path.expanduser(candidate)
        if os.path.exists(expanded):
            paths.append(os.path.abspath(expanded))
        index += 1
    return sorted(dict.fromkeys(paths))


def _prompt_tokens(text: str) -> list[str]:
    try:
        return shlex.split(text)
    except ValueError:
        return text.split()


def _combine_with_following_path_tokens(candidate: str, following: list[str]) -> tuple[str, int] | None:
    """Recover paths that were split by whitespace/newlines inside a prompt.

    Example:
    `/Users/me/Clinic_Research` `/magma-cai/data/file.csv`
    should become
    `/Users/me/Clinic_Research/magma-cai/data/file.csv`.
    """
    base = os.path.expanduser(candidate)
    if not os.path.isabs(base) or not os.path.isdir(base):
        return None
    current = base
    consumed = 0
    for raw in following:
        token = raw.strip().rstrip(TRAILING_PUNCTUATION)
        if not token or not token.startswith(os.sep):
            break
        if os.path.exists(os.path.expanduser(token)):
            break
        next_candidate = os.path.join(current, token.lstrip(os.sep))
        if not os.path.exists(next_candidate):
            break
        current = next_candidate
        consumed += 1
        if os.path.isfile(current):
            break
    return (current, consumed) if consumed else None


def inspect_path(path: str) -> dict[str, Any]:
    p = Path(path)
    info: dict[str, Any] = {
        "path": str(p),
        "exists": p.exists(),
        "kind": "missing",
    }
    if not p.exists():
        return info
    if p.is_dir():
        children = sorted(child.name for child in p.iterdir() if child.name[:1] != ".")
        split_dirs = [name for name in children if name.lower() in {"train", "val", "valid", "validation", "test"}]
        info.update({
            "kind": "directory",
            "children": children[:100],
            "split_directories": split_dirs,
        })
        return info
    info.update({
        "kind": "file",
        "suffix": p.suffix.lower(),
        "size_bytes": p.stat().st_size,
    })
    return info


def inspect_user_prompt(prompt: str) -> dict[str, Any]:
    paths = extract_paths(prompt)
    inspected = [inspect_path(path) for path in paths]
    return {
        "raw_prompt": prompt,
        "paths": paths,
        "inspected_paths": inspected,
    }
