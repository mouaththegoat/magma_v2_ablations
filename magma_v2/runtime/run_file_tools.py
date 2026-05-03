"""MAGMA-owned run-scoped file tools for worker agents."""

from __future__ import annotations

import os
from typing import Any

from strands import tool

from magma_v2.runtime.artifacts import run_dir_from


def _run_dir(tool_context: object | None, artifact_dir: str = "./artifacts") -> str:
    if isinstance(tool_context, dict):
        invocation_state = tool_context.get("invocation_state", {})
    else:
        invocation_state = getattr(tool_context, "invocation_state", {}) if tool_context else {}
    return run_dir_from(invocation_state.get("run_dir") or artifact_dir)


def _resolve_run_path(path: str, tool_context: object | None, artifact_dir: str = "./artifacts") -> tuple[str, str]:
    if not path or not path.strip():
        raise ValueError("path is required")

    run_dir = os.path.realpath(os.path.abspath(_run_dir(tool_context, artifact_dir)))
    requested_path = path.strip()
    candidate = requested_path if os.path.isabs(requested_path) else os.path.join(run_dir, requested_path)
    resolved_path = os.path.realpath(os.path.abspath(candidate))
    if os.path.commonpath([run_dir, resolved_path]) != run_dir:
        raise ValueError(f"path must stay inside the current run directory: {run_dir}")
    return resolved_path, run_dir


@tool(context=True)
def write_run_file(
    path: str,
    content: str,
    artifact_dir: str = "./artifacts",
    tool_context: object | None = None,
) -> dict[str, Any]:
    """Write a file inside the current MAGMA run directory."""
    try:
        resolved_path, run_dir = _resolve_run_path(path, tool_context, artifact_dir)
        os.makedirs(os.path.dirname(resolved_path), exist_ok=True)
        with open(resolved_path, "w", encoding="utf-8") as file:
            file.write(content)
        return {
            "status": "success",
            "path": resolved_path,
            "run_dir": run_dir,
            "bytes_written": len(content.encode("utf-8")),
            "content": [{"text": f"Wrote {resolved_path} ({len(content.encode('utf-8'))} bytes)"}],
        }
    except Exception as exc:
        return {
            "status": "error",
            "content": [{"text": f"Failed to write run file: {exc}"}],
        }
