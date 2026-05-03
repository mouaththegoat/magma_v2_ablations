"""Run-scoped Strands tool consent policy helpers."""

from __future__ import annotations

import os

import strands_tools.file_write as strands_file_write_module


ACTIVE_RUN_DIR_ENV = "MAGMA_ACTIVE_RUN_DIR"

_STRANDS_FILE_WRITE_PATCHED = False


def _is_within_directory(path: str, directory: str) -> bool:
    resolved_path = os.path.realpath(os.path.abspath(os.path.expanduser(path)))
    resolved_directory = os.path.realpath(os.path.abspath(os.path.expanduser(directory)))
    try:
        return os.path.commonpath([resolved_directory, resolved_path]) == resolved_directory
    except ValueError:
        return False


def install_run_scoped_strands_file_write_policy() -> None:
    """Bypass Strands file-write consent only for paths inside the active run directory."""
    global _STRANDS_FILE_WRITE_PATCHED
    if _STRANDS_FILE_WRITE_PATCHED:
        return

    original_file_write = strands_file_write_module.file_write

    def guarded_file_write(tool: dict, **kwargs: object) -> dict:
        tool_input = tool.get("input", {}) if isinstance(tool, dict) else {}
        requested_path = str(tool_input.get("path", "")).strip()
        active_run_dir = os.environ.get(ACTIVE_RUN_DIR_ENV, "").strip()
        previous_bypass = os.environ.get("BYPASS_TOOL_CONSENT")

        try:
            if active_run_dir and requested_path and _is_within_directory(requested_path, active_run_dir):
                os.environ["BYPASS_TOOL_CONSENT"] = "true"
            elif previous_bypass is None:
                os.environ.pop("BYPASS_TOOL_CONSENT", None)
            else:
                os.environ["BYPASS_TOOL_CONSENT"] = previous_bypass
            return original_file_write(tool, **kwargs)
        finally:
            if previous_bypass is None:
                os.environ.pop("BYPASS_TOOL_CONSENT", None)
            else:
                os.environ["BYPASS_TOOL_CONSENT"] = previous_bypass

    strands_file_write_module.file_write = guarded_file_write
    _STRANDS_FILE_WRITE_PATCHED = True


def set_active_run_dir(run_dir: str) -> str | None:
    """Register the current run directory for run-scoped Strands file writes."""
    previous_active_run_dir = os.environ.get(ACTIVE_RUN_DIR_ENV)
    os.environ[ACTIVE_RUN_DIR_ENV] = os.path.abspath(run_dir)
    return previous_active_run_dir


def restore_active_run_dir(previous_active_run_dir: str | None) -> None:
    """Restore the previous active run directory after a run completes."""
    if previous_active_run_dir is None:
        os.environ.pop(ACTIVE_RUN_DIR_ENV, None)
    else:
        os.environ[ACTIVE_RUN_DIR_ENV] = previous_active_run_dir
