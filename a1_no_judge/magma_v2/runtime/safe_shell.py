"""Bounded run-scoped shell wrappers for MAGMA v2."""

from __future__ import annotations

import os
import shlex
from pathlib import Path
from typing import Any

from strands import tool
from strands_tools.shell import shell as strands_shell

from magma_v2.runtime.actions import log_action
from magma_v2.runtime.artifacts import run_dir_from, stage_dir


_INTERACTIVE_COMMANDS = {
    "bash",
    "sh",
    "zsh",
    "fish",
    "python",
    "python3",
    "ipython",
    "bpython",
    "ptpython",
    "vim",
    "vi",
    "nano",
    "emacs",
    "less",
    "more",
    "man",
    "top",
    "htop",
    "screen",
    "tmux",
}


def _run_dir(tool_context: object | None, artifact_dir: str = "./artifacts") -> str:
    if isinstance(tool_context, dict):
        invocation_state = tool_context.get("invocation_state", {})
    else:
        invocation_state = getattr(tool_context, "invocation_state", {}) if tool_context else {}
    return run_dir_from(invocation_state.get("run_dir") or artifact_dir)


def _is_within_directory(path: Path, directory: Path) -> bool:
    try:
        return os.path.commonpath([str(directory), str(path)]) == str(directory)
    except ValueError:
        return False


def _extract_command_texts(command: Any) -> list[str]:
    if isinstance(command, str):
        return [command]
    if isinstance(command, list):
        texts: list[str] = []
        for item in command:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("command"), str):
                texts.append(str(item["command"]))
        return texts
    return []


def _interactive_reason(command_text: str) -> str | None:
    stripped = command_text.strip()
    if not stripped:
        return "empty shell command"
    try:
        tokens = shlex.split(stripped)
    except ValueError:
        tokens = stripped.split()
    if not tokens:
        return "empty shell command"
    first = tokens[0]
    if first in _INTERACTIVE_COMMANDS:
        return f"interactive command {first!r} is not allowed"
    if "exec bash" in stripped or "exec zsh" in stripped or "exec sh" in stripped:
        return "shell replacement is not allowed"
    return None


def build_bounded_shell_tool(*, agent_name: str, stage_name: str, default_timeout: int = 60, max_timeout: int = 120):
    @tool(context=True)
    def bounded_shell(
        command: str | list[str | dict[str, Any]],
        timeout: int | None = None,
        work_dir: str | None = None,
        ignore_errors: bool = False,
        artifact_dir: str = "./artifacts",
        tool_context: object | None = None,
    ) -> dict:
        """Run a bounded non-interactive shell command inside the active run directory."""
        run_dir = Path(_run_dir(tool_context, artifact_dir)).resolve()
        stage_directory = Path(stage_dir(str(run_dir), stage_name)).resolve()
        requested_work_dir = Path(work_dir).expanduser() if work_dir else stage_directory
        if not requested_work_dir.is_absolute():
            requested_work_dir = (stage_directory / requested_work_dir).resolve()
        else:
            requested_work_dir = requested_work_dir.resolve()

        command_texts = _extract_command_texts(command)
        reasons = [reason for text in command_texts if (reason := _interactive_reason(text))]
        if reasons:
            message = "; ".join(reasons)
            log_action(
                str(run_dir),
                agent_name,
                "bounded_shell_rejected",
                success=False,
                error=message,
                metadata={"command": command, "work_dir": str(requested_work_dir)},
            )
            return {"status": "error", "error": message}

        if not _is_within_directory(requested_work_dir, run_dir):
            message = f"work_dir must stay inside the active run directory: {requested_work_dir}"
            log_action(
                str(run_dir),
                agent_name,
                "bounded_shell_rejected",
                success=False,
                error=message,
                metadata={"command": command, "work_dir": str(requested_work_dir)},
            )
            return {"status": "error", "error": message}

        effective_timeout = max(1, min(int(timeout or default_timeout), max_timeout))
        previous_bypass = os.environ.get("BYPASS_TOOL_CONSENT")
        log_action(
            str(run_dir),
            agent_name,
            "bounded_shell_request",
            metadata={
                "command": command,
                "work_dir": str(requested_work_dir),
                "timeout": effective_timeout,
                "ignore_errors": ignore_errors,
                "stage": stage_name,
            },
        )
        try:
            os.environ["BYPASS_TOOL_CONSENT"] = "true"
            result = strands_shell(
                command=command,
                timeout=effective_timeout,
                work_dir=str(requested_work_dir),
                ignore_errors=ignore_errors,
                non_interactive=True,
            )
        finally:
            if previous_bypass is None:
                os.environ.pop("BYPASS_TOOL_CONSENT", None)
            else:
                os.environ["BYPASS_TOOL_CONSENT"] = previous_bypass

        success = not isinstance(result, dict) or result.get("status") != "error"
        error = None
        metadata: dict[str, Any] = {
            "command": command,
            "work_dir": str(requested_work_dir),
            "timeout": effective_timeout,
        }
        if isinstance(result, dict):
            error = str(result.get("error") or result.get("stderr") or "") or None
            metadata["result_keys"] = sorted(result.keys())
        log_action(
            str(run_dir),
            agent_name,
            "bounded_shell_result",
            success=success,
            error=error,
            metadata=metadata,
        )
        return result

    return bounded_shell
