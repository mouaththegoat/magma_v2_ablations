"""Stage-scoped code writing and execution for MAGMA v2."""

from __future__ import annotations

import os
import py_compile
import subprocess
import sys
from pathlib import Path
from typing import Any

from magma_v2.runtime.artifacts import stage_dir


FORBIDDEN_PATH_FRAGMENTS = (
    "./artifacts",
    "artifacts/runs",
    "artifacts\\runs",
)

FORBIDDEN_CODE_FRAGMENTS = {
    "cv='prefit'": (
        "Do not use CalibratedClassifierCV(cv='prefit'). It is incompatible "
        "with newer scikit-learn versions. Use cross-validated calibration or "
        "a validation-set calibrator implemented explicitly."
    ),
    'cv="prefit"': (
        "Do not use CalibratedClassifierCV(cv=\"prefit\"). It is incompatible "
        "with newer scikit-learn versions. Use cross-validated calibration or "
        "a validation-set calibrator implemented explicitly."
    ),
}


def reject_hardcoded_artifacts(code: str) -> str | None:
    if any(fragment in code for fragment in FORBIDDEN_PATH_FRAGMENTS):
        return (
            "Generated code must not hardcode artifact paths. The process cwd "
            "is already the stage directory; write outputs relative to Path.cwd()."
        )
    for fragment, error in FORBIDDEN_CODE_FRAGMENTS.items():
        if fragment in code:
            return error
    return None


def safe_stage_file(run_dir: str, stage: str, filename: str) -> str:
    base = Path(stage_dir(run_dir, stage)).resolve()
    target = (base / Path(filename).name).resolve()
    target.relative_to(base)
    return str(target)


def execute_stage_code(
    run_dir: str,
    stage: str,
    filename: str,
    timeout_seconds: int = 600,
) -> dict[str, Any]:
    stage_directory = stage_dir(run_dir, stage)
    script_path = safe_stage_file(run_dir, stage, filename)
    if not os.path.exists(script_path):
        return {"status": "error", "error": f"Script not found: {script_path}"}

    code = Path(script_path).read_text()
    error = reject_hardcoded_artifacts(code)
    if error:
        return {"status": "error", "error": error}

    try:
        py_compile.compile(script_path, doraise=True)
    except py_compile.PyCompileError as exc:
        log_path = Path(stage_directory) / f"{Path(filename).stem}.log"
        log_path.write_text(
            "STDOUT\n======\n\n\nSTDERR\n======\n"
            + str(exc)
        )
        return {
            "status": "error",
            "returncode": 1,
            "stdout": "",
            "stderr": str(exc)[-10000:],
            "log_path": str(log_path),
            "stage_dir": stage_directory,
            "exception_type": "SyntaxError",
        }

    env = os.environ.copy()
    project_root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = project_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    completed = subprocess.run(
        [sys.executable, "-B", script_path],
        cwd=stage_directory,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_seconds,
    )
    log_path = Path(stage_directory) / f"{Path(filename).stem}.log"
    log_path.write_text(
        "STDOUT\n======\n"
        + completed.stdout
        + "\n\nSTDERR\n======\n"
        + completed.stderr
    )
    return {
        "status": "success" if completed.returncode == 0 else "error",
        "returncode": completed.returncode,
        "stdout": completed.stdout[-10000:],
        "stderr": completed.stderr[-10000:],
        "log_path": str(log_path),
        "stage_dir": stage_directory,
    }
