"""Phase 4 — assemble one evaluation record per ablation variant from its run artifacts.

Produces JSON trace files under ablation_runs/llm_ensemble_eval/ that the three LLM judges
will read. The assembled record bundles the key handoffs, metrics, code snippet, and judge
verdict (if present) into a single compact document.

Usage:
    python eval/assemble_ablation_traces.py \
        --ablation-root /scratch/mma9138/MAGMA/baseline_testing/ablation_runs
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


VARIANTS = [
    "a1_no_judge",
    "a2_judge_no_rl",
    "a3_no_model_worker",
    "a4_no_data_worker",
    "a5_no_validators",
    "a6_no_hil",
]

VARIANT_DESCRIPTIONS = {
    "a1_no_judge": "Full pipeline (data_worker + model_worker + validators + HIL) with no judge call and no reward-driven retry.",
    "a2_judge_no_rl": "Full pipeline with judge running once at the end; verdict written but never used to drive retries.",
    "a3_no_model_worker": "data_worker + HIL intact; model stage replaced by single-shot GPT call (one LLM call → script → execute). No model_worker_validator.",
    "a4_no_data_worker": "No data_worker or DATA_HANDOFF. model_worker must handle all data loading and splits itself. Validators + judge intact.",
    "a5_no_validators": "Full pipeline with both data_worker_validator and model_worker_validator removed. Tests validators' contribution.",
    "a6_no_hil": "Full pipeline with HIL clarification disabled. Pipeline runs unattended on prompt alone.",
}


def _read_json_safe(path: str | Path) -> Any:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


def _read_text_safe(path: str | Path, max_chars: int = 8000) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return p.read_text(encoding="utf-8", errors="replace")[:max_chars]
    except Exception:
        return None


def find_latest_run_dir(artifact_root: str) -> str | None:
    runs_dir = Path(artifact_root) / "runs"
    if not runs_dir.exists():
        return None
    candidates = sorted(
        (d for d in runs_dir.iterdir() if d.is_dir()),
        key=lambda d: d.stat().st_mtime,
    )
    return str(candidates[-1]) if candidates else None


def _find_model_file(run_dir: Path, filename: str) -> Path | None:
    candidate = run_dir / "model" / filename
    return candidate if candidate.exists() else None


def assemble_trace(variant: str, run_dir: str) -> dict[str, Any]:
    rd = Path(run_dir)

    task_input = _read_json_safe(rd / "task_input.json") or {}
    data_handoff = _read_json_safe(rd / "DATA_HANDOFF.json")
    model_handoff = _read_json_safe(rd / "MODEL_HANDOFF.json")
    research_handoff = _read_json_safe(rd / "RESEARCH_HANDOFF.json")
    judge_verdict = _read_json_safe(rd / "judge_verdict.json")
    reward = _read_json_safe(rd / "reward.json")
    data_validation = _read_json_safe(rd / "DATA_VALIDATION_REPORT.json")
    model_validation = _read_json_safe(rd / "MODEL_VALIDATION_REPORT.json")
    final_report = _read_text_safe(rd / "final_report.md")
    recovery_summary = _read_json_safe(rd / "recovery_summary.json")

    metrics = None
    if model_handoff:
        evaluation = model_handoff.get("evaluation_artifacts") or {}
        metrics_path = evaluation.get("metrics_path")
        if metrics_path:
            metrics = _read_json_safe(metrics_path) or _read_json_safe(rd / "model" / "metrics.json")
    if metrics is None:
        metrics = _read_json_safe(_find_model_file(rd, "metrics.json") or "")

    code_snippet = None
    if model_handoff:
        code_artifacts = model_handoff.get("code_artifacts") or {}
        code_path = code_artifacts.get("training_code_path") or code_artifacts.get("train_model_path")
        if code_path:
            code_snippet = _read_text_safe(code_path, max_chars=6000)
    if code_snippet is None:
        code_snippet = _read_text_safe(_find_model_file(rd, "training_code.py") or "", max_chars=6000)

    # Slim down large objects to keep the trace readable for LLM judges
    slim_data_handoff = None
    if data_handoff:
        slim_data_handoff = {k: v for k, v in data_handoff.items() if k not in ("raw_schema", "column_stats", "full_profile")}

    slim_model_handoff = None
    if model_handoff:
        slim_model_handoff = {k: v for k, v in model_handoff.items() if k not in ("full_config", "raw_logs")}

    return {
        "variant": variant,
        "variant_description": VARIANT_DESCRIPTIONS.get(variant, ""),
        "case_study": "cs1",
        "case_study_description": "Sepsis prediction from PhysioNet 2019 (target=SepsisLabel, expected model: LightGBM)",
        "run_dir": str(run_dir),
        "task_input": {
            "raw_prompt": task_input.get("raw_prompt"),
            "reward_lambda": task_input.get("reward_lambda"),
            "created_at": task_input.get("created_at"),
        },
        "pipeline_artifacts": {
            "data_handoff_present": data_handoff is not None,
            "model_handoff_present": model_handoff is not None,
            "research_handoff_present": research_handoff is not None,
            "data_validation_present": data_validation is not None,
            "model_validation_present": model_validation is not None,
            "judge_verdict_present": judge_verdict is not None,
            "metrics_present": metrics is not None,
            "training_code_present": code_snippet is not None,
            "final_report_present": final_report is not None,
        },
        "data_handoff": slim_data_handoff,
        "model_handoff": slim_model_handoff,
        "metrics": metrics,
        "judge_verdict": judge_verdict,
        "reward": reward,
        "data_validation_summary": (data_validation or {}).get("summary") or (data_validation or {}).get("compliance_status"),
        "model_validation_summary": (model_validation or {}).get("summary") or (model_validation or {}).get("compliance_status"),
        "training_code_snippet": code_snippet,
        "final_report": final_report,
        "recovery_summary": (recovery_summary or {}).get("summary"),
    }


def main(ablation_root: str, variants: list[str]) -> None:
    ensemble_dir = Path(ablation_root) / "llm_ensemble_eval"
    ensemble_dir.mkdir(parents=True, exist_ok=True)

    assembled: list[dict[str, Any]] = []
    for variant in variants:
        artifact_root = os.path.join(ablation_root, variant, "cs1")
        run_dir = find_latest_run_dir(artifact_root)
        if run_dir is None:
            print(f"[{variant}] No run directory found — skipping.")
            continue

        print(f"[{variant}] Assembling from: {run_dir}")
        trace = assemble_trace(variant, run_dir)
        assembled.append(trace)

        trace_path = ensemble_dir / f"{variant}_cs1_trace.json"
        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump(trace, f, indent=2, default=str)
        print(f"  Written: {trace_path}")

    index_path = ensemble_dir / "traces_index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(
            [{"variant": t["variant"], "trace_file": f"{t['variant']}_cs1_trace.json"} for t in assembled],
            f,
            indent=2,
        )
    print(f"\nIndex written to: {index_path}")
    print(f"Total traces assembled: {len(assembled)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble LLM-judge traces from ablation run dirs.")
    parser.add_argument(
        "--ablation-root",
        default="/scratch/mma9138/MAGMA/baseline_testing/ablation_runs",
    )
    parser.add_argument("--variants", nargs="+", default=VARIANTS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.ablation_root, args.variants)
