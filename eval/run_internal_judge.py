"""Phase 4 — run the internal TRIPOD judge on each ablation variant's cs1 run dir.

Uses the judge agent from a1_no_judge (the variant where judge is always intact).
Mirrors the pattern from run_judge_on_baselines.py.

Usage:
    cd /home/user/magma_v2_ablations
    PYTHONPATH=a1_no_judge python eval/run_internal_judge.py \
        --ablation-root /scratch/mma9138/MAGMA/baseline_testing/ablation_runs

Outputs: writes/overwrites judge_verdict.json and reward.json in each run_dir.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path


VARIANTS = [
    "a1_no_judge",
    "a2_judge_no_rl",
    "a3_no_model_worker",
    "a4_no_data_worker",
    "a5_no_validators",
    "a6_no_hil",
]

REWARD_LAMBDA = 0.5


def find_latest_run_dir(artifact_root: str) -> str | None:
    runs_dir = Path(artifact_root) / "runs"
    if not runs_dir.exists():
        return None
    candidates = sorted(
        (d for d in runs_dir.iterdir() if d.is_dir()),
        key=lambda d: d.stat().st_mtime,
    )
    return str(candidates[-1]) if candidates else None


async def judge_run_dir(run_dir: str, reward_lambda: float = REWARD_LAMBDA) -> dict:
    """Run the judge agent on an existing run directory and return the verdict."""
    from magma_v2.agents.judge import judge as judge_agent

    invocation_state = {"run_dir": os.path.abspath(run_dir)}
    prompt = (
        f"Evaluate the MAGMA pipeline run at: {os.path.abspath(run_dir)}\n"
        f"Use reward_lambda={reward_lambda} when calling write_judge_verdict.\n"
        "Call read_pipeline_bundle first, then evaluate all artifacts per TRIPOD guidelines, "
        "then call write_judge_verdict with the full verdict JSON."
    )

    text_parts: list[str] = []
    try:
        async for event in judge_agent.stream_async(prompt, invocation_state=invocation_state):
            if "data" in event:
                text_parts.append(str(event["data"]))
    except Exception as exc:
        return {"status": "error", "error": str(exc), "run_dir": run_dir}

    verdict_path = Path(run_dir) / "judge_verdict.json"
    if verdict_path.exists():
        with open(verdict_path) as f:
            verdict = json.load(f)
        return {"status": "success", "run_dir": run_dir, "verdict": verdict}

    return {
        "status": "warning",
        "run_dir": run_dir,
        "message": "judge_verdict.json was not written",
        "judge_text": "".join(text_parts)[:500],
    }


async def main(ablation_root: str, variants: list[str], reward_lambda: float) -> None:
    results = {}

    for variant in variants:
        artifact_root = os.path.join(ablation_root, variant, "cs1")
        run_dir = find_latest_run_dir(artifact_root)
        if run_dir is None:
            print(f"[{variant}] No run directory found under {artifact_root} — skipping.")
            results[variant] = {"status": "skipped", "reason": "no run dir found"}
            continue

        print(f"[{variant}] Judging run dir: {run_dir}")
        result = await judge_run_dir(run_dir, reward_lambda)
        results[variant] = result

        if result.get("status") == "success":
            verdict = result.get("verdict", {})
            print(
                f"  verdict={verdict.get('status')}  "
                f"auroc={verdict.get('auroc')}  "
                f"tripod={verdict.get('tripod_score')}  "
                f"reward={verdict.get('reward')}"
            )
        else:
            print(f"  [{result.get('status')}] {result.get('error') or result.get('message')}")

    summary_path = os.path.join(ablation_root, "internal_judge_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSummary written to: {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run internal TRIPOD judge on ablation runs.")
    parser.add_argument(
        "--ablation-root",
        default="/scratch/mma9138/MAGMA/baseline_testing/ablation_runs",
        help="Root directory containing all variant subdirectories.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=VARIANTS,
        help="Which variants to judge (default: all 6).",
    )
    parser.add_argument(
        "--reward-lambda",
        type=float,
        default=REWARD_LAMBDA,
        help="Lambda for reward = lambda*AUROC + (1-lambda)*TRIPOD.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args.ablation_root, args.variants, args.reward_lambda))
