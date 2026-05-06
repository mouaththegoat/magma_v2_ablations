"""Collect key artifact files from ablation and full-system runs into a clean results directory.

Output structure:
    results_dir/
        ablations/
            cs1/a1_no_judge/{reward.json, judge_verdict.json, metrics.json, final_report.md, ...}
            cs1/a2_judge_no_rl/...
            ...
        full/
            cs1/{reward.json, judge_verdict.json, metrics.json, final_report.md, ...}
            ...

Usage (ablations cs1-4):
    python eval/gather_results.py \
        --ablation-root /scratch/mma9138/MAGMA/baseline_testing/ablation_runs \
        --out-dir /scratch/mma9138/MAGMA/baseline_testing/results \
        --mode ablations --case-studies cs1 cs2 cs3 cs4

Usage (full system cs1-4):
    python eval/gather_results.py \
        --ablation-root /scratch/mma9138/MAGMA/baseline_testing/ablation_runs \
        --out-dir /scratch/mma9138/MAGMA/baseline_testing/results \
        --mode full --case-studies cs1 cs2 cs3 cs4

Usage (both, cs5-10 later):
    python eval/gather_results.py ... --mode both --case-studies cs5 cs6 cs7 cs8 cs9 cs10
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


ABLATION_VARIANTS = [
    "a1_no_judge",
    "a2_judge_no_rl",
    "a3_no_model_worker",
    "a4_no_data_worker",
    "a5_no_validators",
    "a6_no_hil",
]

# Files copied verbatim from run_dir root
ROOT_FILES = [
    "reward.json",
    "judge_verdict.json",
    "final_report.md",
    "DATA_HANDOFF.json",
    "MODEL_HANDOFF.json",
    "DATA_VALIDATION_REPORT.json",
    "MODEL_VALIDATION_REPORT.json",
    "recovery_summary.json",
    "actions.jsonl",
]

# Files copied from run_dir/model/
MODEL_FILES = [
    "metrics.json",
    "run_model.log",
]

# Files copied from run_dir/data/
DATA_FILES = [
    "README.md",
]


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


def find_latest_run_dir(artifact_root: Path) -> Path | None:
    runs_dir = artifact_root / "runs"
    if not runs_dir.exists():
        return None
    candidates = sorted(
        (d for d in runs_dir.iterdir() if d.is_dir()),
        key=lambda d: d.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def copy_run(run_dir: Path, dest: Path) -> dict[str, Any]:
    dest.mkdir(parents=True, exist_ok=True)
    copied, missing = [], []

    for fname in ROOT_FILES:
        src = run_dir / fname
        if src.exists():
            shutil.copy2(src, dest / fname)
            copied.append(fname)
        else:
            missing.append(fname)

    model_dest = dest / "model"
    model_dest.mkdir(exist_ok=True)
    for fname in MODEL_FILES:
        src = run_dir / "model" / fname
        if src.exists():
            shutil.copy2(src, model_dest / fname)
            copied.append(f"model/{fname}")
        else:
            missing.append(f"model/{fname}")

    # Copy predictions directory if present
    for pred_dir_name in ("predictions", "preds"):
        pred_src = run_dir / "model" / pred_dir_name
        if pred_src.exists():
            shutil.copytree(pred_src, dest / "model" / pred_dir_name, dirs_exist_ok=True)
            copied.append(f"model/{pred_dir_name}/")
            break

    # Write a quick summary JSON for easy inspection
    reward = _read_json(run_dir / "reward.json") or {}
    verdict = _read_json(run_dir / "judge_verdict.json") or {}
    metrics_raw = _read_json(run_dir / "model" / "metrics.json") or {}
    test_metrics = metrics_raw.get("metrics", metrics_raw).get("test", {})

    summary = {
        "run_dir": str(run_dir),
        "auroc": reward.get("auroc") or verdict.get("auroc") or test_metrics.get("auroc"),
        "auprc": test_metrics.get("auprc") or test_metrics.get("average_precision"),
        "tripod_score": reward.get("tripod_score") or verdict.get("tripod_score"),
        "reward": reward.get("reward") or reward.get("score"),
        "run_status": verdict.get("status") or ("completed" if reward else "unknown"),
        "copied_files": copied,
        "missing_files": missing,
    }
    (dest / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    return summary


def gather_ablations(ablation_root: Path, out_dir: Path, case_studies: list[str], variants: list[str]) -> None:
    base = out_dir / "ablations"
    for cs in case_studies:
        for variant in variants:
            artifact_root = ablation_root / variant / cs
            run_dir = find_latest_run_dir(artifact_root)
            dest = base / cs / variant
            if run_dir is None:
                print(f"  [skip] {variant}/{cs} — no run dir found")
                dest.mkdir(parents=True, exist_ok=True)
                (dest / "summary.json").write_text(
                    json.dumps({"run_dir": None, "run_status": "no_run_dir"}, indent=2), encoding="utf-8"
                )
                continue
            summary = copy_run(run_dir, dest)
            auroc = f"{summary['auroc']:.4f}" if summary["auroc"] is not None else "—"
            tripod = f"{summary['tripod_score']:.4f}" if summary["tripod_score"] is not None else "—"
            reward = f"{summary['reward']:.4f}" if summary["reward"] is not None else "—"
            print(f"  {variant}/{cs}  auroc={auroc}  tripod={tripod}  reward={reward}")


def gather_full(ablation_root: Path, out_dir: Path, case_studies: list[str]) -> None:
    base = out_dir / "full"
    for cs in case_studies:
        artifact_root = ablation_root / "full" / cs
        run_dir = find_latest_run_dir(artifact_root)
        dest = base / cs
        if run_dir is None:
            print(f"  [skip] full/{cs} — no run dir found")
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "summary.json").write_text(
                json.dumps({"run_dir": None, "run_status": "no_run_dir"}, indent=2), encoding="utf-8"
            )
            continue
        summary = copy_run(run_dir, dest)
        auroc = f"{summary['auroc']:.4f}" if summary["auroc"] is not None else "—"
        tripod = f"{summary['tripod_score']:.4f}" if summary["tripod_score"] is not None else "—"
        reward = f"{summary['reward']:.4f}" if summary["reward"] is not None else "—"
        print(f"  full/{cs}  auroc={auroc}  tripod={tripod}  reward={reward}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation-root", default="/scratch/mma9138/MAGMA/baseline_testing/ablation_runs")
    parser.add_argument("--out-dir", default="/scratch/mma9138/MAGMA/baseline_testing/results")
    parser.add_argument("--mode", choices=["ablations", "full", "both"], default="both")
    parser.add_argument("--case-studies", nargs="+", default=["cs1", "cs2", "cs3", "cs4"])
    parser.add_argument("--variants", nargs="+", default=ABLATION_VARIANTS)
    args = parser.parse_args()

    ab = Path(args.ablation_root)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.mode in ("ablations", "both"):
        print(f"\n=== Gathering ablation results → {out / 'ablations'} ===")
        gather_ablations(ab, out, args.case_studies, args.variants)

    if args.mode in ("full", "both"):
        print(f"\n=== Gathering full-system results → {out / 'full'} ===")
        gather_full(ab, out, args.case_studies)

    print(f"\nDone. Results written to: {out}")


if __name__ == "__main__":
    main()
