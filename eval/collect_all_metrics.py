"""Collect AUROC, AUPRC, TRIPOD, Reward, Failures, and wall-time for every
(case_study, variant) pair and write a single flat CSV.

Usage:
    python eval/collect_all_metrics.py \
        --ablation-root /scratch/mma9138/MAGMA/baseline_testing/ablation_runs \
        --out all_metrics.csv

Output columns:
    case_study, variant, auroc, auprc, tripod_score, reward_score,
    n_failures, n_retries, wall_time_seconds, run_status
"""

from __future__ import annotations

import argparse
import csv
import json
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

CASE_STUDIES = [f"cs{i}" for i in range(1, 11)]

COLUMNS = [
    "case_study",
    "variant",
    "auroc",
    "auprc",
    "tripod_score",
    "reward_score",
    "n_failures",
    "n_retries",
    "wall_time_seconds",
    "run_status",
]


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


def _find_latest_run_dir(artifact_root: Path) -> Path | None:
    runs_dir = artifact_root / "runs"
    if not runs_dir.exists():
        return None
    candidates = sorted(
        (d for d in runs_dir.iterdir() if d.is_dir()),
        key=lambda d: d.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def _metric(metrics: dict | None, split: str, name: str) -> float | None:
    if not metrics:
        return None
    payload = metrics.get("metrics") if isinstance(metrics.get("metrics"), dict) else metrics
    section = (payload or {}).get(split, {})
    if not isinstance(section, dict):
        return None
    aliases = {"auprc": ("auprc", "average_precision", "ap")}.get(name, (name,))
    for alias in aliases:
        v = section.get(alias)
        if v is not None:
            try:
                return round(float(v), 4)
            except (TypeError, ValueError):
                pass
    return None


def _parse_actions(run_dir: Path) -> tuple[int, int]:
    """Return (n_failures, n_retries) by scanning actions.jsonl."""
    path = run_dir / "actions.jsonl"
    if not path.exists():
        return 0, 0
    starts = failures = 0
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                ev = json.loads(line)
                action = ev.get("action", "")
                if action == "start_attempt":
                    starts += 1
                elif action in ("attempt_failed", "code_execution_failed", "failure"):
                    failures += 1
            except Exception:
                pass
    except Exception:
        pass
    retries = max(0, starts - 1)
    return failures, retries


def _run_status(run_dir: Path) -> str:
    for fname in ("recovery_summary.json", "recovery_state.json"):
        rec = _read_json(run_dir / fname)
        if rec:
            state = rec.get("state") or {}
            s = state.get("run_status")
            if s:
                return str(s)
    if _read_json(run_dir / "reward.json"):
        return "completed"
    final = run_dir / "final_report.md"
    if final.exists():
        text = final.read_text(errors="replace")
        return "failed" if ("Failed" in text or "failed" in text) else "completed"
    return "unknown"


def _load_wall_times(ablation_root: Path, case_study: str) -> dict[str, int]:
    tsv = ablation_root / f"ablation_wall_times_{case_study}.tsv"
    if not tsv.exists():
        # fall back to the legacy single-cs file
        tsv = ablation_root / "ablation_wall_times.tsv"
    if not tsv.exists():
        return {}
    times: dict[str, int] = {}
    try:
        for line in tsv.read_text().splitlines()[1:]:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                try:
                    times[parts[0]] = int(parts[3])
                except ValueError:
                    pass
    except Exception:
        pass
    return times


def gather(ablation_root: Path, case_study: str, variant: str, wall_times: dict[str, int]) -> dict:
    row: dict[str, Any] = {c: None for c in COLUMNS}
    row["case_study"] = case_study
    row["variant"] = variant
    row["run_status"] = "no_run_dir"

    artifact_root = ablation_root / variant / case_study
    run_dir = _find_latest_run_dir(artifact_root)
    if run_dir is None:
        return row

    row["run_status"] = _run_status(run_dir)
    row["n_failures"], row["n_retries"] = _parse_actions(run_dir)
    row["wall_time_seconds"] = wall_times.get(variant)

    reward = _read_json(run_dir / "reward.json") or {}
    verdict = _read_json(run_dir / "judge_verdict.json") or {}

    row["auroc"] = reward.get("auroc") or verdict.get("auroc")
    row["reward_score"] = reward.get("reward") or reward.get("score")
    row["tripod_score"] = reward.get("tripod_score") or verdict.get("tripod_score")

    # also check model/metrics.json for AUROC + AUPRC
    metrics = _read_json(run_dir / "model" / "metrics.json")
    if metrics is None:
        mh = _read_json(run_dir / "MODEL_HANDOFF.json") or {}
        mp = (mh.get("evaluation_artifacts") or {}).get("metrics_path")
        if mp:
            metrics = _read_json(Path(mp))

    if row["auroc"] is None:
        row["auroc"] = _metric(metrics, "test", "auroc")
    row["auprc"] = _metric(metrics, "test", "auprc")

    for key in ("auroc", "auprc", "tripod_score", "reward_score"):
        if row[key] is not None:
            try:
                row[key] = round(float(row[key]), 4)
            except (TypeError, ValueError):
                pass

    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ablation-root",
        default="/scratch/mma9138/MAGMA/baseline_testing/ablation_runs",
    )
    parser.add_argument(
        "--case-studies", nargs="+", default=CASE_STUDIES,
        help="Which case studies to collect (default: cs1–cs10)",
    )
    parser.add_argument(
        "--variants", nargs="+", default=VARIANTS,
    )
    parser.add_argument(
        "--out",
        default="all_metrics.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    ab = Path(args.ablation_root)
    rows: list[dict] = []

    for cs in args.case_studies:
        wall_times = _load_wall_times(ab, cs)
        for variant in args.variants:
            row = gather(ab, cs, variant, wall_times)
            rows.append(row)
            auroc = f"{row['auroc']:.4f}" if row["auroc"] is not None else "—"
            auprc = f"{row['auprc']:.4f}" if row["auprc"] is not None else "—"
            print(f"  {cs}/{variant:25s}  auroc={auroc}  auprc={auprc}  [{row['run_status']}]")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows → {out}")


if __name__ == "__main__":
    main()
