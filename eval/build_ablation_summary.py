"""Phase 4 — build ablation_summary.csv, ablation_summary.md, and CHANGES_OVERVIEW.md.

Reads:
  - Each variant's run_dir: judge_verdict.json, reward.json, model/metrics.json, actions.jsonl
  - ablation_runs/ablation_wall_times.tsv  (written by run_ablations_cs1.sbatch)
  - ablation_runs/llm_ensemble_eval/ensemble_scores.json
  - Each variant's CHANGES.md  (from the repo)

Writes:
  - ablation_runs/ablation_summary.csv
  - ablation_runs/ablation_summary.md
  - ablation_runs/CHANGES_OVERVIEW.md

Usage:
    python eval/build_ablation_summary.py \
        --ablation-root /scratch/mma9138/MAGMA/baseline_testing/ablation_runs \
        --repo-root /home/user/magma_v2_ablations
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, UTC
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

VARIANT_LABELS = {
    "a1_no_judge": "A1: no_judge",
    "a2_judge_no_rl": "A2: judge_no_rl",
    "a3_no_model_worker": "A3: no_model_worker",
    "a4_no_data_worker": "A4: no_data_worker",
    "a5_no_validators": "A5: no_validators",
    "a6_no_hil": "A6: no_hil",
}

CSV_COLUMNS = [
    "variant",
    "auroc",
    "auprc",
    "tripod_score_internal",
    "ensemble_score_claude",
    "ensemble_score_gpt",
    "ensemble_score_gemini",
    "ensemble_mean",
    "wall_time_seconds",
    "api_cost_usd",
    "n_retries",
    "status",
]


def _read_json_safe(path: Path) -> Any:
    if not path.exists():
        return None
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


def _get_metric(metrics: dict | None, split: str, name: str) -> float | None:
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


def _count_retries(run_dir: Path) -> int:
    """Count start_attempt events in actions.jsonl as a proxy for retries."""
    actions_path = run_dir / "actions.jsonl"
    if not actions_path.exists():
        return 0
    count = 0
    try:
        for line in actions_path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                ev = json.loads(line)
                if ev.get("action") == "start_attempt":
                    count += 1
            except Exception:
                pass
    except Exception:
        pass
    return max(0, count - 1)  # subtract 1 because the first attempt is not a retry


def _run_status(run_dir: Path) -> str:
    recovery = _read_json_safe(run_dir / "recovery_summary.json")
    if recovery:
        state = recovery.get("state") or {}
        return str(state.get("run_status") or "unknown")
    reward = _read_json_safe(run_dir / "reward.json")
    if reward:
        return "completed"
    final = run_dir / "final_report.md"
    if final.exists():
        text = final.read_text(errors="replace")
        if "Failed" in text or "failed" in text:
            return "failed"
        return "completed"
    return "unknown"


def load_wall_times(ablation_root: Path) -> dict[str, int]:
    tsv_path = ablation_root / "ablation_wall_times.tsv"
    if not tsv_path.exists():
        return {}
    times: dict[str, int] = {}
    try:
        for line in tsv_path.read_text().splitlines()[1:]:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                try:
                    times[parts[0]] = int(parts[3])
                except ValueError:
                    pass
    except Exception:
        pass
    return times


def load_ensemble_scores(ablation_root: Path) -> dict[str, dict]:
    path = ablation_root / "llm_ensemble_eval" / "ensemble_scores.json"
    data = _read_json_safe(path)
    if not isinstance(data, list):
        return {}
    return {r["variant"]: r for r in data if isinstance(r, dict) and "variant" in r}


def gather_row(
    variant: str,
    ablation_root: Path,
    wall_times: dict[str, int],
    ensemble: dict[str, dict],
) -> dict[str, Any]:
    artifact_root = ablation_root / variant / "cs1"
    run_dir = find_latest_run_dir(artifact_root)

    row: dict[str, Any] = {k: None for k in CSV_COLUMNS}
    row["variant"] = variant
    row["status"] = "no_run_dir"

    if run_dir is None:
        return row

    row["status"] = _run_status(run_dir)
    row["n_retries"] = _count_retries(run_dir)

    reward = _read_json_safe(run_dir / "reward.json") or {}
    verdict = _read_json_safe(run_dir / "judge_verdict.json") or {}

    row["auroc"] = reward.get("auroc") or verdict.get("auroc")
    row["tripod_score_internal"] = reward.get("tripod_score") or verdict.get("tripod_score")

    metrics_path = run_dir / "model" / "metrics.json"
    metrics = _read_json_safe(metrics_path)
    if metrics is None and verdict:
        mh_path = run_dir / "MODEL_HANDOFF.json"
        mh = _read_json_safe(mh_path) or {}
        eval_arts = mh.get("evaluation_artifacts") or {}
        mp = eval_arts.get("metrics_path")
        if mp:
            metrics = _read_json_safe(Path(mp))

    row["auprc"] = _get_metric(metrics, "test", "auprc")
    if row["auroc"] is None:
        row["auroc"] = _get_metric(metrics, "test", "auroc")

    if row["auroc"] is not None:
        row["auroc"] = round(float(row["auroc"]), 4)
    if row["tripod_score_internal"] is not None:
        row["tripod_score_internal"] = round(float(row["tripod_score_internal"]), 4)

    row["wall_time_seconds"] = wall_times.get(variant)
    row["api_cost_usd"] = None  # not tracked natively; fill manually from provider dashboards

    ens = ensemble.get(variant, {})
    row["ensemble_score_claude"] = ens.get("overall_score_claude")
    row["ensemble_score_gpt"] = ens.get("overall_score_gpt")
    row["ensemble_score_gemini"] = ens.get("overall_score_gemini")
    row["ensemble_mean"] = ens.get("ensemble_overall_mean")

    return row


def fmt(v: Any, decimals: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


def build_markdown_table(rows: list[dict[str, Any]]) -> str:
    header = (
        "| Variant | AUROC | AUPRC | TRIPOD (internal) | "
        "Ensemble Claude | Ensemble GPT | Ensemble Gemini | Ensemble Mean | "
        "Wall time (s) | Retries | Status |\n"
        "|---------|-------|-------|-------------------|"
        "----------------|--------------|-----------------|---------------|"
        "--------------|---------|--------|\n"
    )
    rows_md = ""
    for r in rows:
        rows_md += (
            f"| {VARIANT_LABELS.get(r['variant'], r['variant'])} "
            f"| {fmt(r['auroc'])} "
            f"| {fmt(r['auprc'])} "
            f"| {fmt(r['tripod_score_internal'])} "
            f"| {fmt(r['ensemble_score_claude'], 2)} "
            f"| {fmt(r['ensemble_score_gpt'], 2)} "
            f"| {fmt(r['ensemble_score_gemini'], 2)} "
            f"| {fmt(r['ensemble_mean'], 2)} "
            f"| {fmt(r['wall_time_seconds'], 0)} "
            f"| {fmt(r['n_retries'], 0)} "
            f"| {r['status']} |\n"
        )
    return header + rows_md


def build_changes_overview(repo_root: Path, variants: list[str]) -> str:
    lines = [
        "# CHANGES_OVERVIEW — MAGMA v2 Ablation Studies",
        "",
        "Summary of all 6 variant modifications. See each variant's `CHANGES.md` for full detail.",
        "",
    ]
    for variant in variants:
        changes_path = repo_root / variant / "CHANGES.md"
        lines.append(f"## {VARIANT_LABELS.get(variant, variant)}")
        if changes_path.exists():
            lines.append("")
            lines.append(changes_path.read_text(encoding="utf-8", errors="replace").strip())
        else:
            lines.append(f"_CHANGES.md not found at {changes_path}_")
        lines.append("")
    return "\n".join(lines) + "\n"


def main(ablation_root: str, repo_root: str, variants: list[str]) -> None:
    ab = Path(ablation_root)
    rr = Path(repo_root)

    wall_times = load_wall_times(ab)
    ensemble = load_ensemble_scores(ab)

    rows = [gather_row(v, ab, wall_times, ensemble) for v in variants]

    csv_path = ab / "ablation_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV written to: {csv_path}")

    md_table = build_markdown_table(rows)
    md_path = ab / "ablation_summary.md"
    generated_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    md_content = (
        "# MAGMA v2 Ablation Study — cs1 Results (PhysioNet 2019 Sepsis)\n\n"
        f"_Generated: {generated_at}_\n\n"
        "## Summary Table\n\n"
        + md_table
        + "\n### Notes\n"
        "- AUROC and AUPRC are from `model/metrics.json` test split.\n"
        "- TRIPOD (internal) is from `judge_verdict.json` written by the MAGMA TRIPOD judge.\n"
        "- Ensemble scores are LLM-as-a-judge overall scores (0–10 scale) from Claude Opus 4.7, GPT-5.5, and Gemini 3.1 Pro.\n"
        "- `api_cost_usd` is not tracked natively; fill from provider dashboards.\n"
        "- Retries = number of reward-improvement retries (0 = single run).\n"
    )
    md_path.write_text(md_content, encoding="utf-8")
    print(f"Markdown written to: {md_path}")

    changes_md = build_changes_overview(rr, variants)
    changes_path = ab / "CHANGES_OVERVIEW.md"
    changes_path.write_text(changes_md, encoding="utf-8")
    print(f"CHANGES_OVERVIEW written to: {changes_path}")

    print("\n--- Summary preview ---")
    for r in rows:
        print(
            f"  {r['variant']:25s}  auroc={fmt(r['auroc'])}  auprc={fmt(r['auprc'])}  "
            f"tripod={fmt(r['tripod_score_internal'])}  "
            f"ensemble_mean={fmt(r['ensemble_mean'], 2)}  retries={r['n_retries']}  [{r['status']}]"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ablation summary CSV + Markdown.")
    parser.add_argument("--ablation-root", default="/scratch/mma9138/MAGMA/baseline_testing/ablation_runs")
    parser.add_argument("--repo-root", default="/home/user/magma_v2_ablations")
    parser.add_argument("--variants", nargs="+", default=VARIANTS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.ablation_root, args.repo_root, args.variants)
