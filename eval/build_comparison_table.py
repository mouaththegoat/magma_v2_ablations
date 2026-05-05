"""Build contaminated-vs-clean AUROC/AUPRC comparison table.

Reads ablation_summary_{cs}.csv from both legacy (contaminated) and clean result
directories, then produces a side-by-side comparison table per (case_study, variant).
Highlights the largest deltas — those are the most contaminated baselines.

Usage:
    python eval/build_comparison_table.py \\
        --legacy-root /scratch/mma9138/MAGMA/baseline_testing/ablation_runs_legacy_contaminated \\
        --clean-root  /scratch/mma9138/MAGMA/baseline_testing/ablation_runs \\
        --case-studies cs1 cs2 cs3 cs4 cs5 cs6 cs7 cs8 cs9 cs10 \\
        --out-dir /scratch/mma9138/MAGMA/baseline_testing/ablation_runs

Outputs:
    <out-dir>/comparison_table.csv
    <out-dir>/comparison_table.md
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

VARIANT_LABELS = {
    "a1_no_judge": "A1: no judge",
    "a2_judge_no_rl": "A2: judge, no RL",
    "a3_no_model_worker": "A3: no model worker",
    "a4_no_data_worker": "A4: no data worker",
    "a5_no_validators": "A5: no validators",
    "a6_no_hil": "A6: no HIL",
}

CS_LABELS = {
    "cs1": "CS1: PhysioNet Sepsis",
    "cs2": "CS2: MIMIC-Note COPD",
    "cs3": "CS3: ChestX-ray14",
    "cs4": "CS4: PTB-XL ECG",
    "cs5": "CS5: CXR+EHR Pleural Effusion",
    "cs6": "CS6: EHR+ECG Mortality",
    "cs7": "CS7: Note+EHR Readmission",
    "cs8": "CS8: CXR+ECG Cardiomegaly",
    "cs9": "CS9: CXR+Note Multimodal",
    "cs10": "CS10: ECG+Note Multimodal",
}


def _read_summary_csv(summary_path: Path) -> dict[str, dict[str, Any]]:
    """Read ablation_summary_{cs}.csv; returns {variant: row_dict}."""
    if not summary_path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    try:
        with open(summary_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                v = row.get("variant", "").strip()
                if v:
                    rows[v] = row
    except Exception:
        pass
    return rows


def _float_or_none(val: Any) -> float | None:
    try:
        return round(float(val), 4)
    except (TypeError, ValueError):
        return None


def _read_run_metrics(root: Path, variant: str, case_study: str) -> dict[str, float | None]:
    """Try to read test AUROC/AUPRC from the agent-generated metrics artifacts.

    For clean runs, the agent writes test-set metrics to the held-out artifact.
    Look in the latest run_dir for model/metrics.json and reward.json.
    """
    artifact_root = root / variant / case_study
    runs_dir = artifact_root / "runs"
    if not runs_dir.exists():
        return {}
    candidates = sorted(
        (d for d in runs_dir.iterdir() if d.is_dir()),
        key=lambda d: d.stat().st_mtime,
    )
    if not candidates:
        return {}
    run_dir = candidates[-1]

    result: dict[str, float | None] = {}

    # Try model/metrics.json: look for test split explicitly
    metrics_path = run_dir / "model" / "metrics.json"
    if not metrics_path.exists():
        metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        try:
            m = json.loads(metrics_path.read_text(encoding="utf-8", errors="replace"))
            payload = m.get("metrics") if isinstance(m.get("metrics"), dict) else m
            for split_name in ("test", "held_out", "heldout"):
                section = (payload or {}).get(split_name, {})
                if isinstance(section, dict) and section:
                    for auroc_key in ("auroc", "roc_auc", "auc"):
                        v = section.get(auroc_key)
                        if v is not None:
                            result["test_auroc"] = _float_or_none(v)
                            break
                    for auprc_key in ("auprc", "average_precision", "ap"):
                        v = section.get(auprc_key)
                        if v is not None:
                            result["test_auprc"] = _float_or_none(v)
                            break
                    if result:
                        break
        except Exception:
            pass

    # Also try reward.json for the val AUROC
    reward_path = run_dir / "reward.json"
    if reward_path.exists():
        try:
            r = json.loads(reward_path.read_text(encoding="utf-8", errors="replace"))
            result["val_auroc_from_reward"] = _float_or_none(r.get("auroc"))
        except Exception:
            pass

    return result


def build_comparison(
    legacy_root: Path,
    clean_root: Path,
    case_studies: list[str],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    for cs in case_studies:
        legacy_csv = legacy_root / f"ablation_summary_{cs}.csv"
        clean_csv = clean_root / f"ablation_summary_{cs}.csv"

        legacy_data = _read_summary_csv(legacy_csv)
        clean_data = _read_summary_csv(clean_csv)

        for variant in VARIANTS:
            legacy_row = legacy_data.get(variant, {})
            clean_row = clean_data.get(variant, {})
            clean_test = _read_run_metrics(clean_root, variant, cs)

            leg_auroc = _float_or_none(legacy_row.get("auroc"))
            leg_auprc = _float_or_none(legacy_row.get("auprc"))

            # For clean runs prefer explicit test metrics, fall back to summary CSV
            cln_auroc = clean_test.get("test_auroc") or _float_or_none(clean_row.get("auroc"))
            cln_auprc = clean_test.get("test_auprc") or _float_or_none(clean_row.get("auprc"))

            delta_auroc: float | None = None
            delta_auprc: float | None = None
            if leg_auroc is not None and cln_auroc is not None:
                delta_auroc = round(leg_auroc - cln_auroc, 4)
            if leg_auprc is not None and cln_auprc is not None:
                delta_auprc = round(leg_auprc - cln_auprc, 4)

            rows.append({
                "case_study": cs,
                "case_study_label": CS_LABELS.get(cs, cs),
                "variant": variant,
                "variant_label": VARIANT_LABELS.get(variant, variant),
                "legacy_auroc": leg_auroc,
                "legacy_auprc": leg_auprc,
                "clean_val_auroc": _float_or_none(clean_row.get("auroc")),
                "clean_test_auroc": cln_auroc,
                "clean_test_auprc": cln_auprc,
                "delta_auroc_legacy_minus_clean": delta_auroc,
                "delta_auprc_legacy_minus_clean": delta_auprc,
                "legacy_status": legacy_row.get("status", "missing"),
                "clean_status": clean_row.get("status", "missing"),
            })

    # Write CSV
    csv_path = out_dir / "comparison_table.csv"
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Comparison CSV written to: {csv_path}")
    else:
        print("WARNING: no rows produced — check that summary CSVs exist in both roots.")

    # Write Markdown
    md_path = out_dir / "comparison_table.md"
    _write_markdown(rows, md_path, legacy_root, clean_root)
    print(f"Comparison Markdown written to: {md_path}")


def _write_markdown(rows: list[dict], path: Path, legacy_root: Path, clean_root: Path) -> None:
    lines: list[str] = []
    lines.append("# Contaminated vs Clean AUROC/AUPRC Comparison")
    lines.append("")
    lines.append(f"**Legacy (contaminated) root:** `{legacy_root}`")
    lines.append(f"**Clean root:** `{clean_root}`")
    lines.append("")
    lines.append(
        "Positive delta = contaminated AUROC was *higher* than clean AUROC "
        "(upward bias from test-set leakage). "
        "Largest positive deltas are the most contaminated baselines."
    )
    lines.append("")

    # Group by case study
    by_cs: dict[str, list[dict]] = {}
    for r in rows:
        by_cs.setdefault(r["case_study"], []).append(r)

    for cs, cs_rows in by_cs.items():
        lines.append(f"## {CS_LABELS.get(cs, cs)}")
        lines.append("")
        header = (
            "| Variant | Legacy AUROC | Clean Test AUROC | Δ AUROC "
            "| Legacy AUPRC | Clean Test AUPRC | Δ AUPRC |"
        )
        sep = "|---|---|---|---|---|---|---|"
        lines.append(header)
        lines.append(sep)

        for r in cs_rows:
            def fmt(v: Any) -> str:
                return f"{v:.4f}" if isinstance(v, float) else "—"

            delta_a = r["delta_auroc_legacy_minus_clean"]
            delta_p = r["delta_auprc_legacy_minus_clean"]

            # Bold large deltas (>= 0.02)
            def fmt_delta(v: Any) -> str:
                if v is None:
                    return "—"
                s = f"{v:+.4f}"
                return f"**{s}**" if abs(v) >= 0.02 else s

            lines.append(
                f"| {r['variant_label']} "
                f"| {fmt(r['legacy_auroc'])} "
                f"| {fmt(r['clean_test_auroc'])} "
                f"| {fmt_delta(delta_a)} "
                f"| {fmt(r['legacy_auprc'])} "
                f"| {fmt(r['clean_test_auprc'])} "
                f"| {fmt_delta(delta_p)} |"
            )
        lines.append("")

    # Summary: largest contamination gaps
    lines.append("## Largest Contamination Gaps (Δ AUROC ≥ 0.02)")
    lines.append("")
    large = [
        r for r in rows
        if isinstance(r.get("delta_auroc_legacy_minus_clean"), float)
        and abs(r["delta_auroc_legacy_minus_clean"]) >= 0.02
    ]
    large.sort(key=lambda r: -abs(r["delta_auroc_legacy_minus_clean"]))

    if large:
        lines.append("| Case Study | Variant | Δ AUROC |")
        lines.append("|---|---|---|")
        for r in large:
            lines.append(
                f"| {r['case_study_label']} "
                f"| {r['variant_label']} "
                f"| {r['delta_auroc_legacy_minus_clean']:+.4f} |"
            )
    else:
        lines.append("*No gaps ≥ 0.02 AUROC detected — or clean results not yet available.*")

    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build contaminated vs clean comparison table.")
    parser.add_argument(
        "--legacy-root",
        default="/scratch/mma9138/MAGMA/baseline_testing/ablation_runs_legacy_contaminated",
        help="Root directory containing old (contaminated) ablation results.",
    )
    parser.add_argument(
        "--clean-root",
        default="/scratch/mma9138/MAGMA/baseline_testing/ablation_runs",
        help="Root directory for new (clean) ablation results.",
    )
    parser.add_argument(
        "--case-studies",
        nargs="+",
        default=["cs1", "cs2", "cs3", "cs4", "cs5", "cs6", "cs7", "cs8", "cs9", "cs10"],
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (defaults to --clean-root).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.clean_root)
    build_comparison(
        legacy_root=Path(args.legacy_root),
        clean_root=Path(args.clean_root),
        case_studies=args.case_studies,
        out_dir=out_dir,
    )
