"""Generate comparison figures for MAGMA v2 ablation study results.

Produces 3 figures (one per metric: AUROC, TRIPOD, Reward), each showing
all ablation variants + full system across case studies as grouped bar charts.

Data can be supplied two ways:
  1. Via --metrics-csv (output of collect_all_metrics.py) — preferred once runs complete.
  2. Hardcoded below — edit the RESULTS dict to fill in placeholders (None).

Usage:
    python eval/plot_ablations.py --out-dir figures/
    python eval/plot_ablations.py --metrics-csv all_metrics.csv --out-dir figures/

To fill in placeholder values: find every `None` in the RESULTS dict below
and replace with the actual float from your results directory or timing TSV.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ---------------------------------------------------------------------------
# Hardcoded results (from screenshot + placeholders for missing/pending values)
# Format: RESULTS[variant][case_study] = {"auroc": float|None, "tripod": float|None, "reward": float|None}
# None = not yet available or failed run (shown as empty bar with hatching)
# "na" = intentionally not applicable (A3: no model worker)
# ---------------------------------------------------------------------------
RESULTS: dict[str, dict[str, dict[str, Any]]] = {
    "A1: No Judge": {
        "CS1": {"auroc": 0.7999, "tripod": None, "reward": None},   # tripod/reward: run judge (see README)
        "CS2": {"auroc": 0.9605, "tripod": None, "reward": None},
        "CS3": {"auroc": 0.6376, "tripod": None, "reward": None},
        "CS4": {"auroc": 0.9085, "tripod": None, "reward": None},
    },
    "A2: Judge\nNo RL": {
        "CS1": {"auroc": 0.8019, "tripod": 0.73684, "reward": 0.76935},
        "CS2": {"auroc": 0.95513, "tripod": 0.73684, "reward": 0.84598},
        "CS3": {"auroc": None,    "tripod": None,    "reward": None},    # FAILED — rerun
        "CS4": {"auroc": 0.87411, "tripod": 0.84210, "reward": 0.85810},
    },
    "A3: No Model\nWorker": {
        "CS1": {"auroc": "na", "tripod": "na", "reward": "na"},
        "CS2": {"auroc": "na", "tripod": "na", "reward": "na"},
        "CS3": {"auroc": "na", "tripod": "na", "reward": "na"},
        "CS4": {"auroc": "na", "tripod": "na", "reward": "na"},
    },
    "A4: No Data\nWorker": {
        "CS1": {"auroc": 0.86328, "tripod": 0.78947, "reward": 0.82637},
        "CS2": {"auroc": None,    "tripod": None,    "reward": None},    # FAILED — rerun
        "CS3": {"auroc": 0.62157, "tripod": 0.78947, "reward": 0.70552},
        "CS4": {"auroc": 0.87545, "tripod": 0.73684, "reward": 0.80615},
    },
    "A5: No\nValidators": {
        "CS1": {"auroc": 0.82459, "tripod": 0.89473, "reward": 0.85966},
        "CS2": {"auroc": 0.92154, "tripod": 0.89473, "reward": 0.90813},
        "CS3": {"auroc": None,    "tripod": None,    "reward": None},    # FAILED — rerun
        "CS4": {"auroc": 0.89485, "tripod": 0.73684, "reward": 0.81584},
    },
    "A6: No HIL": {
        "CS1": {"auroc": 0.83595, "tripod": 0.94736, "reward": 0.89166},
        "CS2": {"auroc": 0.95833, "tripod": 0.84210, "reward": 0.90021},
        "CS3": {"auroc": 0.53752, "tripod": 0.94736, "reward": 0.74244},
        "CS4": {"auroc": 0.90141, "tripod": 0.89473, "reward": 0.89807},
    },
    "MAGMA\n(Full)": {
        "CS1": {"auroc": None, "tripod": None, "reward": None},  # PLACEHOLDER — replace when run completes
        "CS2": {"auroc": None, "tripod": None, "reward": None},  # FAILED — rerun run_full_cs2.sh
        "CS3": {"auroc": None, "tripod": None, "reward": None},  # FAILED — rerun run_full_cs3.sh
        "CS4": {"auroc": None, "tripod": None, "reward": None},  # PLACEHOLDER — replace when run completes
    },
}

CASE_STUDIES = ["CS1", "CS2", "CS3", "CS4"]
VARIANT_KEYS = list(RESULTS.keys())

# Color palette — fire/ember tones from project color scheme
COLORS = [
    "#bc2e16",  # A1 — dark red
    "#6f7073",  # A2 — slate gray
    "#af6f70",  # A3 — muted rose
    "#f64000",  # A4 — bright ember orange
    "#565d5d",  # A5 — dark gray
    "#e66d1d",  # A6 — warm orange
    "#261e1b",  # MAGMA (full) — near black, bold
]


def load_from_csv(csv_path: Path) -> None:
    """Overwrite RESULTS entries from collect_all_metrics.py CSV output."""
    variant_map = {
        "a1_no_judge":       "A1: No Judge",
        "a2_judge_no_rl":    "A2: Judge\nNo RL",
        "a3_no_model_worker":"A3: No Model\nWorker",
        "a4_no_data_worker": "A4: No Data\nWorker",
        "a5_no_validators":  "A5: No\nValidators",
        "a6_no_hil":         "A6: No HIL",
        "full":              "MAGMA\n(Full)",
    }
    cs_map = {f"cs{i}": f"CS{i}" for i in range(1, 11)}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vkey = variant_map.get(row.get("variant", ""))
            cskey = cs_map.get(row.get("case_study", ""))
            if not vkey or not cskey:
                continue
            if vkey not in RESULTS:
                RESULTS[vkey] = {}
            if cskey not in RESULTS[vkey]:
                RESULTS[vkey][cskey] = {}
            for metric, col in [("auroc", "auroc"), ("tripod", "tripod_score"), ("reward", "reward_score")]:
                raw = row.get(col, "")
                if raw and raw not in ("", "None", "—"):
                    try:
                        RESULTS[vkey][cskey][metric] = float(raw)
                    except ValueError:
                        pass


def _val(v: Any) -> float | None:
    if v == "na" or v is None:
        return None
    return float(v)


def plot_metric(
    metric: str,
    ylabel: str,
    out_path: Path,
    figsize: tuple = (14, 5),
    ylim: tuple = (0.0, 1.05),
) -> None:
    n_cs = len(CASE_STUDIES)
    n_variants = len(VARIANT_KEYS)
    bar_w = 0.11
    group_gap = 0.15
    group_w = n_variants * bar_w + group_gap
    x_centers = np.arange(n_cs) * group_w

    fig, ax = plt.subplots(figsize=figsize)

    for vi, (vkey, color) in enumerate(zip(VARIANT_KEYS, COLORS)):
        offsets = x_centers + (vi - n_variants / 2 + 0.5) * bar_w
        for ci, cs in enumerate(CASE_STUDIES):
            val = _val(RESULTS.get(vkey, {}).get(cs, {}).get(metric))
            is_na = RESULTS.get(vkey, {}).get(cs, {}).get(metric) == "na"
            x = offsets[ci]
            if is_na:
                # Draw tiny marker for "N/A by design"
                ax.bar(x, 0.01, width=bar_w * 0.9, color=color, alpha=0.2, edgecolor="none")
                ax.text(x, 0.02, "N/A", ha="center", va="bottom", fontsize=5, color=color, alpha=0.7)
            elif val is None:
                # Missing/failed — hatched placeholder
                ax.bar(x, 0.05, width=bar_w * 0.9, color="none", edgecolor=color,
                       linewidth=0.8, hatch="///", alpha=0.6)
                ax.text(x, 0.06, "?", ha="center", va="bottom", fontsize=6, color=color)
            else:
                lw = 1.5 if "MAGMA" in vkey else 0.5
                ec = "black" if "MAGMA" in vkey else color
                ax.bar(x, val, width=bar_w * 0.9, color=color, edgecolor=ec,
                       linewidth=lw, alpha=0.88 if "MAGMA" not in vkey else 1.0)

    ax.set_xticks(x_centers)
    ax.set_xticklabels(CASE_STUDIES, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_ylim(ylim)
    ax.set_xlim(-group_w * 0.3, x_centers[-1] + group_w * 0.7)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_patches = [
        mpatches.Patch(facecolor=c, label=v.replace("\n", " "), alpha=0.88)
        for v, c in zip(VARIANT_KEYS, COLORS)
    ]
    ax.legend(handles=legend_patches, fontsize=8, ncol=4,
              loc="upper right", framealpha=0.9, edgecolor="0.8")

    # Legend for placeholder symbols
    na_patch = mpatches.Patch(facecolor="white", edgecolor="gray", hatch="///", label="Failed / pending (?)")
    ax.legend(
        handles=legend_patches + [na_patch],
        fontsize=8, ncol=4, loc="lower right", framealpha=0.9, edgecolor="0.8",
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-csv", default=None,
                        help="Path to collect_all_metrics.py CSV output (optional)")
    parser.add_argument("--out-dir", default="figures")
    parser.add_argument("--case-studies", nargs="+", default=CASE_STUDIES)
    args = parser.parse_args()

    if args.metrics_csv:
        print(f"Loading metrics from CSV: {args.metrics_csv}")
        load_from_csv(Path(args.metrics_csv))

    # Restrict to requested case studies
    global CASE_STUDIES
    CASE_STUDIES = [c.upper() if not c.startswith("CS") else c for c in args.case_studies]
    CASE_STUDIES = [f"CS{c}" if c.isdigit() else c for c in CASE_STUDIES]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    plot_metric("auroc",  "AUROC",         out / "fig_auroc.png")
    plot_metric("tripod", "TRIPOD Score",   out / "fig_tripod.png")
    plot_metric("reward", "Reward",         out / "fig_reward.png")

    print(f"\nAll figures written to: {out}/")
    print("\nTo fill in placeholder values (None), edit the RESULTS dict in this file")
    print("or pass --metrics-csv <path> to load from collect_all_metrics.py output.")


if __name__ == "__main__":
    main()
