"""Phase 4 — aggregate LLM ensemble scores across the three judge outputs.

Reads judge_claude_scores.json, judge_gpt_scores.json, judge_gemini_scores.json from
llm_ensemble_eval/ and produces ensemble_scores.json with per-variant means.

Usage:
    python eval/compute_ablation_scores.py \
        --ablation-root /scratch/mma9138/MAGMA/baseline_testing/ablation_runs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCORE_KEYS = [
    "data_quality_score",
    "model_quality_score",
    "clinical_reporting_score",
    "pipeline_robustness_score",
    "overall_score",
]

JUDGE_FILES = {
    "claude": "judge_claude_scores.json",
    "gpt": "judge_gpt_scores.json",
    "gemini": "judge_gemini_scores.json",
}

VARIANTS = [
    "a1_no_judge",
    "a2_judge_no_rl",
    "a3_no_model_worker",
    "a4_no_data_worker",
    "a5_no_validators",
    "a6_no_hil",
]


def _load_scores(ensemble_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """Returns {judge_name: {variant: score_dict}}."""
    all_scores: dict[str, dict[str, dict[str, Any]]] = {}
    for judge_name, filename in JUDGE_FILES.items():
        path = ensemble_dir / filename
        if not path.exists():
            print(f"  [{judge_name}] Score file not found: {path}")
            all_scores[judge_name] = {}
            continue
        with open(path) as f:
            records = json.load(f)
        all_scores[judge_name] = {r["variant"]: r for r in records if isinstance(r, dict) and "variant" in r}
    return all_scores


def _mean_or_none(values: list[float | None]) -> float | None:
    valid = [v for v in values if v is not None and isinstance(v, (int, float))]
    if not valid:
        return None
    return round(sum(valid) / len(valid), 4)


def compute(ablation_root: str, variants: list[str]) -> None:
    ensemble_dir = Path(ablation_root) / "llm_ensemble_eval"
    all_scores = _load_scores(ensemble_dir)

    results: list[dict[str, Any]] = []
    for variant in variants:
        row: dict[str, Any] = {"variant": variant}

        for judge_name in JUDGE_FILES:
            judge_data = all_scores.get(judge_name, {}).get(variant, {})
            row[f"overall_score_{judge_name}"] = judge_data.get("overall_score")
            row[f"model_quality_{judge_name}"] = judge_data.get("model_quality_score")
            row[f"clinical_reporting_{judge_name}"] = judge_data.get("clinical_reporting_score")
            row[f"data_quality_{judge_name}"] = judge_data.get("data_quality_score")
            row[f"pipeline_robustness_{judge_name}"] = judge_data.get("pipeline_robustness_score")

        for key in SCORE_KEYS:
            values = [
                all_scores.get(judge, {}).get(variant, {}).get(key)
                for judge in JUDGE_FILES
            ]
            row[f"{key}_ensemble_mean"] = _mean_or_none(values)

        row["ensemble_overall_mean"] = row["overall_score_ensemble_mean"]
        results.append(row)

        print(
            f"  {variant}: "
            f"overall(claude={row.get('overall_score_claude')}, "
            f"gpt={row.get('overall_score_gpt')}, "
            f"gemini={row.get('overall_score_gemini')}) "
            f"mean={row.get('ensemble_overall_mean')}"
        )

    out_path = ensemble_dir / "ensemble_scores.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nEnsemble scores written to: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate LLM ensemble scores.")
    parser.add_argument("--ablation-root", default="/scratch/mma9138/MAGMA/baseline_testing/ablation_runs")
    parser.add_argument("--variants", nargs="+", default=VARIANTS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    compute(args.ablation_root, args.variants)
