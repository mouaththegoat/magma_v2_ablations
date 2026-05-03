"""Phase 4 — LLM-as-a-judge: Claude Opus 4.7.

Reads assembled traces from llm_ensemble_eval/, calls Claude Opus 4.7 for each, writes scores.

Usage:
    python eval/judge_ablations_claude.py \
        --ablation-root /scratch/mma9138/MAGMA/baseline_testing/ablation_runs
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import anthropic

from _judge_common import JUDGE_SYSTEM_PROMPT, build_judge_user_prompt, parse_judge_response

MODEL = "claude-opus-4-7"
VARIANTS = [
    "a1_no_judge",
    "a2_judge_no_rl",
    "a3_no_model_worker",
    "a4_no_data_worker",
    "a5_no_validators",
    "a6_no_hil",
]


def judge_trace(client: anthropic.Anthropic, trace: dict, variant: str) -> dict:
    user_prompt = build_judge_user_prompt(trace)
    for attempt in range(3):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=2048,
                system=JUDGE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            raw_text = response.content[0].text if response.content else ""
            result = parse_judge_response(raw_text, variant)
            result["model"] = MODEL
            result["input_tokens"] = response.usage.input_tokens
            result["output_tokens"] = response.usage.output_tokens
            return result
        except anthropic.RateLimitError:
            wait = 2 ** (attempt + 2)
            print(f"  Rate limit — retrying in {wait}s")
            time.sleep(wait)
        except Exception as exc:
            return {"variant": variant, "error": str(exc), "model": MODEL}
    return {"variant": variant, "error": "max retries exceeded", "model": MODEL}


def main(ablation_root: str, variants: list[str]) -> None:
    ensemble_dir = Path(ablation_root) / "llm_ensemble_eval"
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    results = []

    for variant in variants:
        trace_path = ensemble_dir / f"{variant}_cs1_trace.json"
        if not trace_path.exists():
            print(f"[{variant}] Trace not found at {trace_path} — skipping.")
            continue

        with open(trace_path) as f:
            trace = json.load(f)

        print(f"[{variant}] Calling {MODEL} ...")
        verdict = judge_trace(client, trace, variant)
        results.append(verdict)

        score = verdict.get("overall_score", "?")
        print(f"  overall_score={score}  model_quality={verdict.get('model_quality_score', '?')}  clinical_reporting={verdict.get('clinical_reporting_score', '?')}")

    out_path = ensemble_dir / "judge_claude_scores.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nScores written to: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Claude Opus 4.7 judge for ablation traces.")
    parser.add_argument("--ablation-root", default="/scratch/mma9138/MAGMA/baseline_testing/ablation_runs")
    parser.add_argument("--variants", nargs="+", default=VARIANTS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.ablation_root, args.variants)
