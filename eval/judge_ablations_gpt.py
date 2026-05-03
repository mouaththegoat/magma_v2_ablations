"""Phase 4 — LLM-as-a-judge: GPT-5.5.

Reads assembled traces from llm_ensemble_eval/, calls GPT-5.5 for each, writes scores.

Usage:
    python eval/judge_ablations_gpt.py \
        --ablation-root /scratch/mma9138/MAGMA/baseline_testing/ablation_runs
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import openai

from _judge_common import JUDGE_SYSTEM_PROMPT, build_judge_user_prompt, parse_judge_response

MODEL = "gpt-5.5"
VARIANTS = [
    "a1_no_judge",
    "a2_judge_no_rl",
    "a3_no_model_worker",
    "a4_no_data_worker",
    "a5_no_validators",
    "a6_no_hil",
]


def judge_trace(client: openai.OpenAI, trace: dict, variant: str) -> dict:
    user_prompt = build_judge_user_prompt(trace)
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                max_tokens=2048,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
            raw_text = response.choices[0].message.content or ""
            result = parse_judge_response(raw_text, variant)
            result["model"] = MODEL
            result["input_tokens"] = response.usage.prompt_tokens if response.usage else None
            result["output_tokens"] = response.usage.completion_tokens if response.usage else None
            return result
        except openai.RateLimitError:
            wait = 2 ** (attempt + 2)
            print(f"  Rate limit — retrying in {wait}s")
            time.sleep(wait)
        except Exception as exc:
            return {"variant": variant, "error": str(exc), "model": MODEL}
    return {"variant": variant, "error": "max retries exceeded", "model": MODEL}


def main(ablation_root: str, variants: list[str]) -> None:
    ensemble_dir = Path(ablation_root) / "llm_ensemble_eval"
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
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

    out_path = ensemble_dir / "judge_gpt_scores.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nScores written to: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPT-5.5 judge for ablation traces.")
    parser.add_argument("--ablation-root", default="/scratch/mma9138/MAGMA/baseline_testing/ablation_runs")
    parser.add_argument("--variants", nargs="+", default=VARIANTS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.ablation_root, args.variants)
