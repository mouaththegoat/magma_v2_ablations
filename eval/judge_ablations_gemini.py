"""Phase 4 — LLM-as-a-judge: Gemini 3.1 Pro.

Reads assembled traces from llm_ensemble_eval/, calls Gemini 3.1 Pro for each, writes scores.

Usage:
    python eval/judge_ablations_gemini.py \
        --ablation-root /scratch/mma9138/MAGMA/baseline_testing/ablation_runs
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import google.generativeai as genai

from _judge_common import JUDGE_SYSTEM_PROMPT, build_judge_user_prompt, parse_judge_response

MODEL = "gemini-3.1-pro"
VARIANTS = [
    "a1_no_judge",
    "a2_judge_no_rl",
    "a3_no_model_worker",
    "a4_no_data_worker",
    "a5_no_validators",
    "a6_no_hil",
]


def judge_trace(model: genai.GenerativeModel, trace: dict, variant: str) -> dict:
    user_prompt = build_judge_user_prompt(trace)
    full_prompt = JUDGE_SYSTEM_PROMPT + "\n\n" + user_prompt
    for attempt in range(3):
        try:
            response = model.generate_content(
                full_prompt,
                generation_config=genai.GenerationConfig(
                    max_output_tokens=2048,
                    response_mime_type="application/json",
                ),
            )
            raw_text = response.text or ""
            result = parse_judge_response(raw_text, variant)
            result["model"] = MODEL
            if hasattr(response, "usage_metadata"):
                result["input_tokens"] = getattr(response.usage_metadata, "prompt_token_count", None)
                result["output_tokens"] = getattr(response.usage_metadata, "candidates_token_count", None)
            return result
        except Exception as exc:
            err = str(exc)
            if "429" in err or "quota" in err.lower() or "rate" in err.lower():
                wait = 2 ** (attempt + 2)
                print(f"  Rate limit — retrying in {wait}s")
                time.sleep(wait)
            else:
                return {"variant": variant, "error": err, "model": MODEL}
    return {"variant": variant, "error": "max retries exceeded", "model": MODEL}


def main(ablation_root: str, variants: list[str], case_study: str = "cs1") -> None:
    ensemble_dir = Path(ablation_root) / "llm_ensemble_eval"
    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))
    gemini_model = genai.GenerativeModel(MODEL)
    results = []

    for variant in variants:
        trace_path = ensemble_dir / f"{variant}_{case_study}_trace.json"
        if not trace_path.exists():
            print(f"[{variant}] Trace not found at {trace_path} — skipping.")
            continue

        with open(trace_path) as f:
            trace = json.load(f)

        print(f"[{variant}] Calling {MODEL} ...")
        verdict = judge_trace(gemini_model, trace, variant)
        results.append(verdict)

        score = verdict.get("overall_score", "?")
        print(f"  overall_score={score}  model_quality={verdict.get('model_quality_score', '?')}  clinical_reporting={verdict.get('clinical_reporting_score', '?')}")

    out_path = ensemble_dir / f"judge_gemini_scores_{case_study}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nScores written to: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gemini 3.1 Pro judge for ablation traces.")
    parser.add_argument("--ablation-root", default="/scratch/mma9138/MAGMA/baseline_testing/ablation_runs")
    parser.add_argument("--variants", nargs="+", default=VARIANTS)
    parser.add_argument("--case-study", default="cs1")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.ablation_root, args.variants, args.case_study)
