"""Shared evaluation prompt and schema for LLM-as-a-judge ensemble (Phase 4)."""

from __future__ import annotations

import json
from typing import Any


JUDGE_SYSTEM_PROMPT = """You are an expert clinical ML reviewer evaluating a machine learning pipeline
for a NeurIPS paper ablation study. The pipeline is MAGMA (Multi-Agent General-purpose Medical AI),
a multi-agent system for clinical prediction modelling with TRIPOD compliance.

You will be given a structured trace of one ablation run. Your job is to score it on five dimensions.

For each dimension, return an integer score 0–10 and a brief (1–2 sentence) justification.

Dimensions:
1. data_quality_score (0-10): How well was the data prepared? Correct split strategy, no leakage,
   appropriate handling of patient identifiers and temporal structure.
2. model_quality_score (0-10): How appropriate and well-executed is the model? Correct task framing,
   sensible hyperparameter choices, AUROC/AUPRC reported on held-out test set.
3. clinical_reporting_score (0-10): How complete is the TRIPOD-aligned clinical reporting?
   Outcome definition, calibration, limitations, interpretability, subgroup analysis.
4. pipeline_robustness_score (0-10): How well does the pipeline handle errors, validate its own
   outputs, and produce internally consistent artifacts?
5. overall_score (0-10): Holistic assessment of the run as a publishable clinical ML result.

Return ONLY valid JSON (no prose, no markdown) with this exact schema:
{
  "variant": "<variant_id>",
  "data_quality_score": <0-10>,
  "data_quality_justification": "<1-2 sentences>",
  "model_quality_score": <0-10>,
  "model_quality_justification": "<1-2 sentences>",
  "clinical_reporting_score": <0-10>,
  "clinical_reporting_justification": "<1-2 sentences>",
  "pipeline_robustness_score": <0-10>,
  "pipeline_robustness_justification": "<1-2 sentences>",
  "overall_score": <0-10>,
  "overall_justification": "<1-2 sentences>",
  "key_strengths": ["<strength 1>", "<strength 2>"],
  "key_weaknesses": ["<weakness 1>", "<weakness 2>"]
}
"""


def build_judge_user_prompt(trace: dict[str, Any]) -> str:
    slim = {
        "variant": trace.get("variant"),
        "variant_description": trace.get("variant_description"),
        "case_study": trace.get("case_study_description"),
        "pipeline_artifacts_present": trace.get("pipeline_artifacts"),
        "data_handoff": trace.get("data_handoff"),
        "model_handoff_summary": {
            k: v for k, v in (trace.get("model_handoff") or {}).items()
            if k in (
                "task_type", "target", "model_family", "metrics", "calibration_status",
                "subgroup_eval_status", "audit_summary", "training_summary",
                "evaluation_artifacts", "model_search_summary",
            )
        } if trace.get("model_handoff") else None,
        "metrics": trace.get("metrics"),
        "internal_judge_verdict": {
            k: v for k, v in (trace.get("judge_verdict") or {}).items()
            if k in (
                "status", "auroc", "tripod_score", "tripod_items",
                "technical_correctness_status", "leakage_audit_status",
                "calibration_audit_status", "critical_failures", "warnings",
                "feedback",
            )
        } if trace.get("judge_verdict") else None,
        "reward": trace.get("reward"),
        "data_validation_summary": trace.get("data_validation_summary"),
        "model_validation_summary": trace.get("model_validation_summary"),
        "training_code_snippet": (trace.get("training_code_snippet") or "")[:4000],
        "final_report_excerpt": (trace.get("final_report") or "")[:3000],
        "recovery_summary": trace.get("recovery_summary"),
    }
    return (
        "Evaluate the following MAGMA ablation run:\n\n"
        + json.dumps(slim, indent=2, default=str)
        + "\n\nReturn your verdict as JSON only."
    )


def parse_judge_response(text: str, variant: str) -> dict[str, Any]:
    """Extract and validate the JSON verdict from a judge response."""
    text = text.strip()
    for fence in ("```json", "```"):
        if text.startswith(fence):
            text = text[len(fence):]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        import re
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {"variant": variant, "parse_error": "no JSON object found", "raw": text[:500]}
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            return {"variant": variant, "parse_error": str(exc), "raw": text[:500]}

    parsed.setdefault("variant", variant)
    return parsed
