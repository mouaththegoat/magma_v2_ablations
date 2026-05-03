# eval/ — Phase 4 Evaluation Pipeline

## Scripts

| Script | Purpose |
|--------|---------|
| `run_internal_judge.py` | Runs the MAGMA TRIPOD judge (from a1_no_judge) on each ablation run dir. Writes `judge_verdict.json` + `reward.json` in place. |
| `assemble_ablation_traces.py` | Assembles one compact JSON trace per variant for the LLM ensemble judges. Outputs to `llm_ensemble_eval/`. |
| `judge_ablations_claude.py` | Calls Claude Opus 4.7 on each assembled trace. Writes `judge_claude_scores.json`. |
| `judge_ablations_gpt.py` | Calls GPT-5.5 on each assembled trace. Writes `judge_gpt_scores.json`. |
| `judge_ablations_gemini.py` | Calls Gemini 3.1 Pro on each assembled trace. Writes `judge_gemini_scores.json`. |
| `compute_ablation_scores.py` | Aggregates the three judge files into `ensemble_scores.json` with per-variant means. |
| `build_ablation_summary.py` | Reads all artifacts and writes `ablation_summary.csv`, `ablation_summary.md`, `CHANGES_OVERVIEW.md`. |
| `_judge_common.py` | Shared system prompt, user prompt builder, and response parser for LLM judges. |
| `run_full_eval_pipeline.sh` | Chains all 5 steps above in order. |

## Prerequisites

```bash
pip install anthropic openai google-generativeai
```

API keys needed (in `/scratch/mma9138/MAGMA/.env` or environment):
- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`
- `GOOGLE_API_KEY`

## Run order

```bash
# After all 6 MAGMA ablation runs complete:
cd /home/user/magma_v2_ablations
bash eval/run_full_eval_pipeline.sh

# Or step by step (internal judge needs PYTHONPATH pointing at a1_no_judge):
PYTHONPATH=a1_no_judge python eval/run_internal_judge.py
python eval/assemble_ablation_traces.py
python eval/judge_ablations_claude.py
python eval/judge_ablations_gpt.py
python eval/judge_ablations_gemini.py
python eval/compute_ablation_scores.py
python eval/build_ablation_summary.py
```

## Output directory structure

```
ablation_runs/
  a1_no_judge/cs1/runs/<ts>/          ← MAGMA run artifacts
  a2_judge_no_rl/cs1/runs/<ts>/
  a3_no_model_worker/cs1/runs/<ts>/
  a4_no_data_worker/cs1/runs/<ts>/
  a5_no_validators/cs1/runs/<ts>/
  a6_no_hil/cs1/runs/<ts>/
  llm_ensemble_eval/
    a1_no_judge_cs1_trace.json        ← assembled trace per variant
    ...
    judge_claude_scores.json          ← raw Claude verdicts
    judge_gpt_scores.json
    judge_gemini_scores.json
    ensemble_scores.json              ← aggregated means
  ablation_summary.csv                ← one row per variant
  ablation_summary.md                 ← paper-ready markdown table
  CHANGES_OVERVIEW.md                 ← all 6 CHANGES.md concatenated
  ablation_wall_times.tsv             ← written by SLURM script
  internal_judge_summary.json         ← written by run_internal_judge.py
```

## CSV columns

```
variant, auroc, auprc, tripod_score_internal,
ensemble_score_claude, ensemble_score_gpt, ensemble_score_gemini, ensemble_mean,
wall_time_seconds, api_cost_usd, n_retries, status
```

`api_cost_usd` is not tracked natively by MAGMA; fill from provider dashboards manually.
