# A2 — judge_no_rl

## Ablation description
Full MAGMA pipeline including the judge agent, but the judge verdict is never used to drive
retry / reward-improvement loops. The judge runs once at the end of the single attempt, writes
`judge_verdict.json` and `reward.json` for external evaluation, and is then ignored. No
improvement_plan or judge_retry_signals are ever injected into a subsequent orchestrator prompt.

## Files modified

### `magma_v2/main.py`
- Replaced `should_retry = attempt < max_attempts and below_threshold and ...` with
  `should_retry = False` (line ~361).
- Effect: the outer retry loop always terminates after the first attempt regardless of the
  reward value. The judge still runs (the orchestrator calls `call_judge` as normal) and writes
  its artifacts, but nothing reads them to trigger a second run.

## What is removed
- Reward-threshold-driven retry loop.
- `_build_retry_context()` injection (only activated when `attempt > 1`, which never occurs).

## What is preserved
- data_worker, data_worker_validator, model_worker, model_worker_validator, judge
- judge_verdict.json + reward.json are written and available for post-hoc analysis
- HIL clarification loop
- Error repair routing within the single attempt

## How to invoke
```bash
cd /path/to/magma_v2_ablations/a2_judge_no_rl
PYTHONPATH=. python -m magma_v2.main \
  "Build a clinical prediction model using /path/to/data/sepsis_combined.csv. \
   Target is SepsisLabel (1 = sepsis onset, 0 = no sepsis)." \
  --artifact-root /scratch/mma9138/MAGMA/baseline_testing/ablation_runs/a2_judge_no_rl/cs1
```

Run directory output: `ablation_runs/a2_judge_no_rl/cs1/<run_id>/`
Expected artifacts: `DATA_HANDOFF.json`, `MODEL_HANDOFF.json`, `judge_verdict.json`,
  `reward.json`, `final_report.md`
Difference vs baseline: judge verdict is written but never used for retry.
