# A5 — no_validators

## Ablation description
Both data_worker_validator and model_worker_validator are removed from the pipeline.
All other agents (data_worker, model_worker, judge, HIL, reward/retry loop) are intact.
Tests whether validator agents improve output quality vs. simply adding latency and cost.

## Files modified

### `magma_v2/agents/orchestrator.py`
- Removed `call_data_worker_validator` from orchestrator `tools=[...]` (line ~831).
- Removed `call_model_worker_validator` from orchestrator `tools=[...]` (line ~834).
- Updated ORCHESTRATOR_PROMPT step 3: skip data_worker_validator; go straight from
  DATA_HANDOFF check to step 4 (literature / model).
- Updated ORCHESTRATOR_PROMPT step 6: skip model_worker_validator; go straight from
  MODEL_HANDOFF to call_judge.

## What is removed
- data_worker_validator execution; DATA_VALIDATION_REPORT.json will not be written.
- model_worker_validator execution; MODEL_VALIDATION_REPORT.json will not be written.

## What is preserved
- data_worker, model_worker, judge, HIL, reward/retry loop, error repair routing
- All handoffs (DATA_HANDOFF.json, MODEL_HANDOFF.json) still required
- judge still evaluates correctness, leakage, and TRIPOD compliance

## How to invoke
```bash
cd /path/to/magma_v2_ablations/a5_no_validators
PYTHONPATH=. python -m magma_v2.main \
  "Build a clinical prediction model using /path/to/data/sepsis_combined.csv. \
   Target is SepsisLabel (1 = sepsis onset, 0 = no sepsis)." \
  --artifact-root /scratch/mma9138/MAGMA/baseline_testing/ablation_runs/a5_no_validators/cs1
```

Run directory output: `ablation_runs/a5_no_validators/cs1/<run_id>/`
Expected artifacts: `DATA_HANDOFF.json`, `MODEL_HANDOFF.json`, `judge_verdict.json`,
  `reward.json`, `final_report.md`
Missing vs baseline: `DATA_VALIDATION_REPORT.json`, `MODEL_VALIDATION_REPORT.json`
