# A4 — no_data_worker

## Ablation description
The data_worker stage is skipped entirely. No DATA_HANDOFF.json is written. The model_worker
must infer data loading, splits, target derivation, and leakage controls from the user prompt
and dataset file directly. All other agents (model_worker, validators, judge, HIL, retry loop)
are intact.

## Files modified

### `magma_v2/agents/orchestrator.py`
- Removed `call_data_worker` and `call_data_worker_validator` from orchestrator `tools=[...]`
  (lines ~829-830). The tool functions remain in the file (dead code) to avoid churn; removing
  them from the tools list is sufficient to prevent the orchestrator from calling them.
- Updated ORCHESTRATOR_PROMPT steps 2–3:
  - Step 2: note data_worker is disabled; skip data stage.
  - Step 3: note data_worker_validator is disabled; inspect_run_contract only; DATA_HANDOFF
    absent is expected and not an error.
  - Step 5: instruct model_worker to load and prepare data itself (no DATA_HANDOFF.json).
  - Step 7 routing: data-related errors (path_resolution_error, split_error, etc.) are routed
    back to model_worker instead of call_data_worker.

## What is removed
- data_worker execution; DATA_HANDOFF.json will not be written.
- data_worker_validator execution; DATA_VALIDATION_REPORT.json will not be written.

## What is preserved
- model_worker (must do data loading), model_worker_validator, judge, HIL, reward/retry loop
- Error repair routing remains active but data-owned errors are re-routed to model_worker

## Known constraints
- model_worker must read the dataset path from the user prompt directly.
- Without DATA_HANDOFF.json, judge's data_gate signal will report "unknown/failed" — this is
  expected and will lower the TRIPOD score intentionally.
- plan_initial_human_clarification always runs the planner (DATA_HANDOFF.json never exists to
  short-circuit it) — the cs1 prompt is complete so it should return needs_human_input=False.

## How to invoke
```bash
cd /path/to/magma_v2_ablations/a4_no_data_worker
PYTHONPATH=. python -m magma_v2.main \
  "Build a clinical prediction model using /path/to/data/sepsis_combined.csv. \
   Target is SepsisLabel (1 = sepsis onset, 0 = no sepsis)." \
  --artifact-root /scratch/mma9138/MAGMA/baseline_testing/ablation_runs/a4_no_data_worker/cs1
```

Run directory output: `ablation_runs/a4_no_data_worker/cs1/<run_id>/`
Expected artifacts: `MODEL_HANDOFF.json`, `model/metrics.json`, `judge_verdict.json`,
  `reward.json`, `final_report.md`
Missing vs baseline: `DATA_HANDOFF.json`, `DATA_VALIDATION_REPORT.json`
