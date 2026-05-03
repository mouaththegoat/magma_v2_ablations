# A3 — no_model_worker

## Ablation description
The agentic model_worker is replaced by a single-shot GPT-5.4 call that receives the data
context and writes one complete Python training script. The script is executed directly.
model_worker_validator is also removed because there is no agentic model worker to validate.
All other agents (data_worker, data_worker_validator, judge, HIL) are intact.

## Files modified

### `magma_v2/agents/model_worker.py`
- Added `single_shot_model_worker(instructions, run_dir)` async function at end of file.
  - Creates a no-tool strands Agent using `create_agent_model("model_worker")`.
  - Sends one prompt: writes a complete training script (load data, split, train LightGBM,
    write metrics.json + test_predictions.csv + training_code.py to model/).
  - Executes the script via subprocess; writes model/train_model.log.
  - Returns {"status": "success"|"error", "script_path", "log_path", "model_dir"}.

### `magma_v2/agents/orchestrator.py`
- Added `single_shot_model_worker` to imports from `magma_v2.agents.model_worker`.
- Removed `model_worker_validator` import.
- Replaced `call_model_worker` body: uses `single_shot_model_worker(instructions, run_dir)`
  instead of `_run_agent(model_worker, ...)`.
- Removed entire `call_model_worker_validator` function definition.
- Removed `call_model_worker_validator` from orchestrator `tools=[...]` list.
- Removed unused `build_model_validation_bundle` / `normalize_validation_report` import.
- Updated ORCHESTRATOR_PROMPT step 6: skip validator, proceed directly to call_judge.

## What is removed
- Iterative agentic model_worker (no tool use, no code iteration, no repair loops).
- model_worker_validator and MODEL_VALIDATION_REPORT.json.

## What is preserved
- data_worker, data_worker_validator, judge, HIL, reward/retry loop
- MODEL_HANDOFF.json is still required (produced by the executed script or the grounded
  recovery fallback in _ensure_stage_outcome)
- judge still evaluates artifacts; reward loop still runs

## How to invoke
```bash
cd /path/to/magma_v2_ablations/a3_no_model_worker
PYTHONPATH=. python -m magma_v2.main \
  "Build a clinical prediction model using /path/to/data/sepsis_combined.csv. \
   Target is SepsisLabel (1 = sepsis onset, 0 = no sepsis)." \
  --artifact-root /scratch/mma9138/MAGMA/baseline_testing/ablation_runs/a3_no_model_worker/cs1
```

Run directory output: `ablation_runs/a3_no_model_worker/cs1/<run_id>/`
Expected artifacts: `DATA_HANDOFF.json`, `model/training_code.py`, `model/metrics.json`,
  `model/test_predictions.csv`, `judge_verdict.json`, `reward.json`, `final_report.md`
Missing vs baseline: `MODEL_VALIDATION_REPORT.json`
