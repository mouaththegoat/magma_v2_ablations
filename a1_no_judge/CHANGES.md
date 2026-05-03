# A1 — no_judge

## Ablation description
Full MAGMA pipeline (orchestrator + data_worker + model_worker + validators + HIL) with the judge
agent removed entirely. No TRIPOD verdict, no reward.json, no retry signals fed back.

## Files modified

### `magma_v2/agents/orchestrator.py`
- Removed `call_judge` from the orchestrator `tools=[...]` list (line ~835).
- Removed `from magma_v2.agents.judge import judge` import (no longer needed).

### `magma_v2/main.py`
- Changed `below_threshold` logic (line ~357):
  - Before: `reward is None or reward < REWARD_THRESHOLD`
  - After:   `isinstance(reward, float) and reward < REWARD_THRESHOLD`
  - Effect: when no judge runs (reward=None), `below_threshold=False`, so the retry loop exits
    after the first attempt instead of looping up to MAX_RETRIES times.

## What is removed
- The judge call that writes `judge_verdict.json` and `reward.json`.
- Reward-threshold-driven retry logic (no reward → no retry signal → single run).
- `improvement_plan` / `judge_retry_signals` injection into the next attempt's prompt
  (only activated on attempt ≥ 2, which never occurs here).

## What is preserved
- data_worker, data_worker_validator, model_worker, model_worker_validator
- HIL clarification loop (plan_initial_human_clarification + ask_human_question)
- Error repair routing (decide_repair_route, write_terminal_failure_report)
- final_report.md written by orchestrator

## How to invoke
```bash
cd /path/to/magma_v2_ablations/a1_no_judge
PYTHONPATH=. python -m magma_v2.main \
  "Build a clinical prediction model using /path/to/data/sepsis_combined.csv. \
   Target is SepsisLabel (1 = sepsis onset, 0 = no sepsis)." \
  --artifact-root /scratch/mma9138/MAGMA/baseline_testing/ablation_runs/a1_no_judge/cs1
```

Run directory output: `ablation_runs/a1_no_judge/cs1/<run_id>/`
Expected artifacts: `DATA_HANDOFF.json`, `MODEL_HANDOFF.json`, `final_report.md`
Missing (vs baseline): `judge_verdict.json`, `reward.json`
