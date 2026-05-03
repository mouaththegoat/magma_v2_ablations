# A6 — no_hil

## Ablation description
The human-in-the-loop (HIL) clarification system is disabled entirely. The pipeline runs
with whatever information is in the user prompt, making its own inferences for any ambiguities
instead of pausing to ask. All other agents (data_worker + validators, model_worker +
validators, judge, reward/retry loop) are intact.

## Files modified

### `magma_v2/main.py`
- Replaced the inner `while True` HIL loop (lines ~281–351) with a direct orchestrator call.
  - Removed: `await plan_initial_human_clarification(...)` call.
  - Removed: `ask_human_question(assessment, ...)` pre-orchestrator block.
  - Removed: post-orchestrator `pending_human_input` check and `write_human_input_response`.
  - `human_answers` is always `""` (no prior answers injected).
- Removed unused imports: `ask_human_question`, `plan_initial_human_clarification`,
  `human_response_as_prompt_text`, `pending_human_input`, `write_human_input_response`.

### `magma_v2/agents/orchestrator.py`
- Removed `ask_human_question` from orchestrator `tools=[...]` list (line ~834).
- Updated ORCHESTRATOR_PROMPT HIL policy: replaced the ask-the-human instructions with an
  explicit prohibition on asking for clarification; instructs the orchestrator to proceed on
  best-effort inference and document any ambiguities as limitations in the final report.

## What is removed
- `plan_initial_human_clarification` pre-flight check (HIL planner agent).
- `ask_human_question` orchestrator tool (cannot pause for human input).
- All post-orchestrator human-input collection loops.

## What is preserved
- data_worker, data_worker_validator, model_worker, model_worker_validator, judge
- reward/retry loop with full judge feedback
- Error repair routing (decide_repair_route) — but human-blocking errors will not be routed
  to a human; the orchestrator must make a best-effort decision or declare terminal failure.

## Known constraints
- Any ambiguity in the user prompt that would normally trigger a clarification question will
  now cause the orchestrator to infer or fail. For well-specified prompts like cs1, this
  should be transparent.

## How to invoke
```bash
cd /path/to/magma_v2_ablations/a6_no_hil
PYTHONPATH=. python -m magma_v2.main \
  "Build a clinical prediction model using /path/to/data/sepsis_combined.csv. \
   Target is SepsisLabel (1 = sepsis onset, 0 = no sepsis)." \
  --artifact-root /scratch/mma9138/MAGMA/baseline_testing/ablation_runs/a6_no_hil/cs1
```

Run directory output: `ablation_runs/a6_no_hil/cs1/<run_id>/`
Expected artifacts: identical to baseline — DATA_HANDOFF.json, MODEL_HANDOFF.json,
  judge_verdict.json, reward.json, final_report.md
Missing vs baseline: HUMAN_INPUT_REQUEST.json (never written)
