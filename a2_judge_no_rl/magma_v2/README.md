# MAGMA v2

MAGMA v2 is a code-first clinical AI pipeline. Users provide a natural-language task. Agents inspect the inputs, generate Python scripts, execute them inside stage directories, write structured handoffs, and produce a clinical/TRIPOD-aware judge verdict.

## Run

```bash
uv run python -m magma_v2.main
```

Running without a prompt uses the built-in default request (case study 1).

Default terminal output is compact. Use verbose mode when you want to see streamed agent text:

```bash
uv run python -m magma_v2.main --ui verbose
```

With a custom request:

```bash
uv run python -m magma_v2.main "Build a clinical prediction model from data/all_patients.csv. Infer the target and patient/group identifiers from the file."
```

### Case Study Prompts

Case study 2 (clinical NLP):

```bash
uv run python -m magma_v2.main "Build a binary clinical NLP phenotyping model using /Users/yousef_aljazzazi/Documents/NYUAD/Clinic_Research/magma-cai/data/case_study_2_dummy/all_patients.csv. Target is copd_label (1=has COPD/bronchiectasis, 0=does not). Use split column for train/val/test, prevent patient leakage with patient_id, handle class imbalance, report AUROC and AUPRC on test, and save prediction artifacts."
```

Case study 3 (clinical imaging):

```bash
uv run python -m magma_v2.main "Build a binary chest X-ray classifier for pneumonia using /Users/yousef_aljazzazi/Documents/NYUAD/Clinic_Research/magma-cai/data/case_study_3_dummy/Data_Entry_2017.csv with images under /Users/yousef_aljazzazi/Documents/NYUAD/Clinic_Research/magma-cai/data/case_study_3_dummy/images and images_001/images through images_004/images. Derive label from Finding Labels (Pneumonia present=1), use train_val_list.txt and test_list.txt, split train/val by Patient ID to avoid leakage, handle class imbalance, fine-tune a pretrained CNN, and report test AUROC and AUPRC."
```

Case study 4 (clinical ECG):

```bash
uv run python -m magma_v2.main "Build a binary ECG classifier for arrhythmia using /Users/yousef_aljazzazi/Documents/NYUAD/Clinic_Research/magma-cai/data/case_study_4_dummy/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptbxl_database.csv and WFDB records in records100 and records500. Define arrhythmia from scp_codes (e.g., AFIB/STACH/SARRH positive), use strat_fold with train=1-8, val=9, test=10 while preventing patient leakage via patient_id, handle class imbalance, and report test AUROC and AUPRC with prediction artifacts."
```

## Artifact Layout

```text
artifacts/runs/<run_id>/
  task_input.json
  actions.jsonl
  DATA_HANDOFF.json
  RESEARCH_HANDOFF.json   # optional, non-blocking literature/method evidence
  MODEL_HANDOFF.json
  judge_verdict.json
  reward.json
  final_report.md

  data/
    run_data.py
    optional flat helper modules
    README.md
    generated data files

  model/
    run_model.py
    optional flat helper modules
    README.md
    generated model files

  scratch/
    optional temporary files
```

## Model/Judge Audit Contract

`MODEL_HANDOFF.json` is not just a pointer to a model file. Supported model runs should also declare the generated code, predictions, metrics, calibration report, subgroup report, error examples, feature-drop/leakage report, and execution logs under `model/`.

Generated code is allowed to be multi-file. Agents should keep files flat under
`data/` or `model/`, declare one entrypoint such as `run_data.py` or
`run_model.py`, and record every helper module in `code_artifacts.code_paths`.
This keeps execution simple while avoiding giant single-file scripts for ECG,
imaging, and multimodal workflows.

The judge treats those files as evidence. It reads the actual artifacts and generated code, then compares them against `DATA_HANDOFF.json` and `MODEL_HANDOFF.json`. Missing predictions or metrics, handoff/artifact disagreements, target leakage, hardcoded run paths, substitute handoffs, and incompatible calibration patterns are hard correctness failures. Missing optional analyses can be warnings or provisional results when the unavailable reason is explicit.

When adding a new model artifact, add it to `evaluation_artifacts` or `code_artifacts`, include it in `produced_files`, and add a judge-side consistency check if it can falsify a worker claim.

## Strands Tools vs MAGMA Tools

The data and model workers expose Strands generic tools for routine file and
shell work:

- `file_read`, `file_write`, and `editor` for inspecting and editing run-local files.
- `shell` for exploratory operating-system checks when direct file tools are not enough.

MAGMA-specific tools still own the pipeline contract:

- `execute_data_code` and `execute_model_code` run generated entrypoints inside
  `data/` or `model/` and capture logs, stdout/stderr, and structured recovery
  metadata.
- `write_data_handoff`, `write_model_handoff`, `write_*_error`, and
  `write_grounded_model_handoff` write canonical communication artifacts.
- The orchestrator, recovery state, human-loop artifacts, and judge audit remain
  MAGMA-owned because they encode clinical workflow semantics, not generic file
  operations.

Prefer Strands tools for generic file editing and inspection. Prefer MAGMA tools
when the result must affect stage status, recovery routing, handoff validity, or
judge-readable evidence.

The old MAGMA code-writing/listing helpers are intentionally no longer exposed
to worker agents. Workers should create or edit `data/` and `model/` files with
Strands file tools, then execute the declared entrypoint with MAGMA
`execute_data_code` or `execute_model_code`.

## Literature Worker

`literature_worker` is an optional, non-blocking evidence layer. It is triggered
when the request or data handoff suggests a multimodal, SOTA, literature-backed,
foundation-model, pretrained, or evidence-sensitive modeling decision.

For safety, the literature worker does not receive shell, curl, browser,
Strands file tools, or arbitrary filesystem access. It only has wrapped tools
that:

- read bounded context from `task_input.json` and `DATA_HANDOFF.json`;
- query fixed allowlisted literature metadata providers from a semantic query
  plan;
- write `RESEARCH_HANDOFF.json` and `research_report.md`.

`RESEARCH_HANDOFF.json` centers `method_cards`, recommendations, evidence gaps,
and limitations. Papers are supporting evidence, not the main product. The
model worker may use method-card `method_id` values to ground architecture or
evaluation choices, and the judge warns when a run claims literature/SOTA
grounding without a supporting research handoff. Missing research evidence is
not a hard failure for ordinary baseline runs.

## Reward

The judge writes `reward.json` using:

```text
RT = lambda * AUROC + (1 - lambda) * TRIPODScore
```

Default `lambda` is `0.5`, configurable with `--reward-lambda`.
