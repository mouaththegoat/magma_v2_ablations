# Path Migration Inventory

Tracks every old data path reference and its replacement under the new partitioned layout
(`exposed/` for training+validation, `held_out/` for final test evaluation only).

Generated as part of the clean-rerun migration. All paths are now centralised in
`data_paths.sh` and sourced by every run script.

---

## Summary

| # | File | Old path | New path | Action |
|---|------|----------|----------|--------|
| 1 | `run_ablations_cs1.sbatch` | `/scratch/mma9138/MAGMA/baseline_testing/case_study_1_input/all_patients.csv` | `${EXPOSED_PHYSIONET}` (PSV directory) | Updated |
| 2 | `run_ablations_cs2.sbatch` | `/scratch/mma9138/MAGMA/baseline_testing/copd_task/copd_task.csv` | `${EXPOSED_MIMIC_NOTE_DISCHARGE}` | Updated |
| 3 | `run_ablations_cs2.sh` | `/scratch/mma9138/MAGMA/baseline_testing/copd_task/copd_task.csv` | `${EXPOSED_MIMIC_NOTE_DISCHARGE}` | Updated |
| 4 | `run_ablations_cs3.sbatch` | *(new file)* | `${EXPOSED_CHESTXRAY14}` | Created |
| 5 | `run_ablations_cs4.sbatch` | *(new file)* | `${EXPOSED_PTBXL}` | Created |
| 6 | `run_ablations_cs5.sbatch` | *(new file)* | `${EXPOSED_MIMIC_CXR}` + `${EXPOSED_MIMIC_IV}` | Created |
| 7 | `run_ablations_cs6.sbatch` | *(new file)* | `${EXPOSED_MIMIC_IV}` + `${MIMIC_ECG_SRC}` (filtered) | Created |
| 8 | `run_ablations_cs7.sbatch` | *(new file)* | `${EXPOSED_MIMIC_NOTE_DISCHARGE}` + `${EXPOSED_MIMIC_IV}` | Created |
| 9 | `run_ablations_cs8.sbatch` | *(new file)* | `${EXPOSED_MIMIC_CXR}` + `${MIMIC_ECG_SRC}` (filtered) | Created |
| 10 | `run_ablations_cs9.sbatch` | *(new file)* | `${EXPOSED_MIMIC_CXR}` + `${EXPOSED_MIMIC_NOTE_RADIOLOGY}` | Created |
| 11 | `run_ablations_cs10.sbatch` | *(new file)* | `${MIMIC_ECG_SRC}` (filtered) + `${EXPOSED_MIMIC_NOTE_DISCHARGE}` | Created |

---

## Detailed Change Log

### CS1: PhysioNet 2019 Sepsis

| Item | Detail |
|------|--------|
| **Old path** | `/scratch/mma9138/MAGMA/baseline_testing/case_study_1_input/all_patients.csv` |
| **Old format** | Single merged CSV with a pre-defined `split` column |
| **New path** | `/scratch/fs999/shamoutlab/data/magma/exposed/physionet_2019/` |
| **New format** | Per-patient `.psv` files (pipe-separated) under `training_setA/` and `training_setB/`. Patient ID in filename: `pXXXXXX.psv`. **No** pre-existing split column — baselines create their own train/val split from `exposed/`. |
| **Held-out** | `/scratch/fs999/shamoutlab/data/magma/held_out/physionet_2019/` — same structure, touched once at final eval |
| **Prompt change** | Removed "use split column directly". Added per-patient PSV format description, seed=42 split instruction, held-out evaluation instruction |

### CS2: MIMIC-Note COPD/bronchiectasis

| Item | Detail |
|------|--------|
| **Old path** | `/scratch/mma9138/MAGMA/baseline_testing/copd_task/copd_task.csv` |
| **Old format** | Single CSV with a pre-defined `split` column (train/val/test) |
| **New path** | `/scratch/fs999/shamoutlab/data/magma/exposed/mimic_note_discharge/discharge.csv` |
| **New format** | Full MIMIC-IV discharge CSV filtered to exposed subjects. RFC-4180 quoting (embedded newlines in text). **No** split column — baselines do their own train/val split. |
| **Held-out** | `/scratch/fs999/shamoutlab/data/magma/held_out/mimic_note_discharge/discharge.csv` |
| **Prompt change** | Removed "A pre-defined split column is present … use it directly". Added embedded-newline warning (always use `pandas.read_csv`), seed=42 split instruction, held-out evaluation instruction |

### CS3: ChestX-ray14 (NIH)

*(New case study — no previous run to compare against)*

| Item | Detail |
|------|--------|
| **Path** | `exposed/chestxray14/` → `held_out/chestxray14/` |
| **Key notes** | Images are flat under `images/` (not `images_001/` shards). No `train_list.txt` / `test_list.txt` — deliberately omitted (would leak official NIH splits). Multi-label: 14 disease categories. |

### CS4: PTB-XL ECG

*(New case study)*

| Item | Detail |
|------|--------|
| **Path** | `exposed/ptbxl/` → `held_out/ptbxl/` |
| **Key notes** | `strat_fold` column in `ptbxl_database.csv` must **not** be used for splitting — it is the original PTB-XL stratification, not the new patient-level partition. Baselines must use `seed=42` split on `exposed/`. |

### CS5: MIMIC-CXR + MIMIC-IV (Pleural Effusion)

*(New case study)*

| Item | Detail |
|------|--------|
| **CXR path** | `exposed/mimic_cxr/` → `held_out/mimic_cxr/` |
| **EHR path** | `exposed/mimic_iv/` → `held_out/mimic_iv/` |
| **Key notes** | CXR images are **flat** under `resized/<dicom_id>.jpg` (not in `p10/pXXX/sNNN/` hierarchy). Labels derived from ICD codes in `hosp/diagnoses_icd.csv`. **Deprecated path** `/scratch/fs999/shamoutlab/data/mimic-iv-extracted/` must never be read. Large tables (`labevents.csv` ~13 GB, `chartevents.csv` ~28 GB) must be chunked. |

### CS6–CS10: New multimodal case studies

*(All new — no legacy comparison data)*

| CS | Modalities | Key constraint |
|----|-----------|----------------|
| CS6 | MIMIC-IV + MIMIC-ECG | ECG not partitioned — filter by `mimic_subjects.csv` split at runtime |
| CS7 | MIMIC-Note discharge + MIMIC-IV | 30-day readmission label derived from `core/admissions.csv` |
| CS8 | MIMIC-CXR + MIMIC-ECG | ECG not partitioned — filter at runtime |
| CS9 | MIMIC-CXR + MIMIC-Note radiology | Cross-modality consistent via shared `subject_id` |
| CS10 | MIMIC-ECG + MIMIC-Note discharge | ECG not partitioned — filter at runtime |

---

## MIMIC-IV: Deprecated Tree Refactor

**DO NOT use:** `/scratch/fs999/shamoutlab/data/mimic-iv-extracted/`

This is the pre-baked MIMIC-IV extraction tree with hard-coded `train/val/test` splits —
the **primary source of the original contamination**. Any baseline that previously read
from this tree must be refactored to:

1. Read raw tables from `exposed/mimic_iv/` (filtered `core/`, `hosp/`, `icu/` CSVs)
2. Re-derive task-specific labels and features at runtime

Affected case studies and required derivations:

| Case Study | Task | Label derivation |
|------------|------|-----------------|
| CS5 | Pleural effusion | ICD codes in `hosp/diagnoses_icd.csv` (ICD-9: 511.x; ICD-10: J90, J91.x) |
| CS6 | In-hospital mortality | `core/admissions.csv` → `hospital_expire_flag == 1` |
| CS7 | 30-day readmission | `core/admissions.csv` → next admission within 30 days of `dischtime`, excluding transfers |

---

## Paths Not Found in Repo Code

The following paths were searched in the codebase but found **only in shell prompts**
(natural language strings passed to MAGMA agents), not in Python module code:

- `/scratch/fs999/shamoutlab/data/mimic-note/`
- `/scratch/fs999/shamoutlab/data/nih_chest_xrays/`
- `/scratch/mma9138/MAGMA/baseline_testing/ptbxl/`
- `/scratch/fs999/shamoutlab/data/physionet.org/files/mimic-cxr-jpg/`
- `/scratch/fs999/shamoutlab/data/physionet.org/files/mimic-iv-1.0/`
- `/scratch/fs999/shamoutlab/data/mimic-iv-ecg/`
- `/scratch/fs999/shamoutlab/data/mimic-iv-extracted/`

**Reason:** MAGMA is an agent-based system. Data loading code is generated at runtime by the
agents from the natural language prompt. The agent receives the dataset path in the prompt
and writes Python code to load it. There is no static data-loading module in the repo to
update — all path migration happens at the prompt level (i.e., the run scripts).

---

## Files with Only Runtime/Infra Paths (not data paths — no change needed)

These files reference `/scratch/mma9138/MAGMA/...` only for:
- SLURM log output paths (`--output`, `--error`)
- `ABLATION_ROOT` (artifact storage, not data)
- `REPO_ROOT` (code location, not data)
- `.env` (API keys)

None of these require changes for the data migration.
