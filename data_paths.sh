#!/bin/bash
# =============================================================================
# data_paths.sh — Centralized data path configuration for MAGMA ablation runs
#
# Source this file from any run script:
#   source "$(dirname "$0")/data_paths.sh"
#
# All baselines must:
#   - Use EXPOSED_* paths for training, validation, and model selection ONLY
#   - Touch HELD_OUT_* paths EXACTLY ONCE for final test evaluation
#   - Never read from DEPRECATED_MIMIC_EXTRACTED (old contaminated tree)
#   - Use SPLIT_ASSIGNMENTS CSVs for audit purposes only — never for runtime routing
# =============================================================================

MAGMA_DATA_ROOT="/scratch/fs999/shamoutlab/data/magma"
EXPOSED="${MAGMA_DATA_ROOT}/exposed"
HELD_OUT="${MAGMA_DATA_ROOT}/held_out"

# Audit-only split assignment CSVs (do NOT use at runtime for routing)
SPLIT_ASSIGNMENTS="/scratch/mma9138/MAGMA/dataset_splits/_split_assignments"
MIMIC_SUBJECTS_CSV="${SPLIT_ASSIGNMENTS}/mimic_subjects.csv"
PHYSIONET_SUBJECTS_CSV="${SPLIT_ASSIGNMENTS}/physionet2019_patients.csv"
CHESTXRAY_SUBJECTS_CSV="${SPLIT_ASSIGNMENTS}/chestxray14_patients.csv"
PTBXL_SUBJECTS_CSV="${SPLIT_ASSIGNMENTS}/ptbxl_patients.csv"

# ---------------------------------------------------------------------------
# EXPOSED paths (train + val combined — the only data baselines see)
# ---------------------------------------------------------------------------

# CS1: PhysioNet 2019 Sepsis
# Per-patient .psv files (pipe-separated). Patient ID in filename: pXXXXXX.psv
# Subdirs: training_setA/ and training_setB/ (hospital-of-origin balanced across splits)
EXPOSED_PHYSIONET="${EXPOSED}/physionet_2019"

# CS2, CS7, CS9, CS10: MIMIC-Note discharge summaries
# RFC-4180 CSV; text field contains embedded newlines — always use pandas.read_csv
EXPOSED_MIMIC_NOTE_DISCHARGE="${EXPOSED}/mimic_note_discharge/discharge.csv"

# CS9, CS10 (radiology track): MIMIC-Note radiology reports
EXPOSED_MIMIC_NOTE_RADIOLOGY="${EXPOSED}/mimic_note_radiology/radiology.csv"

# CS3: ChestX-ray14 (NIH)
# Data_Entry_2017.csv + flat images/ directory (NOT images_001/ shards)
# BBox_List_2017.csv also present. train/val/test list files deliberately OMITTED.
EXPOSED_CHESTXRAY14="${EXPOSED}/chestxray14"

# CS4: PTB-XL ECG
# ptbxl_database.csv, records100/, records500/ (preserving NNNNN/ substructure)
# scp_statements.csv (SCP code → label mapping). DO NOT use strat_fold for splitting.
EXPOSED_PTBXL="${EXPOSED}/ptbxl"

# CS5, CS8, CS9: MIMIC-CXR
# CAUTION: flat layout — all images under resized/<dicom_id>.jpg (not p10/pXXX/sNNN/)
# Filtered CSVs: metadata, chexpert, negbio, reports (has embedded newlines), etc.
# Legacy split files (mimic-cxr-2.0.0-split.csv etc.) deliberately OMITTED.
EXPOSED_MIMIC_CXR="${EXPOSED}/mimic_cxr"

# CS5, CS6, CS7: MIMIC-IV raw v1.0
# 19 patient-keyed CSVs across core/, hosp/, icu/ filtered by subject_id.
# 5 dimension/lookup tables symlinked (d_hcpcs, d_icd_*, d_labitems, d_items).
# CRITICAL: do NOT use /scratch/fs999/shamoutlab/data/mimic-iv-extracted/ — deprecated.
EXPOSED_MIMIC_IV="${EXPOSED}/mimic_iv"

# CS6, CS8, CS10: MIMIC-ECG (NOT YET PARTITIONED — use original source)
# Filter ECG records at runtime by subject_id against MIMIC_SUBJECTS_CSV split column
# (exposed = split IN {train, val}). This enforces cross-modality consistency.
MIMIC_ECG_SRC="/scratch/fs999/shamoutlab/data/mimic-iv-ecg"

# ---------------------------------------------------------------------------
# HELD-OUT paths (test set — touch EXACTLY ONCE per baseline, at final eval)
# ---------------------------------------------------------------------------

HELD_OUT_PHYSIONET="${HELD_OUT}/physionet_2019"
HELD_OUT_MIMIC_NOTE_DISCHARGE="${HELD_OUT}/mimic_note_discharge/discharge.csv"
HELD_OUT_MIMIC_NOTE_RADIOLOGY="${HELD_OUT}/mimic_note_radiology/radiology.csv"
HELD_OUT_CHESTXRAY14="${HELD_OUT}/chestxray14"
HELD_OUT_PTBXL="${HELD_OUT}/ptbxl"
HELD_OUT_MIMIC_CXR="${HELD_OUT}/mimic_cxr"
HELD_OUT_MIMIC_IV="${HELD_OUT}/mimic_iv"

# MIMIC-ECG held-out: not a directory but a filtered view — filter from MIMIC_ECG_SRC
# at runtime using MIMIC_SUBJECTS_CSV where split == 'test'
HELD_OUT_MIMIC_ECG_FILTER="${MIMIC_SUBJECTS_CSV}"  # use split==test rows as filter keys

# ---------------------------------------------------------------------------
# DEPRECATED — do not use
# ---------------------------------------------------------------------------
# /scratch/fs999/shamoutlab/data/mimic-iv-extracted/
# This is the pre-baked extracted MIMIC-IV tree that caused the original
# train/val/test contamination. Any code reading from here must be refactored
# to use EXPOSED_MIMIC_IV and re-derive labels/features from raw tables.
DEPRECATED_MIMIC_EXTRACTED="/scratch/fs999/shamoutlab/data/mimic-iv-extracted"

# ---------------------------------------------------------------------------
# Expected record counts (for sanity-checking partition integrity)
# ---------------------------------------------------------------------------
# MIMIC-IV patients.csv:   exposed=324,946  held_out=57,332
# MIMIC-CXR metadata.csv:  exposed=319,083  held_out=58,027
# MIMIC-Note discharge:    exposed=281,934  held_out=49,859
# MIMIC-Note radiology:    exposed=1,967,379 held_out=353,976
# ChestX-ray14 images:     exposed=95,811   held_out=16,309
# PTB-XL ECG records:      exposed=18,493   held_out=3,306
# PhysioNet 2019 .psv:     exposed=34,285   held_out=6,051
