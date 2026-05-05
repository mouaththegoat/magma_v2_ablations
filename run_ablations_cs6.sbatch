#!/bin/bash
# =============================================================================
# MAGMA v2 Ablation Studies — Case Study 6 (cs6: In-hospital mortality — MIMIC-IV + MIMIC-ECG)
# Runs all 6 ablation variants sequentially in a single SLURM job.
# =============================================================================
#SBATCH --job-name=magma_ablations_cs6
#SBATCH --output=/scratch/mma9138/MAGMA/baseline_testing/ablation_runs/logs/ablations_cs6_%j.out
#SBATCH --error=/scratch/mma9138/MAGMA/baseline_testing/ablation_runs/logs/ablations_cs6_%j.err
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --account=mma9138
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:1

# =============================================================================
# Configuration
# =============================================================================
REPO_ROOT="/scratch/mma9138/MAGMA/ablations"
source "${REPO_ROOT}/data_paths.sh"
ABLATION_ROOT="/scratch/mma9138/MAGMA/baseline_testing/ablation_runs"
MAX_ATTEMPTS=3
REWARD_LAMBDA=0.5

# cs6 canonical prompt (clean run — multimodal MIMIC-IV EHR + MIMIC-ECG)
# MIMIC-ECG is not yet partitioned — filter at runtime by subject_id from mimic_subjects.csv
# DO NOT use mimic-iv-extracted/ — derive labels from raw MIMIC-IV tables
CS6_PROMPT="Build a multimodal in-hospital mortality prediction model using MIMIC-IV EHR data \
and MIMIC-ECG waveforms. \
\
EHR data (training/validation): ${EXPOSED_MIMIC_IV} \
  Key tables: core/admissions.csv (hospital_expire_flag is the mortality label), \
              core/patients.csv, icu/icustays.csv, icu/chartevents.csv (large ~28 GB — always chunk), \
              hosp/labevents.csv (large ~13 GB — always chunk; chunk_size=500000). \
  Dimension tables (lookup only): icu/d_items.csv, hosp/d_labitems.csv. \
  DO NOT read from /scratch/fs999/shamoutlab/data/mimic-iv-extracted/ — that path is deprecated. \
  Derive in-hospital mortality label from core/admissions.csv (hospital_expire_flag == 1). \
  Use first ICU admission per patient if multiple exist. \
\
ECG data (training/validation — MIMIC-ECG not yet partitioned, filter manually): \
  Source: ${MIMIC_ECG_SRC} \
  Filter to exposed subjects: load ${MIMIC_SUBJECTS_CSV}, keep rows where split IN ('train', 'val'). \
  Join to admissions by subject_id to get ECG waveforms for the relevant ICU stays. \
\
Patient identifier: subject_id. \
Join key: subject_id (MIMIC-IV and MIMIC-ECG share subject_id space). \
There is NO pre-defined split column — create your own patient-level train/val split \
from the exposed data using seed=42 (80/20 stratified on mortality). \
\
Model: multimodal (EHR tabular features + ECG waveform embedding). \
Evaluate AUROC and AUPRC for in-hospital mortality on the validation set. \
Perform TRIPOD assessment, calibrate, check for temporal leakage, and write a final report. \
FINAL EVALUATION (touch EXACTLY ONCE, after all training and model selection is complete): \
  evaluate the final model using: \
    EHR held-out: ${HELD_OUT_MIMIC_IV} \
    ECG held-out: filter ${MIMIC_ECG_SRC} by subjects in ${MIMIC_SUBJECTS_CSV} where split == 'test'. \
  Report test AUROC and AUPRC separately from validation metrics. \
Log the timestamp of your single held-out evaluation to held_out_access.log in the artifact directory. \
Infer all remaining decisions from the data and this prompt — do not pause for clarification."

# =============================================================================
# Environment setup
# =============================================================================
PYTHON="/share/apps/NYUAD5/miniconda/3-4.11.0/envs/jupyter/bin/python"

if [ ! -x "${PYTHON}" ]; then
    echo "ERROR: Python binary not found at ${PYTHON}" >&2
    exit 1
fi
if ! "${PYTHON}" -c "import strands_tools" 2>/dev/null; then
    echo "ERROR: strands_tools not importable from ${PYTHON}" >&2
    exit 1
fi

export PYTHONWARNINGS="ignore"

if [ -f "/scratch/mma9138/MAGMA/.env" ]; then
    set -a
    source /scratch/mma9138/MAGMA/.env
    set +a
fi

mkdir -p "${ABLATION_ROOT}/logs"

echo "Running on host: $(hostname)"
echo "SLURM partition: ${SLURM_JOB_PARTITION:-unknown}"
echo "SLURM job ID: ${SLURM_JOB_ID:-unknown}"

TIMING_FILE="${ABLATION_ROOT}/ablation_wall_times_cs6.tsv"
echo -e "variant\tstart_epoch\tend_epoch\twall_seconds\tstatus" > "${TIMING_FILE}"

run_variant() {
    local VARIANT="$1"
    local VARIANT_DIR="${REPO_ROOT}/${VARIANT}"
    local ARTIFACT_ROOT="${ABLATION_ROOT}/${VARIANT}/cs6"

    echo ""
    echo "============================================================"
    echo "  Starting variant: ${VARIANT}"
    echo "  $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "============================================================"

    mkdir -p "${ARTIFACT_ROOT}"
    local T_START; T_START=$(date +%s)

    (
        cd "${VARIANT_DIR}"
        "${PYTHON}" -m magma_v2.main \
            "${CS6_PROMPT}" \
            --artifact-root "${ARTIFACT_ROOT}" \
            --reward-lambda "${REWARD_LAMBDA}" \
            --max-attempts "${MAX_ATTEMPTS}" \
            --ui compact
    )
    local EXIT_CODE=$?

    local T_END; T_END=$(date +%s)
    local WALL=$(( T_END - T_START ))
    local STATUS="success"
    [ "${EXIT_CODE}" -ne 0 ] && STATUS="failed(exit=${EXIT_CODE})"

    echo -e "${VARIANT}\t${T_START}\t${T_END}\t${WALL}\t${STATUS}" >> "${TIMING_FILE}"
    echo ""
    echo "  Finished ${VARIANT} in ${WALL}s  [${STATUS}]"
    echo "  Artifacts: ${ARTIFACT_ROOT}/runs/"
}

VARIANTS=(a1_no_judge a2_judge_no_rl a3_no_model_worker a4_no_data_worker a5_no_validators a6_no_hil)

echo "MAGMA ablation run (CS6 — clean) started at $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "Exposed MIMIC-IV: ${EXPOSED_MIMIC_IV}"
echo "ECG source (filtered at runtime): ${MIMIC_ECG_SRC}"
echo "Held-out MIMIC-IV: ${HELD_OUT_MIMIC_IV}"
echo "Ablation root: ${ABLATION_ROOT}"
echo ""

for VARIANT in "${VARIANTS[@]}"; do
    run_variant "${VARIANT}"
done

echo ""
echo "============================================================"
echo "  All variants complete at $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "  Wall times written to: ${TIMING_FILE}"
echo "============================================================"
