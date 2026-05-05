#!/bin/bash
# =============================================================================
# MAGMA v2 Ablation Studies — Case Study 7 (cs7: 30-day readmission — MIMIC-Note + MIMIC-IV)
# Runs all 6 ablation variants sequentially in a single SLURM job.
# =============================================================================
#SBATCH --job-name=magma_ablations_cs7
#SBATCH --output=/scratch/mma9138/MAGMA/baseline_testing/ablation_runs/logs/ablations_cs7_%j.out
#SBATCH --error=/scratch/mma9138/MAGMA/baseline_testing/ablation_runs/logs/ablations_cs7_%j.err
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

# cs7 canonical prompt (clean run — multimodal MIMIC-Note discharge + MIMIC-IV EHR)
# 30-day readmission: derive from core/admissions.csv — next admission within 30 days,
# excluding same-day transfers (admittime of next admission > dischtime of current by > 0 days).
CS7_PROMPT="Build a multimodal 30-day hospital readmission prediction model using MIMIC-IV \
discharge notes and EHR data. \
\
NOTES data (training/validation): ${EXPOSED_MIMIC_NOTE_DISCHARGE} \
  (MIMIC-IV discharge summaries; RFC-4180 CSV, text field has embedded newlines — \
   always use pandas.read_csv, never wc -l or awk). \
\
EHR data (training/validation): ${EXPOSED_MIMIC_IV} \
  Key tables: core/admissions.csv (for label derivation — see below), \
              core/patients.csv, hosp/diagnoses_icd.csv, hosp/labevents.csv (large ~13 GB — chunk). \
  DO NOT read from /scratch/fs999/shamoutlab/data/mimic-iv-extracted/ — that path is deprecated. \
\
Label derivation (30-day readmission): \
  For each admission in core/admissions.csv, compute whether the patient (subject_id) \
  had a subsequent admission within 30 days of discharge (dischtime). \
  Exclude same-day transfers: next admittime must be > dischtime by at least 1 day. \
  Exclude admissions where the patient died in-hospital (hospital_expire_flag == 1). \
  All admissions for a patient are in the same split bucket (exposed or held_out), \
  so window logic is self-consistent. \
\
Join key: subject_id (and hadm_id for notes). \
Patient identifier: subject_id. \
There is NO pre-defined split column — create your own patient-level train/val split \
from the exposed data using seed=42 (80/20 stratified on readmission). \
\
Model: multimodal (discharge note text features + EHR tabular features). \
Evaluate AUROC and AUPRC for 30-day readmission on the validation set. \
Perform TRIPOD assessment, calibrate, check for text/temporal leakage, and write a final report. \
FINAL EVALUATION (touch EXACTLY ONCE, after all training and model selection is complete): \
  evaluate the final model on: \
    Notes held-out: ${HELD_OUT_MIMIC_NOTE_DISCHARGE} \
    EHR held-out: ${HELD_OUT_MIMIC_IV} \
  Report test AUROC and AUPRC separately from validation metrics. \
Log the timestamp of your single held-out access to held_out_access.log in the artifact directory. \
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

TIMING_FILE="${ABLATION_ROOT}/ablation_wall_times_cs7.tsv"
echo -e "variant\tstart_epoch\tend_epoch\twall_seconds\tstatus" > "${TIMING_FILE}"

run_variant() {
    local VARIANT="$1"
    local VARIANT_DIR="${REPO_ROOT}/${VARIANT}"
    local ARTIFACT_ROOT="${ABLATION_ROOT}/${VARIANT}/cs7"

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
            "${CS7_PROMPT}" \
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

echo "MAGMA ablation run (CS7 — clean) started at $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "Exposed notes: ${EXPOSED_MIMIC_NOTE_DISCHARGE}"
echo "Exposed MIMIC-IV: ${EXPOSED_MIMIC_IV}"
echo "Held-out notes: ${HELD_OUT_MIMIC_NOTE_DISCHARGE}"
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
