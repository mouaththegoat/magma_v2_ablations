#!/bin/bash
# =============================================================================
# MAGMA v2 Ablation Studies — Case Study 5 (cs5: Pleural effusion — MIMIC-CXR + MIMIC-IV)
# Runs all 6 ablation variants sequentially in a single SLURM job.
# =============================================================================
#SBATCH --job-name=magma_ablations_cs5
#SBATCH --output=/scratch/mma9138/MAGMA/baseline_testing/ablation_runs/logs/ablations_cs5_%j.out
#SBATCH --error=/scratch/mma9138/MAGMA/baseline_testing/ablation_runs/logs/ablations_cs5_%j.err
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

# cs5 canonical prompt (clean run — multimodal CXR + EHR)
# CRITICAL: images are FLAT under mimic_cxr/resized/<dicom_id>.jpg (not p10/pXXX/sNNN/)
# CRITICAL: do NOT use mimic-iv-extracted/ — derive labels from raw MIMIC-IV tables
CS5_PROMPT="Build a multimodal pleural effusion prediction model using MIMIC-CXR and MIMIC-IV EHR data. \
\
IMAGING data (training/validation): ${EXPOSED_MIMIC_CXR} \
  Files: mimic-cxr-2.0.0-metadata.csv (DICOM-level, has subject_id and study_id), \
         mimic-cxr-2.0.0-chexpert.csv (study-level CheXpert labels including Pleural Effusion), \
         mimic_cxr_reports.csv (radiology reports; embedded newlines — use pandas.read_csv), \
         resized/ (flat directory: all images as <dicom_id>.jpg — NOT in p10/pXXX/sNNN/ hierarchy). \
  Legacy split files (mimic-cxr-2.0.0-split.csv etc.) are deliberately absent. \
\
EHR data (training/validation): ${EXPOSED_MIMIC_IV} \
  Key tables: hosp/labevents.csv (large ~13 GB — always chunk; chunk_size=500000), \
              hosp/diagnoses_icd.csv, hosp/admissions.csv, \
              icu/icustays.csv, core/patients.csv. \
  Dimension tables (lookup, no patient data): hosp/d_labitems.csv, hosp/d_icd_diagnoses.csv. \
  DO NOT read from /scratch/fs999/shamoutlab/data/mimic-iv-extracted/ — that path is deprecated. \
  Derive pleural effusion labels from ICD codes in diagnoses_icd.csv \
  (ICD-9: 511.x, 511.9; ICD-10: J90, J91.x) joined by subject_id/hadm_id. \
\
Join key: subject_id (present in both MIMIC-CXR metadata and MIMIC-IV tables). \
Patient identifier: subject_id. \
There is NO pre-defined split column — create your own patient-level train/val split \
using seed=42 (80/20 stratified). Cross-modality consistency is automatic: \
subjects in the exposed CXR bucket are exactly the subjects in the exposed MIMIC-IV bucket. \
\
Model: multimodal (image + EHR tabular features). \
Evaluate AUROC and AUPRC for pleural effusion on the validation set. \
Perform TRIPOD assessment, calibrate, check for leakage, and write a final report. \
FINAL EVALUATION (touch EXACTLY ONCE, after all training and model selection is complete): \
  evaluate the final model on: \
    CXR held-out: ${HELD_OUT_MIMIC_CXR} \
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

TIMING_FILE="${ABLATION_ROOT}/ablation_wall_times_cs5.tsv"
echo -e "variant\tstart_epoch\tend_epoch\twall_seconds\tstatus" > "${TIMING_FILE}"

run_variant() {
    local VARIANT="$1"
    local VARIANT_DIR="${REPO_ROOT}/${VARIANT}"
    local ARTIFACT_ROOT="${ABLATION_ROOT}/${VARIANT}/cs5"

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
            "${CS5_PROMPT}" \
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

echo "MAGMA ablation run (CS5 — clean) started at $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "Exposed CXR: ${EXPOSED_MIMIC_CXR}"
echo "Exposed MIMIC-IV: ${EXPOSED_MIMIC_IV}"
echo "Held-out CXR: ${HELD_OUT_MIMIC_CXR}"
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
