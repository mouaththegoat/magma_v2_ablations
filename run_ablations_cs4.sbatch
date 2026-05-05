#!/bin/bash
# =============================================================================
# MAGMA v2 Ablation Studies — Case Study 4 (cs4: PTB-XL ECG arrhythmia classification)
# Runs all 6 ablation variants sequentially in a single SLURM job.
# =============================================================================
#SBATCH --job-name=magma_ablations_cs4
#SBATCH --output=/scratch/mma9138/MAGMA/baseline_testing/ablation_runs/logs/ablations_cs4_%j.out
#SBATCH --error=/scratch/mma9138/MAGMA/baseline_testing/ablation_runs/logs/ablations_cs4_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
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

# cs4 canonical prompt (clean run)
# Key: DO NOT use strat_fold column for splitting — use the on-disk exposed/held_out partition.
CS4_PROMPT="Build a multi-label ECG arrhythmia classification model. \
Training/validation data directory: ${EXPOSED_PTBXL} \
  (PTB-XL dataset). \
Files present: ptbxl_database.csv (patient-level metadata, patient_id column), \
  records100/ (100 Hz WFDB signals), records500/ (500 Hz WFDB signals), \
  scp_statements.csv (SCP code → human-readable label mapping, needed for label derivation). \
Patient identifier column: patient_id. \
Target: multi-label arrhythmia classification using the SCP codes from ptbxl_database.csv \
  joined with scp_statements.csv (focus on rhythm and form statements; \
  derive binary labels for the 5 super-classes: NORM, MI, STTC, CD, HYP). \
IMPORTANT: the strat_fold column in ptbxl_database.csv is the original PTB-XL stratification \
  and must NOT be used for train/val/test splitting in this run. \
  Create your own patient-level train/val split from the exposed data using seed=42 (80/20). \
Use wfdb to load signals; prefer 100 Hz records for speed. \
Use a 1-D ResNet or CNN-LSTM backbone. \
Evaluate macro-averaged AUROC and per-class AUROC on the validation set. \
Perform TRIPOD assessment, calibrate, check for leakage, and write a final report. \
FINAL EVALUATION (touch EXACTLY ONCE, after all training and model selection is complete): \
  evaluate the final model on ${HELD_OUT_PTBXL} (same directory structure). \
  Report test AUROC separately from validation metrics. \
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

TIMING_FILE="${ABLATION_ROOT}/ablation_wall_times_cs4.tsv"
echo -e "variant\tstart_epoch\tend_epoch\twall_seconds\tstatus" > "${TIMING_FILE}"

run_variant() {
    local VARIANT="$1"
    local VARIANT_DIR="${REPO_ROOT}/${VARIANT}"
    local ARTIFACT_ROOT="${ABLATION_ROOT}/${VARIANT}/cs4"

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
            "${CS4_PROMPT}" \
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

echo "MAGMA ablation run (CS4 — clean) started at $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "Exposed data: ${EXPOSED_PTBXL}"
echo "Held-out data: ${HELD_OUT_PTBXL}"
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
