#!/bin/bash
# =============================================================================
# MAGMA v2 Ablation Studies — Case Study 3 (cs3: ChestX-ray14 multi-label classification)
# Runs all 6 ablation variants sequentially in a single SLURM job.
# =============================================================================
#SBATCH --job-name=magma_ablations_cs3
#SBATCH --output=/scratch/mma9138/MAGMA/baseline_testing/ablation_runs/logs/ablations_cs3_%j.out
#SBATCH --error=/scratch/mma9138/MAGMA/baseline_testing/ablation_runs/logs/ablations_cs3_%j.err
#SBATCH --time=48:00:00
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

# cs3 canonical prompt (clean run)
# Data layout: Data_Entry_2017.csv + flat images/ directory (NOT images_001/ shards)
# BBox_List_2017.csv also present. train/val/test list files are deliberately absent.
CS3_PROMPT="Build a multi-label chest X-ray disease classification model. \
Training/validation data directory: ${EXPOSED_CHESTXRAY14} \
  (NIH ChestX-ray14 dataset). \
Files present: Data_Entry_2017.csv (patient-level labels, Patient ID column), \
  images/ (flat directory of all X-ray JPEGs — NOT in images_001/ shards), \
  BBox_List_2017.csv (bounding box annotations, optional). \
Target: predict the 14 thoracic disease labels in the Finding Labels column \
  (pipe-separated, e.g. 'Atelectasis|Cardiomegaly|No Finding'). \
Patient identifier column: Patient ID. Images named: {Patient ID}_{view index}.png. \
There are NO official train/val/test list files — create your own patient-level split \
using seed=42 (70/15/15 stratified on multi-label presence). \
Use a DenseNet-121 or ResNet-50 pre-trained on ImageNet as backbone. \
Evaluate per-label AUROC (14 labels) and report macro-averaged AUROC and AUPRC on the validation set. \
Perform TRIPOD assessment, calibrate predictions, check for leakage, and write a final report. \
FINAL EVALUATION (touch EXACTLY ONCE, after all training and model selection is complete): \
  evaluate the final model on ${HELD_OUT_CHESTXRAY14} (same directory structure). \
  Report test AUROC (macro and per-label) separately from validation metrics. \
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

TIMING_FILE="${ABLATION_ROOT}/ablation_wall_times_cs3.tsv"
echo -e "variant\tstart_epoch\tend_epoch\twall_seconds\tstatus" > "${TIMING_FILE}"

# =============================================================================
# Helper: run one ablation variant
# =============================================================================
run_variant() {
    local VARIANT="$1"
    local VARIANT_DIR="${REPO_ROOT}/${VARIANT}"
    local ARTIFACT_ROOT="${ABLATION_ROOT}/${VARIANT}/cs3"

    echo ""
    echo "============================================================"
    echo "  Starting variant: ${VARIANT}"
    echo "  $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "============================================================"

    mkdir -p "${ARTIFACT_ROOT}"

    local T_START
    T_START=$(date +%s)

    (
        cd "${VARIANT_DIR}"
        "${PYTHON}" -m magma_v2.main \
            "${CS3_PROMPT}" \
            --artifact-root "${ARTIFACT_ROOT}" \
            --reward-lambda "${REWARD_LAMBDA}" \
            --max-attempts "${MAX_ATTEMPTS}" \
            --ui compact
    )
    local EXIT_CODE=$?

    local T_END
    T_END=$(date +%s)
    local WALL=$(( T_END - T_START ))
    local STATUS="success"
    if [ "${EXIT_CODE}" -ne 0 ]; then
        STATUS="failed(exit=${EXIT_CODE})"
    fi

    echo -e "${VARIANT}\t${T_START}\t${T_END}\t${WALL}\t${STATUS}" >> "${TIMING_FILE}"

    echo ""
    echo "  Finished ${VARIANT} in ${WALL}s  [${STATUS}]"
    echo "  Artifacts: ${ARTIFACT_ROOT}/runs/"
}

# =============================================================================
# Run all 6 variants — sequential
# =============================================================================
VARIANTS=(
    a1_no_judge
    a2_judge_no_rl
    a3_no_model_worker
    a4_no_data_worker
    a5_no_validators
    a6_no_hil
)

echo "MAGMA ablation run (CS3 — clean) started at $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "Exposed data: ${EXPOSED_CHESTXRAY14}"
echo "Held-out data: ${HELD_OUT_CHESTXRAY14}"
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
