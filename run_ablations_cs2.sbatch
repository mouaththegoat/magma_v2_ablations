#!/bin/bash
# =============================================================================
# MAGMA v2 Ablation Studies — Case Study 2 (cs2: COPD/bronchiectasis phenotyping)
# Runs all 6 ablation variants sequentially in a single SLURM job.
# =============================================================================
#SBATCH --job-name=magma_ablations_cs2
#SBATCH --output=/scratch/mma9138/MAGMA/baseline_testing/ablation_runs/logs/ablations_cs2_%j.out
#SBATCH --error=/scratch/mma9138/MAGMA/baseline_testing/ablation_runs/logs/ablations_cs2_%j.err
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

# cs2 canonical prompt (clean run — exposed/ for training, held_out/ for final eval)
CS2_PROMPT="Build a clinical NLP phenotyping model for COPD/bronchiectasis \
using ${EXPOSED_MIMIC_NOTE_DISCHARGE} \
(MIMIC-IV discharge summaries; RFC-4180 CSV with embedded newlines — use pandas.read_csv; \
target copd_label; each row is one admission; no patient ID column; no pre-defined split — \
use seed=42 80/20 stratified split; minority class, handle imbalance; \
TF-IDF + logistic regression baseline; check for text leakage; TRIPOD, calibration, final report). \
Final test evaluation: use ${HELD_OUT_MIMIC_NOTE_DISCHARGE} exactly once after training; \
report test AUROC and AUPRC separately; log access to held_out_access.log. \
Infer all remaining decisions — do not pause for clarification."

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

TIMING_FILE="${ABLATION_ROOT}/ablation_wall_times_cs2.tsv"
echo -e "variant\tstart_epoch\tend_epoch\twall_seconds\tstatus" > "${TIMING_FILE}"

# =============================================================================
# Helper: run one ablation variant
# =============================================================================
run_variant() {
    local VARIANT="$1"
    local VARIANT_DIR="${REPO_ROOT}/${VARIANT}"
    local ARTIFACT_ROOT="${ABLATION_ROOT}/${VARIANT}/cs2"

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
            "${CS2_PROMPT}" \
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

echo "MAGMA ablation run (CS2 — clean) started at $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "Exposed data: ${EXPOSED_MIMIC_NOTE_DISCHARGE}"
echo "Held-out data: ${HELD_OUT_MIMIC_NOTE_DISCHARGE}"
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
