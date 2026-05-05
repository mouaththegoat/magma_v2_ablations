#!/bin/bash
# =============================================================================
# MAGMA v2 Ablation Studies — Case Study 1 (cs1: PhysioNet 2019 Sepsis)
# Runs all 6 ablation variants sequentially in a single SLURM job.
# =============================================================================
#SBATCH --job-name=magma_ablations_cs1
#SBATCH --output=/scratch/mma9138/MAGMA/baseline_testing/ablation_runs/logs/ablations_cs1_%j.out
#SBATCH --error=/scratch/mma9138/MAGMA/baseline_testing/ablation_runs/logs/ablations_cs1_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --account=mma9138
# Partition: find yours with `sinfo -s` then uncomment the line below.
# Common names: gpu, standard, compute, normal, general, shared
# #SBATCH --partition=gpu

# =============================================================================
# Configuration
# =============================================================================
REPO_ROOT="/scratch/mma9138/MAGMA/ablations"
source "${REPO_ROOT}/data_paths.sh"
ABLATION_ROOT="/scratch/mma9138/MAGMA/baseline_testing/ablation_runs"
MAX_ATTEMPTS=3          # applied to variants that have a reward loop (A1/A2 ignore via code)
REWARD_LAMBDA=0.5

# cs1 canonical prompt (clean run — exposed/ only for training, held_out/ for final eval)
# Data layout: per-patient .psv (pipe-separated) files under training_setA/ and training_setB/
# Patient ID is encoded in the filename (pXXXXXX.psv). SepsisLabel is a column in each file.
CS1_PROMPT="Build a clinical prediction model using the PhysioNet 2019 Sepsis dataset. \
Training/validation data directory: ${EXPOSED_PHYSIONET} \
  (contains training_setA/ and training_setB/ subdirectories; \
   each file is pXXXXXX.psv, pipe-separated, one row per hour per patient). \
Target column: SepsisLabel (1 = sepsis onset within the next 6 hours, 0 = no sepsis). \
Patient identifier: encoded in the filename (strip .psv suffix). \
There is NO pre-defined split column — create your own train/val split from the exposed data \
using seed=42 (e.g. 80/20 patient-level stratified split). \
The positive class is rare (~2%); handle class imbalance with class_weight=balanced or equivalent. \
Use a clinically aware workflow: temporal leakage check, patient-level split (not row-level), \
AUROC and AUPRC on the validation set, calibration, TRIPOD assessment. \
FINAL EVALUATION (touch EXACTLY ONCE, after all training and model selection is complete): \
  evaluate the final selected model on the held-out test set at ${HELD_OUT_PHYSIONET} \
  (same directory structure). Report test AUROC and AUPRC separately. \
Log the timestamp of your single held-out access to a file named held_out_access.log \
in the artifact directory. \
Infer all remaining decisions from the data and this prompt — do not pause for clarification."

# =============================================================================
# Environment setup
# =============================================================================
PYTHON="/share/apps/NYUAD5/miniconda/3-4.11.0/envs/jupyter/bin/python"

# Verify the Python binary exists and has the required packages
if [ ! -x "${PYTHON}" ]; then
    echo "ERROR: Python binary not found at ${PYTHON}" >&2
    exit 1
fi
if ! "${PYTHON}" -c "import strands_tools" 2>/dev/null; then
    echo "ERROR: strands_tools not importable from ${PYTHON}" >&2
    exit 1
fi

export PYTHONWARNINGS="ignore"

# Load API keys from /scratch/mma9138/MAGMA/.env
# File format (one per line):
#   OPENAI_API_KEY=sk-...
#   ANTHROPIC_API_KEY=sk-ant-...
#   GOOGLE_API_KEY=AI...
if [ -f "/scratch/mma9138/MAGMA/.env" ]; then
    set -a
    source /scratch/mma9138/MAGMA/.env
    set +a
fi

mkdir -p "${ABLATION_ROOT}/logs"

# Print partition info to help debug scheduling
echo "Running on host: $(hostname)"
echo "SLURM partition: ${SLURM_JOB_PARTITION:-unknown}"
echo "SLURM job ID: ${SLURM_JOB_ID:-unknown}"

# Timing file written alongside the run artifacts
TIMING_FILE="${ABLATION_ROOT}/ablation_wall_times.tsv"
echo -e "variant\tstart_epoch\tend_epoch\twall_seconds\tstatus" > "${TIMING_FILE}"

# =============================================================================
# Helper: run one ablation variant
# =============================================================================
run_variant() {
    local VARIANT="$1"           # e.g. a1_no_judge
    local VARIANT_DIR="${REPO_ROOT}/${VARIANT}"
    local ARTIFACT_ROOT="${ABLATION_ROOT}/${VARIANT}/cs1"

    echo ""
    echo "============================================================"
    echo "  Starting variant: ${VARIANT}"
    echo "  $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "============================================================"

    mkdir -p "${ARTIFACT_ROOT}"

    local T_START
    T_START=$(date +%s)

    # cd into variant dir so sys.path[0] resolves to the variant's magma_v2,
    # not the base repo's magma_v2 (which would shadow all variant changes).
    (
        cd "${VARIANT_DIR}"
        "${PYTHON}" -m magma_v2.main \
            "${CS1_PROMPT}" \
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

echo "MAGMA ablation run (CS1 — clean) started at $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "Exposed data: ${EXPOSED_PHYSIONET}"
echo "Held-out data: ${HELD_OUT_PHYSIONET}"
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

