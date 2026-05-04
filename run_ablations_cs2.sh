#!/bin/bash
# =============================================================================
# MAGMA v2 Ablation Studies — Case Study 2 (cs2: COPD/bronchiectasis phenotyping)
# Interactive version — run inside salloc or screen/tmux so HIL questions
# can be answered by the user at the terminal.
#
# Usage:
#   salloc --partition=nvidia --gres=gpu:1 --cpus-per-task=8 --mem=32G \
#          --account=mma9138 --time=48:00:00
#   # then on the allocated node:
#   screen -S cs2
#   cd /scratch/mma9138/MAGMA/ablations
#   source /scratch/mma9138/MAGMA/.env
#   bash run_ablations_cs2.sh
# =============================================================================

set -e

REPO_ROOT="/scratch/mma9138/MAGMA/ablations"
DATASET_PATH="/scratch/mma9138/MAGMA/baseline_testing/copd_task/copd_task.csv"
ABLATION_ROOT="/scratch/mma9138/MAGMA/baseline_testing/ablation_runs"
MAX_ATTEMPTS=3
REWARD_LAMBDA=0.5

CS2_PROMPT="Build a clinical NLP phenotyping model using ${DATASET_PATH}. \
The target column is copd_label (1 = patient has COPD or bronchiectasis, 0 = does not). \
Each row is one hospital admission; the 'text' column contains the full discharge summary note. \
There is no patient identifier column — each row is independent. \
A pre-defined split column is present with values train/val/test — use it directly, do not re-split. \
The positive class (copd_label=1) is the minority class; handle class imbalance appropriately. \
Evaluate on the test split and report AUROC and AUPRC. Perform TRIPOD assessment, \
calibrate the model, and write a final report."

PYTHON="/share/apps/NYUAD5/miniconda/3-4.11.0/envs/jupyter/bin/python"

if [ ! -x "${PYTHON}" ]; then
    echo "ERROR: Python binary not found at ${PYTHON}" >&2; exit 1
fi

if [ -f "/scratch/mma9138/MAGMA/.env" ]; then
    set -a; source /scratch/mma9138/MAGMA/.env; set +a
fi

mkdir -p "${ABLATION_ROOT}/logs"

TIMING_FILE="${ABLATION_ROOT}/ablation_wall_times_cs2.tsv"
echo -e "variant\tstart_epoch\tend_epoch\twall_seconds\tstatus" > "${TIMING_FILE}"

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
    local T_START; T_START=$(date +%s)

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

echo "MAGMA ablation run (CS2) started at $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "Running interactively — you can answer HIL questions as they appear."
echo "Dataset: ${DATASET_PATH}"
echo ""

for VARIANT in "${VARIANTS[@]}"; do
    run_variant "${VARIANT}"
done

echo ""
echo "============================================================"
echo "  All variants complete at $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "  Wall times written to: ${TIMING_FILE}"
echo "============================================================"
