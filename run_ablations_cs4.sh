#!/bin/bash
# =============================================================================
# MAGMA v2 Ablation Studies — Case Study 4 (cs4: PTB-XL ECG)
# Interactive version — run inside salloc or screen/tmux.
#
# Usage:
#   salloc --partition=nvidia --gres=gpu:1 --cpus-per-task=8 --mem=32G \
#          --account=mma9138 --time=48:00:00
#   screen -S cs4
#   cd /scratch/mma9138/MAGMA/ablations
#   source /scratch/mma9138/MAGMA/.env
#   bash run_ablations_cs4.sh
# =============================================================================

set -e

REPO_ROOT="/scratch/mma9138/MAGMA/ablations"
source "${REPO_ROOT}/data_paths.sh"
ABLATION_ROOT="/scratch/mma9138/MAGMA/baseline_testing/ablation_runs"
MAX_ATTEMPTS=3
REWARD_LAMBDA=0.5

PYTHON="/share/apps/NYUAD5/miniconda/3-4.11.0/envs/jupyter/bin/python"

if [ ! -x "${PYTHON}" ]; then
    echo "ERROR: Python binary not found at ${PYTHON}" >&2; exit 1
fi

if [ -f "/scratch/mma9138/MAGMA/.env" ]; then
    set -a; source /scratch/mma9138/MAGMA/.env; set +a
fi

mkdir -p "${ABLATION_ROOT}/logs"

TIMING_FILE="${ABLATION_ROOT}/ablation_wall_times_cs4.tsv"
echo -e "variant\tstart_epoch\tend_epoch\twall_seconds\tstatus" > "${TIMING_FILE}"

run_variant() {
    local VARIANT="$1"
    local VARIANT_DIR="${REPO_ROOT}/${VARIANT}"
    local ARTIFACT_ROOT="${ABLATION_ROOT}/${VARIANT}/cs4"
    local out_path="${ARTIFACT_ROOT}"

    local PROMPT="I have an ECG dataset for multi-label arrhythmia classification. Here are the full details:

DATASET:
- Directory: ${EXPOSED_PTBXL}
- Metadata: ptbxl_database.csv (patient_id column, scp_codes column with arrhythmia annotations)
- Label mapping: scp_statements.csv (SCP code to human-readable label, needed for deriving targets)
- Signals: records100/ (100 Hz WFDB format) and records500/ (500 Hz)
- Important: DO NOT use the strat_fold column for splitting

TASK:
- Multi-label arrhythmia classification; derive 5 super-class binary labels: NORM, MI, STTC, CD, HYP
- Patient identifier: patient_id

REQUIREMENTS:
1. Split data into train/val/test by patient_id using seed=42
2. Load signals with wfdb; prefer 100 Hz records for speed
3. Train a 1-D ResNet or CNN-LSTM on the train set
4. Tune or validate using the val set
5. Evaluate on the test set and report macro-averaged AUROC and per-class AUROC
6. Save final predictions to ${out_path}

CONSTRAINTS:
- Python 3.11"

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
            "${PROMPT}" \
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

echo "MAGMA ablation run (CS4 — clean, interactive) started at $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "Running interactively — you can answer HIL questions as they appear."
echo "Exposed data: ${EXPOSED_PTBXL}"
echo ""

for VARIANT in "${VARIANTS[@]}"; do
    run_variant "${VARIANT}"
done

echo ""
echo "============================================================"
echo "  All variants complete at $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "  Wall times written to: ${TIMING_FILE}"
echo "============================================================"
