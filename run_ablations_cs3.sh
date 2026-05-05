#!/bin/bash
# =============================================================================
# MAGMA v2 Ablation Studies — Case Study 3 (cs3: ChestX-ray14)
# Interactive version — run inside salloc or screen/tmux.
#
# Usage:
#   salloc --partition=nvidia --gres=gpu:1 --cpus-per-task=8 --mem=64G \
#          --account=mma9138 --time=48:00:00
#   screen -S cs3
#   cd /scratch/mma9138/MAGMA/ablations
#   source /scratch/mma9138/MAGMA/.env
#   bash run_ablations_cs3.sh
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

TIMING_FILE="${ABLATION_ROOT}/ablation_wall_times_cs3.tsv"
echo -e "variant\tstart_epoch\tend_epoch\twall_seconds\tstatus" > "${TIMING_FILE}"

run_variant() {
    local VARIANT="$1"
    local VARIANT_DIR="${REPO_ROOT}/${VARIANT}"
    local ARTIFACT_ROOT="${ABLATION_ROOT}/${VARIANT}/cs3"
    local out_path="${ARTIFACT_ROOT}"

    local PROMPT="I have a medical imaging dataset for multi-label classification. Here are the full details:

DATASET:
- Directory: ${EXPOSED_CHESTXRAY14}
- Metadata: Data_Entry_2017.csv (Patient ID column, Finding Labels column with pipe-separated disease labels)
- Images: flat under images/ directory (NOT in images_001/ shards)
- 14 disease labels in Finding Labels (e.g. Atelectasis|Cardiomegaly|No Finding)
- No train/val/test list files present

TASK:
- Multi-label chest X-ray disease classification (14 thoracic conditions)
- Patient identifier: Patient ID column; image filename: {Patient_ID}_{view_index}.png

REQUIREMENTS:
1. Split data into train/val/test by Patient ID using seed=42
2. Train a DenseNet-121 or ResNet-50 (ImageNet pre-trained) on the train set
3. Tune or validate using the val set
4. Evaluate on the test set: report per-label AUROC and macro-averaged AUROC and AUPRC
5. Handle multi-label imbalance appropriately
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

echo "MAGMA ablation run (CS3 — clean, interactive) started at $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "Running interactively — you can answer HIL questions as they appear."
echo "Exposed data: ${EXPOSED_CHESTXRAY14}"
echo ""

for VARIANT in "${VARIANTS[@]}"; do
    run_variant "${VARIANT}"
done

echo ""
echo "============================================================"
echo "  All variants complete at $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "  Wall times written to: ${TIMING_FILE}"
echo "============================================================"
