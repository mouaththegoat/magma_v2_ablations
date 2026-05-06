#!/bin/bash
# Rerun only the failed ablation variants for CS2 (a4_no_data_worker)
# Appends to the existing timing TSV rather than overwriting it.

REPO_ROOT="/scratch/mma9138/MAGMA/ablations"
source "${REPO_ROOT}/data_paths.sh"
ABLATION_ROOT="/scratch/mma9138/MAGMA/baseline_testing/ablation_runs"
MAX_ATTEMPTS=3
REWARD_LAMBDA=0.5
PYTHON="/share/apps/NYUAD5/miniconda/3-4.11.0/envs/jupyter/bin/python"

if [ ! -x "${PYTHON}" ]; then echo "ERROR: Python not found at ${PYTHON}" >&2; exit 1; fi
if [ -f "/scratch/mma9138/MAGMA/.env" ]; then set -a; source /scratch/mma9138/MAGMA/.env; set +a; fi

TIMING_FILE="${ABLATION_ROOT}/ablation_wall_times_cs2.tsv"
mkdir -p "${ABLATION_ROOT}/logs"

run_variant() {
    local VARIANT="$1"
    local VARIANT_DIR="${REPO_ROOT}/${VARIANT}"
    local ARTIFACT_ROOT="${ABLATION_ROOT}/${VARIANT}/cs2"
    local out_path="${ARTIFACT_ROOT}"

    local PROMPT="I have a clinical NLP dataset for binary phenotyping. Here are the full details:

DATASET:
- File path: ${EXPOSED_MIMIC_NOTE_DISCHARGE}
- Format: RFC-4180 CSV; text field contains embedded newlines — always use pandas.read_csv
- Target column: copd_label (1 = COPD or bronchiectasis, 0 = does not)
- Each row is one hospital admission; text column is the full discharge summary
- No patient identifier column — each row is independent

TASK:
- Phenotype COPD/bronchiectasis from discharge note text
- This is a clinical NLP binary classification task

REQUIREMENTS:
1. Split data into train/val/test using seed=42 (stratified)
2. Train a model on the train set (TF-IDF + logistic regression baseline)
3. Tune or validate using the val set
4. Evaluate on the test set and report AUROC and AUPRC
5. Handle class imbalance (copd_label=1 is minority class)
6. Check for text leakage (direct copd/bronchiectasis mentions)
7. Save final predictions to ${out_path}

CONSTRAINTS:
- Python 3.11"

    echo ""
    echo "============================================================"
    echo "  Rerunning: ${VARIANT} / cs2"
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
    echo "  Finished ${VARIANT} in ${WALL}s  [${STATUS}]"
}

echo "CS2 ablation reruns started at $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
run_variant "a4_no_data_worker"
echo "Done at $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
