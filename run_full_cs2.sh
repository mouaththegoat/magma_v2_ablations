#!/bin/bash
# MAGMA v2 Full System — CS2 (COPD/bronchiectasis NLP)
# Usage: salloc ...; screen -S full_cs2; bash run_full_cs2.sh

set -e
REPO_ROOT="/scratch/mma9138/MAGMA/ablations"
source "${REPO_ROOT}/data_paths.sh"
ABLATION_ROOT="/scratch/mma9138/MAGMA/baseline_testing/ablation_runs"
MAX_ATTEMPTS=3
REWARD_LAMBDA=0.5
PYTHON="/share/apps/NYUAD5/miniconda/3-4.11.0/envs/jupyter/bin/python"

if [ ! -x "${PYTHON}" ]; then echo "ERROR: Python not found at ${PYTHON}" >&2; exit 1; fi
if [ -f "/scratch/mma9138/MAGMA/.env" ]; then set -a; source /scratch/mma9138/MAGMA/.env; set +a; fi

ARTIFACT_ROOT="${ABLATION_ROOT}/full/cs2"
TIMING_FILE="${ABLATION_ROOT}/full_wall_times_cs2.tsv"
mkdir -p "${ARTIFACT_ROOT}" "${ABLATION_ROOT}/logs"
echo -e "case_study\tstart_epoch\tend_epoch\twall_seconds\tstatus" > "${TIMING_FILE}"

PROMPT="I have a clinical NLP dataset for binary phenotyping. Here are the full details:

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
7. Save final predictions to ${ARTIFACT_ROOT}

CONSTRAINTS:
- Python 3.11"

echo "MAGMA full system — CS2 started at $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
T_START=$(date +%s)

(
    cd "${REPO_ROOT}"
    "${PYTHON}" -m magma_v2.main \
        "${PROMPT}" \
        --artifact-root "${ARTIFACT_ROOT}" \
        --reward-lambda "${REWARD_LAMBDA}" \
        --max-attempts "${MAX_ATTEMPTS}" \
        --ui compact
)
EXIT_CODE=$?

T_END=$(date +%s)
WALL=$(( T_END - T_START ))
STATUS="success"; [ "${EXIT_CODE}" -ne 0 ] && STATUS="failed(exit=${EXIT_CODE})"
echo -e "cs2\t${T_START}\t${T_END}\t${WALL}\t${STATUS}" >> "${TIMING_FILE}"
echo "Finished CS2 in ${WALL}s  [${STATUS}]"
echo "Artifacts: ${ARTIFACT_ROOT}/runs/"
