#!/bin/bash
# MAGMA v2 Full System — CS1 (PhysioNet 2019 Sepsis)
# Usage: salloc ...; screen -S full_cs1; bash run_full_cs1.sh

set -e
REPO_ROOT="/scratch/mma9138/MAGMA/ablations"
source "${REPO_ROOT}/data_paths.sh"
ABLATION_ROOT="/scratch/mma9138/MAGMA/baseline_testing/ablation_runs"
MAX_ATTEMPTS=3
REWARD_LAMBDA=0.5
PYTHON="/share/apps/NYUAD5/miniconda/3-4.11.0/envs/jupyter/bin/python"

if [ ! -x "${PYTHON}" ]; then echo "ERROR: Python not found at ${PYTHON}" >&2; exit 1; fi
if [ -f "/scratch/mma9138/MAGMA/.env" ]; then set -a; source /scratch/mma9138/MAGMA/.env; set +a; fi

ARTIFACT_ROOT="${ABLATION_ROOT}/full/cs1"
TIMING_FILE="${ABLATION_ROOT}/full_wall_times_cs1.tsv"
mkdir -p "${ARTIFACT_ROOT}" "${ABLATION_ROOT}/logs"
echo -e "case_study\tstart_epoch\tend_epoch\twall_seconds\tstatus" > "${TIMING_FILE}"

PROMPT="I have a clinical dataset for a binary classification task. Here are the full details:

DATASET:
- File path: ${EXPOSED_PHYSIONET}/aggregated.csv (single CSV; all patients merged, one row per hour per patient)
- Target column: SepsisLabel (1 = sepsis onset, 0 = no sepsis)
- Patient identifier: patient_id column

TASK:
- Predict SepsisLabel from the physiological features
- This is a structured EHR prediction task
- Handle the longitudinal structure appropriately

REQUIREMENTS:
1. Split data into train/val/test by patient_id using seed=42 (patient-level, not row-level)
2. Train a model on the train set
3. Tune or validate using the val set
4. Evaluate on the test set and report AUROC and AUPRC
5. Handle class imbalance appropriately
6. Save final predictions to ${ARTIFACT_ROOT}

CONSTRAINTS:
- Python 3.11"

echo "MAGMA full system — CS1 started at $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
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
echo -e "cs1\t${T_START}\t${T_END}\t${WALL}\t${STATUS}" >> "${TIMING_FILE}"
echo "Finished CS1 in ${WALL}s  [${STATUS}]"
echo "Artifacts: ${ARTIFACT_ROOT}/runs/"
