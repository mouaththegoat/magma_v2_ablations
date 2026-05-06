#!/bin/bash
# MAGMA v2 Full System — CS4 (PTB-XL ECG)
# Usage: salloc ...; screen -S full_cs4; bash run_full_cs4.sh

set -e
REPO_ROOT="/scratch/mma9138/MAGMA/ablations"
source "${REPO_ROOT}/data_paths.sh"
ABLATION_ROOT="/scratch/mma9138/MAGMA/baseline_testing/ablation_runs"
MAX_ATTEMPTS=3
REWARD_LAMBDA=0.5
PYTHON="/share/apps/NYUAD5/miniconda/3-4.11.0/envs/jupyter/bin/python"

if [ ! -x "${PYTHON}" ]; then echo "ERROR: Python not found at ${PYTHON}" >&2; exit 1; fi
if [ -f "/scratch/mma9138/MAGMA/.env" ]; then set -a; source /scratch/mma9138/MAGMA/.env; set +a; fi

ARTIFACT_ROOT="${ABLATION_ROOT}/full/cs4"
TIMING_FILE="${ABLATION_ROOT}/full_wall_times_cs4.tsv"
mkdir -p "${ARTIFACT_ROOT}" "${ABLATION_ROOT}/logs"
echo -e "case_study\tstart_epoch\tend_epoch\twall_seconds\tstatus" > "${TIMING_FILE}"

PROMPT="I have an ECG dataset for multi-label arrhythmia classification. Here are the full details:

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
6. Save final predictions to ${ARTIFACT_ROOT}

CONSTRAINTS:
- Python 3.11"

echo "MAGMA full system — CS4 started at $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
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
echo -e "cs4\t${T_START}\t${T_END}\t${WALL}\t${STATUS}" >> "${TIMING_FILE}"
echo "Finished CS4 in ${WALL}s  [${STATUS}]"
echo "Artifacts: ${ARTIFACT_ROOT}/runs/"
