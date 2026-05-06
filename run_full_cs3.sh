#!/bin/bash
# MAGMA v2 Full System — CS3 (ChestX-ray14)
# Usage: salloc ...; screen -S full_cs3; bash run_full_cs3.sh

set -e
REPO_ROOT="/scratch/mma9138/MAGMA/ablations"
source "${REPO_ROOT}/data_paths.sh"
ABLATION_ROOT="/scratch/mma9138/MAGMA/baseline_testing/ablation_runs"
MAX_ATTEMPTS=3
REWARD_LAMBDA=0.5
PYTHON="/share/apps/NYUAD5/miniconda/3-4.11.0/envs/jupyter/bin/python"

if [ ! -x "${PYTHON}" ]; then echo "ERROR: Python not found at ${PYTHON}" >&2; exit 1; fi
if [ -f "/scratch/mma9138/MAGMA/.env" ]; then set -a; source /scratch/mma9138/MAGMA/.env; set +a; fi

ARTIFACT_ROOT="${ABLATION_ROOT}/full/cs3"
TIMING_FILE="${ABLATION_ROOT}/full_wall_times_cs3.tsv"
mkdir -p "${ARTIFACT_ROOT}" "${ABLATION_ROOT}/logs"
echo -e "case_study\tstart_epoch\tend_epoch\twall_seconds\tstatus" > "${TIMING_FILE}"

PROMPT="I have a medical imaging dataset for multi-label classification. Here are the full details:

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
6. Save final predictions to ${ARTIFACT_ROOT}

CONSTRAINTS:
- Python 3.11"

echo "MAGMA full system — CS3 started at $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
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
echo -e "cs3\t${T_START}\t${T_END}\t${WALL}\t${STATUS}" >> "${TIMING_FILE}"
echo "Finished CS3 in ${WALL}s  [${STATUS}]"
echo "Artifacts: ${ARTIFACT_ROOT}/runs/"
