#!/bin/bash
# =============================================================================
# MAGMA v2 Full System — CS1–CS4
# Runs the original (non-ablated) MAGMA v2 on case studies 1–4 sequentially.
# =============================================================================
#SBATCH --job-name=magma_full_cs1_cs4
#SBATCH --output=/scratch/mma9138/MAGMA/baseline_testing/ablation_runs/logs/full_cs1_cs4_%j.out
#SBATCH --error=/scratch/mma9138/MAGMA/baseline_testing/ablation_runs/logs/full_cs1_cs4_%j.err
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --account=mma9138
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:1

REPO_ROOT="/scratch/mma9138/MAGMA/ablations"
source "${REPO_ROOT}/data_paths.sh"
ABLATION_ROOT="/scratch/mma9138/MAGMA/baseline_testing/ablation_runs"
MAX_ATTEMPTS=3
REWARD_LAMBDA=0.5

PYTHON="/share/apps/NYUAD5/miniconda/3-4.11.0/envs/jupyter/bin/python"

if [ ! -x "${PYTHON}" ]; then
    echo "ERROR: Python binary not found at ${PYTHON}" >&2; exit 1
fi
if ! "${PYTHON}" -c "import strands_tools" 2>/dev/null; then
    echo "ERROR: strands_tools not importable from ${PYTHON}" >&2; exit 1
fi

export PYTHONWARNINGS="ignore"

if [ -f "/scratch/mma9138/MAGMA/.env" ]; then
    set -a; source /scratch/mma9138/MAGMA/.env; set +a
fi

mkdir -p "${ABLATION_ROOT}/logs"

echo "Running on host: $(hostname)"
echo "SLURM partition: ${SLURM_JOB_PARTITION:-unknown}"
echo "SLURM job ID: ${SLURM_JOB_ID:-unknown}"

TIMING_FILE="${ABLATION_ROOT}/full_wall_times_cs1_cs4.tsv"
echo -e "case_study\tstart_epoch\tend_epoch\twall_seconds\tstatus" > "${TIMING_FILE}"

run_case_study() {
    local CS="$1"
    local ARTIFACT_ROOT="${ABLATION_ROOT}/full/${CS}"
    local out_path="${ARTIFACT_ROOT}"

    case "${CS}" in
        cs1)
            local PROMPT="I have a clinical dataset for a binary classification task. Here are the full details:

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
6. Save final predictions to ${out_path}

CONSTRAINTS:
- Python 3.11"
            ;;
        cs2)
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
            ;;
        cs3)
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
            ;;
        cs4)
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
            ;;
    esac

    echo ""
    echo "============================================================"
    echo "  Starting full system: ${CS}"
    echo "  $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "============================================================"

    mkdir -p "${ARTIFACT_ROOT}"
    local T_START; T_START=$(date +%s)

    (
        cd "${REPO_ROOT}"
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

    echo -e "${CS}\t${T_START}\t${T_END}\t${WALL}\t${STATUS}" >> "${TIMING_FILE}"
    echo ""
    echo "  Finished ${CS} in ${WALL}s  [${STATUS}]"
    echo "  Artifacts: ${ARTIFACT_ROOT}/runs/"
}

echo "MAGMA full system run (CS1–CS4) started at $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo ""

for CS in cs1 cs2 cs3 cs4; do
    run_case_study "${CS}"
done

echo ""
echo "============================================================"
echo "  All case studies complete at $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "  Wall times written to: ${TIMING_FILE}"
echo "============================================================"
