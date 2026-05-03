#!/bin/bash
# =============================================================================
# Phase 4 — full evaluation pipeline for MAGMA ablation studies (cs1)
#
# Runs sequentially:
#   1. Internal TRIPOD judge on all 6 ablation run dirs
#   2. Trace assembly for LLM ensemble
#   3. LLM-as-a-judge (Claude Opus 4.7, GPT-5.5, Gemini 3.1 Pro)
#   4. Score aggregation
#   5. Summary CSV + Markdown + CHANGES_OVERVIEW
#
# Usage (from repo root, after MAGMA runs complete):
#   bash eval/run_full_eval_pipeline.sh
#
# Or override paths:
#   ABLATION_ROOT=/other/path REPO_ROOT=/other/repo bash eval/run_full_eval_pipeline.sh
# =============================================================================

set -e

ABLATION_ROOT="${ABLATION_ROOT:-/scratch/mma9138/MAGMA/baseline_testing/ablation_runs}"
REPO_ROOT="${REPO_ROOT:-/scratch/mma9138/MAGMA/ablations}"
EVAL_DIR="${REPO_ROOT}/eval"
PYTHON="${PYTHON:-/share/apps/NYUAD5/miniconda/3-4.11.0/envs/jupyter/bin/python}"

# Load API keys
if [ -f "/scratch/mma9138/MAGMA/.env" ]; then
    set -a; source /scratch/mma9138/MAGMA/.env; set +a
fi

echo "============================================================"
echo "  MAGMA Ablation Evaluation Pipeline"
echo "  Ablation root: ${ABLATION_ROOT}"
echo "  $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "============================================================"

# ---- Step 1: Internal TRIPOD judge ------------------------------------------
echo ""
echo "[Step 1/5] Running internal TRIPOD judge ..."
PYTHONPATH="${REPO_ROOT}/a1_no_judge" "${PYTHON}" "${EVAL_DIR}/run_internal_judge.py" \
    --ablation-root "${ABLATION_ROOT}"

# ---- Step 2: Assemble traces ------------------------------------------------
echo ""
echo "[Step 2/5] Assembling LLM ensemble traces ..."
"${PYTHON}" "${EVAL_DIR}/assemble_ablation_traces.py" \
    --ablation-root "${ABLATION_ROOT}"

# ---- Step 3a: Claude judge --------------------------------------------------
echo ""
echo "[Step 3a/5] Claude Opus 4.7 judge ..."
"${PYTHON}" "${EVAL_DIR}/judge_ablations_claude.py" \
    --ablation-root "${ABLATION_ROOT}"

# ---- Step 3b: GPT judge -----------------------------------------------------
echo ""
echo "[Step 3b/5] GPT-5.5 judge ..."
"${PYTHON}" "${EVAL_DIR}/judge_ablations_gpt.py" \
    --ablation-root "${ABLATION_ROOT}"

# ---- Step 3c: Gemini judge --------------------------------------------------
echo ""
echo "[Step 3c/5] Gemini 3.1 Pro judge ..."
"${PYTHON}" "${EVAL_DIR}/judge_ablations_gemini.py" \
    --ablation-root "${ABLATION_ROOT}"

# ---- Step 4: Compute ensemble scores ----------------------------------------
echo ""
echo "[Step 4/5] Aggregating ensemble scores ..."
"${PYTHON}" "${EVAL_DIR}/compute_ablation_scores.py" \
    --ablation-root "${ABLATION_ROOT}"

# ---- Step 5: Build summary --------------------------------------------------
echo ""
echo "[Step 5/5] Building summary CSV + Markdown ..."
"${PYTHON}" "${EVAL_DIR}/build_ablation_summary.py" \
    --ablation-root "${ABLATION_ROOT}" \
    --repo-root "${REPO_ROOT}"

echo ""
echo "============================================================"
echo "  Evaluation complete at $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "  Outputs:"
echo "    ${ABLATION_ROOT}/ablation_summary.csv"
echo "    ${ABLATION_ROOT}/ablation_summary.md"
echo "    ${ABLATION_ROOT}/CHANGES_OVERVIEW.md"
echo "    ${ABLATION_ROOT}/llm_ensemble_eval/"
echo "============================================================"
