#!/bin/bash
# =============================================================================
# move_legacy_results.sh — Preserve contaminated results before clean rerun
#
# Moves the existing ablation_runs/ directory (which contains results from runs
# on un-frozen, potentially contaminated splits) to a sibling directory named
# ablation_runs_legacy_contaminated/. The new clean runs then populate a fresh
# ablation_runs/ directory.
#
# Run this ONCE on the HPC before submitting any of the clean sbatch jobs.
# After running, verify the move with the listing printed at the end.
#
# Usage:
#   bash move_legacy_results.sh
#   # or override the default root:
#   ABLATION_ROOT=/other/path bash move_legacy_results.sh
# =============================================================================

set -e

ABLATION_ROOT="${ABLATION_ROOT:-/scratch/mma9138/MAGMA/baseline_testing/ablation_runs}"
LEGACY_ROOT="${ABLATION_ROOT%/}_legacy_contaminated"

if [ ! -d "${ABLATION_ROOT}" ]; then
    echo "INFO: ${ABLATION_ROOT} does not exist — nothing to move."
    exit 0
fi

if [ -d "${LEGACY_ROOT}" ]; then
    echo "ERROR: Legacy destination already exists: ${LEGACY_ROOT}"
    echo "  Remove or rename it manually before re-running this script."
    exit 1
fi

echo "============================================================"
echo "  Moving contaminated results to legacy directory"
echo "  FROM: ${ABLATION_ROOT}"
echo "  TO:   ${LEGACY_ROOT}"
echo "  $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "============================================================"

mv "${ABLATION_ROOT}" "${LEGACY_ROOT}"

# Re-create an empty logs/ directory for the new clean runs
mkdir -p "${ABLATION_ROOT}/logs"

echo ""
echo "Move complete. Legacy contents:"
ls -lh "${LEGACY_ROOT}/" | head -30
echo ""
echo "Fresh ablation root ready at: ${ABLATION_ROOT}"
echo ""
echo "Next steps:"
echo "  1. Submit clean jobs:  sbatch run_ablations_cs1.sbatch"
echo "                         sbatch run_ablations_cs2.sbatch  (or use .sh for interactive)"
echo "                         ... through cs10"
echo "  2. After all runs complete, build comparison table:"
echo "     python eval/build_comparison_table.py \\"
echo "         --legacy-root ${LEGACY_ROOT} \\"
echo "         --clean-root  ${ABLATION_ROOT}"
