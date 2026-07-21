#!/bin/bash
# ==============================================================================
# Master SLURM Orchestrator via batch-infer start alphafold3_datafill_predictions
# Follows exact jurgjn/batch-infer workflow & Inductive Bio AF3 docking + GNINA refinement framework
# ==============================================================================

set -e

# Setup logging
mkdir -p logs alphafold3_jsons alphafold3_predictions data/processed
LOG_FILE="logs/pipeline_submission.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================================================"
echo " Starting Submission at $(date) on host $(hostname)"
echo " Workspace: $(pwd)"
echo "========================================================================"

# Load Euler modules (with eth_proxy for network access on computing nodes)
module load eth_proxy 2>/dev/null || true
module load stack/2024-06 2>/dev/null || module load stack/2024-05 2>/dev/null || true
module load gcc python openbabel cuda 2>/dev/null || true

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Ensure batch-infer package activate symlinks exist
mkdir -p venv/lib/python3.10/.venv/bin 2>/dev/null || true
ln -sf ../../../bin/activate venv/lib/python3.10/.venv/bin/activate 2>/dev/null || true
ln -sf config/config.yaml config.yaml 2>/dev/null || true

# Run python diagnostic check
python3 scripts/diagnose_environment.py || echo "Warning: Diagnostics reported missing items."

# Count total input JSON files
TOTAL_INPUTS=$(ls -1 alphafold3_jsons/*.json 2>/dev/null | wc -l)

if [ "$TOTAL_INPUTS" -eq 0 ]; then
    echo "Error: No input JSON files found in alphafold3_jsons/"
    exit 1
fi

if ! command -v batch-infer &> /dev/null; then
    echo "Error: 'batch-infer' CLI tool not found on PATH! Ensure venv is activated."
    exit 1
fi

echo "========================================================================"
echo " Submitting batch-infer for $TOTAL_INPUTS AF3 input pairs"
echo "========================================================================"

# Launch batch-infer start
echo "Executing: batch-infer start alphafold3_datafill_predictions"
batch-infer start alphafold3_datafill_predictions

echo "========================================================================"
echo " batch-infer start submission successfully triggered at $(date)!"
echo " Track status with: squeue -u \$USER"
echo " Full submission log saved to: $LOG_FILE"
echo "========================================================================"
