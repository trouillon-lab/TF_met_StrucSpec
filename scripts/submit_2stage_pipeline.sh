#!/bin/bash
# ==============================================================================
# Master SLURM Orchestrator via batch-infer start alphafold3_datafill_predictions
# Portable and fully automated for ETH Euler HPC / Slurm clusters.
# ==============================================================================

set -e

# Setup logging
mkdir -p logs alphafold3_jsons alphafold3_predictions data/processed software
LOG_FILE="logs/pipeline_submission.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================================================"
echo " Starting Submission at $(date) on host $(hostname)"
echo " Workspace: $(pwd)"
echo "========================================================================"

# Load Euler modules
module load eth_proxy 2>/dev/null || true
module load stack/2025-06 python/3.13.0 2>/dev/null || module load stack/2024-06 2>/dev/null || true
module load gcc python openbabel cuda 2>/dev/null || true

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# ------------------------------------------------------------------------------
# Auto-setup batch-infer workflow repository if missing
# ------------------------------------------------------------------------------
PY_VER_DIR=$(ls -d venv/lib/python3.*/ 2>/dev/null | head -n 1)

if [ -n "$PY_VER_DIR" ] && [ ! -d "${PY_VER_DIR}workflow" ]; then
    echo "Notice: Setting up batch-infer workflow repository..."
    if [ ! -d "software/batch-infer" ]; then
        git clone -b develop https://github.com/jurgjn/batch-infer.git software/batch-infer 2>/dev/null || true
    fi
    if [ -d "software/batch-infer/workflow" ]; then
        ln -sf "$(pwd)/software/batch-infer/workflow" "${PY_VER_DIR}workflow" 2>/dev/null || true
        ln -sf "$(pwd)/software/batch-infer/workflow" "workflow" 2>/dev/null || true
    fi
    pip install --no-deps --ignore-requires-python -e software/batch-infer 2>/dev/null || true
fi

# Ensure package activate symlinks exist
if [ -n "$PY_VER_DIR" ]; then
    mkdir -p "${PY_VER_DIR}.venv/bin" 2>/dev/null || true
    ln -sf "$(pwd)/venv/bin/activate" "${PY_VER_DIR}.venv/bin/activate" 2>/dev/null || true
fi

JSON_DIR="${1:-alphafold3_jsons}"
PRED_DIR="${2:-alphafold3_predictions}"
SCORES_CSV="${3:-data/processed/gnina_scores.csv}"
REDOCKED_DIR="${4:-data/processed/gnina_redocked_structures}"

mkdir -p "$JSON_DIR" "$PRED_DIR" "$(dirname "$SCORES_CSV")" "$REDOCKED_DIR"

if [ "$JSON_DIR" != "alphafold3_jsons" ]; then
    echo "Symlinking alphafold3_jsons -> $JSON_DIR"
    ln -sfn "$JSON_DIR" alphafold3_jsons
fi
if [ "$PRED_DIR" != "alphafold3_predictions" ]; then
    echo "Symlinking alphafold3_predictions -> $PRED_DIR"
    ln -sfn "$PRED_DIR" alphafold3_predictions
fi

# Clear cached TSV to force Snakemake to rescan input JSONs
rm -f alphafold3_predictions/.alphafold3_datafill_predictions.tsv

# Ensure root config.yaml symlink
ln -sf config/config.yaml config.yaml 2>/dev/null || true

# Run python diagnostic check
python3 scripts/diagnose_environment.py || echo "Warning: Diagnostics reported missing items."

# Count total input JSON files
TOTAL_INPUTS=$(ls -1 "$JSON_DIR"/*.json 2>/dev/null | wc -l)

if [ "$TOTAL_INPUTS" -eq 0 ]; then
    echo "Error: No input JSON files found in $JSON_DIR/"
    exit 1
fi

if ! command -v batch-infer &> /dev/null; then
    echo "Error: 'batch-infer' CLI tool not found on PATH! Ensure venv is activated."
    exit 1
fi

echo "========================================================================"
echo " Submitting batch-infer for $TOTAL_INPUTS AF3 input pairs ($JSON_DIR -> $PRED_DIR)"
echo "========================================================================"

# Launch batch-infer start
echo "Executing: batch-infer start alphafold3_datafill_predictions"
batch-infer start alphafold3_datafill_predictions

echo "========================================================================"
echo " batch-infer start submission successfully triggered at $(date)!"
echo " Track status with: squeue -u \$USER"
echo " Full submission log saved to: $LOG_FILE"
echo "========================================================================"
