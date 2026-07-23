#!/bin/bash
#SBATCH --job-name=gnina_rescore
#SBATCH --output=logs/gnina_%A_%a.out
#SBATCH --error=logs/gnina_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --gpus=1
#SBATCH --array=1-15

# ==============================================================================
# SLURM Job Array Submission Script for Gnina Rescoring on ETH Euler HPC
# ==============================================================================

echo "Starting Gnina rescoring job array task: ${SLURM_ARRAY_TASK_ID} of 15 (Job ID: ${SLURM_ARRAY_JOB_ID}) at $(date)"

# Load required modules on Euler
module load gcc python openbabel 2>/dev/null
module load gnina 2>/dev/null

# Activate Python virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

PRED_DIR="${1:-${PRED_DIR:-alphafold3_predictions}}"
SCORES_CSV="${2:-${SCORES_CSV:-data/processed/gnina_scores.csv}}"
REDOCKED_DIR="${3:-${REDOCKED_DIR:-data/processed/gnina_redocked_structures}}"

# Run the python rescoring script for this batch chunk
python scripts/rescore_gnina.py \
    --config config/config.yaml \
    --predictions-dir "$PRED_DIR" \
    --output "$SCORES_CSV" \
    --save-dir "$REDOCKED_DIR" \
    --batch-index ${SLURM_ARRAY_TASK_ID} \
    --total-batches 15

echo "Gnina rescoring batch task ${SLURM_ARRAY_TASK_ID} finished at $(date)"
