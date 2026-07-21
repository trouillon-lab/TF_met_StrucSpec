#!/bin/bash
#SBATCH --job-name=gnina_rescore
#SBATCH --output=logs/gnina_%j.out
#SBATCH --error=logs/gnina_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --gpus=1

# ==============================================================================
# SLURM Submission Script for Gnina Rescoring on ETH Euler HPC
# ==============================================================================

echo "Starting Gnina rescoring job: $SLURM_JOB_ID at $(date)"

# Load required modules on Euler (fallback silently if loaded already or missing)
module load gcc python openbabel 2>/dev/null

# Load Gnina module if available, or assume it's pre-installed and on PATH
module load gnina 2>/dev/null

# Activate Python virtual environment in workspace root
if [ -f "venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Run the python rescoring script
python scripts/rescore_gnina.py \
    --config config/config.yaml \
    --predictions-dir alphafold3_predictions \
    --output data/processed/gnina_scores.csv

echo "Gnina rescoring job finished at $(date)"
