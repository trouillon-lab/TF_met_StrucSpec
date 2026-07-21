#!/bin/bash
#SBATCH --job-name=af3_gnina_pipeline
#SBATCH --output=logs/pipeline_%j.out
#SBATCH --error=logs/pipeline_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --gpus=1

# ==============================================================================
# Master SLURM Pipeline Script: AF3 (batch-infer) -> GNINA Rescoring -> Ranking
# Follows Inductive Bio framework (AF3 structure prediction + GNINA refinement)
# ==============================================================================

echo "Starting AF3 + GNINA Pipeline Job: $SLURM_JOB_ID on host $(hostname) at $(date)"

# Load Euler modules if available
module load gcc python openbabel 2>/dev/null

# Activate Python virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Ensure output and log directories exist
mkdir -p logs alphafold3_jsons alphafold3_datafill alphafold3_missing alphafold3_predictions data/processed

# Run master pipeline script
python scripts/run_pipeline_euler.py \
    --jsons-dir alphafold3_jsons \
    --datafill-dir alphafold3_datafill \
    --predictions-dir alphafold3_predictions \
    --config config/config.yaml

echo "Pipeline Job Finished at $(date)"
