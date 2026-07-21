#!/bin/bash
# ==============================================================================
# Master 2-Stage SLURM Orchestrator with Verbose Logging & Error Capture
# Stage 1: Parallel CPU MSA Datafill (reuse MSAs across TFs)
# Stage 2: Parallel GPU SLURM Array (concurrent AF3 GPU predictions + GNINA rescoring)
# ==============================================================================

set -e

# Setup logging
mkdir -p logs alphafold3_jsons alphafold3_datafill alphafold3_missing alphafold3_predictions data/processed/array_scores
LOG_FILE="logs/pipeline_submission.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================================================"
echo " Starting Submission at $(date) on host $(hostname)"
echo " Workspace: $(pwd)"
echo "========================================================================"

# Load modules and activate venv
module load gcc python openbabel 2>/dev/null || true

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Run python diagnostic check first
python3 scripts/diagnose_environment.py || echo "Warning: Diagnostics reported missing items."

# Count total input JSON files
TOTAL_INPUTS=$(ls -1 alphafold3_jsons/*.json 2>/dev/null | wc -l)

if [ "$TOTAL_INPUTS" -eq 0 ]; then
    echo "Error: No input JSON files found in alphafold3_jsons/"
    exit 1
fi

if ! command -v sbatch &> /dev/null; then
    echo "Error: 'sbatch' command not found! Ensure you are logged into ETH Euler HPC login node."
    exit 1
fi

echo "========================================================================"
echo " Submitting 2-Stage Parallel AF3 + GNINA Pipeline for $TOTAL_INPUTS pairs"
echo "========================================================================"

# ------------------------------------------------------------------------------
# STAGE 1: Launch Parallel CPU Datafill (MSA generation & lookup)
# ------------------------------------------------------------------------------
echo "[Stage 1] Submitting CPU Datafill job (parallel MSA generation)..."

DATAFILL_CMD="sbatch --parsable --job-name=af3_datafill --output=logs/datafill_%j.out --error=logs/datafill_%j.err --time=02:00:00 --cpus-per-task=8 --mem-per-cpu=2G --wrap=\"batch-infer alphafold3 datafill --config config/config.yaml --input-dir alphafold3_jsons --output-dir alphafold3_datafill --missing-dir alphafold3_missing\""

echo "Executing: $DATAFILL_CMD"
DATAFILL_JOB_OUT=$(eval "$DATAFILL_CMD")

if [ -z "$DATAFILL_JOB_OUT" ]; then
    echo "Error: Stage 1 Datafill submission failed!"
    exit 1
fi

echo " -> Submitted Stage 1 Datafill Job ID: $DATAFILL_JOB_OUT"

# ------------------------------------------------------------------------------
# STAGE 2: Launch Parallel GPU Array (depends on Stage 1 completion)
# ------------------------------------------------------------------------------
echo "[Stage 2] Submitting Parallel GPU Array (--array=1-$TOTAL_INPUTS)..."

ARRAY_CMD="sbatch --parsable --dependency=afterok:$DATAFILL_JOB_OUT --array=1-$TOTAL_INPUTS scripts/submit_array_task.sh"

echo "Executing: $ARRAY_CMD"
ARRAY_JOB_OUT=$(eval "$ARRAY_CMD")

if [ -z "$ARRAY_JOB_OUT" ]; then
    echo "Error: Stage 2 GPU Array submission failed!"
    exit 1
fi

echo " -> Submitted Stage 2 GPU Job Array ID: $ARRAY_JOB_OUT (dependent on $DATAFILL_JOB_OUT)"

echo "========================================================================"
echo " Pipeline submission successfully completed at $(date)!"
echo " Track Stage 1 Datafill log: tail -f logs/datafill_${DATAFILL_JOB_OUT}.out"
echo " Track Stage 2 GPU Array tasks: squeue -u \$USER"
echo " Full submission log saved to: $LOG_FILE"
echo "========================================================================"
