#!/bin/bash
# ==============================================================================
# Master 2-Stage SLURM Orchestrator:
# Stage 1: Parallel CPU MSA Datafill (reuse MSAs across TFs)
# Stage 2: Parallel GPU SLURM Array (concurrent AF3 GPU predictions + GNINA rescoring)
# ==============================================================================

set -e

# Load modules and activate venv
module load gcc python openbabel 2>/dev/null

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

mkdir -p logs alphafold3_jsons alphafold3_datafill alphafold3_missing alphafold3_predictions data/processed/array_scores

# Count total input JSON files
TOTAL_INPUTS=$(ls -1 alphafold3_jsons/*.json 2>/dev/null | wc -l)

if [ "$TOTAL_INPUTS" -eq 0 ]; then
    echo "Error: No input JSON files found in alphafold3_jsons/"
    exit 1
fi

echo "========================================================================"
echo " Launching 2-Stage Parallel AF3 + GNINA Pipeline for $TOTAL_INPUTS pairs"
echo "========================================================================"

# ------------------------------------------------------------------------------
# STAGE 1: Launch Parallel CPU Datafill (MSA generation & lookup)
# ------------------------------------------------------------------------------
echo "Stage 1: Submitting CPU Datafill jobs (parallel MSA generation)..."

DATAFILL_JOB_OUT=$(sbatch --parsable \
    --job-name=af3_datafill \
    --output=logs/datafill_%j.out \
    --error=logs/datafill_%j.err \
    --time=02:00:00 \
    --cpus-per-task=8 \
    --mem-per-cpu=2G \
    --wrap="batch-infer alphafold3 datafill --config config/config.yaml --input-dir alphafold3_jsons --output-dir alphafold3_datafill --missing-dir alphafold3_missing" 2>/dev/null || echo "LOCAL_DATAFILL")

echo "Submitted Stage 1 Datafill Job ID: $DATAFILL_JOB_OUT"

# ------------------------------------------------------------------------------
# STAGE 2: Launch Parallel GPU Array (depends on Stage 1 completion)
# ------------------------------------------------------------------------------
echo "Stage 2: Submitting Parallel GPU Array (--array=1-$TOTAL_INPUTS)..."

if [ "$DATAFILL_JOB_OUT" != "LOCAL_DATAFILL" ]; then
    ARRAY_JOB_OUT=$(sbatch --parsable \
        --dependency=afterok:$DATAFILL_JOB_OUT \
        --array=1-$TOTAL_INPUTS \
        scripts/submit_array_task.sh)
    echo "Submitted Stage 2 GPU Job Array ID: $ARRAY_JOB_OUT (dependent on $DATAFILL_JOB_OUT)"
else
    # Dry-run or local fallback
    echo "Running local fallback execution..."
    python scripts/run_pipeline_euler.py
fi

echo "========================================================================"
echo " Pipeline submission complete!"
echo " Track status with: squeue -u \$USER"
echo "========================================================================"
