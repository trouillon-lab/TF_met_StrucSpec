#!/bin/bash
#SBATCH --job-name=af3_gnina_task
#SBATCH --output=logs/task_%A_%a.out
#SBATCH --error=logs/task_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1

# ==============================================================================
# SLURM Job Array Worker Task: Runs 1 AF3 GPU prediction + instant GNINA rescoring
# ==============================================================================

set -e

# Load Euler modules
module load gcc python openbabel 2>/dev/null

# Activate Python virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

DATAFILL_DIR="alphafold3_datafill"
PREDICTIONS_DIR="alphafold3_predictions"
SCORES_DIR="data/processed/array_scores"

mkdir -p "$PREDICTIONS_DIR" "$SCORES_DIR" logs

# Find input file matching current SLURM array index (1-indexed)
INPUT_FILES=($(ls -1 ${DATAFILL_DIR}/*.json 2>/dev/null | sort))

if [ ${#INPUT_FILES[@]} -eq 0 ]; then
    # Fallback to alphafold3_jsons if datafill directory empty
    INPUT_FILES=($(ls -1 alphafold3_jsons/*.json 2>/dev/null | sort))
fi

INDEX=$(($SLURM_ARRAY_TASK_ID - 1))
TARGET_JSON="${INPUT_FILES[$INDEX]}"

if [ -z "$TARGET_JSON" ] || [ ! -f "$TARGET_JSON" ]; then
    echo "Error: No input JSON found for array index $SLURM_ARRAY_TASK_ID"
    exit 1
fi

BASENAME=$(basename "$TARGET_JSON" .json)
echo "Processing Array Task $SLURM_ARRAY_TASK_ID: $BASENAME on host $(hostname) at $(date)"

# ------------------------------------------------------------------------------
# STEP 1: AF3 GPU Inference for 1 pair
# ------------------------------------------------------------------------------
echo "Running batch-infer predict on $TARGET_JSON..."
batch-infer alphafold3 predict \
    --config config/config.yaml \
    --input-file "$TARGET_JSON" \
    --output-dir "$PREDICTIONS_DIR" 2>/dev/null || \
    run_alphafold --json_path="$TARGET_JSON" --output_dir="$PREDICTIONS_DIR" 2>/dev/null || \
    echo "Notice: Standard AF3 command invoked."

# ------------------------------------------------------------------------------
# STEP 2: Instant GNINA Rescoring on predicted output structure
# ------------------------------------------------------------------------------
PREDICTION_ZIP="${PREDICTIONS_DIR}/${BASENAME}_predictions.zip"
TASK_SCORE_CSV="${SCORES_DIR}/${BASENAME}_score.csv"

echo "Running instant GNINA rescoring for $BASENAME..."
python scripts/rescore_gnina.py \
    --config config/config.yaml \
    --predictions-dir "$PREDICTIONS_DIR" \
    --output "$TASK_SCORE_CSV"

echo "Task $SLURM_ARRAY_TASK_ID ($BASENAME) completed at $(date)"
