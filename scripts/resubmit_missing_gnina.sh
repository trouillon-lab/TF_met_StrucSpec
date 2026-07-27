#!/bin/bash
# ==============================================================================
# Helper Script: Resubmit missing/timed-out GNINA chunks with 1-hour walltime
# ==============================================================================

PRED_DIR="${1:-alphafold3_predictions_dataset3}"
SCORES_CSV="${2:-data/processed/gnina_scores.csv}"
REDOCKED_DIR="${3:-data/processed/gnina_redocked_dataset3}"
TARGET_CHUNK_SIZE=15

NUM_INPUTS=$(ls -1 "$PRED_DIR"/*.zip 2>/dev/null | wc -l)
if [ "$NUM_INPUTS" -eq 0 ]; then exit 0; fi

N_BATCHES=$(( (NUM_INPUTS + TARGET_CHUNK_SIZE - 1) / TARGET_CHUNK_SIZE ))

MISSING_BATCHES=()
for i in $(seq 1 $N_BATCHES); do
    CHUNK_FILE="data/processed/gnina_chunk_$(printf "%03d" $i).csv"
    if [ ! -s "$CHUNK_FILE" ]; then
        MISSING_BATCHES+=("$i")
    fi
done

if [ ${#MISSING_BATCHES[@]} -eq 0 ]; then
    echo "All $N_BATCHES GNINA chunks completed successfully!"
    python scripts/rescore_gnina.py --output "$SCORES_CSV" --merge
else
    MISSING_LIST=$(IFS=,; echo "${MISSING_BATCHES[*]}")
    echo "Found ${#MISSING_BATCHES[@]} missing/incomplete GNINA chunks: $MISSING_LIST"
    echo "Resubmitting missing chunks with 1-hour walltime..."
    sbatch --time=01:00:00 --array="$MISSING_LIST" scripts/submit_gnina.sh "$PRED_DIR" "$SCORES_CSV" "$REDOCKED_DIR"
fi
