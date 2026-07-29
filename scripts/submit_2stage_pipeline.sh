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
    if [ -e "alphafold3_jsons" ] && [ ! -L "alphafold3_jsons" ]; then
        rm -rf alphafold3_jsons
    fi
    ln -sfn "$JSON_DIR" alphafold3_jsons
fi
if [ "$PRED_DIR" != "alphafold3_predictions" ]; then
    echo "Symlinking alphafold3_predictions -> $PRED_DIR"
    if [ -e "alphafold3_predictions" ] && [ ! -L "alphafold3_predictions" ]; then
        rm -rf alphafold3_predictions
    fi
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
echo " Submitting Automated 3-Stage AF3 Pipeline for $TOTAL_INPUTS input pairs"
echo "========================================================================"

# Step 1: Identify missing TFs requiring MSAs
echo "[Stage 1] Submitting missing sequence identification..."
STAGE1_OUTPUT=$(batch-infer start alphafold3_datafill_missing 2>&1)

# Extract job ID from batch-infer's lock file only if it actually submitted a job.
# Validate it looks like a SLURM job number before proceeding, to avoid silently
# submitting downstream stages without the correct dependency.
if echo "$STAGE1_OUTPUT" | grep -q '\.lock'; then
    JOB1=$(grep -oE '^[0-9]+$' .batch-infer.lock 2>/dev/null || echo "")
    if [ -z "$JOB1" ] || ! echo "$JOB1" | grep -qE '^[0-9]+$'; then
        echo "ERROR: batch-infer reported a lock file but could not extract a valid SLURM job ID."
        echo "  Lock file contents: $(cat .batch-infer.lock 2>/dev/null || echo 'not found')"
        echo "  batch-infer output:"
        echo "$STAGE1_OUTPUT"
        exit 1
    fi
else
    JOB1=""
fi

if [ -n "$JOB1" ]; then
    echo "  -> Stage 1 Lock Job ID: $JOB1"
    
    # Step 2: Calculate CPU MSAs (dependent on Stage 1 completing)
    echo "[Stage 2] Submitting CPU MSA calculations (dependent on Stage 1: $JOB1)..."
    JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 --job-name=af3_stage2_msas --output=logs/stage2_msas_%j.out --error=logs/stage2_msas_%j.err --time=04:00:00 --cpus-per-task=2 --mem-per-cpu=2G --wrap="batch-infer start alphafold3_datafill_msas")
    echo "  -> Stage 2 Trigger Job ID: $JOB2"
    
    # Step 3: Launch GPU Predictions (dependent on Stage 2 completing)
    echo "[Stage 3] Submitting GPU Predictions (dependent on Stage 2: $JOB2)..."
    JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 --job-name=AF3_Pred_Trig --output=logs/stage3_preds_%j.out --error=logs/stage3_preds_%j.err --time=04:00:00 --cpus-per-task=2 --mem-per-cpu=2G --wrap="batch-infer start alphafold3_datafill_predictions . --keep-going")
    echo "  -> Stage 3 Trigger Job ID: $JOB3"
    
    # Step 4: Calculate dynamic array task count for GNINA rescoring (~15 pairs per job)
    TARGET_CHUNK_SIZE=15
    NUM_INPUTS=$(ls -1 "$JSON_DIR"/*.json 2>/dev/null | wc -l)
    if [ "$NUM_INPUTS" -eq 0 ]; then NUM_INPUTS=6000; fi
    N_BATCHES=$(( (NUM_INPUTS + TARGET_CHUNK_SIZE - 1) / TARGET_CHUNK_SIZE ))
    if [ "$N_BATCHES" -lt 1 ]; then N_BATCHES=1; fi
    if [ "$N_BATCHES" -gt 500 ]; then N_BATCHES=500; fi

    echo "[Stage 4] Submitting GNINA Array: $NUM_INPUTS pairs / $TARGET_CHUNK_SIZE per chunk = $N_BATCHES jobs (dependent on Stage 3: $JOB3)..."
    JOB4=$(sbatch --parsable --array=1-${N_BATCHES} --dependency=afterok:$JOB3 scripts/submit_gnina.sh "$PRED_DIR" "$SCORES_CSV" "$REDOCKED_DIR")
    echo "  -> Stage 4 GNINA Array Job ID: $JOB4"
else
    echo "[Stage 1] All sequences indexed! Submitting GPU predictions & chaining GNINA..."
    STAGE3_OUTPUT=$(batch-infer start alphafold3_datafill_predictions . --keep-going 2>&1)
    if echo "$STAGE3_OUTPUT" | grep -q '\.lock'; then
        JOB3=$(grep -oE '^[0-9]+$' .batch-infer.lock 2>/dev/null || echo "")
        if [ -z "$JOB3" ] || ! echo "$JOB3" | grep -qE '^[0-9]+$'; then
            echo "ERROR: batch-infer reported a lock file but could not extract a valid SLURM job ID."
            echo "  Lock file contents: $(cat .batch-infer.lock 2>/dev/null || echo 'not found')"
            echo "  batch-infer output:"
            echo "$STAGE3_OUTPUT"
            exit 1
        fi
    else
        JOB3=""
    fi
    
    TARGET_CHUNK_SIZE=15
    NUM_INPUTS=$(ls -1 "$JSON_DIR"/*.json 2>/dev/null | wc -l)
    if [ "$NUM_INPUTS" -eq 0 ]; then NUM_INPUTS=6000; fi
    N_BATCHES=$(( (NUM_INPUTS + TARGET_CHUNK_SIZE - 1) / TARGET_CHUNK_SIZE ))
    if [ "$N_BATCHES" -lt 1 ]; then N_BATCHES=1; fi
    if [ "$N_BATCHES" -gt 500 ]; then N_BATCHES=500; fi

    if [ -n "$JOB3" ]; then
        JOB4=$(sbatch --parsable --array=1-${N_BATCHES} --dependency=afterok:$JOB3 scripts/submit_gnina.sh "$PRED_DIR" "$SCORES_CSV" "$REDOCKED_DIR")
        echo "  -> Stage 4 GNINA Array Job ID: $JOB4 (dependent on Stage 3: $JOB3)"
        
        JOB5=$(sbatch --parsable --dependency=afterany:$JOB4 --job-name=gnina_cleanup --output=logs/gnina_cleanup_%j.out --error=logs/gnina_cleanup_%j.err --time=00:30:00 --wrap="bash scripts/resubmit_missing_gnina.sh '$PRED_DIR' '$SCORES_CSV' '$REDOCKED_DIR'")
        echo "  -> Stage 5 Auto-Cleanup & Merge Job ID: $JOB5 (dependent on Stage 4: $JOB4)"
    else
        JOB4=$(sbatch --parsable --array=1-${N_BATCHES} scripts/submit_gnina.sh "$PRED_DIR" "$SCORES_CSV" "$REDOCKED_DIR")
        echo "  -> Stage 4 GNINA Array Job ID: $JOB4"
        
        JOB5=$(sbatch --parsable --dependency=afterany:$JOB4 --job-name=gnina_cleanup --output=logs/gnina_cleanup_%j.out --error=logs/gnina_cleanup_%j.err --time=00:30:00 --wrap="bash scripts/resubmit_missing_gnina.sh '$PRED_DIR' '$SCORES_CSV' '$REDOCKED_DIR'")
        echo "  -> Stage 5 Auto-Cleanup & Merge Job ID: $JOB5 (dependent on Stage 4: $JOB4)"
    fi
fi

echo "========================================================================"
echo " Fully automated 5-Stage Self-Healing Pipeline successfully chained at $(date)!"
echo " Track status with: squeue -u \$USER"
echo " Full submission log saved to: $LOG_FILE"
echo "========================================================================"

