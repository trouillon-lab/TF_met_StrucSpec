#!/bin/bash
#SBATCH --job-name=gnina_fallback
#SBATCH --output=logs/gnina_fallback_%j.out
#SBATCH --error=logs/gnina_fallback_%j.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=512M
#SBATCH --partition=normal.4h

# ==============================================================================
# One-off fallback pass for pairs the redock escalation chain (1h->2h->4h) gave
# up on. These are almost always enormous, highly-flexible ligands (LPS/lipid-A
# glycolipids) where a full conformational redocking search is combinatorially
# intractable, not pairs that just need more walltime -- so instead of raising
# the time limit further (which would just time out again), this rescores them
# with GNINA's --minimize mode: a local energy relaxation of the AF3-predicted
# pose, not a global search, so it stays fast regardless of ligand size.
#
# Deliberately kept separate from gnina_check_and_retry.sh's self-chaining
# escalation (rather than editing it in place) because SLURM freezes a batch
# script's content at sbatch time -- jobs already queued from that chain are
# running an old copy, and this avoids any risk of a live version mismatch.
#
# Usage: sbatch scripts/gnina_fallback_pass.sh <label> <pool_dir> <chronic_file>
# ==============================================================================
set -euo pipefail

LABEL="$1"
POOL_DIR="$2"
CHRONIC_FILE="$3"

module load gcc python openbabel 2>/dev/null || true
if [ -f "venv/bin/activate" ]; then source venv/bin/activate; fi

if [ ! -s "$CHRONIC_FILE" ]; then
    echo "[$(date)] No chronic pairs listed in $CHRONIC_FILE. Nothing to do."
    exit 0
fi

TS=$(date +%s)
FALLBACK_DIR="data/processed/gnina_retry_pools/${LABEL}_fallback_${TS}"
CHUNK_DIR="data/processed/gnina_chunks_${LABEL}_fallback_${TS}"
mkdir -p "$FALLBACK_DIR" "$CHUNK_DIR"

COUNT=0
while IFS= read -r pair; do
    [ -z "$pair" ] && continue
    src="${POOL_DIR}/${pair}.zip"
    if [ -f "$src" ]; then
        ln -sf "$(pwd)/${src}" "$FALLBACK_DIR/${pair}.zip"
        COUNT=$((COUNT + 1))
    else
        echo "[$(date)] WARNING: $src not found, skipping."
    fi
done < "$CHRONIC_FILE"

if [ "$COUNT" -eq 0 ]; then
    echo "[$(date)] No valid pairs found to fall back. Exiting."
    exit 0
fi

echo "[$(date)] Rescoring $COUNT chronic '$LABEL' pairs with --mode minimize (fast local relaxation, no global search) -> $CHUNK_DIR"
JOB_ID=$(sbatch --parsable --array=1-"$COUNT" --time=00:30:00 \
    scripts/submit_gnina.sh "$FALLBACK_DIR" "$CHUNK_DIR/gnina_scores.csv" \
    "data/processed/gnina_redocked_${LABEL}_fallback" minimize)
echo "[$(date)] Submitted fallback array: job $JOB_ID"
