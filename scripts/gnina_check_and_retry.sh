#!/bin/bash
#SBATCH --job-name=gnina_check
#SBATCH --output=logs/gnina_check_%j.out
#SBATCH --error=logs/gnina_check_%j.err
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=512M
#SBATCH --partition=normal.4h

# ==============================================================================
# Self-chaining GNINA retry watchdog.
#
# Must be submitted with --dependency=afterany:<job1>[:<job2>...] on the GNINA
# array job(s) that just finished for this pool, so it fires only once every
# task in those arrays has reached a terminal state. It then:
#   1. Computes which pairs in the pool are still unscored.
#   2. Escalates each missing pair's walltime tier (2h, then 4h -- the
#      gpu.4h partition ceiling) and resubmits ONLY those pairs, one-pair-
#      per-job (never chunked -- that's what caused silent data loss before).
#      Never cancels or duplicates anything still running/pending elsewhere.
#   3. Chains the NEXT round of itself via --dependency=afterany on the retry
#      array(s) it just submitted.
#   4. When nothing is left to retry (or the remaining pairs have exhausted
#      the 4h ceiling and are logged as chronic failures), merges everything
#      scored so far, marks this pool DONE, and -- once BOTH pools are DONE --
#      triggers the final evaluation job exactly once.
#
# Usage: sbatch --dependency=afterany:<jobids> scripts/gnina_check_and_retry.sh \
#          <label> <pool_dir> <score_glob1> [<score_glob2> ...]
# ==============================================================================
set -euo pipefail

LABEL="$1"
POOL_DIR="$2"
shift 2
SCORE_GLOBS=("$@")

module load gcc python 2>/dev/null || true
if [ -f "venv/bin/activate" ]; then source venv/bin/activate; fi

STATE_DIR="data/processed/gnina_watchdog"
mkdir -p "$STATE_DIR" logs
RETRY_STATE="$STATE_DIR/${LABEL}_retry_state.tsv"
CHRONIC_FILE="$STATE_DIR/${LABEL}_chronic_failures.txt"
PLAN_FILE="$STATE_DIR/${LABEL}_plan.json"

echo "[$(date)] Checking pool '$LABEL' (dir=$POOL_DIR, globs=${SCORE_GLOBS[*]})"

python3 scripts/gnina_retry_logic.py \
    --label "$LABEL" \
    --pool-dir "$POOL_DIR" \
    --retry-state "$RETRY_STATE" \
    --chronic-file "$CHRONIC_FILE" \
    --score-glob "${SCORE_GLOBS[@]}" \
    --plan-out "$PLAN_FILE"

STATUS=$(python3 -c "import json; print(json.load(open('$PLAN_FILE'))['status'])")

if [ "$STATUS" == "done" ]; then
    echo "[$(date)] Pool '$LABEL' has nothing left to retry. Finalizing."

    if [ "$LABEL" == "d3gap" ]; then
        python3 scripts/gnina_merge_final.py \
            --base-csv data/processed/gnina_scores.csv \
            --score-glob "${SCORE_GLOBS[@]}" \
            --output data/processed/gnina_scores_merged.csv
    elif [ "$LABEL" == "d4" ]; then
        python3 scripts/gnina_merge_final.py \
            --score-glob "${SCORE_GLOBS[@]}" \
            --output data/processed/gnina_scores_dataset4.csv
    else
        echo "[$(date)] WARNING: unknown label '$LABEL', skipping merge."
    fi

    touch "$STATE_DIR/${LABEL}_DONE"

    # Figure out the other pool's label and check if it's done too.
    if [ "$LABEL" == "d3gap" ]; then OTHER="d4"; else OTHER="d3gap"; fi
    if [ -f "$STATE_DIR/${OTHER}_DONE" ]; then
        # Atomic exclusive check so a near-simultaneous finish on both pools
        # only triggers the final evaluation job once.
        if mkdir "$STATE_DIR/FINAL_TRIGGER_LOCK" 2>/dev/null; then
            echo "[$(date)] Both pools done. Submitting final evaluation."
            sbatch scripts/gnina_final_eval.sh
        else
            echo "[$(date)] Both pools done, but final eval already triggered by the other pool's checker."
        fi
    else
        echo "[$(date)] Pool '$LABEL' done; waiting on '$OTHER' before final evaluation."
    fi
    exit 0
fi

echo "[$(date)] Pool '$LABEL' still has stragglers. Submitting escalated retries."

JOB_IDS=()
while IFS=$'\t' read -r tier pairs_csv; do
    [ -z "$tier" ] && continue
    TS=$(date +%s)
    RETRY_DIR="data/processed/gnina_retry_pools/${LABEL}_tier${tier}h_${TS}"
    CHUNK_DIR="data/processed/gnina_chunks_${LABEL}_retry_tier${tier}h_${TS}"
    mkdir -p "$RETRY_DIR" "$CHUNK_DIR"

    IFS=',' read -r -a PAIR_ARR <<< "$pairs_csv"
    for pair in "${PAIR_ARR[@]}"; do
        ln -sf "$(pwd)/${POOL_DIR}/${pair}.zip" "$RETRY_DIR/${pair}.zip"
    done
    COUNT=${#PAIR_ARR[@]}

    echo "[$(date)] Tier ${tier}h: resubmitting $COUNT pairs one-pair-per-job -> $RETRY_DIR"
    JOB_ID=$(sbatch --parsable --array=1-"$COUNT" --time="${tier}:00:00" \
        scripts/submit_gnina.sh "$RETRY_DIR" "$CHUNK_DIR/gnina_scores.csv" \
        "data/processed/gnina_redocked_${LABEL}_retry")
    echo "[$(date)]   -> job $JOB_ID"
    JOB_IDS+=("$JOB_ID")
done < <(python3 -c "
import json
plan = json.load(open('$PLAN_FILE'))
for tier, pairs in plan['tier_groups'].items():
    print(f'{tier}\t' + ','.join(pairs))
")

if [ ${#JOB_IDS[@]} -eq 0 ]; then
    echo "[$(date)] ERROR: status was 'retry' but no jobs were submitted. Not chaining further."
    exit 1
fi

DEP=$(IFS=:; echo "${JOB_IDS[*]}")
NEXT=$(sbatch --parsable --dependency=afterany:"$DEP" scripts/gnina_check_and_retry.sh "$LABEL" "$POOL_DIR" "${SCORE_GLOBS[@]}")
echo "[$(date)] Chained next check for '$LABEL' as job $NEXT (after: $DEP)"
