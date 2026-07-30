#!/bin/bash
#SBATCH --job-name=gnina_final_eval
#SBATCH --output=logs/gnina_final_eval_%j.out
#SBATCH --error=logs/gnina_final_eval_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=normal.4h

# ==============================================================================
# Runs once both the D3-gap and D4 GNINA retry chains report DONE (triggered
# by scripts/gnina_check_and_retry.sh). Re-scores D3/D4 with the now-complete
# GNINA CSVs and re-runs the ground-truth recovery analysis.
# ==============================================================================
set -euo pipefail

module load gcc python 2>/dev/null || true
if [ -f "venv/bin/activate" ]; then source venv/bin/activate; fi

echo "[$(date)] Running full evaluation for Dataset 3..."
python3 scripts/run_full_evaluation.py \
    --pred-dirs alphafold3_predictions_dataset3 \
    --gnina-csvs data/processed/gnina_scores_merged.csv \
    --out-dir results/dataset3 \
    --label "Dataset 3"

echo "[$(date)] Running full evaluation for Dataset 4..."
python3 scripts/run_full_evaluation.py \
    --pred-dirs alphafold3_predictions_dataset4 \
    --gnina-csvs data/processed/gnina_scores_dataset4.csv \
    --out-dir results/dataset4 \
    --label "Dataset 4"

echo "[$(date)] Running ground truth recovery (AF3_Score)..."
python3 scripts/ground_truth_recovery.py

echo "[$(date)] Running ground truth recovery (Consensus_Score)..."
python3 scripts/ground_truth_recovery.py --score-col Consensus_Score

touch data/processed/gnina_watchdog/ALL_DONE
echo "[$(date)] All done. See results/dataset3, results/dataset4, results/ground_truth_recovery."
