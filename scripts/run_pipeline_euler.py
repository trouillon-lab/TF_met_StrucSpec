#!/usr/bin/env python3
"""
Automated Euler HPC Pipeline Runner.
Integrates `batch-infer start` AlphaFold 3 execution with GNINA rescoring & ranking.
Following the exact jurgjn/batch-infer workflow & Inductive Bio AF3 docking + GNINA refinement framework.
"""

import os
import sys
import argparse
import subprocess
import shutil

def run_command(cmd, dry_run=False):
    """Executes a shell command or prints it if dry_run is True."""
    print(f" Executing: {' '.join(cmd)}")
    if dry_run:
        return True
    res = subprocess.run(cmd, text=True)
    if res.returncode != 0:
        print(f"Error: Command failed with exit code {res.returncode}", file=sys.stderr)
        return False
    return True

def run_pipeline(input_jsons_dir="alphafold3_jsons",
                 predictions_dir="alphafold3_predictions",
                 config_path="config/config.yaml",
                 scores_output="data/processed/gnina_scores.csv",
                 ranked_output="data/processed/ranked_candidates.csv",
                 dry_run=False,
                 skip_af3=False):
    """
    Executes end-to-end virtual screening pipeline on Euler HPC.
    """
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(os.path.dirname(scores_output), exist_ok=True)

    # -------------------------------------------------------------------------
    # STEP 1: AF3 Execution via batch-infer start
    # -------------------------------------------------------------------------
    if not skip_af3:
        print("\n=== [Step 1/3] Running batch-infer AlphaFold 3 (datafill + predictions) ===")
        # batch-infer start alphafold3_datafill_predictions
        af3_cmd = [
            "batch-infer", "start", "alphafold3_datafill_predictions"
        ]
        if not shutil.which("batch-infer") and not dry_run:
            print("Warning: 'batch-infer' executable not found. Ensure active venv.", file=sys.stderr)
        if not run_command(af3_cmd, dry_run=dry_run):
            print("Notice: If batch-infer is submitting asynchronously via sbatch, verify with squeue.")
    else:
        print("\n=== [Step 1/3] Skipping AF3 Execution ===")

    # -------------------------------------------------------------------------
    # STEP 2: GNINA Refinement & Rescoring (Inductive Bio framework)
    # -------------------------------------------------------------------------
    print("\n=== [Step 2/3] Running GNINA Refinement & Rescoring ===")
    gnina_cmd = [
        sys.executable, "scripts/rescore_gnina.py",
        "--config", config_path,
        "--predictions-dir", predictions_dir,
        "--output", scores_output
    ]
    if not run_command(gnina_cmd, dry_run=dry_run):
        return False

    # -------------------------------------------------------------------------
    # STEP 3: Candidate Ranking & Diagnostic Summary
    # -------------------------------------------------------------------------
    print("\n=== [Step 3/3] Ranking Candidates & Generating Diagnostic Summary ===")
    rank_cmd = [
        sys.executable, "scripts/rank_candidates.py",
        "--gnina-scores", scores_output,
        "--output", ranked_output
    ]
    if not run_command(rank_cmd, dry_run=dry_run):
        return False

    print("\n Pipeline completed successfully!")
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run full AF3 + GNINA Virtual Screening Pipeline on Euler.")
    parser.add_argument('--jsons-dir', default='alphafold3_jsons', help="Input JSONs directory")
    parser.add_argument('--predictions-dir', default='alphafold3_predictions', help="AF3 predictions directory")
    parser.add_argument('--config', default='config/config.yaml', help="Pipeline config YAML")
    parser.add_argument('--dry-run', action='store_true', help="Print commands without executing")
    parser.add_argument('--skip-af3', action='store_true', help="Skip AF3 GPU inference step")

    args = parser.parse_args()

    success = run_pipeline(
        input_jsons_dir=args.jsons_dir,
        predictions_dir=args.predictions_dir,
        config_path=args.config,
        dry_run=args.dry_run,
        skip_af3=args.skip_af3
    )

    sys.exit(0 if success else 1)
