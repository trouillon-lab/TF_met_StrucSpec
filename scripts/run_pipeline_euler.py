#!/usr/bin/env python3
"""
Automated Euler HPC Pipeline Runner.
Integrates `batch-infer` AlphaFold 3 MSA datafill & GPU inference with GNINA rescoring & ranking.
Following the Inductive Bio AF3 docking + GNINA refinement framework.
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
                 datafill_dir="alphafold3_datafill",
                 missing_dir="alphafold3_missing",
                 predictions_dir="alphafold3_predictions",
                 config_path="config/config.yaml",
                 scores_output="data/processed/gnina_scores.csv",
                 ranked_output="data/processed/ranked_candidates.csv",
                 dry_run=False,
                 skip_datafill=False,
                 skip_af3=False):
    """
    Executes end-to-end virtual screening pipeline on Euler HPC.
    """
    os.makedirs(datafill_dir, exist_ok=True)
    os.makedirs(missing_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(os.path.dirname(scores_output), exist_ok=True)

    # -------------------------------------------------------------------------
    # STEP 1: AF3 Datafill (MSA Generation / Cache Lookup via batch-infer)
    # -------------------------------------------------------------------------
    if not skip_datafill:
        print("\n=== [Step 1/4] Running batch-infer AlphaFold 3 Datafill (MSA reuse) ===")
        datafill_cmd = [
            "batch-infer", "alphafold3", "datafill",
            "--config", config_path,
            "--input-dir", input_jsons_dir,
            "--output-dir", datafill_dir,
            "--missing-dir", missing_dir
        ]
        if not shutil.which("batch-infer") and not dry_run:
            print("Warning: 'batch-infer' command not found on PATH. Ensure module/venv is loaded.", file=sys.stderr)
        if not run_command(datafill_cmd, dry_run=dry_run):
            return False
    else:
        print("\n=== [Step 1/4] Skipping AF3 Datafill ===")

    # -------------------------------------------------------------------------
    # STEP 2: AF3 Inference (GPU Prediction via batch-infer)
    # -------------------------------------------------------------------------
    if not skip_af3:
        print("\n=== [Step 2/4] Running batch-infer AlphaFold 3 Inference (GPU) ===")
        predict_cmd = [
            "batch-infer", "alphafold3", "predict",
            "--config", config_path,
            "--input-dir", datafill_dir if os.path.exists(datafill_dir) else input_jsons_dir,
            "--output-dir", predictions_dir
        ]
        if not run_command(predict_cmd, dry_run=dry_run):
            return False
    else:
        print("\n=== [Step 2/4] Skipping AF3 Inference ===")

    # -------------------------------------------------------------------------
    # STEP 3: GNINA Refinement & Rescoring (Inductive Bio framework)
    # -------------------------------------------------------------------------
    print("\n=== [Step 3/4] Running GNINA Refinement & Rescoring ===")
    gnina_cmd = [
        sys.executable, "scripts/rescore_gnina.py",
        "--config", config_path,
        "--predictions-dir", predictions_dir,
        "--output", scores_output
    ]
    if not run_command(gnina_cmd, dry_run=dry_run):
        return False

    # -------------------------------------------------------------------------
    # STEP 4: Candidate Ranking & Diagnostic Summary
    # -------------------------------------------------------------------------
    print("\n=== [Step 4/4] Ranking Candidates & Generating Diagnostic Summary ===")
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
    parser.add_argument('--datafill-dir', default='alphafold3_datafill', help="AF3 datafill output directory")
    parser.add_argument('--predictions-dir', default='alphafold3_predictions', help="AF3 predictions directory")
    parser.add_argument('--config', default='config/config.yaml', help="Pipeline config YAML")
    parser.add_argument('--dry-run', action='store_true', help="Print commands without executing")
    parser.add_argument('--skip-datafill', action='store_true', help="Skip AF3 MSA datafill step")
    parser.add_argument('--skip-af3', action='store_true', help="Skip AF3 GPU inference step")

    args = parser.parse_args()

    success = run_pipeline(
        input_jsons_dir=args.jsons_dir,
        datafill_dir=args.datafill_dir,
        predictions_dir=args.predictions_dir,
        config_path=args.config,
        dry_run=args.dry_run,
        skip_datafill=args.skip_datafill,
        skip_af3=args.skip_af3
    )

    sys.exit(0 if success else 1)
