#!/bin/env python3
"""
Diagnostic & Health Check Script for AF3 + GNINA Pipeline on Euler HPC.
Verifies software binaries, Python environment, directory structures, input files, and SLURM tools.
"""

import os
import sys
import glob
import shutil
import subprocess

def check_mark(status):
    return " [OK]" if status else " [FAIL]"

def run_diagnostics():
    print("========================================================================")
    print("       AF3 + GNINA Pipeline Euler Environment Diagnostic Check")
    print("========================================================================")
    
    all_ok = True
    
    # 1. Check Python executable & virtual environment
    py_path = sys.executable
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    print(f"1. Python Interpreter: {py_path}")
    print(f"   Virtual Environment Active: {in_venv}{check_mark(in_venv)}")
    if not in_venv:
        all_ok = False
        
    # 2. Check Python packages
    required_packages = ['yaml', 'requests', 'pytest']
    for pkg in required_packages:
        try:
            __import__(pkg)
            print(f"   Package '{pkg}': Installed{check_mark(True)}")
        except ImportError:
            print(f"   Package '{pkg}': Missing!{check_mark(False)}")
            all_ok = False

    # Check batch_infer & setup dynamic symlinks if needed
    try:
        import batch_infer
        from pathlib import Path
        base_dir = Path(batch_infer.__file__).parent.parent.parent
        print(f"   Package 'batch_infer': Installed at {base_dir}{check_mark(True)}")

        # Ensure .venv symlink inside package base
        venv_act = base_dir / ".venv" / "bin" / "activate"
        if not venv_act.exists():
            (base_dir / ".venv" / "bin").mkdir(parents=True, exist_ok=True)
            real_act = Path(sys.prefix) / "bin" / "activate"
            if real_act.exists():
                os.symlink(str(real_act), str(venv_act))

        # Check workflow directory
        wf_dir = base_dir / "workflow"
        if not wf_dir.exists():
            print(f"   Notice: Setting up batch_infer workflow symlink...")
            # Search site-packages or git clone for workflow
            found_wf = None
            for p in sys.path:
                matches = glob.glob(os.path.join(p, "**", "workflow"), recursive=True)
                if matches and os.path.isdir(matches[0]):
                    found_wf = matches[0]
                    break
            if found_wf:
                os.symlink(found_wf, str(wf_dir))
                print(f"   Workflow symlink created: {wf_dir} -> {found_wf}{check_mark(True)}")
            else:
                print(f"   Warning: workflow directory not found under {base_dir}{check_mark(False)}")

    except ImportError:
        print(f"   Package 'batch_infer': Missing!{check_mark(False)}")

    # 3. Check Binaries (sbatch, batch-infer, gnina, obabel)
    print("\n2. Executable & Binary Checks:")
    sbatch_path = shutil.which("sbatch")
    print(f"   sbatch (SLURM): {sbatch_path or 'Not Found'}{check_mark(bool(sbatch_path))}")
    
    batch_infer_path = shutil.which("batch-infer")
    print(f"   batch-infer: {batch_infer_path or 'Not Found'}{check_mark(bool(batch_infer_path))}")
    
    # Check GNINA binary
    gnina_path = shutil.which("gnina") or (os.path.abspath("bin/gnina") if os.path.exists("bin/gnina") else None)
    gnina_ok = bool(gnina_path and os.access(gnina_path, os.X_OK))
    print(f"   gnina: {gnina_path or 'Not Found / Not Executable'}{check_mark(gnina_ok)}")
    if not gnina_ok:
        all_ok = False
        print("     -> Install GNINA binary via: mkdir -p bin && wget https://github.com/gnina/gnina/releases/download/v1.1/gnina -O bin/gnina && chmod +x bin/gnina")

    # 4. Input File Check
    print("\n3. Pipeline Input Files:")
    json_dir = "alphafold3_jsons"
    json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")] if os.path.exists(json_dir) else []
    print(f"   Input JSON Directory '{json_dir}': Found {len(json_files)} input files{check_mark(len(json_files) > 0)}")
    if len(json_files) == 0:
        all_ok = False
        print("     -> Run: python scripts/prepare_test_subset.py")

    # 5. Directory Structure & Permissions
    print("\n4. Workspace Directory Permissions:")
    dirs_to_check = ["logs", "alphafold3_datafill", "alphafold3_missing", "alphafold3_predictions", "data/processed"]
    for d in dirs_to_check:
        os.makedirs(d, exist_ok=True)
        writable = os.access(d, os.W_OK)
        print(f"   Directory '{d}': Writable{check_mark(writable)}")
        if not writable:
            all_ok = False

    print("\n========================================================================")
    if all_ok:
        print(" DIAGNOSTIC RESULT: ALL CHECKS PASSED. Ready to submit pipeline!")
    else:
        print(" DIAGNOSTIC RESULT: SOME CHECKS FAILED. Fix issues above before submission.")
    print("========================================================================\n")
    return all_ok

if __name__ == '__main__':
    success = run_diagnostics()
    sys.exit(0 if success else 1)
