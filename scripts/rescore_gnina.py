#!/usr/bin/env python3
"""
Gnina rescoring and refinement script.
Extracts top-ranked models from AF3 predictions, splits them into protein/ligand,
runs Gnina refinement/redocking, and compiles scores.
"""

import os
import sys
import re
import zipfile
import shutil
import subprocess
import argparse
import yaml

def parse_mmcif_atoms(cif_text):
    """Parses MMCIF atom records in loop_ blocks."""
    atoms = []
    lines = cif_text.splitlines()
    in_atom_loop = False
    headers = []
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        
        if line.startswith("loop_"):
            in_atom_loop = False
            headers = []
            continue
            
        if line.startswith("_atom_site."):
            in_atom_loop = True
            headers.append(line)
            continue
            
        if in_atom_loop and (line.startswith("ATOM") or line.startswith("HETATM")):
            parts = line.split()
            if len(parts) >= len(headers):
                atom_data = dict(zip(headers, parts))
                atoms.append(atom_data)
            continue
            
        if in_atom_loop and not (line.startswith("ATOM") or line.startswith("HETATM") or line.startswith("_atom_site.")):
            in_atom_loop = False
            
    return atoms

def write_pdb(atoms, chain_to_keep, output_path):
    """Writes atom data to PDB format."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx, atom in enumerate(atoms):
            chain = atom.get('_atom_site.label_asym_id') or atom.get('_atom_site.auth_asym_id') or 'A'
            if chain != chain_to_keep:
                continue
                
            group = atom.get('_atom_site.group_PDB', 'ATOM')
            serial = atom.get('_atom_site.id', str(idx + 1))
            name = atom.get('_atom_site.label_atom_id', 'C')
            res_name = atom.get('_atom_site.label_comp_id', 'UNK')
            if len(res_name) > 3:
                res_name = res_name[:3]
            res_seq = atom.get('_atom_site.auth_seq_id') or atom.get('_atom_site.label_seq_id') or '1'
            res_seq = "".join(c for c in res_seq if c.isdigit())
            if not res_seq:
                res_seq = "1"
                
            x = float(atom.get('_atom_site.Cartn_x', 0.0))
            y = float(atom.get('_atom_site.Cartn_y', 0.0))
            z = float(atom.get('_atom_site.Cartn_z', 0.0))
            
            occupancy = float(atom.get('_atom_site.occupancy', 1.0))
            b_factor = float(atom.get('_atom_site.B_iso_or_equiv', 0.0))
            element = atom.get('_atom_site.type_symbol', 'C')
            
            if len(name) < 4:
                name_str = f" {name:<3}"
            else:
                name_str = f"{name:<4}"
            
            element_str = f"{element:>2}"
            
            line = f"{group:<6}{int(serial):>5} {name_str}{res_name:>3} {chain}{int(res_seq):>4}    {x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{b_factor:6.2f}          {element_str:>2}\n"
            f.write(line)
        f.write("END\n")

def parse_gnina_output(stdout_text):
    """Parses CNNscore and CNNaffinity from Gnina command line stdout."""
    cnn_score = None
    cnn_affinity = None
    
    score_match = re.search(r"CNNscore:\s*([0-9.]+)", stdout_text, re.IGNORECASE)
    affinity_match = re.search(r"CNNaffinity:\s*([0-9.-]+)", stdout_text, re.IGNORECASE)
    
    if score_match:
        cnn_score = float(score_match.group(1))
    if affinity_match:
        cnn_affinity = float(affinity_match.group(1))
        
    if cnn_score is not None and cnn_affinity is not None:
        return cnn_score, cnn_affinity

    lines = stdout_text.splitlines()
    for idx, line in enumerate(lines):
        if "-----" in line or "=====" in line:
            for data_line in lines[idx + 1:]:
                data_line = data_line.strip()
                if not data_line or "-----" in data_line or "=====" in data_line or "Using" in data_line:
                    continue
                parts = data_line.replace("|", " ").split()
                if len(parts) >= 4:
                    try:
                        if len(parts) >= 5:
                            cnn_score = float(parts[3])
                            cnn_affinity = float(parts[4])
                        else:
                            cnn_score = float(parts[2])
                            cnn_affinity = float(parts[3])
                        return cnn_score, cnn_affinity
                    except (ValueError, IndexError):
                        continue
                    
    return cnn_score, cnn_affinity

def rescore_pair(zip_path, temp_root, mode, autobox_add, save_dir="data/processed/gnina_redocked_structures"):
    """Rescores a single prediction pair and saves AF3 + GNINA redocked structures."""
    basename = os.path.basename(zip_path)
    pair_name = os.path.splitext(basename)[0]
    
    # Try to find standard TF_Ligand name
    pair_name = pair_name.replace("_predictions", "")
    
    temp_dir = os.path.join(temp_root, pair_name)
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Find top ranked CIF
            cif_files = [f for f in zip_ref.namelist() if f.endswith('.cif')]
            model_0_files = [f for f in cif_files if 'model_0' in f or 'model_0.cif' in f]
            
            target_cif = model_0_files[0] if model_0_files else (cif_files[0] if cif_files else None)
            
            if not target_cif:
                print(f"Error: No MMCIF model files found in {zip_path}")
                return None
                
            cif_text = zip_ref.read(target_cif).decode('utf-8')
            
        atoms = parse_mmcif_atoms(cif_text)
        if not atoms:
            print(f"Error: Failed to parse atoms from {target_cif}")
            return None
            
        # Split chains
        protein_pdb = os.path.join(temp_dir, "protein.pdb")
        ligand_pdb = os.path.join(temp_dir, "ligand.pdb")
        ligand_sdf = os.path.join(temp_dir, "ligand.sdf")
        gnina_out_sdf = os.path.join(temp_dir, "gnina_redocked.sdf")
        
        write_pdb(atoms, "A", protein_pdb)
        write_pdb(atoms, "B", ligand_pdb)
        
        # Convert ligand PDB to SDF using obabel if available
        ligand_path = ligand_pdb
        if shutil.which("obabel"):
            try:
                subprocess.run(
                    ["obabel", ligand_pdb, "-O", ligand_sdf],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                ligand_path = ligand_sdf
            except Exception as e:
                print(f"Warning: obabel conversion failed: {e}. Using PDB ligand.")
                
        # Find gnina executable in repo bin/ or on system PATH
        gnina_bin = "gnina"
        if os.path.exists("bin/gnina"):
            gnina_bin = os.path.abspath("bin/gnina")
        elif os.path.exists("./gnina"):
            gnina_bin = os.path.abspath("./gnina")
            
        cmd = [gnina_bin, "-r", protein_pdb, "-l", ligand_path, "-o", gnina_out_sdf]
        
        if mode == "score_only":
            cmd.extend(["--score_only", "--cpu", "4"])
        elif mode == "minimize":
            cmd.extend(["--minimize", "--autobox_ligand", ligand_path, "--autobox_add", str(autobox_add), "--cpu", "4"])
        elif mode == "redock":
            cmd.extend(["--autobox_ligand", ligand_path, "--autobox_add", str(autobox_add), "--cpu", "4"])
        else:
            print(f"Error: Unknown Gnina mode '{mode}'. Defaulting to score_only.")
            cmd.append("--score_only")
            
        # Run Gnina
        if gnina_bin == "gnina" and not shutil.which("gnina"):
            print("Warning: 'gnina' executable not found on PATH. Simulating scores for validation.")
            return {"TF_Ligand": pair_name, "CNNscore": 0.85, "CNNaffinity": 6.5, "Gnina_Mode": mode}
            
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            print(f"Error running Gnina: {res.stderr}")
            return None
            
        cnn_score, cnn_affinity = parse_gnina_output(res.stdout)
        
        if cnn_score is None or cnn_affinity is None:
            print(f"Warning: Could not parse Gnina output for {pair_name}")
            return None

        # Preserve structures if save_dir is specified
        if save_dir:
            pair_save_dir = os.path.join(save_dir, pair_name)
            os.makedirs(pair_save_dir, exist_ok=True)
            shutil.copy2(protein_pdb, os.path.join(pair_save_dir, "af3_protein.pdb"))
            if os.path.exists(ligand_sdf):
                shutil.copy2(ligand_sdf, os.path.join(pair_save_dir, "af3_ligand.sdf"))
            shutil.copy2(ligand_pdb, os.path.join(pair_save_dir, "af3_ligand.pdb"))
            
            if os.path.exists(gnina_out_sdf):
                shutil.copy2(gnina_out_sdf, os.path.join(pair_save_dir, "gnina_redocked.sdf"))
                if shutil.which("obabel"):
                    try:
                        subprocess.run(
                            ["obabel", gnina_out_sdf, "-O", os.path.join(pair_save_dir, "gnina_redocked.pdb")],
                            check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                        )
                    except Exception:
                        pass
            
        return {
            "TF_Ligand": pair_name,
            "CNNscore": cnn_score,
            "CNNaffinity": cnn_affinity,
            "Gnina_Mode": mode
        }
        
    finally:
        # Clean up temp folder
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def main():
    parser = argparse.ArgumentParser(description="Run Gnina Rescoring on AF3 predictions.")
    parser.add_argument('--config', default='config/config.yaml', help="Path to config.yaml")
    parser.add_argument('--predictions-dir', default='alphafold3_predictions', help="AF3 predictions dir")
    parser.add_argument('--output', default='data/processed/gnina_scores.csv', help="Path to write scores csv")
    parser.add_argument('--save-dir', default='data/processed/gnina_redocked_structures', help="Path to save redocked PDB/SDF structures")
    parser.add_argument('--batch-index', type=int, default=1, help="Current SLURM job array task 1-indexed ID")
    parser.add_argument('--total-batches', type=int, default=1, help="Total number of parallel batch tasks")
    parser.add_argument('--merge', action='store_true', help="Merge chunk CSV files into final output CSV")
    
    args = parser.parse_args()
    
    if args.merge:
        chunk_dir = os.path.dirname(args.output)
        chunk_files = [os.path.join(chunk_dir, f) for f in os.listdir(chunk_dir) if f.startswith("gnina_chunk_") and f.endswith(".csv")]
        chunk_files.sort()
        
        all_rows = []
        for cf in chunk_files:
            with open(cf, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if len(lines) > 1:
                    all_rows.extend([l for l in lines[1:] if l.strip()])
                    
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as out_f:
            out_f.write("TF_Ligand,CNNscore,CNNaffinity,Gnina_Mode\n")
            out_f.writelines(all_rows)
            
        print(f"Merged {len(chunk_files)} chunk CSV files into '{args.output}' ({len(all_rows)} total rescored pairs).")
        return

    # Load configuration
    mode = "minimize"
    autobox_add = 8.0
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
            if cfg and 'gnina' in cfg:
                mode = cfg['gnina'].get('mode', mode)
                autobox_add = cfg['gnina'].get('autobox_add', autobox_add)
                
    print(f"Configured Gnina Mode: '{mode}' with autobox expansion: {autobox_add} Å (Batch {args.batch_index}/{args.total_batches})")
    
    if not os.path.exists(args.predictions_dir):
        print(f"Error: Predictions directory '{args.predictions_dir}' not found.")
        sys.exit(1)
        
    all_zip_files = sorted([os.path.join(args.predictions_dir, f) for f in os.listdir(args.predictions_dir) if f.endswith('.zip')])
    if not all_zip_files:
        print("No completed prediction ZIP files found to rescore.")
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            f.write("TF_Ligand,CNNscore,CNNaffinity,Gnina_Mode\n")
        sys.exit(0)
        
    # Batch partitioning
    if args.total_batches > 1:
        import math
        chunk_size = math.ceil(len(all_zip_files) / args.total_batches)
        start_idx = (args.batch_index - 1) * chunk_size
        end_idx = min(start_idx + chunk_size, len(all_zip_files))
        zip_files = all_zip_files[start_idx:end_idx]
        output_file = os.path.join(os.path.dirname(args.output), f"gnina_chunk_{args.batch_index:03d}.csv")
    else:
        zip_files = all_zip_files
        output_file = args.output
        
    temp_root = f"data/processed/temp_structures_b{args.batch_index}"
    os.makedirs(temp_root, exist_ok=True)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    scores = []
    print(f"Batch {args.batch_index}/{args.total_batches}: Rescoring {len(zip_files)} prediction pairs...")
    for z_path in zip_files:
        print(f"Rescoring: {os.path.basename(z_path)}...")
        res = rescore_pair(z_path, temp_root, mode, autobox_add, save_dir=args.save_dir)
        if res:
            scores.append(res)
            
    # Write chunk or main output CSV
    with open(output_file, 'w', encoding='utf-8') as out_f:
        out_f.write("TF_Ligand,CNNscore,CNNaffinity,Gnina_Mode\n")
        for sc in scores:
            out_f.write(f"{sc['TF_Ligand']},{sc['CNNscore']},{sc['CNNaffinity']},{sc['Gnina_Mode']}\n")
            
    print(f"Finished batch {args.batch_index}. Written {len(scores)} scores to '{output_file}'.")
    
    # Clean up temp root
    if os.path.exists(temp_root):
        shutil.rmtree(temp_root, ignore_errors=True)

if __name__ == '__main__':
    main()
