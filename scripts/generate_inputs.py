#!/usr/bin/env python3
"""
Automated Input Generator for AF3 virtual screening pipeline.
Reads a pairings CSV file and outputs valid AlphaFold 3 input JSON files.
"""

import os
import sys
import csv
import json
import argparse
import re

def clean_filename(name):
    """Sanitize strings for safe filesystem names."""
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', str(name))

def generate_json_inputs(csv_path, output_dir):
    """
    Reads pairings.csv and generates JSON structures.
    """
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}", file=sys.stderr)
        return False

    os.makedirs(output_dir, exist_ok=True)
    count = 0

    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        # Verify required headers
        required_headers = ['TF_Name', 'TF_Sequence', 'Ligand_Name', 'Ligand_SMILES']
        for header in required_headers:
            if header not in reader.fieldnames:
                print(f"Error: Missing required column '{header}' in CSV file.", file=sys.stderr)
                return False

        for row in reader:
            tf_name = row['TF_Name'].strip()
            tf_seq = row['TF_Sequence'].strip()
            ligand_name = row['Ligand_Name'].strip()
            ligand_smiles = row['Ligand_SMILES'].strip()

            if not tf_name or not tf_seq or not ligand_name or not ligand_smiles:
                print(f"Warning: Skipping row with empty values: {row}", file=sys.stderr)
                continue

            clean_tf = clean_filename(tf_name)
            clean_ligand = clean_filename(ligand_name)
            filename = f"{clean_tf}_{clean_ligand}.json"
            
            # Construct standard AF3 JSON
            af3_data = {
                "dialect": "alphafold3",
                "version": 2,
                "name": f"{clean_tf}_{clean_ligand}",
                "sequences": [
                    {
                        "protein": {
                            "id": "A",
                            "sequence": tf_seq
                        }
                    },
                    {
                        "ligand": {
                            "id": "B",
                            "smiles": ligand_smiles
                        }
                    }
                ],
                "modelSeeds": [1]
            }

            output_path = os.path.join(output_dir, filename)
            with open(output_path, 'w', encoding='utf-8') as out_f:
                json.dump(af3_data, out_f, indent=2)
            
            count += 1

    print(f"Successfully generated {count} AlphaFold 3 input JSON files in '{output_dir}'.")
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate AlphaFold 3 input JSON files from a pairings CSV.")
    parser.add_argument('--csv', default='data/raw/pairings.csv', help="Path to input pairings CSV file")
    parser.add_argument('--out-dir', default='alphafold3_jsons', help="Directory to save JSON outputs")
    
    args = parser.parse_args()
    
    success = generate_json_inputs(args.csv, args.out_dir)
    sys.exit(0 if success else 1)
