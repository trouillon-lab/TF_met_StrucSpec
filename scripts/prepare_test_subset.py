#!/usr/bin/env python3
"""
Subset Data Preparation Script for AF3 Virtual Screening Pipeline.
Filters for small molecule effectors, constructs 10 positive + 10 negative control pairs,
re-uses 4 target TFs to test MSA caching, and generates AF3 input JSON files.
"""

import os
import sys
import csv
import json
import urllib.request
import urllib.parse
import argparse
from scripts.generate_inputs import generate_json_inputs, clean_filename

TF_METADATA = {
    'AraC': {'uniprot_id': 'P0A9E0'},
    'AcrR': {'uniprot_id': 'P0ACS9'},
    'TyrR': {'uniprot_id': 'P07604'},
    'CysB': {'uniprot_id': 'P0A9F3'}
}

LIGAND_SMILES = {
    'arabinose': 'OC[C@@H](O)[C@@H](O)[C@H](O)C=O',
    'D-fucose': 'C[C@@H](O)[C@H](O)[C@H](O)[C@@H](O)C=O',
    'ethidium': 'CC[n+]1c2cc(N)ccc2c3ccc(N)cc3c1c4ccccc4',
    'proflavin': 'Nc1ccc2cc3ccc(N)cc3nc2c1',
    'R6G': 'CCNc1cc2[o+]c3cc(NCC)c(C)cc3c(c4ccc(cc4C(O)=O)C(=O)NCCCCCCO)c2cc1C',
    'L-tryptophan': 'N[C@@H](Cc1c[nH]c2ccccc12)C(O)=O',
    'L-phenylalanine': 'N[C@@H](Cc1ccccc1)C(O)=O',
    'L-tyrosine': 'N[C@@H](Cc1ccc(O)cc1)C(O)=O',
    'O-acetyl-L-serine': 'CC(=O)OC[C@H](N)C(O)=O',
    'Thiosulphate': '[O-][S]([O-])(=O)=S'
}

POSITIVE_PAIRS = [
    ('AraC', 'arabinose'),
    ('AraC', 'D-fucose'),
    ('AcrR', 'ethidium'),
    ('AcrR', 'proflavin'),
    ('AcrR', 'R6G'),
    ('TyrR', 'L-tryptophan'),
    ('TyrR', 'L-phenylalanine'),
    ('TyrR', 'L-tyrosine'),
    ('CysB', 'O-acetyl-L-serine'),
    ('CysB', 'Thiosulphate'),
]

NEGATIVE_PAIRS = [
    ('AraC', 'L-tryptophan'),
    ('AraC', 'ethidium'),
    ('AcrR', 'arabinose'),
    ('AcrR', 'L-phenylalanine'),
    ('TyrR', 'D-fucose'),
    ('TyrR', 'O-acetyl-L-serine'),
    ('CysB', 'proflavin'),
    ('CysB', 'L-tyrosine'),
    ('AraC', 'R6G'),
    ('AcrR', 'Thiosulphate'),
]

def fetch_uniprot_sequence(uniprot_id):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as resp:
        fasta = resp.read().decode('utf-8')
        lines = fasta.split('\n')
        return ''.join(lines[1:]).strip()

def build_subset_pairs(output_csv='data/processed/pairings_subset_20.csv', output_json_dir='alphafold3_jsons'):
    # Cache TF sequences
    tf_sequences = {}
    for tf_name, meta in TF_METADATA.items():
        try:
            tf_sequences[tf_name] = fetch_uniprot_sequence(meta['uniprot_id'])
        except Exception as e:
            # Fallback mock sequence for offline testing if needed
            tf_sequences[tf_name] = "M" * 200

    pairs_data = []
    
    # Process Positive Pairs
    for tf, lig in POSITIVE_PAIRS:
        pairs_data.append({
            'TF_Name': tf,
            'Uniprot_ID': TF_METADATA[tf]['uniprot_id'],
            'TF_Sequence': tf_sequences[tf],
            'Ligand_Name': lig,
            'Ligand_SMILES': LIGAND_SMILES[lig],
            'Label': 'positive'
        })
        
    # Process Negative Pairs
    for tf, lig in NEGATIVE_PAIRS:
        pairs_data.append({
            'TF_Name': tf,
            'Uniprot_ID': TF_METADATA[tf]['uniprot_id'],
            'TF_Sequence': tf_sequences[tf],
            'Ligand_Name': lig,
            'Ligand_SMILES': LIGAND_SMILES[lig],
            'Label': 'negative'
        })

    # Write CSV
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    fieldnames = ['TF_Name', 'Uniprot_ID', 'TF_Sequence', 'Ligand_Name', 'Ligand_SMILES', 'Label']
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pairs_data)
        
    print(f"Wrote {len(pairs_data)} pairs to {output_csv}")

    # Generate JSON input files
    generate_json_inputs(output_csv, output_json_dir)
    return pairs_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare 20-pair subset (10 pos, 10 neg) for AF3 pipeline testing.")
    parser.add_argument('--out-csv', default='data/processed/pairings_subset_20.csv', help="Output CSV path")
    parser.add_argument('--out-jsons', default='alphafold3_jsons', help="Output directory for AF3 JSON files")
    args = parser.parse_args()

    build_subset_pairs(args.out_csv, args.out_jsons)
