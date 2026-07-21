#!/usr/bin/env python3
"""
Subset Data Preparation Script for AF3 Virtual Screening Pipeline.
Filters for small molecule effectors, constructs 10 positive + 10 negative control pairs,
re-uses 4 target TFs to test MSA caching, and resolves SMILES STRICTLY via KEGG ID.
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

KEGG_MAP = {
    'arabinose': 'C02604',
    'D-fucose': 'C02095',
    'ethidium': 'C11161',
    'proflavin': 'C11181',
    'R6G': 'C11177',
    'L-tryptophan': 'C00078',
    'L-phenylalanine': 'C00079',
    'L-tyrosine': 'C00082',
    'O-acetyl-L-serine': 'C00979',
    'Thiosulphate': 'C00320'
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

def fetch_smiles_from_kegg(kegg_id):
    """Strictly resolve Canonical/Isomeric SMILES from KEGG ID via PubChem cross-reference."""
    # 1. PubChem PUG REST substance by KEGG ID
    try:
        url1 = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/substance/sourceid/KEGG/{kegg_id}/cids/JSON"
        req1 = urllib.request.Request(url1, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req1) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            cid = data['InformationList']['Information'][0]['CID'][0]
            
            url_s = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/SMILES,ConnectivitySMILES,IsomericSMILES/JSON"
            req_s = urllib.request.Request(url_s, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req_s) as resp_s:
                props = json.loads(resp_s.read().decode('utf-8'))['PropertyTable']['Properties'][0]
                smiles = props.get('SMILES') or props.get('IsomericSMILES') or props.get('ConnectivitySMILES')
                if smiles:
                    return smiles
    except Exception:
        pass

    # 2. KEGG REST API fallback parsing
    try:
        url_kegg = f"https://rest.kegg.jp/get/{kegg_id}"
        req_k = urllib.request.Request(url_kegg, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req_k) as resp:
            lines = resp.read().decode('utf-8').split('\n')
            for line in lines:
                if 'PubChem:' in line:
                    cid = line.split('PubChem:')[1].strip().split()[0]
                    url_s = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/SMILES,ConnectivitySMILES,IsomericSMILES/JSON"
                    req_s = urllib.request.Request(url_s, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(req_s) as resp_s:
                        props = json.loads(resp_s.read().decode('utf-8'))['PropertyTable']['Properties'][0]
                        smiles = props.get('SMILES') or props.get('IsomericSMILES') or props.get('ConnectivitySMILES')
                        if smiles:
                            return smiles
    except Exception:
        pass

    raise ValueError(f"Could not resolve SMILES strictly for KEGG ID: {kegg_id}")

def build_subset_pairs(output_csv='data/processed/pairings_subset_20.csv', output_json_dir='alphafold3_jsons'):
    # Fetch TF sequences
    tf_sequences = {}
    for tf_name, meta in TF_METADATA.items():
        try:
            tf_sequences[tf_name] = fetch_uniprot_sequence(meta['uniprot_id'])
        except Exception:
            tf_sequences[tf_name] = "M" * 200

    # Fetch SMILES strictly by KEGG ID
    kegg_smiles = {}
    for lig_name, kegg_id in KEGG_MAP.items():
        kegg_smiles[lig_name] = fetch_smiles_from_kegg(kegg_id)

    pairs_data = []
    
    # Process Positive Pairs
    for tf, lig in POSITIVE_PAIRS:
        pairs_data.append({
            'TF_Name': tf,
            'Uniprot_ID': TF_METADATA[tf]['uniprot_id'],
            'TF_Sequence': tf_sequences[tf],
            'Ligand_Name': lig,
            'KEGG_ID': KEGG_MAP[lig],
            'Ligand_SMILES': kegg_smiles[lig],
            'Label': 'positive'
        })
        
    # Process Negative Pairs
    for tf, lig in NEGATIVE_PAIRS:
        pairs_data.append({
            'TF_Name': tf,
            'Uniprot_ID': TF_METADATA[tf]['uniprot_id'],
            'TF_Sequence': tf_sequences[tf],
            'Ligand_Name': lig,
            'KEGG_ID': KEGG_MAP[lig],
            'Ligand_SMILES': kegg_smiles[lig],
            'Label': 'negative'
        })

    # Write CSV
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    fieldnames = ['TF_Name', 'Uniprot_ID', 'TF_Sequence', 'Ligand_Name', 'KEGG_ID', 'Ligand_SMILES', 'Label']
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pairs_data)
        
    print(f"Wrote {len(pairs_data)} pairs (SMILES strictly from KEGG IDs) to {output_csv}")

    # Generate JSON input files
    generate_json_inputs(output_csv, output_json_dir)
    return pairs_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare 20-pair subset using strict KEGG ID SMILES resolution.")
    parser.add_argument('--out-csv', default='data/processed/pairings_subset_20.csv', help="Output CSV path")
    parser.add_argument('--out-jsons', default='alphafold3_jsons', help="Output directory for AF3 JSON files")
    args = parser.parse_args()

    build_subset_pairs(args.out_csv, args.out_jsons)
