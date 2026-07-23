#!/usr/bin/env python3
"""
Subset Data Preparation Script for AF3 Virtual Screening Pipeline.
Filters raw curated dataset for small molecule effectors, constructs a 20-pair test subset
(10 positive + 10 negative pairs) using dynamic API/CSV metadata lookup without hardcoded dictionaries.
"""

import os
import sys
import csv
import json
import random
import argparse
import pandas as pd
from scripts.generate_inputs import generate_json_inputs
from scripts.prepare_score2_dataset import load_cache, save_cache, resolve_uniprot_by_gene
from scripts.cache_data import fetch_smiles_from_kegg

def build_subset_pairs(
    raw_csv='data/raw/tf_effectors_curated_2607.csv',
    output_csv='data/processed/pairings_subset_20.csv',
    output_json_dir='alphafold3_jsons'
):
    """Builds a 20-pair test subset dynamically from curated raw dataset CSV."""
    if not os.path.exists(raw_csv):
        raise FileNotFoundError(f"Raw CSV dataset file '{raw_csv}' not found.")
        
    cache = load_cache()
    df = pd.read_csv(raw_csv)
    
    # Filter small molecule effectors
    df['Effector type'] = df['Effector type'].astype(str).str.strip()
    sm_df = df[df['Effector type'] == 'small molecule'].copy()
    
    # Target subset TFs (AraC, AcrR, TyrR, CysB)
    target_tfs = ['AraC', 'AcrR', 'TyrR', 'CysB']
    subset_df = sm_df[sm_df['Transcription factor'].isin(target_tfs)].copy()
    
    # Deduplicate TF-effector pairs
    subset_df = subset_df.drop_duplicates(subset=['Transcription factor', 'kegg_id']).copy()
    
    # Select 10 distinct positive pairs
    pos_pairs = subset_df.head(10).copy()
    pos_pairs['Label'] = 'positive'
    
    # Generate 10 distinct decoy negative pairs
    all_tfs = list(set(sm_df['Transcription factor']))
    all_keggs = sm_df[['kegg_id', 'effector_name']].drop_duplicates().to_dict('records')
    
    pos_set = set(zip(pos_pairs['Transcription factor'], pos_pairs['kegg_id']))
    all_known_set = set(zip(sm_df['Transcription factor'], sm_df['kegg_id']))
    
    neg_rows = []
    attempts = 0
    random_gen = random.Random(42)
    
    while len(neg_rows) < 10 and attempts < 10000:
        attempts += 1
        tf = random_gen.choice(target_tfs)
        kegg_item = random_gen.choice(all_keggs)
        k_id = kegg_item['kegg_id']
        eff_name = kegg_item['effector_name']
        
        cand = (tf, k_id)
        if cand not in all_known_set and cand not in [(r['Transcription factor'], r['kegg_id']) for r in neg_rows]:
            # Get UniProt ID for TF
            uid_matches = sm_df[sm_df['Transcription factor'] == tf]['Uniprot ID'].tolist()
            uid = uid_matches[0] if uid_matches else "P0A9E0"
            neg_rows.append({
                'Transcription factor': tf,
                'Uniprot ID': uid,
                'effector_name': eff_name,
                'kegg_id': k_id,
                'Label': 'negative'
            })
                
    neg_df = pd.DataFrame(neg_rows)
    combined_df = pd.concat([pos_pairs, neg_df], ignore_index=True)
    
    # Build processed pair records
    pairs_data = []
    for _, row in combined_df.iterrows():
        tf_name = str(row['Transcription factor']).strip()
        uid = str(row['Uniprot ID']).strip()
        lig_name = str(row['effector_name']).strip()
        kegg_id = str(row['kegg_id']).strip()
        label = str(row['Label']).strip()
        
        # Dynamic UniProt Sequence resolution
        if uid in cache.get('uniprot', {}) and cache['uniprot'][uid]:
            seq = cache['uniprot'][uid]
        else:
            acc, seq = resolve_uniprot_by_gene(tf_name, cache)
            
        # Dynamic KEGG SMILES resolution
        smiles = cache.get('kegg_smiles', {}).get(kegg_id)
        if not smiles:
            from scripts.cache_data import fetch_smiles_from_kegg
            smiles = fetch_smiles_from_kegg(kegg_id)
            if smiles:
                cache.setdefault('kegg_smiles', {})[kegg_id] = smiles
                save_cache(cache)
            else:
                smiles = "C"
                
        pairs_data.append({
            'TF_Name': tf_name,
            'Uniprot_ID': uid,
            'TF_Sequence': seq,
            'Ligand_Name': lig_name,
            'KEGG_ID': kegg_id,
            'Ligand_SMILES': smiles,
            'Label': label
        })
        
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    out_df = pd.DataFrame(pairs_data)
    out_df.to_csv(output_csv, index=False)
    print(f"Exported subset dataset ({len(out_df)} pairs) to '{output_csv}'.")
    
    generate_json_inputs(output_csv, output_json_dir)
    return pairs_data

def main():
    parser = argparse.ArgumentParser(description="Prepare subset pairings dataset using dynamic metadata lookup.")
    parser.add_argument('--raw-csv', default='data/raw/tf_effectors_curated_2607.csv', help="Raw curated dataset path")
    parser.add_argument('--out-csv', default='data/processed/pairings_subset_20.csv', help="Output CSV path")
    parser.add_argument('--out-jsons', default='alphafold3_jsons', help="Output directory for AF3 JSON files")
    args = parser.parse_args()

    build_subset_pairs(raw_csv=args.raw_csv, output_csv=args.out_csv, output_json_dir=args.out_jsons)

if __name__ == '__main__':
    main()
