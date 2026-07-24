#!/usr/bin/env python3
"""
Dataset 3 (Weekend Batch) Generator & AF3 Input JSON Creator
1. Parses consensus_rank_interval.csv, excluding all pairs previously run in Dataset 1 & Dataset 2.
2. Selects top N (default 150) positive pairs moving down the consensus rank (scores 1.9786 -> 1.8167).
3. Samples size-matched decoy negative pairs (150) strictly non-existing in consensus_rank_interval.csv.
4. Dynamically resolves UniProt sequences and BiGG/PubChem SMILES using local cache & REST APIs.
5. Outputs pairings_dataset3_weekend.csv and generates AF3 input JSON files in alphafold3_jsons_dataset3/.
"""

import os
import sys
import ast
import json
import csv
import re
import random
import urllib.request
import urllib.parse
import argparse
import pandas as pd

CACHE_FILE = 'data/processed/cache_sequences_smiles.json'

def clean_filename(name):
    """Sanitize strings for safe filesystem names."""
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', str(name))

def sanitize_key(s):
    return re.sub(r'[^a-zA-Z0-9]', '', str(s)).lower()

def parse_pair(s):
    """Parses ('TF', 'BIGG') python string literal."""
    try:
        val = ast.literal_eval(s)
        return str(val[0]).strip(), str(val[1]).strip()
    except Exception:
        return None, None

def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {'uniprot': {}, 'kegg_smiles': {}, 'gene_uniprot': {}, 'bigg_smiles': {}}

def save_cache(cache):
    os.makedirs(os.path.dirname(os.path.abspath(CACHE_FILE)), exist_ok=True)
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2)

def resolve_uniprot_by_gene(gene_name, cache):
    gene_uniprot_cache = cache.setdefault('gene_uniprot', {})
    if gene_name in gene_uniprot_cache:
        acc, seq = gene_uniprot_cache[gene_name]
        return acc, seq

    gene_query = 'ihfA' if gene_name == 'IHF' else ('ygfI' if gene_name == 'SrsR' else gene_name)
    url = f"https://rest.uniprot.org/uniprotkb/search?query=gene_exact:{gene_query}+AND+organism_id:83333+AND+reviewed:true&format=json"
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            if data.get('results'):
                res = data['results'][0]
                acc = res['primaryAccession']
                seq = res['sequence']['value']
                gene_uniprot_cache[gene_name] = [acc, seq]
                save_cache(cache)
                return acc, seq
    except Exception as e:
        print(f"Warning: UniProt REST API lookup failed for gene '{gene_name}': {e}", file=sys.stderr)
        
    return "UNKNOWN_ACC", "M"

def resolve_smiles_by_bigg(bigg_id, cache):
    bigg_cache = cache.setdefault('bigg_smiles', {})
    if bigg_id in bigg_cache:
        name, smiles = bigg_cache[bigg_id]
        return name, smiles

    clean_id = bigg_id.rsplit('_', 1)[0] if bigg_id.endswith(('_c', '_e', '_p')) else bigg_id
    url = f"http://bigg.ucsd.edu/api/v2/universal/metabolites/{clean_id}"
    
    name, smiles = clean_id, None
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            name = data.get('name', clean_id)
            links = data.get('database_links', {})
            
            if 'PubChem Compound' in links:
                cid = links['PubChem Compound'][0]['id']
                pc_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES,SMILES/JSON"
                try:
                    with urllib.request.urlopen(urllib.request.Request(pc_url, headers={'User-Agent': 'Mozilla/5.0'})) as pc_resp:
                        pc_data = json.loads(pc_resp.read().decode('utf-8'))
                        props = pc_data['PropertyTable']['Properties'][0]
                        smiles = props.get('CanonicalSMILES') or props.get('SMILES')
                except Exception:
                    pass
                    
            if not smiles and 'InChI Key' in links:
                inchikey = links['InChI Key'][0]['id']
                pc_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{inchikey}/property/CanonicalSMILES,SMILES/JSON"
                try:
                    with urllib.request.urlopen(urllib.request.Request(pc_url, headers={'User-Agent': 'Mozilla/5.0'})) as pc_resp:
                        pc_data = json.loads(pc_resp.read().decode('utf-8'))
                        props = pc_data['PropertyTable']['Properties'][0]
                        smiles = props.get('CanonicalSMILES') or props.get('SMILES')
                except Exception:
                    pass
                    
            if not smiles and name:
                pc_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{urllib.parse.quote(name)}/property/CanonicalSMILES,SMILES/JSON"
                try:
                    with urllib.request.urlopen(urllib.request.Request(pc_url, headers={'User-Agent': 'Mozilla/5.0'})) as pc_resp:
                        pc_data = json.loads(pc_resp.read().decode('utf-8'))
                        props = pc_data['PropertyTable']['Properties'][0]
                        smiles = props.get('CanonicalSMILES') or props.get('SMILES')
                except Exception:
                    pass
    except Exception as e:
        print(f"Warning: BiGG API lookup failed for '{bigg_id}': {e}", file=sys.stderr)

    if not smiles:
        smiles = "C"
        
    bigg_cache[bigg_id] = [name, smiles]
    save_cache(cache)
    return name, smiles

def prepare_weekend_dataset(
    raw_csv='data/raw/consensus_rank_interval.csv',
    n_positives=150,
    out_csv='data/processed/pairings_dataset3_weekend.csv',
    out_json_dir='alphafold3_jsons_dataset3',
    random_seed=42
):
    """Parses un-evaluated consensus rank interval pairs and generates weekend dataset & JSONs."""
    if not os.path.exists(raw_csv):
        raise FileNotFoundError(f"Raw consensus file '{raw_csv}' not found.")
        
    random.seed(random_seed)
    cache = load_cache()
    
    # 1. Collect all previously evaluated pair keys (Dataset 1 & Dataset 2)
    existing_keys = set()
    prev_files = [
        'data/processed/pairings_subset_20.csv',
        'data/processed/pairings_remaining_248.csv',
        'data/processed/pairings_score2_benchmark.csv'
    ]
    for pf in prev_files:
        if os.path.exists(pf):
            with open(pf, 'r', encoding='utf-8') as f:
                for r in csv.DictReader(f):
                    tf = r['TF_Name'].strip()
                    lig = r['Ligand_Name'].strip()
                    kegg = r.get('KEGG_ID', '').strip()
                    existing_keys.add(sanitize_key(f"{tf}_{lig}"))
                    if kegg:
                        existing_keys.add(sanitize_key(f"{tf}_{kegg}"))
                        
    # 2. Parse consensus_rank_interval.csv
    df = pd.read_csv(raw_csv)
    df['tf'], df['bigg'] = zip(*df['tf_met_pair'].apply(parse_pair))
    df['pair_key'] = [sanitize_key(f"{tf}_{bigg}") for tf, bigg in zip(df['tf'], df['bigg'])]
    
    all_consensus_pairs_set = set(zip(df['tf'], df['bigg']))
    all_tfs_global = sorted(df['tf'].unique())
    all_biggs_global = sorted(df['bigg'].unique())
    
    # Filter out already run pairs
    df['already_run'] = df['pair_key'].isin(existing_keys)
    unrun_df = df[~df['already_run']].sort_values('score', ascending=False)
    
    print(f"Total consensus candidate pool: {len(df)} pairs.")
    print(f"Previously evaluated pairs: {df['already_run'].sum()}. Un-evaluated remaining: {len(unrun_df)}.")
    
    # Select top N positive pairs moving down consensus rank
    top_pos_df = unrun_df.head(n_positives).copy()
    pos_pairs_list = list(zip(top_pos_df['tf'], top_pos_df['bigg']))
    
    score_max = top_pos_df['score'].max()
    score_min = top_pos_df['score'].min()
    print(f"Selected {len(pos_pairs_list)} positive pairs moving down consensus rank (Score interval: {score_max:.4f} -> {score_min:.4f}).")
    
    # 3. Sample equal number of size-matched decoy negative pairs
    # Decoy negative condition: pair (TF, BiGG) does NOT exist in consensus_rank_interval.csv AND has not been evaluated
    tfs_pool = sorted(top_pos_df['tf'].unique())
    biggs_pool = sorted(top_pos_df['bigg'].unique())
    
    neg_pairs = set()
    attempts = 0
    
    while len(neg_pairs) < len(pos_pairs_list) and attempts < 100000:
        attempts += 1
        tf_cand = random.choice(tfs_pool)
        bigg_cand = random.choice(biggs_pool)
        cand = (tf_cand, bigg_cand)
        cand_key = sanitize_key(f"{tf_cand}_{bigg_cand}")
        if cand not in all_consensus_pairs_set and cand_key not in existing_keys and cand not in neg_pairs:
            neg_pairs.add(cand)
            
    while len(neg_pairs) < len(pos_pairs_list) and attempts < 200000:
        attempts += 1
        tf_cand = random.choice(all_tfs_global)
        bigg_cand = random.choice(all_biggs_global)
        cand = (tf_cand, bigg_cand)
        cand_key = sanitize_key(f"{tf_cand}_{bigg_cand}")
        if cand not in all_consensus_pairs_set and cand_key not in existing_keys and cand not in neg_pairs:
            neg_pairs.add(cand)
            
    print(f"Sampled {len(neg_pairs)} size-matched decoy negative pairs (strictly non-existing in consensus_rank_interval.csv).")
    
    # 4. Resolve sequences & SMILES and build output rows
    output_rows = []
    
    print("Resolving sequences and SMILES for positive pairs...")
    for tf_name, bigg_id in pos_pairs_list:
        acc, seq = resolve_uniprot_by_gene(tf_name, cache)
        ligand_name, smiles = resolve_smiles_by_bigg(bigg_id, cache)
        output_rows.append({
            'TF_Name': tf_name,
            'Uniprot_ID': acc,
            'TF_Sequence': seq,
            'Ligand_Name': f"{bigg_id}_{clean_filename(ligand_name)}",
            'KEGG_ID': bigg_id,
            'Ligand_SMILES': smiles,
            'Label': 'positive'
        })
        
    print("Resolving sequences and SMILES for decoy negative pairs...")
    for tf_name, bigg_id in sorted(neg_pairs):
        acc, seq = resolve_uniprot_by_gene(tf_name, cache)
        ligand_name, smiles = resolve_smiles_by_bigg(bigg_id, cache)
        output_rows.append({
            'TF_Name': tf_name,
            'Uniprot_ID': acc,
            'TF_Sequence': seq,
            'Ligand_Name': f"{bigg_id}_{clean_filename(ligand_name)}",
            'KEGG_ID': bigg_id,
            'Ligand_SMILES': smiles,
            'Label': 'negative'
        })
        
    # Export CSV
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    out_df = pd.DataFrame(output_rows)
    out_df.to_csv(out_csv, index=False)
    print(f"Exported Dataset 3 weekend benchmark CSV to '{out_csv}' ({len(out_df)} total pairs: {len(pos_pairs_list)} Positives, {len(neg_pairs)} Negatives).")
    
    # 5. Generate AF3 Input JSON files
    os.makedirs(out_json_dir, exist_ok=True)
    json_count = 0
    
    for row in output_rows:
        clean_tf = clean_filename(row['TF_Name'])
        clean_lig = clean_filename(row['KEGG_ID'])
        job_name = f"{clean_tf}_{clean_lig}"
        
        af3_data = {
            "dialect": "alphafold3",
            "version": 2,
            "name": job_name,
            "sequences": [
                {
                    "protein": {
                        "id": "A",
                        "sequence": row['TF_Sequence']
                    }
                },
                {
                    "ligand": {
                        "id": "B",
                        "smiles": row['Ligand_SMILES']
                    }
                }
            ],
            "modelSeeds": [1]
        }
        
        json_path = os.path.join(out_json_dir, f"{job_name}.json")
        with open(json_path, 'w', encoding='utf-8') as out_f:
            json.dump(af3_data, out_f, indent=2)
        json_count += 1
        
    print(f"Exported {json_count} AlphaFold 3 input JSON files to '{out_json_dir}'.")
    return out_csv, len(pos_pairs_list), len(neg_pairs)

def main():
    parser = argparse.ArgumentParser(description="Prepare Dataset 3 (Weekend Batch) AF3 input JSONs from consensus rank interval.")
    parser.add_argument('--raw-csv', default='data/raw/consensus_rank_interval.csv', help="Path to raw consensus_rank_interval.csv")
    parser.add_argument('--n-positives', type=int, default=150, help="Number of top un-evaluated positive pairs to select")
    parser.add_argument('--out-csv', default='data/processed/pairings_dataset3_weekend.csv', help="Output processed CSV")
    parser.add_argument('--out-json-dir', default='alphafold3_jsons_dataset3', help="Output AF3 JSON directory")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for decoy negative sampling")
    
    args = parser.parse_args()
    
    prepare_weekend_dataset(
        raw_csv=args.raw_csv,
        n_positives=args.n_positives,
        out_csv=args.out_csv,
        out_json_dir=args.out_json_dir,
        random_seed=args.seed
    )

if __name__ == '__main__':
    main()
