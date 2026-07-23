#!/usr/bin/env python3
"""
Score-2 Consensus Dataset Processor & AF3 Input JSON Generator
Parses consensus_rank_interval.csv, extracts score=2 positive pairs, generates an equal amount of
strictly non-existing decoy negative pairs, dynamically resolves gene names to UniProt sequences
and BiGG IDs to SMILES via UniProt/BiGG/PubChem REST APIs, caches resolved entities in JSON,
and outputs AF3 input JSON files for cluster execution.
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

def parse_pair(s):
    """Parses ('TF', 'BIGG') python string literal."""
    try:
        val = ast.literal_eval(s)
        return str(val[0]).strip(), str(val[1]).strip()
    except Exception:
        return None, None

def load_cache():
    """Loads local JSON cache file."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {'uniprot': {}, 'kegg_smiles': {}, 'gene_uniprot': {}, 'bigg_smiles': {}}

def save_cache(cache):
    """Saves updated cache to local JSON file."""
    os.makedirs(os.path.dirname(os.path.abspath(CACHE_FILE)), exist_ok=True)
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2)

def resolve_uniprot_by_gene(gene_name, cache):
    """
    Dynamically queries UniProt REST API for a gene name in E. coli MG1655 (organism 83333).
    Caches result to prevent redundant API calls.
    """
    gene_uniprot_cache = cache.setdefault('gene_uniprot', {})
    if gene_name in gene_uniprot_cache:
        acc, seq = gene_uniprot_cache[gene_name]
        return acc, seq

    # Known gene alias mappings for E. coli MG1655
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
    """
    Dynamically queries BiGG API & PubChem PUG REST API for a BiGG metabolite ID.
    Resolves name and canonical SMILES string, caching the result.
    """
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
            
            # 1. PubChem CID lookup
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
                    
            # 2. InChI Key lookup
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
                    
            # 3. Name lookup on PubChem
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
        smiles = "C"  # Methane fallback if completely unresolvable via API
        
    bigg_cache[bigg_id] = [name, smiles]
    save_cache(cache)
    return name, smiles

def process_consensus_dataset(
    raw_csv='data/raw/consensus_rank_interval.csv',
    out_csv='data/processed/pairings_score2_benchmark.csv',
    out_json_dir='alphafold3_jsons_score2',
    random_seed=42
):
    """Processes consensus dataset, selects score=2 positives & random non-existing negatives, and writes AF3 JSONs."""
    if not os.path.exists(raw_csv):
        raise FileNotFoundError(f"Input CSV file '{raw_csv}' not found.")
        
    random.seed(random_seed)
    cache = load_cache()
    
    raw_df = pd.read_csv(raw_csv)
    raw_df['tf'], raw_df['bigg'] = zip(*raw_df['tf_met_pair'].apply(parse_pair))
    
    # 1. Entire dataset pairs set
    all_pairs_set = set(zip(raw_df['tf'], raw_df['bigg']))
    all_tfs_global = sorted(raw_df['tf'].unique())
    all_biggs_global = sorted(raw_df['bigg'].unique())
    
    # 2. Select positive pairs (score == 2.0)
    score2_df = raw_df[raw_df['score'] == 2.0].copy()
    pos_pairs_set = set(zip(score2_df['tf'], score2_df['bigg']))
    
    tfs_pool = sorted(score2_df['tf'].unique())
    biggs_pool = sorted(score2_df['bigg'].unique())
    
    print(f"Found {len(score2_df)} positive pairs with score == 2.0 ({len(tfs_pool)} TFs, {len(biggs_pool)} BiGG IDs).")
    
    # 3. Sample equal number of decoy negative pairs (NOT in score 2 and NOT in entire dataset)
    neg_pairs = set()
    attempts = 0
    
    # First attempt from score 2 TFs and BiGG IDs pool
    while len(neg_pairs) < len(score2_df) and attempts < 100000:
        attempts += 1
        tf_cand = random.choice(tfs_pool)
        bigg_cand = random.choice(biggs_pool)
        cand = (tf_cand, bigg_cand)
        if cand not in all_pairs_set and cand not in pos_pairs_set and cand not in neg_pairs:
            neg_pairs.add(cand)
            
    # Fallback to global TF and BiGG pool if needed
    while len(neg_pairs) < len(score2_df) and attempts < 200000:
        attempts += 1
        tf_cand = random.choice(all_tfs_global)
        bigg_cand = random.choice(all_biggs_global)
        cand = (tf_cand, bigg_cand)
        if cand not in all_pairs_set and cand not in pos_pairs_set and cand not in neg_pairs:
            neg_pairs.add(cand)
            
    if len(neg_pairs) < len(score2_df):
        print(f"Warning: Could only sample {len(neg_pairs)} unique non-existing negative pairs from pool.", file=sys.stderr)
        
    print(f"Successfully sampled {len(neg_pairs)} decoy negative pairs (strictly non-existing in entire dataset).")
    
    # 4. Build output dataset rows dynamically via REST APIs
    output_rows = []
    
    # Process Positives
    for tf_name, bigg_id in sorted(pos_pairs_set):
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
        
    # Process Negatives
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
    print(f"Exported combined benchmark dataset to '{out_csv}' ({len(out_df)} total pairs).")
    
    # 5. Generate AF3 Input JSONs
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
    return out_csv, len(pos_pairs_set), len(neg_pairs)

def main():
    parser = argparse.ArgumentParser(description="Process consensus rank interval dataset and generate AF3 JSONs dynamically.")
    parser.add_argument('--raw-csv', default='data/raw/consensus_rank_interval.csv', help="Input consensus CSV")
    parser.add_argument('--out-csv', default='data/processed/pairings_score2_benchmark.csv', help="Output processed CSV")
    parser.add_argument('--out-json-dir', default='alphafold3_jsons_score2', help="Output AF3 JSON directory")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for negative pair sampling")
    
    args = parser.parse_args()
    
    process_consensus_dataset(
        raw_csv=args.raw_csv,
        out_csv=args.out_csv,
        out_json_dir=args.out_json_dir,
        random_seed=args.seed
    )

if __name__ == '__main__':
    main()
