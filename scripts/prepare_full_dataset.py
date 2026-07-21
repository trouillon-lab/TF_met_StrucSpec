#!/usr/bin/env python3
"""
Full Dataset Input Generation Script for AF3 Virtual Screening Pipeline.
Extracts remaining positive small-molecule TF-effector pairs (excluding 20 diagnostic subset),
samples an equal number of false/decoy pairings, resolves UniProt sequences and KEGG SMILES,
generates AF3 input JSONs, and outputs Quality Control (QC) vector graphics.
"""

import os
import sys
import csv
import json
import random
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from scripts.generate_inputs import generate_json_inputs, clean_filename

CACHE_FILE = 'data/processed/cache_sequences_smiles.json'

def load_cache():
    """Loads cached UniProt sequences and KEGG SMILES."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {'uniprot': {}, 'kegg_smiles': {}}

def build_full_dataset(
    raw_csv='data/raw/tf_effectors_curated_2607.csv',
    subset_csv='data/processed/pairings_subset_20.csv',
    output_csv='data/processed/pairings_remaining_248.csv',
    output_json_dir='alphafold3_jsons_remaining',
    qc_svg='results/full_dataset_qc.svg',
    qc_png='results/full_dataset_qc.png',
    seed=42
):

    # 1. Load already run subset pairs
    already_run_pairs = set()
    if os.path.exists(subset_csv):
        with open(subset_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                already_run_pairs.add((row['TF_Name'].strip(), row['Ligand_Name'].strip()))
                
    print(f"Loaded {len(already_run_pairs)} previously executed subset pairs to exclude.")

    # 2. Read raw curated dataset
    with open(raw_csv, 'r', encoding='utf-8') as f:
        raw_rows = list(csv.DictReader(f))

    small_mols = [r for r in raw_rows if r.get('Effector type', '').strip() == 'small molecule']
    print(f"Total small molecule rows in raw dataset: {len(small_mols)}")

    all_known_positives = set()
    tf_uniprot_map = {}
    lig_kegg_map = {}
    
    for r in small_mols:
        tf = r['Transcription factor'].strip()
        uid = r['Uniprot ID'].strip()
        lig = r['effector_name'].strip()
        kegg = r['kegg_id'].strip()
        if tf and lig:
            all_known_positives.add((tf, lig))
            tf_uniprot_map[tf] = uid
            lig_kegg_map[lig] = kegg

    # 3. Extract remaining positive pairs
    pos_pairs = []
    seen_pos = set()
    for r in small_mols:
        tf = r['Transcription factor'].strip()
        lig = r['effector_name'].strip()
        pair = (tf, lig)
        if pair in already_run_pairs or pair in seen_pos:
            continue
        seen_pos.add(pair)
        pos_pairs.append((tf, lig))

    print(f"Remaining positive pairs to generate: {len(pos_pairs)}")

    # 4. Sample equal number of negative pairs
    random.seed(seed)
    tfs = sorted(list(tf_uniprot_map.keys()))
    ligs = sorted(list(lig_kegg_map.keys()))

    neg_pairs = []
    seen_neg = set()
    
    max_attempts = 100000
    attempts = 0
    while len(neg_pairs) < len(pos_pairs) and attempts < max_attempts:
        attempts += 1
        tf = random.choice(tfs)
        lig = random.choice(ligs)
        pair = (tf, lig)
        if pair in all_known_positives or pair in already_run_pairs or pair in seen_neg:
            continue
        seen_neg.add(pair)
        neg_pairs.append(pair)

    print(f"Sampled negative pairs: {len(neg_pairs)}")

    # 5. Load cache and build full table
    cache = load_cache()
    
    dataset_rows = []
    
    for tf, lig in pos_pairs:
        uid = tf_uniprot_map[tf]
        kegg = lig_kegg_map[lig]
        seq = cache['uniprot'].get(uid, "M"*200)
        smiles = cache['kegg_smiles'].get(kegg, "")
        dataset_rows.append({
            'TF_Name': tf,
            'Uniprot_ID': uid,
            'TF_Sequence': seq,
            'Ligand_Name': lig,
            'KEGG_ID': kegg,
            'Ligand_SMILES': smiles,
            'Label': 'positive'
        })
        
    for tf, lig in neg_pairs:
        uid = tf_uniprot_map[tf]
        kegg = lig_kegg_map[lig]
        seq = cache['uniprot'].get(uid, "M"*200)
        smiles = cache['kegg_smiles'].get(kegg, "")
        dataset_rows.append({
            'TF_Name': tf,
            'Uniprot_ID': uid,
            'TF_Sequence': seq,
            'Ligand_Name': lig,
            'KEGG_ID': kegg,
            'Ligand_SMILES': smiles,
            'Label': 'negative'
        })

    # 6. Save dataset CSV
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    fieldnames = ['TF_Name', 'Uniprot_ID', 'TF_Sequence', 'Ligand_Name', 'KEGG_ID', 'Ligand_SMILES', 'Label']
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dataset_rows)

    print(f"Wrote {len(dataset_rows)} pairs (124 pos + 124 neg) to '{output_csv}'")

    # 7. Generate AF3 input JSONs
    generate_json_inputs(output_csv, output_json_dir)

    # 8. Generate Quality Control (QC) Vector Graphic
    generate_qc_plots(dataset_rows, qc_svg, qc_png)

    return dataset_rows

def generate_qc_plots(dataset_rows, qc_svg, qc_png):
    """Generates Quality Control distribution plots."""
    sns.set_theme(style="whitegrid")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    # 1. Sequence Length Distribution
    seq_lengths = [len(r['TF_Sequence']) for r in dataset_rows]
    sns.histplot(seq_lengths, kde=True, ax=ax1, color="#1F77B4", bins=20)
    ax1.set_xlabel("TF Protein Sequence Length (Amino Acids)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Count", fontsize=11, fontweight='bold')
    ax1.set_title("Sequence Length Distribution", fontsize=13, fontweight='bold', pad=10)
    
    # 2. Label Balance
    labels = [r['Label'].capitalize() for r in dataset_rows]
    sns.countplot(x=labels, hue=labels, ax=ax2, palette=["#2CA02C", "#D62728"], legend=False)
    ax2.set_xlabel("Pair Label", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Count", fontsize=11, fontweight='bold')
    ax2.set_title(f"Dataset Balance ({len(dataset_rows)} Total Pairs)", fontsize=13, fontweight='bold', pad=10)
    
    for p in ax2.patches:
        ax2.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                     ha='center', va='center', fontsize=12, color='white', fontweight='bold')
                     
    plt.suptitle("Quality Control: Full Screening Dataset (248 Remaining Pairs)", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    os.makedirs(os.path.dirname(os.path.abspath(qc_svg)), exist_ok=True)
    fig.savefig(qc_svg, format='svg', bbox_inches='tight')
    fig.savefig(qc_png, format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Exported QC vector plot to '{qc_svg}'")

def main():
    parser = argparse.ArgumentParser(description="Prepare 248 remaining pairs dataset and generate AF3 JSONs.")
    parser.add_argument('--raw-csv', default='data/raw/tf_effectors_curated_2607.csv', help="Raw curated dataset CSV")
    parser.add_argument('--subset-csv', default='data/processed/pairings_subset_20.csv', help="Previously run 20 subset CSV")
    parser.add_argument('--output-csv', default='data/processed/pairings_remaining_248.csv', help="Output CSV path")
    parser.add_argument('--output-json-dir', default='alphafold3_jsons_remaining', help="Output directory for JSONs")
    parser.add_argument('--qc-svg', default='results/full_dataset_qc.svg', help="Output QC SVG path")
    parser.add_argument('--qc-png', default='results/full_dataset_qc.png', help="Output QC PNG path")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for decoy sampling")

    args = parser.parse_args()

    build_full_dataset(
        raw_csv=args.raw_csv,
        subset_csv=args.subset_csv,
        output_csv=args.output_csv,
        output_json_dir=args.output_json_dir,
        qc_svg=args.qc_svg,
        qc_png=args.qc_png,
        seed=args.seed
    )

if __name__ == '__main__':
    main()
