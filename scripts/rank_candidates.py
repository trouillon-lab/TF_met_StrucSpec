#!/usr/bin/env python3
"""
Scoring and Ranking script for AF3 Virtual Screening.
Combines AF3 structural confidence with Gnina rescoring, ranks candidates,
and outputs validation statistics/plots for true positives.
"""

import os
import sys
import re
import csv
import json
import zipfile
import argparse

from pipeline_utils import (
    GNINA_FALLBACK,
    sanitize_key,
    parse_af3_summary,
    load_gnina_scores,
    generate_svg_slopegraph,
)


def load_gt_map():
    """Loads ground truth mapping (TF_Name, Ligand_Name, Is_Positive) from dataset CSV files."""
    gt_map = {}
    csv_files = [
        'data/processed/pairings_subset_20.csv',
        'data/processed/pairings_remaining_248.csv',
        'data/processed/pairings_score2_benchmark.csv'
    ]
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            with open(csv_file, 'r', encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    tf = row['TF_Name'].strip()
                    lig = row['Ligand_Name'].strip()
                    kegg = row.get('KEGG_ID', '').strip()
                    is_pos = row.get('Label', '').strip().lower() == 'positive'

                    item = {
                        "tf_name": tf,
                        "ligand_name": lig,
                        "is_positive": is_pos
                    }
                    gt_map[sanitize_key(f"{tf}_{lig}")] = item
                    if kegg:
                        gt_map[sanitize_key(f"{tf}_{kegg}")] = item
    return gt_map


def load_true_positives(pairings_csv=None):
    """Compatibility wrapper returning set of true positive TF_Ligand keys."""
    gt = load_gt_map()
    return {f"{v['tf_name']}_{v['ligand_name']}" for v in gt.values() if v['is_positive']}


def main():
    parser = argparse.ArgumentParser(description="Rank candidates and compile virtual screening reports.")
    parser.add_argument('--pairings', default='data/raw/pairings.csv', help="Path to raw pairings.csv")
    parser.add_argument('--predictions-dir', default='alphafold3_predictions', help="Path to predictions zip dir")
    parser.add_argument('--gnina-scores', default='data/processed/gnina_scores.csv', help="Path to gnina scores csv")
    parser.add_argument('--output-report', default='results/ranked_pairings_report.csv', help="Path to final ranked report csv")
    parser.add_argument('--svg-output', default='results/true_positive_comparison.svg', help="Path to validation SVG output")

    args = parser.parse_args()

    # 1. Load Ground Truth Mapping
    gt_map = load_gt_map()
    pos_cnt = sum(1 for v in gt_map.values() if v['is_positive'])
    print(f"Loaded ground truth mapping for {len(gt_map)} dataset pairs ({pos_cnt} true positives).")

    # 2. Load Gnina scores (sanitized keys; missing pairs use GNINA_FALLBACK)
    gnina_data = load_gnina_scores([args.gnina_scores])
    if not gnina_data:
        print(f"Warning: Gnina scores CSV '{args.gnina_scores}' not found or empty. "
              "Gnina rescoring details will be excluded.")

    # 3. Parse AF3 predictions
    if not os.path.exists(args.predictions_dir):
        print(f"Error: Predictions directory '{args.predictions_dir}' not found.")
        sys.exit(1)

    zip_files = [f for f in os.listdir(args.predictions_dir) if f.endswith('.zip')]

    compiled_results = []

    for z_file in zip_files:
        zip_path = os.path.join(args.predictions_dir, z_file)
        basename = os.path.splitext(z_file)[0]
        pair_name = basename.replace("_predictions", "")
        s_key = sanitize_key(pair_name)

        if s_key in gt_map:
            tf_name = gt_map[s_key]['tf_name']
            ligand_name = gt_map[s_key]['ligand_name']
            is_tp = gt_map[s_key]['is_positive']
        else:
            parts = pair_name.split('_', 1)
            tf_name = parts[0] if len(parts) > 0 else "Unknown_TF"
            ligand_name = parts[1] if len(parts) > 1 else "Unknown_Ligand"
            is_tp = False

        af3_metrics = parse_af3_summary(zip_path)
        if not af3_metrics:
            continue

        pae = af3_metrics['pae_min']
        iptm = af3_metrics['iptm']
        clash = af3_metrics['has_clash']

        # Solution 2: iptm / (1.0 + pae)
        af3_score = iptm / (1.0 + pae)
        if clash:
            af3_score = 0.0

        gn = gnina_data.get(s_key, GNINA_FALLBACK)
        cnn_s = gn['CNNscore']
        cnn_a = gn['CNNaffinity']
        cnn_vs = gn['CNN_VS']

        consensus_score = af3_score * cnn_vs

        compiled_results.append({
            "TF_Name": tf_name,
            "Ligand_Name": ligand_name,
            "TF_Ligand": pair_name,
            "AF3_ipTM": iptm,
            "AF3_PAE_min": pae,
            "AF3_Has_Clash": clash,
            "AF3_Score": af3_score,
            "Gnina_CNNscore": cnn_s,
            "Gnina_CNNaffinity": cnn_a,
            "Gnina_CNN_VS": cnn_vs,
            "Consensus_Score": consensus_score,
            "Is_True_Positive": is_tp
        })

    if not compiled_results:
        print("No predictions parsed successfully. Exiting.")
        sys.exit(0)

    # 4. Group by TF and compute per-TF ranks
    tf_groups = {}
    for res in compiled_results:
        tf_groups.setdefault(res['TF_Name'], []).append(res)

    ranked_results = []
    validation_data = []

    for tf, candidates in tf_groups.items():
        candidates.sort(key=lambda x: x['AF3_Score'], reverse=True)
        for idx, item in enumerate(candidates):
            item['AF3_Rank'] = idx + 1

        candidates.sort(key=lambda x: x['Consensus_Score'], reverse=True)
        for idx, item in enumerate(candidates):
            item['Consensus_Rank'] = idx + 1
            ranked_results.append(item)

            if item['Is_True_Positive']:
                validation_data.append({
                    "TF_Ligand": item['TF_Ligand'],
                    "af3_rank": item['AF3_Rank'],
                    "consensus_rank": item['Consensus_Rank'],
                    "total_candidates": len(candidates)
                })

    # 5. Export Report CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.output_report)), exist_ok=True)
    with open(args.output_report, mode='w', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "TF_Name", "Ligand_Name", "TF_Ligand", "AF3_ipTM", "AF3_PAE_min",
            "AF3_Has_Clash", "AF3_Score", "AF3_Rank", "Gnina_CNNscore",
            "Gnina_CNNaffinity", "Gnina_CNN_VS", "Consensus_Score", "Consensus_Rank", "Is_True_Positive"
        ])
        writer.writeheader()
        writer.writerows(ranked_results)

    print(f"Final rankings report written to '{args.output_report}'.")

    # 6. Generate Validation Slopegraph
    generate_svg_slopegraph(validation_data, args.svg_output)

    # 7. Print Terminal Summary
    print("\n" + "="*80)
    print(f"{'TF Name':<15} | {'Top Ligand (Consensus)':<25} | {'AF3 Rank':<10} | {'Cons Rank':<10} | {'Cons Score':<10}")
    print("-"*80)
    for tf, candidates in tf_groups.items():
        top_cand = candidates[0]
        print(f"{tf:<15} | {top_cand['Ligand_Name']:<25} | #{top_cand['AF3_Rank']:<9} | #{top_cand['Consensus_Rank']:<9} | {top_cand['Consensus_Score']:.4f}")
    print("="*80 + "\n")

    if validation_data:
        print("True Positive Rankings Summary:")
        print("-"*80)
        print(f"{'TF_Ligand':<30} | {'AF3 Alone Rank':<15} | {'Consensus Rank':<15} | {'Improvement?':<15}")
        print("-"*80)
        for val in validation_data:
            diff = val['af3_rank'] - val['consensus_rank']
            status = "IMPROVED" if diff > 0 else ("DECLINED" if diff < 0 else "NO CHANGE")
            print(f"{val['TF_Ligand']:<30} | #{val['af3_rank']:<14} | #{val['consensus_rank']:<14} | {status:<15} (change: {diff:+})")
        print("="*80 + "\n")


if __name__ == '__main__':
    main()
