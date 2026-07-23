#!/usr/bin/env python3
"""
Scoring and Ranking script for AF3 Virtual Screening.
Combines AF3 structural confidence with Gnina rescoring, ranks candidates,
and outputs validation statistics/plots for true positives.
"""

import os
import sys
import csv
import json
import zipfile
import argparse

def parse_af3_summary(zip_path):
    """Parses chain_pair_iptm, chain_pair_pae_min, and has_clash from AF3 zip."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Find confidence json
            conf_files = [f for f in zip_ref.namelist() if f.endswith('summary_confidences.json')]
            if not conf_files:
                return None
                
            conf_text = zip_ref.read(conf_files[0]).decode('utf-8')
            data = json.loads(conf_text)
            
            iptm = data.get('chain_pair_iptm')
            pae_min = data.get('chain_pair_pae_min')
            has_clash = data.get('has_clash', False)
            
            # Extract [0][1] for protein chain A and ligand chain B interaction
            val_iptm = None
            if iptm and isinstance(iptm, list):
                if len(iptm) > 0 and isinstance(iptm[0], list) and len(iptm[0]) > 1:
                    val_iptm = iptm[0][1]
                elif len(iptm) > 1:
                    val_iptm = iptm[0]  # Fallback if flat
            
            val_pae = None
            if pae_min and isinstance(pae_min, list):
                if len(pae_min) > 0 and isinstance(pae_min[0], list) and len(pae_min[0]) > 1:
                    val_pae = pae_min[0][1]
                elif len(pae_min) > 1:
                    val_pae = pae_min[0]
            
            # Fallback/default if missing
            if val_iptm is None:
                val_iptm = 0.5
            if val_pae is None:
                val_pae = 15.0
                
            return {
                "iptm": val_iptm,
                "pae_min": val_pae,
                "has_clash": bool(has_clash)
            }
    except Exception as e:
        print(f"Warning: Failed to parse summary confidences from {zip_path}: {e}")
        return None

def generate_svg_slopegraph(validation_data, output_path):
    """
    Generates a pure-Python vector SVG slopegraph comparing AF3 ranks vs. Consensus ranks.
    """
    # SVG Dimensions
    width = 600
    height = 500
    padding_top = 80
    padding_bottom = 50
    padding_left = 120
    padding_right = 120
    
    svg = []
    # SVG Header
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%">')
    # Background and Styles
    svg.append('<rect width="100%" height="100%" fill="#111827"/>') # Modern dark slate background
    svg.append('<style>')
    svg.append('  .title { font-family: "Outfit", "Inter", sans-serif; font-size: 18px; fill: #F3F4F6; font-weight: bold; text-anchor: middle; }')
    svg.append('  .axis-label { font-family: "Inter", sans-serif; font-size: 14px; fill: #9CA3AF; font-weight: 600; text-anchor: middle; }')
    svg.append('  .rank-val { font-family: monospace; font-size: 12px; fill: #D1D5DB; font-weight: bold; }')
    svg.append('  .line-active { stroke: #10B981; stroke-width: 3; stroke-linecap: round; transition: all 0.3s; }') # Green for improvement
    svg.append('  .line-inactive { stroke: #EF4444; stroke-width: 3; stroke-linecap: round; transition: all 0.3s; }') # Red for decline
    svg.append('  .line-neutral { stroke: #6B7280; stroke-width: 2; stroke-dasharray: 4; stroke-linecap: round; }') # Neutral
    svg.append('  .node-dot { stroke-width: 2; fill: #1F2937; }')
    svg.append('  .label-text { font-family: sans-serif; font-size: 11px; fill: #E5E7EB; }')
    svg.append('</style>')
    
    # Title
    svg.append(f'<text x="{width/2}" y="35" class="title">True Positive Rank Validation (AF3 vs. Consensus)</text>')
    svg.append(f'<text x="{width/2}" y="55" font-family="sans-serif" font-size="12" fill="#9CA3AF" text-anchor="middle">Lower rank is better (Rank 1 = Top Candidate)</text>')
    
    # Left Axis (AF3 alone)
    svg.append(f'<text x="{padding_left}" y="{padding_top - 15}" class="axis-label">AF3 Alone</text>')
    svg.append(f'<line x1="{padding_left}" y1="{padding_top}" x2="{padding_left}" y2="{height - padding_bottom}" stroke="#4B5563" stroke-width="2"/>')
    
    # Right Axis (Consensus)
    svg.append(f'<text x="{width - padding_right}" y="{padding_top - 15}" class="axis-label">Consensus (AF3+Gnina)</text>')
    svg.append(f'<line x1="{width - padding_right}" y1="{padding_top}" x2="{width - padding_right}" y2="{height - padding_bottom}" stroke="#4B5563" stroke-width="2"/>')
    
    if not validation_data:
        svg.append(f'<text x="{width/2}" y="{height/2}" font-family="sans-serif" font-size="14" fill="#9CA3AF" text-anchor="middle">No True Positive data available to display</text>')
        svg.append('</svg>')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(svg))
        return

    # Draw lines and points
    # Max candidates sets the scale boundary
    max_rank = max(max(item['af3_rank'], item['consensus_rank'], 5) for item in validation_data)
    min_rank = 1
    
    y_start = padding_top
    y_end = height - padding_bottom
    y_span = y_end - y_start
    
    def get_y(rank):
        # Invert scale so Rank 1 is at the top (y_start) and max_rank is at the bottom (y_end)
        if max_rank == min_rank:
            return y_start + y_span / 2
        return y_start + ((rank - min_rank) / (max_rank - min_rank)) * y_span

    for item in validation_data:
        af3_r = item['af3_rank']
        cons_r = item['consensus_rank']
        label = item['TF_Ligand']
        
        y_af3 = get_y(af3_r)
        y_cons = get_y(cons_r)
        
        # Color based on performance improvement
        if cons_r < af3_r:
            line_class = "line-active"
            dot_color = "#10B981"
        elif cons_r > af3_r:
            line_class = "line-inactive"
            dot_color = "#EF4444"
        else:
            line_class = "line-neutral"
            dot_color = "#9CA3AF"
            
        # Draw slope line
        svg.append(f'  <line x1="{padding_left}" y1="{y_af3}" x2="{width - padding_right}" y2="{y_cons}" class="{line_class}"/>')
        
        # Left dot & rank text
        svg.append(f'  <circle cx="{padding_left}" cy="{y_af3}" r="5" class="node-dot" stroke="{dot_color}"/>')
        svg.append(f'  <text x="{padding_left - 12}" y="{y_af3 + 4}" class="rank-val" text-anchor="end">#{af3_r}</text>')
        svg.append(f'  <text x="{padding_left - 35}" y="{y_af3 + 4}" class="label-text" text-anchor="end">{label}</text>')
        
        # Right dot & rank text
        svg.append(f'  <circle cx="{width - padding_right}" cy="{y_cons}" r="5" class="node-dot" stroke="{dot_color}"/>')
        svg.append(f'  <text x="{width - padding_right + 12}" y="{y_cons + 4}" class="rank-val" text-anchor="start">#{cons_r}</text>')
        svg.append(f'  <text x="{width - padding_right + 35}" y="{y_cons + 4}" class="label-text" text-anchor="start">{label}</text>')
        
    svg.append('</svg>')
    
    # Ensure dir exists and write
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(svg))
    print(f"Generated validation SVG plot at '{output_path}'.")

import re

def sanitize_key(s):
    return re.sub(r'[^a-zA-Z0-9]', '', s).lower()

def load_gt_map():
    """Loads ground truth mapping (TF_Name, Ligand_Name, Is_Positive) from dataset CSV files."""
    gt_map = {}
    for csv_file in ['data/processed/pairings_subset_20.csv', 'data/processed/pairings_remaining_248.csv']:
        if os.path.exists(csv_file):
            with open(csv_file, 'r', encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    tf = row['TF_Name'].strip()
                    lig = row['Ligand_Name'].strip()
                    is_pos = row.get('Label', '').strip().lower() == 'positive'
                    s_key = sanitize_key(f"{tf}_{lig}")
                    gt_map[s_key] = {
                        "tf_name": tf,
                        "ligand_name": lig,
                        "is_positive": is_pos
                    }
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
    
    # 2. Load Gnina scores
    gnina_data = {}
    if os.path.exists(args.gnina_scores):
        with open(args.gnina_scores, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                pair = row['TF_Ligand']
                cnn_s = float(row['CNNscore'])
                cnn_a = float(row['CNNaffinity'])
                gnina_data[pair] = {
                    "CNNscore": cnn_s,
                    "CNNaffinity": cnn_a,
                    "CNN_VS": cnn_s * cnn_a
                }
    else:
        print(f"Warning: Gnina scores CSV '{args.gnina_scores}' not found. Gnina rescoring details will be excluded.")
        
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
        
        # Match pair against gt_map using sanitized key
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
            
        # Compute AF3 Score using Solution 2: denominator starts at 1.0 (iptm / (1.0 + pae))
        pae = af3_metrics['pae_min']
        iptm = af3_metrics['iptm']
        clash = af3_metrics['has_clash']
        
        af3_score = iptm / (1.0 + pae)
        if clash:
            af3_score = 0.0
            
        # Fetch Gnina metrics
        gnina_metrics = gnina_data.get(pair_name, {"CNNscore": 0.0, "CNNaffinity": 0.0, "CNN_VS": 0.0})
        cnn_s = gnina_metrics['CNNscore']
        cnn_a = gnina_metrics['CNNaffinity']
        cnn_vs = gnina_metrics['CNN_VS']
        
        # Consensus Score = AF3_Score * CNN_VS
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
        
    # 4. Group by TF and compute ranks
    # Group results
    tf_groups = {}
    for res in compiled_results:
        tf = res['TF_Name']
        if tf not in tf_groups:
            tf_groups[tf] = []
        tf_groups[tf].append(res)
        
    ranked_results = []
    validation_data = []
    
    for tf, candidates in tf_groups.items():
        # Rank by AF3 alone (descending)
        candidates.sort(key=lambda x: x['AF3_Score'], reverse=True)
        for idx, item in enumerate(candidates):
            item['AF3_Rank'] = idx + 1
            
        # Rank by Consensus (descending)
        candidates.sort(key=lambda x: x['Consensus_Score'], reverse=True)
        for idx, item in enumerate(candidates):
            item['Consensus_Rank'] = idx + 1
            ranked_results.append(item)
            
            # If it's a true positive, save for validation plot
            if item['Is_True_Positive']:
                validation_data.append({
                    "TF_Ligand": item['TF_Ligand'],
                    "af3_rank": item['AF3_Rank'],
                    "consensus_rank": item['Consensus_Rank'],
                    "total_candidates": len(candidates)
                })

    # 5. Export Report CSV
    os.makedirs(os.path.dirname(args.output_report), exist_ok=True)
    with open(args.output_report, mode='w', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "TF_Name", "Ligand_Name", "TF_Ligand", "AF3_ipTM", "AF3_PAE_min", 
            "AF3_Has_Clash", "AF3_Score", "AF3_Rank", "Gnina_CNNscore", 
            "Gnina_CNNaffinity", "Gnina_CNN_VS", "Consensus_Score", "Consensus_Rank", "Is_True_Positive"
        ])
        writer.writeheader()
        for row in ranked_results:
            writer.writerow(row)
            
    print(f"Final rankings report written to '{args.output_report}'.")
    
    # 6. Generate Validation Plot
    generate_svg_slopegraph(validation_data, args.svg_output)
    
    # 7. Print Terminal Summary
    print("\n" + "="*80)
    print(f"{'TF Name':<15} | {'Top Ligand (Consensus)':<25} | {'AF3 Rank':<10} | {'Cons Rank':<10} | {'Cons Score':<10}")
    print("-"*80)
    for tf, candidates in tf_groups.items():
        # Sorted by consensus
        top_cand = candidates[0]
        print(f"{tf:<15} | {top_cand['Ligand_Name']:<25} | #{top_cand['AF3_Rank']:<9} | #{top_cand['Consensus_Rank']:<9} | {top_cand['Consensus_Score']:.4f}")
    print("="*80 + "\n")
    
    # Print validation summary
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
