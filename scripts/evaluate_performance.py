#!/usr/bin/env python3
"""
Primitive Classification & Ranking Evaluation Script
Evaluates ROC AUC, PR AUC, and Top-1 Accuracy comparing AF3 alone, GNINA alone, and Consensus (AF3+GNINA).
"""

import os
import csv

def trapz(y, x):
    """Pure Python trapezoidal integration for (x, y) coordinates."""
    return sum(0.5 * (x[i] - x[i-1]) * (y[i] + y[i-1]) for i in range(1, len(x)))

def compute_roc_auc(y_true, y_scores):
    """Computes Area Under ROC Curve via trapezoidal integration."""
    pairs = sorted(zip(y_scores, y_true), key=lambda x: x[0], reverse=True)
    y_true_sorted = [p[1] for p in pairs]
    
    n_pos = sum(y_true_sorted)
    n_neg = len(y_true_sorted) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0
        
    tpr = [0.0]
    fpr = [0.0]
    
    tp = 0
    fp = 0
    
    for label in y_true_sorted:
        if label:
            tp += 1
        else:
            fp += 1
        tpr.append(tp / n_pos)
        fpr.append(fp / n_neg)
        
    return float(trapz(tpr, fpr))

def compute_pr_auc(y_true, y_scores):
    """Computes Area Under Precision-Recall Curve."""
    pairs = sorted(zip(y_scores, y_true), key=lambda x: x[0], reverse=True)
    y_true_sorted = [p[1] for p in pairs]
    
    n_pos = sum(y_true_sorted)
    if n_pos == 0:
        return 0.0
        
    precisions = [1.0]
    recalls = [0.0]
    
    tp = 0
    fp = 0
    
    for label in y_true_sorted:
        if label:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / (tp + fp))
        recalls.append(tp / n_pos)
        
    return float(trapz(precisions, recalls))

def main():
    report_csv = 'results/ranked_pairings_report.csv'
    
    # Ground truth positive controls in 20-pair diagnostic dataset
    true_positive_pairs = {
        "AraC_D-fucose",
        "AcrR_proflavin",
        "CysB_Thiosulphate",
        "TyrR_L-phenylalanine"
    }
    
    if not os.path.exists(report_csv):
        print(f"Error: '{report_csv}' not found. Please run scripts/rank_candidates.py first.")
        return
        
    rows = []
    with open(report_csv, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            r['is_tp'] = 1 if r['TF_Ligand'] in true_positive_pairs else 0
            r['AF3_Score'] = float(r['AF3_Score'])
            r['AF3_ipTM'] = float(r['AF3_ipTM'])
            r['Gnina_CNNscore'] = float(r['Gnina_CNNscore'])
            r['Gnina_CNNaffinity'] = float(r['Gnina_CNNaffinity'])
            r['Gnina_CNN_VS'] = float(r['Gnina_CNN_VS'])
            r['Consensus_Score'] = float(r['Consensus_Score'])
            rows.append(r)
            
    y_true = [r['is_tp'] for r in rows]
    
    scores = {
        "AF3 Alone (AF3_Score)": [r['AF3_Score'] for r in rows],
        "AF3 ipTM": [r['AF3_ipTM'] for r in rows],
        "GNINA CNNscore": [r['Gnina_CNNscore'] for r in rows],
        "GNINA CNNaffinity": [r['Gnina_CNNaffinity'] for r in rows],
        "GNINA VS Score": [r['Gnina_CNN_VS'] for r in rows],
        "Consensus (AF3 + GNINA)": [r['Consensus_Score'] for r in rows]
    }
    
    print("\n" + "="*85)
    print(f"{'Method / Scoring Metric':<30} | {'ROC AUC':<10} | {'PR AUC':<10} | {'Top-1 Accuracy':<15}")
    print("-"*85)
    
    # Calculate Top-1 accuracy per TF
    tf_groups = {}
    for r in rows:
        tf = r['TF_Name']
        if tf not in tf_groups:
            tf_groups[tf] = []
        tf_groups[tf].append(r)
        
    for name, s_list in scores.items():
        roc_auc = compute_roc_auc(y_true, s_list)
        pr_auc = compute_pr_auc(y_true, s_list)
        
        # Top 1 accuracy calculation
        top1_count = 0
        key_map = {
            "AF3 Alone (AF3_Score)": "AF3_Score",
            "AF3 ipTM": "AF3_ipTM",
            "GNINA CNNscore": "Gnina_CNNscore",
            "GNINA CNNaffinity": "Gnina_CNNaffinity",
            "GNINA VS Score": "Gnina_CNN_VS",
            "Consensus (AF3 + GNINA)": "Consensus_Score"
        }
        k = key_map[name]
        for tf, items in tf_groups.items():
            sorted_items = sorted(items, key=lambda x: x[k], reverse=True)
            if sorted_items[0]['is_tp'] == 1:
                top1_count += 1
                
        top1_acc = (top1_count / len(tf_groups)) * 100.0
        print(f"{name:<30} | {roc_auc:<10.4f} | {pr_auc:<10.4f} | {top1_count}/{len(tf_groups)} ({top1_acc:.0f}%)")
        
    print("="*85 + "\n")

if __name__ == '__main__':
    main()
