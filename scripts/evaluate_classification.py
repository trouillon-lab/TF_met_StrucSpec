#!/usr/bin/env python3
"""
Comprehensive Virtual Screening Classification & Multi-Binder Ranking Evaluator
Analyzes score distributions, optimal decision thresholds (Youden's J-index), Sensitivity/Specificity,
F1-score, and per-TF multi-binder ranking performance (Recall@K, mAP) across AF3, GNINA, and Consensus scores.
Strictly defines True Positives as pairings in raw curated CSV (small molecules) and True Negatives as all other pairings.
"""

import os
import sys
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix, f1_score

def load_ground_truth_positives(raw_csv='data/raw/tf_effectors_curated_2607.csv'):
    """Loads set of known small-molecule true positive (TF, Ligand) pairs from raw curated CSV."""
    if not os.path.exists(raw_csv):
        raise FileNotFoundError(f"Raw dataset file '{raw_csv}' not found.")
        
    positives = set()
    with open(raw_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('Effector type', '').strip() == 'small molecule':
                tf = row['Transcription factor'].strip()
                lig = row['effector_name'].strip()
                if tf and lig:
                    positives.add((tf, lig))
    return positives

def compute_optimal_threshold(y_true, y_scores):
    """Computes optimal decision threshold using Youden's J-index = Sensitivity + Specificity - 1."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_thresh = float(thresholds[best_idx])
    
    # Compute metrics at best threshold
    y_pred = (y_scores >= best_thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    bal_acc = (sens + spec) / 2.0
    acc = (tp + tn) / len(y_true)
    
    return best_thresh, {
        'threshold': best_thresh,
        'sensitivity': float(sens),
        'specificity': float(spec),
        'precision': float(prec),
        'f1': float(f1),
        'balanced_accuracy': float(bal_acc),
        'accuracy': float(acc),
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn)
    }

def evaluate_classification_metrics(y_true, metrics_data):
    """Evaluates ROC AUC, PR AUC, and optimal threshold metrics for all scoring methods."""
    results = {}
    for name, y_scores in metrics_data.items():
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = float(auc(fpr, tpr))
        pr_auc = float(average_precision_score(y_true, y_scores))
        
        opt_thresh, opt_metrics = compute_optimal_threshold(y_true, y_scores)
        
        results[name] = {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'optimal_metrics': opt_metrics,
            'y_scores': y_scores
        }
    return results

def evaluate_multibinder_ranking(rows, score_col='Consensus_Score'):
    """Evaluates per-TF multi-binder ranking metrics (Recall@1, Recall@2, Recall@3, AP)."""
    tf_groups = {}
    for r in rows:
        tf = r['TF_Name']
        if tf not in tf_groups:
            tf_groups[tf] = []
        tf_groups[tf].append(r)
        
    per_tf_results = {}
    for tf, items in tf_groups.items():
        # Sort items descending by score
        sorted_items = sorted(items, key=lambda x: float(x[score_col]), reverse=True)
        n_total_pos = sum(x['is_tp'] for x in items)
        if n_total_pos == 0:
            continue
            
        rec_at_1 = sum(x['is_tp'] for x in sorted_items[:1]) / n_total_pos
        rec_at_2 = sum(x['is_tp'] for x in sorted_items[:2]) / n_total_pos
        rec_at_3 = sum(x['is_tp'] for x in sorted_items[:3]) / n_total_pos
        
        # Calculate Average Precision for this TF
        hits = 0
        sum_precisions = 0.0
        for i, item in enumerate(sorted_items):
            if item['is_tp'] == 1:
                hits += 1
                sum_precisions += hits / (i + 1)
        ap = sum_precisions / n_total_pos if n_total_pos > 0 else 0.0
        
        per_tf_results[tf] = {
            'total_binders': n_total_pos,
            'total_screened': len(items),
            'recall_at_1': rec_at_1,
            'recall_at_2': rec_at_2,
            'recall_at_3': rec_at_3,
            'ap': ap
        }
    return per_tf_results

def plot_score_distributions(rows, out_svg='results/classification_score_distributions.svg', out_png='results/classification_score_distributions.png'):
    """Generates dual violin/KDE score distribution plot comparing Positives vs Negatives."""
    sns.set_theme(style="whitegrid")
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)
    
    metrics_to_plot = [
        ("AF3 Alone", "AF3_Score", "#1F77B4"),
        ("GNINA CNNscore", "Gnina_CNNscore", "#FF7F0E"),
        ("Consensus (AF3 + GNINA)", "Consensus_Score", "#2CA02C")
    ]
    
    for idx, (title, col, color) in enumerate(metrics_to_plot):
        ax = axes[idx]
        pos_scores = [float(r[col]) for r in rows if r['is_tp'] == 1]
        neg_scores = [float(r[col]) for r in rows if r['is_tp'] == 0]
        
        data = [
            {'Score': s, 'Group': 'True Positive'} for s in pos_scores
        ] + [
            {'Score': s, 'Group': 'True Negative'} for s in neg_scores
        ]
        
        import pandas as pd
        df_data = pd.DataFrame(data)
        
        sns.violinplot(
            data=df_data, x='Group', y='Score', hue='Group',
            palette={'True Positive': color, 'True Negative': '#7F7F7F'},
            inner='quartile', ax=ax, legend=False
        )
        sns.stripplot(
            data=df_data, x='Group', y='Score', color='black',
            alpha=0.6, jitter=0.15, size=6, ax=ax
        )
        
        # Calculate optimal threshold
        y_true = np.array([1]*len(pos_scores) + [0]*len(neg_scores))
        y_scores = np.array(pos_scores + neg_scores)
        opt_thresh, opt_m = compute_optimal_threshold(y_true, y_scores)
        
        ax.axhline(opt_thresh, color='red', linestyle='--', linewidth=1.8, label=f"Opt Thresh = {opt_thresh:.3f}")
        ax.set_title(f"{title}\n(Opt Thresh: {opt_thresh:.3f} | F1: {opt_m['f1']:.2f})", fontsize=12, fontweight='bold')
        ax.set_xlabel("", fontsize=11)
        ax.set_ylabel(title if idx == 0 else "", fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', frameon=True, fontsize=9.5)
        
    plt.suptitle("Score Distributions & Decision Thresholds: True Positives vs True Negatives", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(os.path.abspath(out_svg)), exist_ok=True)
    fig.savefig(out_svg, format='svg', bbox_inches='tight')
    fig.savefig(out_png, format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Exported score distribution vector plot to '{out_svg}'")

def run_classification_analysis(
    report_csv='results/ranked_pairings_report.csv',
    raw_csv='data/raw/tf_effectors_curated_2607.csv',
    out_dist_svg='results/classification_score_distributions.svg',
    out_dist_png='results/classification_score_distributions.png'
):
    """Executes full classification analysis workflow."""
    if not os.path.exists(report_csv):
        print(f"Error: '{report_csv}' not found. Run scripts/rank_candidates.py first.")
        return None
        
    pos_pairs = load_ground_truth_positives(raw_csv)
    print(f"Loaded {len(pos_pairs)} ground-truth small-molecule positive pairs from raw dataset.")
    
    rows = []
    with open(report_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            tf = r['TF_Name'].strip()
            lig = r['Ligand_Name'].strip()
            r['is_tp'] = 1 if (tf, lig) in pos_pairs else 0
            rows.append(r)
            
    y_true = np.array([r['is_tp'] for r in rows])
    n_pos = int(sum(y_true))
    n_neg = int(len(y_true) - n_pos)
    print(f"Dataset Evaluation: {n_pos} True Positives, {n_neg} True Negatives (Total {len(y_true)} pairs)")
    
    metrics_data = {
        "AF3 ipTM": np.array([float(r['AF3_ipTM']) for r in rows]),
        "AF3 Alone (ipTM / PAE_min)": np.array([float(r['AF3_Score']) for r in rows]),
        "GNINA CNNscore": np.array([float(r['Gnina_CNNscore']) for r in rows]),
        "GNINA VS Score": np.array([float(r['Gnina_CNN_VS']) for r in rows]),
        "Consensus (AF3 + GNINA)": np.array([float(r['Consensus_Score']) for r in rows])
    }
    
    eval_results = evaluate_classification_metrics(y_true, metrics_data)
    
    print("\n" + "="*110)
    print(f"{'Method / Scoring Metric':<30} | {'ROC AUC':<8} | {'PR AUC':<8} | {'Opt Thresh':<10} | {'Sens':<7} | {'Spec':<7} | {'F1':<7} | {'Bal Acc':<7}")
    print("-"*110)
    for name, res in eval_results.items():
        m = res['optimal_metrics']
        print(f"{name:<30} | {res['roc_auc']:<8.4f} | {res['pr_auc']:<8.4f} | {m['threshold']:<10.4f} | {m['sensitivity']:<7.2f} | {m['specificity']:<7.2f} | {m['f1']:<7.2f} | {m['balanced_accuracy']:<7.2f}")
    print("="*110 + "\n")
    
    # Evaluate Multi-Binder Ranking
    print("="*80)
    print("Multi-Binder Per-TF Specificity Ranking Performance (Consensus Score)")
    print("-"*80)
    print(f"{'TF Name':<12} | {'True Binders':<14} | {'Recall@1':<10} | {'Recall@2':<10} | {'Average Precision':<18}")
    print("-"*80)
    mb_res = evaluate_multibinder_ranking(rows, score_col='Consensus_Score')
    for tf, m in mb_res.items():
        print(f"{tf:<12} | {m['total_binders']:<14} | {m['recall_at_1']:<10.2f} | {m['recall_at_2']:<10.2f} | {m['ap']:<18.4f}")
    print("="*80 + "\n")
    
    # Generate distribution plots
    plot_score_distributions(rows, out_dist_svg, out_dist_png)
    
    return {
        'eval_results': eval_results,
        'multibinder_results': mb_res
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate classification thresholds and multi-binder ranking.")
    parser.add_argument('--report-csv', default='results/ranked_pairings_report.csv', help="Report CSV")
    parser.add_argument('--raw-csv', default='data/raw/tf_effectors_curated_2607.csv', help="Raw curated dataset CSV")
    parser.add_argument('--out-dist-svg', default='results/classification_score_distributions.svg', help="Output distribution SVG path")
    parser.add_argument('--out-dist-png', default='results/classification_score_distributions.png', help="Output distribution PNG path")
    
    args = parser.parse_args()
    
    run_classification_analysis(
        report_csv=args.report_csv,
        raw_csv=args.raw_csv,
        out_dist_svg=args.out_dist_svg,
        out_dist_png=args.out_dist_png
    )

if __name__ == '__main__':
    main()
