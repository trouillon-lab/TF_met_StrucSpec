#!/usr/bin/env python3
"""
Shifted Inverse Consensus Scoring & Refinement Evaluator
Evaluates AF3 ipTM, AF3 Alone, GNINA VS, Current Consensus, and Shifted Inverse Consensus
formulations with dampening constant epsilon in [0.1, 0.2, 0.3, 0.5] to eliminate PAE_min asymptotes.
"""

import os
import sys
import warnings
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix, f1_score

def load_and_preprocess_report(report_csv='results/ranked_pairings_report.csv'):
    """Loads report CSV and standardizes column names."""
    if not os.path.exists(report_csv):
        raise FileNotFoundError(f"Report file '{report_csv}' not found. Run scripts/rank_candidates.py first.")
        
    raw_df = pd.read_csv(report_csv)
    
    df = pd.DataFrame({
        'tf_name': raw_df['TF_Name'].astype(str).str.strip(),
        'compound_id': raw_df['Ligand_Name'].astype(str).str.strip(),
        'label': raw_df['Is_True_Positive'].astype(str).str.lower().isin(['true', '1', 'yes', 't', 'positive']).astype(int),
        'ipTM': raw_df['AF3_ipTM'].astype(float),
        'PAE_min': raw_df['AF3_PAE_min'].astype(float),
        'CNNscore': raw_df['Gnina_CNNscore'].astype(float),
        'CNNaffinity': raw_df['Gnina_CNNaffinity'].astype(float),
        'AF3_Score': raw_df['AF3_Score'].astype(float),
        'Gnina_CNN_VS': raw_df['Gnina_CNN_VS'].astype(float),
        'Consensus_Score': raw_df['Consensus_Score'].astype(float)
    })
    
    return df

def engineer_shifted_features(df, eps_values=[0.1, 0.2, 0.3, 0.5]):
    """Engineers Shifted Inverse Consensus scores across a range of dampening constants (epsilon)."""
    df = df.copy()
    
    # GNINA VS product
    gnina_vs = df['CNNscore'] * df['CNNaffinity']
    
    for eps in eps_values:
        col_name = f'S_Shifted_eps_{eps}'
        shifted_af3 = df['ipTM'] / (df['PAE_min'] + eps)
        df[col_name] = shifted_af3 * gnina_vs
        df[col_name] = np.nan_to_num(df[col_name].values, nan=0.0, posinf=0.0, neginf=0.0)
        
    return df

def compute_metrics_table(y_true, scores_dict):
    """Calculates summary metrics table across all evaluated scoring methods."""
    metrics_list = []
    
    for name, y_scores in scores_dict.items():
        fpr, tpr, roc_thresh = roc_curve(y_true, y_scores)
        roc_auc_val = auc(fpr, tpr)
        
        precision, recall, pr_thresh = precision_recall_curve(y_true, y_scores)
        pr_auc_val = average_precision_score(y_true, y_scores)
        
        # Max F1 Score across threshold sweep
        f1_array = []
        for t in np.linspace(np.min(y_scores), np.max(y_scores), 200):
            y_pred = (y_scores >= t).astype(int)
            f1_array.append(f1_score(y_true, y_pred, zero_division=0))
        max_f1 = np.max(f1_array)
        
        # Youden's J optimal threshold for Balanced Accuracy
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_thresh = roc_thresh[best_idx]
        
        y_pred_opt = (y_scores >= best_thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_opt, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        bal_acc = (sens + spec) / 2.0
        
        metrics_list.append({
            'Scoring Method': name,
            'ROC AUC': float(roc_auc_val),
            'PR AUC': float(pr_auc_val),
            'Max F1': float(max_f1),
            'Balanced Accuracy': float(bal_acc),
            'Optimal Threshold': float(best_thresh)
        })
        
    return pd.DataFrame(metrics_list)

def select_best_epsilon(y_true, df, eps_values=[0.1, 0.2, 0.3, 0.5]):
    """Selects the best performing epsilon based on combined ROC AUC and PR AUC performance."""
    best_eps = eps_values[0]
    best_score = -1.0
    eps_metrics = {}
    
    for eps in eps_values:
        col_name = f'S_Shifted_eps_{eps}'
        y_scores = df[col_name].values
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc_val = auc(fpr, tpr)
        pr_auc_val = average_precision_score(y_true, y_scores)
        
        score = roc_auc_val + pr_auc_val
        eps_metrics[eps] = (roc_auc_val, pr_auc_val)
        
        if score > best_score:
            best_score = score
            best_eps = eps
            
    return best_eps, eps_metrics

def plot_roc_pr_grid(y_true, scores_dict, out_svg='results/advanced_scoring_eval.svg', out_png='results/advanced_scoring_eval.png'):
    """Generates 1x2 publication-grade ROC & PR grid plot comparing top scoring methods."""
    sns.set_theme(style="whitegrid")
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    
    colors = {
        "AF3 ipTM": "#7F7F7F",                               # Gray
        "AF3 Alone (ipTM / PAE_min)": "#FF7F0E",             # Orange
        "GNINA VS (CNNscore * CNNaffinity)": "#9467BD",       # Purple
        "Current Consensus": "#17BECF",                     # Cyan
    }
    
    linestyles = {
        "AF3 ipTM": ":",
        "AF3 Alone (ipTM / PAE_min)": "--",
        "GNINA VS (CNNscore * CNNaffinity)": "-.",
        "Current Consensus": "--",
    }
    
    for name, y_scores in scores_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc_val = auc(fpr, tpr)
        
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc_val = average_precision_score(y_true, y_scores)
        
        if "Shifted Inverse Consensus" in name:
            c = "#2CA02C"  # Emerald Green (Highlight)
            ls = "-"
            lw = 3.0
        else:
            c = colors.get(name, "#333333")
            ls = linestyles.get(name, "-")
            lw = 2.0
            
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_val:.4f})", color=c, linestyle=ls, linewidth=lw)
        ax_pr.plot(recall, precision, label=f"{name} (AUC = {pr_auc_val:.4f})", color=c, linestyle=ls, linewidth=lw)
        
    # ROC Panel Details
    ax_roc.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5, label="Random Chance (AUC = 0.5000)")
    ax_roc.set_xlim([-0.02, 1.02])
    ax_roc.set_ylim([-0.02, 1.02])
    ax_roc.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12, fontweight='bold')
    ax_roc.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12, fontweight='bold')
    ax_roc.set_title("Receiver Operating Characteristic (ROC)", fontsize=14, fontweight='bold', pad=12)
    ax_roc.legend(loc="lower right", frameon=True, facecolor='white', framealpha=0.95, fontsize=8.8)
    ax_roc.grid(True, linestyle=':', alpha=0.6)
    
    # PR Panel Details
    baseline_pr = np.sum(y_true) / len(y_true)
    ax_pr.plot([0, 1], [baseline_pr, baseline_pr], 'k--', lw=1.5, alpha=0.7, label=f"Random Classifier (Baseline = {baseline_pr:.3f})")
    ax_pr.set_xlim([-0.02, 1.02])
    ax_pr.set_ylim([-0.02, 1.02])
    ax_pr.set_xlabel("Recall (Sensitivity)", fontsize=12, fontweight='bold')
    ax_pr.set_ylabel("Precision (Positive Predictive Value)", fontsize=12, fontweight='bold')
    ax_pr.set_title("Precision-Recall (PR) Curve", fontsize=14, fontweight='bold', pad=12)
    ax_pr.legend(loc="lower right", frameon=True, facecolor='white', framealpha=0.95, fontsize=8.8)
    ax_pr.grid(True, linestyle=':', alpha=0.6)
    
    plt.suptitle("Virtual Screening Evaluation: Shifted Inverse Consensus Formulation", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    os.makedirs(os.path.dirname(os.path.abspath(out_svg)), exist_ok=True)
    fig.savefig(out_svg, format='svg', bbox_inches='tight')
    fig.savefig(out_png, format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Exported advanced evaluation vector plot to '{out_svg}'")
    print(f"Exported advanced evaluation PNG plot to '{out_png}'")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Shifted Inverse Consensus scoring methods.")
    parser.add_argument('--report-csv', default='results/ranked_pairings_report.csv', help="Report CSV")
    parser.add_argument('--out-svg', default='results/advanced_scoring_eval.svg', help="Output SVG path")
    parser.add_argument('--out-png', default='results/advanced_scoring_eval.png', help="Output PNG path")
    
    args = parser.parse_args()
    
    df = load_and_preprocess_report(args.report_csv)
    eps_values = [0.1, 0.2, 0.3, 0.5]
    df = engineer_shifted_features(df, eps_values=eps_values)
    
    y_true = df['label'].values
    
    best_eps, eps_metrics = select_best_epsilon(y_true, df, eps_values=eps_values)
    print(f"Selected Optimal Epsilon: eps = {best_eps}")
    
    # Build complete dictionary including all epsilon variants + baseline methods
    all_scores_dict = {
        "AF3 ipTM": df['ipTM'].values,
        "AF3 Alone (ipTM / PAE_min)": df['AF3_Score'].values,
        "GNINA VS (CNNscore * CNNaffinity)": df['Gnina_CNN_VS'].values,
        "Current Consensus": df['Consensus_Score'].values
    }
    
    for eps in eps_values:
        all_scores_dict[f"Shifted Inverse Consensus (eps = {eps})"] = df[f'S_Shifted_eps_{eps}'].values
        
    metrics_table = compute_metrics_table(y_true, all_scores_dict)
    
    print("\n" + "="*105)
    print("VIRTUAL SCREENING SCORING METHOD COMPARISON SUMMARY TABLE")
    print("="*105)
    print(f"{'Scoring Method':<45} | {'ROC AUC':<9} | {'PR AUC':<9} | {'Max F1':<9} | {'Bal Acc':<9}")
    print("-"*105)
    for _, row in metrics_table.iterrows():
        print(f"{row['Scoring Method']:<45} | {row['ROC AUC']:<9.4f} | {row['PR AUC']:<9.4f} | {row['Max F1']:<9.4f} | {row['Balanced Accuracy']:<9.4f}")
    print("="*105 + "\n")
    
    # Dict for plotting overlay (Baseline methods + Best Shifted Inverse Consensus variant)
    plot_scores_dict = {
        "AF3 ipTM": df['ipTM'].values,
        "AF3 Alone (ipTM / PAE_min)": df['AF3_Score'].values,
        "GNINA VS (CNNscore * CNNaffinity)": df['Gnina_CNN_VS'].values,
        "Current Consensus": df['Consensus_Score'].values,
        f"Shifted Inverse Consensus (eps = {best_eps})": df[f'S_Shifted_eps_{best_eps}'].values
    }
    
    plot_roc_pr_grid(y_true, plot_scores_dict, out_svg=args.out_svg, out_png=args.out_png)

if __name__ == '__main__':
    main()
