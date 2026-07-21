#!/usr/bin/env python3
"""
Publication-Grade ROC and Precision-Recall (PR) Curve Plotter
Evaluates AF3 ipTM, AF3 Alone, GNINA CNNscore, GNINA VS Score, and Consensus Score
across the 20-pair TF-small molecule diagnostic dataset.
"""

import os
import sys
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# Ground Truth 10 Positive Pairs
TRUE_POSITIVES = {
    "AraC_arabinose",
    "AraC_D-fucose",
    "AcrR_ethidium",
    "AcrR_proflavin",
    "AcrR_R6G",
    "TyrR_L-tryptophan",
    "TyrR_L-phenylalanine",
    "TyrR_L-tyrosine",
    "CysB_O-acetyl-L-serine",
    "CysB_Thiosulphate"
}

def load_data(report_csv='results/ranked_pairings_report.csv'):
    """Loads scores and ground-truth labels from report CSV."""
    if not os.path.exists(report_csv):
        raise FileNotFoundError(f"Report file '{report_csv}' not found.")
        
    y_true = []
    metrics_data = {
        "AF3 ipTM": [],
        "AF3 Alone (ipTM / PAE_min)": [],
        "GNINA CNNscore": [],
        "GNINA VS Score": [],
        "Consensus (AF3 + GNINA)": []
    }
    
    with open(report_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pair = row['TF_Ligand']
            label = 1 if (pair in TRUE_POSITIVES or row.get('Is_True_Positive', '').lower() in ['true', '1', 'yes']) else 0
            y_true.append(label)
            
            metrics_data["AF3 ipTM"].append(float(row['AF3_ipTM']))
            metrics_data["AF3 Alone (ipTM / PAE_min)"].append(float(row['AF3_Score']))
            metrics_data["GNINA CNNscore"].append(float(row['Gnina_CNNscore']))
            metrics_data["GNINA VS Score"].append(float(row['Gnina_CNN_VS']))
            metrics_data["Consensus (AF3 + GNINA)"].append(float(row['Consensus_Score']))
            
    return np.array(y_true), {k: np.array(v) for k, v in metrics_data.items()}

def compute_roc_pr(y_true, y_scores):
    """Computes ROC and PR points and AUC metrics using scikit-learn."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)
    
    return fpr, tpr, float(roc_auc), recall, precision, float(pr_auc)

def generate_plots(y_true, metrics_data, svg_path='results/roc_pr_curves.svg', png_path='results/roc_pr_curves.png'):
    """Generates dual-panel ROC and PR publication-quality curves."""
    sns.set_theme(style="whitegrid")
    
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    
    # Custom harmonious color palette
    colors = {
        "AF3 ipTM": "#7F7F7F",                  # Neutral Gray
        "AF3 Alone (ipTM / PAE_min)": "#1F77B4",# Steel Blue
        "GNINA CNNscore": "#FF7F0E",            # Vivid Orange
        "GNINA VS Score": "#9467BD",            # Muted Purple
        "Consensus (AF3 + GNINA)": "#2CA02C"    # Forest Green (Highlight)
    }
    
    styles = {
        "AF3 ipTM": "--",
        "AF3 Alone (ipTM / PAE_min)": "-.",
        "GNINA CNNscore": ":",
        "GNINA VS Score": "--",
        "Consensus (AF3 + GNINA)": "-"
    }
    
    linewidths = {
        "AF3 ipTM": 1.8,
        "AF3 Alone (ipTM / PAE_min)": 2.2,
        "GNINA CNNscore": 2.2,
        "GNINA VS Score": 2.2,
        "Consensus (AF3 + GNINA)": 3.0
    }
    
    print("\n" + "="*80)
    print(f"{'Method / Scoring Metric':<32} | {'ROC AUC':<10} | {'PR AUC':<10}")
    print("-"*80)
    
    for name, y_scores in metrics_data.items():
        fpr, tpr, roc_auc, recall, precision, pr_auc = compute_roc_pr(y_true, y_scores)
        print(f"{name:<32} | {roc_auc:<10.4f} | {pr_auc:<10.4f}")
        
        color = colors.get(name, "#333333")
        linestyle = styles.get(name, "-")
        lw = linewidths.get(name, 2.0)
        
        # Plot ROC curve
        ax_roc.plot(
            fpr, tpr,
            label=f"{name} (AUC = {roc_auc:.4f})",
            color=color,
            linestyle=linestyle,
            linewidth=lw
        )
        
        # Plot PR curve
        ax_pr.plot(
            recall, precision,
            label=f"{name} (AUC = {pr_auc:.4f})",
            color=color,
            linestyle=linestyle,
            linewidth=lw
        )
        
    print("="*80 + "\n")

    # ROC Panel Details
    ax_roc.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5, label="Random Chance (AUC = 0.5000)")
    ax_roc.set_xlim([-0.02, 1.02])
    ax_roc.set_ylim([-0.02, 1.02])
    ax_roc.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12, fontweight='bold')
    ax_roc.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12, fontweight='bold')
    ax_roc.set_title("Receiver Operating Characteristic (ROC)", fontsize=14, fontweight='bold', pad=12)
    ax_roc.legend(loc="lower right", frameon=True, facecolor='white', framealpha=0.9, fontsize=9.5)
    ax_roc.grid(True, linestyle=':', alpha=0.6)

    # PR Panel Details
    baseline_pr = np.sum(y_true) / len(y_true)
    ax_pr.plot([0, 1], [baseline_pr, baseline_pr], 'k--', lw=1.5, alpha=0.5, label=f"Random Chance (AUC = {baseline_pr:.2f})")
    ax_pr.set_xlim([-0.02, 1.02])
    ax_pr.set_ylim([-0.02, 1.02])
    ax_pr.set_xlabel("Recall (Sensitivity)", fontsize=12, fontweight='bold')
    ax_pr.set_ylabel("Precision (Positive Predictive Value)", fontsize=12, fontweight='bold')
    ax_pr.set_title("Precision-Recall (PR) Curve", fontsize=14, fontweight='bold', pad=12)
    ax_pr.legend(loc="lower left", frameon=True, facecolor='white', framealpha=0.9, fontsize=9.5)
    ax_pr.grid(True, linestyle=':', alpha=0.6)

    plt.suptitle("Virtual Screening Evaluation: TF-Small Molecule Specificity (20 Diagnostic Pairs)", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save vector SVG and raster PNG
    os.makedirs(os.path.dirname(os.path.abspath(svg_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(png_path)), exist_ok=True)
    
    fig.savefig(svg_path, format='svg', bbox_inches='tight')
    fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Exported vector plot to '{svg_path}'")
    print(f"Exported PNG figure to '{png_path}'")

def main():
    parser = argparse.ArgumentParser(description="Generate publication-grade ROC & PR curves.")
    parser.add_argument('--report', default='results/ranked_pairings_report.csv', help="Path to ranked pairings report CSV")
    parser.add_argument('--svg-out', default='results/roc_pr_curves.svg', help="Output SVG path")
    parser.add_argument('--png-out', default='results/roc_pr_curves.png', help="Output PNG path")
    
    args = parser.parse_args()
    
    y_true, metrics_data = load_data(args.report)
    generate_plots(y_true, metrics_data, svg_path=args.svg_out, png_path=args.png_out)

if __name__ == '__main__':
    main()
