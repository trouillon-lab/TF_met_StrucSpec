#!/usr/bin/env python3
"""
Comprehensive Score Distribution & Correlation Diagnostic Plotter
Plots KDE density histograms, boxplots, and Spearman correlation heatmaps across all 7 score metrics
(ipTM, PAE_min, AF3 Score, CNNscore, CNNaffinity, GNINA VS, Consensus) to analyze mathematical scaling,
skewness, and combination compatibility for virtual screening.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_report_data(report_csv='results/ranked_pairings_report.csv'):
    """Loads ranked pairings report CSV and extracts all 7 score metrics."""
    if not os.path.exists(report_csv):
        raise FileNotFoundError(f"Report file '{report_csv}' not found. Run scripts/rank_candidates.py first.")
        
    raw_df = pd.read_csv(report_csv)
    
    df = pd.DataFrame({
        'TF_Name': raw_df['TF_Name'].astype(str).str.strip(),
        'Ligand_Name': raw_df['Ligand_Name'].astype(str).str.strip(),
        'Is_Positive': raw_df['Is_True_Positive'].astype(str).str.lower().isin(['true', '1', 'yes', 't', 'positive']).astype(int),
        'AF3_ipTM': raw_df['AF3_ipTM'].astype(float),
        'AF3_PAE_min': raw_df['AF3_PAE_min'].astype(float),
        'AF3_Score': raw_df['AF3_Score'].astype(float),
        'Gnina_CNNscore': raw_df['Gnina_CNNscore'].astype(float),
        'Gnina_CNNaffinity': raw_df['Gnina_CNNaffinity'].astype(float),
        'Gnina_CNN_VS': raw_df['Gnina_CNN_VS'].astype(float),
        'Consensus_Score': raw_df['Consensus_Score'].astype(float)
    })
    
    return df

def generate_distribution_diagnostics(
    report_csv='results/ranked_pairings_report.csv',
    out_hist_svg='results/score_distribution_histograms.svg',
    out_hist_png='results/score_distribution_histograms.png',
    out_corr_svg='results/score_correlation_matrix.svg',
    out_corr_png='results/score_correlation_matrix.png'
):
    """Generates distribution KDE histograms and Spearman correlation heatmaps for all scores."""
    df = load_report_data(report_csv)
    
    sns.set_theme(style="whitegrid")
    
    metrics_config = [
        ("AF3 ipTM", "AF3_ipTM", "[0, 1] (Bounded)", "#7F7F7F"),
        ("AF3 PAE_min", "AF3_PAE_min", "Å (Lower is better)", "#17BECF"),
        ("AF3 Score (ipTM / PAE)", "AF3_Score", "Ratio (Unbounded)", "#1F77B4"),
        ("GNINA CNNscore", "Gnina_CNNscore", "[0, 1] (Bounded)", "#FF7F0E"),
        ("GNINA CNNaffinity", "Gnina_CNNaffinity", "pK_d (Unbounded)", "#E377C2"),
        ("GNINA VS (CNNscore * pK_d)", "Gnina_CNN_VS", "Product", "#9467BD"),
        ("Consensus (AF3 * GNINA_VS)", "Consensus_Score", "Composite Product", "#2CA02C")
    ]
    
    # 1. Plot 7-Panel KDE & Histogram Grid
    fig, axes = plt.subplots(2, 4, figsize=(20, 9.5), dpi=300)
    axes_flat = axes.flatten()
    
    for idx, (title, col, scale_desc, color) in enumerate(metrics_config):
        ax = axes_flat[idx]
        
        pos_data = df[df['Is_Positive'] == 1][col]
        neg_data = df[df['Is_Positive'] == 0][col]
        
        sns.histplot(pos_data, kde=True, ax=ax, color=color, label="True Positive", stat="density", common_norm=False, alpha=0.45, bins=25)
        sns.histplot(neg_data, kde=True, ax=ax, color="#444444", label="Decoy Negative", stat="density", common_norm=False, alpha=0.35, bins=25)
        
        pos_mean, pos_med = pos_data.mean(), pos_data.median()
        neg_mean, neg_med = neg_data.mean(), neg_data.median()
        skew_val = df[col].skew()
        
        ax.set_title(f"{title}\n({scale_desc} | Skew: {skew_val:+.2f})", fontsize=11, fontweight='bold', pad=8)
        ax.set_xlabel(f"{title} Value", fontsize=10, fontweight='bold')
        ax.set_ylabel("Density", fontsize=10, fontweight='bold')
        ax.legend(loc="upper right", frameon=True, fontsize=8.5)
        
        # Annotate median values
        ax.annotate(f"Pos Med: {pos_med:.2f}\nNeg Med: {neg_med:.2f}", xy=(0.05, 0.75), xycoords='axes fraction',
                    fontsize=8.5, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85))
                    
    # Hide the 8th empty subplot
    fig.delaxes(axes_flat[7])
    
    plt.suptitle("Distribution Behavior & Scale Characteristics across All 7 Virtual Screening Scores (268 Pairs)", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    os.makedirs(os.path.dirname(os.path.abspath(out_hist_svg)), exist_ok=True)
    fig.savefig(out_hist_svg, format='svg', bbox_inches='tight')
    fig.savefig(out_hist_png, format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Exported score distribution histogram vector plot to '{out_hist_svg}'")
    
    # 2. Plot Pairwise Spearman Correlation Heatmap
    fig_corr, ax_corr = plt.subplots(figsize=(9, 7.5), dpi=300)
    
    corr_cols = ['Is_Positive', 'AF3_ipTM', 'AF3_PAE_min', 'AF3_Score', 'Gnina_CNNscore', 'Gnina_CNNaffinity', 'Gnina_CNN_VS', 'Consensus_Score']
    labels_display = ['Ground Truth Label', 'AF3 ipTM', 'AF3 PAE_min', 'AF3 Score (ipTM/PAE)', 'GNINA CNNscore', 'GNINA CNNaffinity', 'GNINA VS Score', 'Consensus Score']
    
    corr_matrix = df[corr_cols].corr(method='spearman')
    corr_matrix.columns = labels_display
    corr_matrix.index = labels_display
    
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="vlag", vmin=-1.0, vmax=1.0, ax=ax_corr, cbar_kws={'label': 'Spearman Rank Correlation (ρ)'}, linewidths=0.5)
    ax_corr.set_title("Pairwise Spearman Rank Correlation Matrix (All Scores & Label)", fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(os.path.abspath(out_corr_svg)), exist_ok=True)
    fig_corr.savefig(out_corr_svg, format='svg', bbox_inches='tight')
    fig_corr.savefig(out_corr_png, format='png', dpi=300, bbox_inches='tight')
    plt.close(fig_corr)
    print(f"Exported score correlation heatmap vector plot to '{out_corr_svg}'")
    
    # 3. Print Statistical Summary Table
    print("\n" + "="*115)
    print("ALL SCORES STATISTICAL DISTRIBUTION SUMMARY TABLE (268 PAIRS)")
    print("="*115)
    print(f"{'Score Metric':<32} | {'Scale Bounds':<20} | {'Mean ± SD':<18} | {'Median [IQR]':<18} | {'Skewness':<8}")
    print("-"*115)
    
    for title, col, scale_desc, _ in metrics_config:
        vals = df[col].values
        q25, q50, q75 = np.percentile(vals, [25, 50, 75])
        mean_val, sd_val = np.mean(vals), np.std(vals)
        skew_val = df[col].skew()
        
        sd_str = f"{sd_val:.2f}"
        iqr_str = f"{q50:.2f} [{q25:.2f}-{q75:.2f}]"
        print(f"{title:<32} | {scale_desc:<20} | {mean_val:.2f} ± {sd_str:<9} | {iqr_str:<18} | {skew_val:+.2f}")
    print("="*115 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Plot distribution histograms and correlation matrices for all scores.")
    parser.add_argument('--report-csv', default='results/ranked_pairings_report.csv', help="Report CSV")
    parser.add_argument('--out-hist-svg', default='results/score_distribution_histograms.svg', help="Output Hist SVG path")
    parser.add_argument('--out-hist-png', default='results/score_distribution_histograms.png', help="Output Hist PNG path")
    parser.add_argument('--out-corr-svg', default='results/score_correlation_matrix.svg', help="Output Corr SVG path")
    parser.add_argument('--out-corr-png', default='results/score_correlation_matrix.png', help="Output Corr PNG path")
    
    args = parser.parse_args()
    
    generate_distribution_diagnostics(
        report_csv=args.report_csv,
        out_hist_svg=args.out_hist_svg,
        out_hist_png=args.out_hist_png,
        out_corr_svg=args.out_corr_svg,
        out_corr_png=args.out_corr_png
    )

if __name__ == '__main__':
    main()
