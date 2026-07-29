#!/usr/bin/env python3
"""
Master Virtual Screening Evaluation & Diagnostics Suite
Runs comprehensive benchmarking (AUROC, PR curves, Score Distributions,
Spearman Correlation, Decision Cutoffs, and True Positive Rank Slopegraphs)
for:
  1. Dataset 1 (First Set - 268 pairs)
  2. Dataset 2 (Score2 Set - 204 pairs)
  3. Combined Dataset (Dataset 1 + Dataset 2 = 472 pairs)

Run with no arguments to execute all three tracks with default paths.
Pass --pred-dirs / --gnina-csvs / --out-dir / --label to run a custom
single-track evaluation (e.g. Dataset 3).
"""

import os
import sys
import csv
import json
import zipfile
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, f1_score
)

from pipeline_utils import (
    GNINA_FALLBACK,
    sanitize_key,
    parse_af3_summary,
    load_gnina_scores,
    generate_svg_slopegraph,
)

sns.set_theme(style="whitegrid")


def load_ground_truth_map():
    """Loads combined ground truth map across all raw and processed dataset files."""
    gt_map = {}

    raw_csv = 'data/raw/tf_effectors_curated_2607.csv'
    if os.path.exists(raw_csv):
        with open(raw_csv, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if row.get('Effector type', '').strip() == 'small molecule':
                    tf = row['Transcription factor'].strip()
                    lig = row['effector_name'].strip()
                    if tf and lig:
                        s_key = sanitize_key(f"{tf}_{lig}")
                        gt_map[s_key] = {'tf_name': tf, 'ligand_name': lig, 'is_positive': True}

    pairings_files = [
        'data/processed/pairings_subset_20.csv',
        'data/processed/pairings_remaining_248.csv',
        'data/processed/pairings_score2_benchmark.csv'
    ]
    for pf in pairings_files:
        if os.path.exists(pf):
            with open(pf, 'r', encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    tf = row['TF_Name'].strip()
                    lig = row['Ligand_Name'].strip()
                    kegg = row.get('KEGG_ID', '').strip()
                    is_pos = row.get('Label', '').strip().lower() == 'positive'

                    item = {'tf_name': tf, 'ligand_name': lig, 'is_positive': is_pos}
                    gt_map[sanitize_key(f"{tf}_{lig}")] = item
                    if kegg:
                        gt_map[sanitize_key(f"{tf}_{kegg}")] = item

    # Dataset 3: keyed by TF_Name + KEGG_ID (BiGG ID), which matches the ZIP filenames
    d3_file = 'data/processed/pairings_dataset3_weekend.csv'
    if os.path.exists(d3_file):
        with open(d3_file, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                tf = row['TF_Name'].strip()
                bigg = row['KEGG_ID'].strip()
                is_pos = row.get('Label', '').strip().lower() == 'positive'
                if tf and bigg:
                    item = {'tf_name': tf, 'ligand_name': bigg, 'is_positive': is_pos}
                    gt_map[sanitize_key(f"{tf}_{bigg}")] = item

    return gt_map


def process_and_rank_dataset(pred_dirs, gnina_csvs, gt_map, out_report_csv):
    """Parses predictions, rescores with Solution 2 + GNINA VS, computes per-TF ranks, writes report CSV."""
    gnina_map = load_gnina_scores(gnina_csvs)

    zip_paths = []
    for pred_dir in pred_dirs:
        if os.path.exists(pred_dir):
            for f in os.listdir(pred_dir):
                if f.endswith('.zip'):
                    zip_paths.append(os.path.join(pred_dir, f))

    compiled_results = []
    seen_keys = set()

    for z_path in sorted(zip_paths):
        basename = os.path.splitext(os.path.basename(z_path))[0]
        pair_name = basename.replace("_predictions", "")
        s_key = sanitize_key(pair_name)

        if s_key in seen_keys:
            continue
        seen_keys.add(s_key)

        if s_key in gt_map:
            tf_name = gt_map[s_key]['tf_name']
            ligand_name = gt_map[s_key]['ligand_name']
            is_tp = gt_map[s_key]['is_positive']
        else:
            parts = pair_name.split('_', 1)
            tf_name = parts[0] if len(parts) > 0 else "Unknown_TF"
            ligand_name = parts[1] if len(parts) > 1 else "Unknown_Ligand"
            is_tp = False

        af3_metrics = parse_af3_summary(z_path)
        if not af3_metrics:
            continue

        iptm = af3_metrics['iptm']
        pae = af3_metrics['pae_min']
        clash = af3_metrics['has_clash']

        # Solution 2: iptm / (1.0 + pae)
        af3_score = iptm / (1.0 + pae)
        if clash:
            af3_score = 0.0

        gn = gnina_map.get(s_key, GNINA_FALLBACK)
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

    # Compute per-TF ranks
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

    os.makedirs(os.path.dirname(os.path.abspath(out_report_csv)), exist_ok=True)
    with open(out_report_csv, 'w', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "TF_Name", "Ligand_Name", "TF_Ligand", "AF3_ipTM", "AF3_PAE_min",
            "AF3_Has_Clash", "AF3_Score", "AF3_Rank", "Gnina_CNNscore",
            "Gnina_CNNaffinity", "Gnina_CNN_VS", "Consensus_Score", "Consensus_Rank", "Is_True_Positive"
        ])
        writer.writeheader()
        writer.writerows(ranked_results)

    return ranked_results, validation_data


def compute_optimal_threshold(y_true, y_scores):
    """Computes Youden's J optimal cutoff and classification metrics.

    Returns (best_thresh, metrics_dict). metrics_dict includes 'roc_auc' so
    callers can avoid recomputing the ROC curve separately.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc_val = auc(fpr, tpr)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_thresh = float(thresholds[best_idx])

    y_pred = (y_scores >= best_thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    bal_acc = (sens + spec) / 2.0

    return best_thresh, {
        'threshold': best_thresh,
        'roc_auc': float(roc_auc_val),
        'sensitivity': float(sens),
        'specificity': float(spec),
        'f1': float(f1),
        'balanced_accuracy': float(bal_acc),
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)
    }


def plot_dataset_diagnostics(df, out_dir, dataset_label):
    """Generates all 5 plot figures for a given dataset."""
    os.makedirs(out_dir, exist_ok=True)
    n_pairs = len(df)
    n_pos = int(df['Is_True_Positive'].sum())
    n_neg = n_pairs - n_pos

    y_true = df['Is_True_Positive'].values

    pae_val = df['AF3_PAE_min'].values
    inv_pae = 1.0 / np.maximum(pae_val, 0.01)

    metrics_map = {
        "AF3 ipTM": df['AF3_ipTM'].values,
        "AF3 PAE_min (Inv: 1/PAE)": inv_pae,
        "AF3 Alone (ipTM / (1+PAE))": df['AF3_Score'].values,
        "GNINA CNNscore": df['Gnina_CNNscore'].values,
        "GNINA CNNaffinity (pK_d)": df['Gnina_CNNaffinity'].values,
        "GNINA VS Score (CNNscore * CNNaffinity)": df['Gnina_CNN_VS'].values,
        "Consensus Score (AF3 * GNINA_VS)": df['Consensus_Score'].values
    }

    # ------------------------------------------------------------------
    # 1. Standard 5-Metric ROC and PR Curves
    # ------------------------------------------------------------------
    fig_5, (ax_roc_5, ax_pr_5) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    core_5_metrics = {
        "AF3 ipTM": df['AF3_ipTM'].values,
        "AF3 Alone (ipTM / (1+PAE))": df['AF3_Score'].values,
        "GNINA CNNscore": df['Gnina_CNNscore'].values,
        "GNINA VS Score": df['Gnina_CNN_VS'].values,
        "Consensus Score (AF3 * GNINA_VS)": df['Consensus_Score'].values
    }

    colors_5 = {
        "AF3 ipTM": "#7F7F7F",
        "AF3 Alone (ipTM / (1+PAE))": "#1F77B4",
        "GNINA CNNscore": "#FF7F0E",
        "GNINA VS Score": "#9467BD",
        "Consensus Score (AF3 * GNINA_VS)": "#2CA02C"
    }

    for name, y_scores in core_5_metrics.items():
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc_val = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc_val = average_precision_score(y_true, y_scores)

        c = colors_5.get(name, "#333333")
        lw = 2.8 if "Consensus" in name else 2.0
        ls = "-" if "Consensus" in name else ("--" if "Alone" in name else ":")

        ax_roc_5.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_val:.4f})", color=c, linestyle=ls, linewidth=lw)
        ax_pr_5.plot(recall, precision, label=f"{name} (AUC = {pr_auc_val:.4f})", color=c, linestyle=ls, linewidth=lw)

    ax_roc_5.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5, label="Random Chance (AUC = 0.5000)")
    ax_roc_5.set_xlim([-0.02, 1.02]); ax_roc_5.set_ylim([-0.02, 1.02])
    ax_roc_5.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight='bold')
    ax_roc_5.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11, fontweight='bold')
    ax_roc_5.set_title(f"Receiver Operating Characteristic (ROC) – {dataset_label}", fontsize=13, fontweight='bold')
    ax_roc_5.legend(loc="lower right", frameon=True, fontsize=8.5)

    base_pr = n_pos / n_pairs
    ax_pr_5.plot([0, 1], [base_pr, base_pr], 'k--', lw=1.5, alpha=0.5, label=f"Random Chance (Baseline = {base_pr:.3f})")
    ax_pr_5.set_xlim([-0.02, 1.02]); ax_pr_5.set_ylim([-0.02, 1.02])
    ax_pr_5.set_xlabel("Recall (Sensitivity)", fontsize=11, fontweight='bold')
    ax_pr_5.set_ylabel("Precision (PPV)", fontsize=11, fontweight='bold')
    ax_pr_5.set_title(f"Precision-Recall (PR) Curve – {dataset_label}", fontsize=13, fontweight='bold')
    ax_pr_5.legend(loc="lower right", frameon=True, fontsize=8.5)

    plt.suptitle(f"Virtual Screening ROC & PR Performance ({dataset_label}: {n_pairs} Pairs)", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig_5.savefig(os.path.join(out_dir, "roc_pr_curves.svg"), format='svg', bbox_inches='tight')
    fig_5.savefig(os.path.join(out_dir, "roc_pr_curves.png"), format='png', dpi=300, bbox_inches='tight')
    plt.close(fig_5)

    # ------------------------------------------------------------------
    # 1b. Advanced 7-Metric ROC and PR Grid
    # ------------------------------------------------------------------
    fig_7, (ax_roc_7, ax_pr_7) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    colors_7 = {
        "AF3 ipTM": "#4C72B0",
        "AF3 PAE_min (Inv: 1/PAE)": "#55A868",
        "AF3 Alone (ipTM / (1+PAE))": "#D55E00",
        "GNINA CNNscore": "#8172B0",
        "GNINA CNNaffinity (pK_d)": "#CCB974",
        "GNINA VS Score (CNNscore * CNNaffinity)": "#64B5CD",
        "Consensus Score (AF3 * GNINA_VS)": "#009E73"
    }

    for name, y_scores in metrics_map.items():
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc_val = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc_val = average_precision_score(y_true, y_scores)

        c = colors_7.get(name, "#333333")
        lw = 2.8 if "Consensus" in name else 2.0
        ls = "-" if "Consensus" in name else ("--" if "Alone" in name else ":")

        ax_roc_7.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_val:.4f})", color=c, linestyle=ls, linewidth=lw)
        ax_pr_7.plot(recall, precision, label=f"{name} (AUC = {pr_auc_val:.4f})", color=c, linestyle=ls, linewidth=lw)

    ax_roc_7.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5, label="Random Chance (AUC = 0.5000)")
    ax_roc_7.set_xlim([-0.02, 1.02]); ax_roc_7.set_ylim([-0.02, 1.02])
    ax_roc_7.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight='bold')
    ax_roc_7.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11, fontweight='bold')
    ax_roc_7.set_title(f"Receiver Operating Characteristic (ROC) – {dataset_label}", fontsize=13, fontweight='bold')
    ax_roc_7.legend(loc="lower right", frameon=True, fontsize=8.2)

    ax_pr_7.plot([0, 1], [base_pr, base_pr], 'k--', lw=1.5, alpha=0.5, label=f"Random Chance (Baseline = {base_pr:.3f})")
    ax_pr_7.set_xlim([-0.02, 1.02]); ax_pr_7.set_ylim([-0.02, 1.02])
    ax_pr_7.set_xlabel("Recall (Sensitivity)", fontsize=11, fontweight='bold')
    ax_pr_7.set_ylabel("Precision (PPV)", fontsize=11, fontweight='bold')
    ax_pr_7.set_title(f"Precision-Recall (PR) Curve – {dataset_label}", fontsize=13, fontweight='bold')
    ax_pr_7.legend(loc="lower right", frameon=True, fontsize=8.2)

    plt.suptitle(f"Advanced 7-Metric Virtual Screening Evaluation ({dataset_label}: {n_pairs} Pairs)", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig_7.savefig(os.path.join(out_dir, "advanced_scoring_eval.svg"), format='svg', bbox_inches='tight')
    fig_7.savefig(os.path.join(out_dir, "advanced_scoring_eval.png"), format='png', dpi=300, bbox_inches='tight')
    plt.close(fig_7)

    # ------------------------------------------------------------------
    # 2. Score Distributions & Decision Cutoffs
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 7, figsize=(28, 4.8), dpi=300)

    dist_metrics = [
        ("AF3 ipTM", df['AF3_ipTM'].values, "#4C72B0"),
        ("AF3 Inv PAE", inv_pae, "#55A868"),
        ("AF3 Alone", df['AF3_Score'].values, "#D55E00"),
        ("GNINA CNNscore", df['Gnina_CNNscore'].values, "#8172B0"),
        ("GNINA CNNaffinity", df['Gnina_CNNaffinity'].values, "#CCB974"),
        ("GNINA VS Score", df['Gnina_CNN_VS'].values, "#64B5CD"),
        ("Consensus Score", df['Consensus_Score'].values, "#009E73")
    ]

    for idx, (title, y_scores, color) in enumerate(dist_metrics):
        ax = axes[idx]
        pos_s = y_scores[y_true == 1]
        neg_s = y_scores[y_true == 0]

        plot_df = pd.DataFrame(
            [{'Score': s, 'Group': 'True Positive'} for s in pos_s] +
            [{'Score': s, 'Group': 'True Negative'} for s in neg_s]
        )

        sns.violinplot(
            data=plot_df, x='Group', y='Score', hue='Group',
            palette={'True Positive': color, 'True Negative': '#444444'},
            inner='quartile', ax=ax, legend=False
        )
        sns.stripplot(
            data=plot_df, x='Group', y='Score', color='black',
            alpha=0.4, jitter=0.15, size=3.5, ax=ax
        )

        opt_t, opt_m = compute_optimal_threshold(y_true, y_scores)
        ax.axhline(opt_t, color='red', linestyle='--', linewidth=1.8, label=f"Cutoff = {opt_t:.3f}")
        ax.set_title(f"{title}\n(Cutoff: {opt_t:.3f} | F1: {opt_m['f1']:.2f})", fontsize=10, fontweight='bold')
        ax.set_xlabel("", fontsize=9)
        ax.set_ylabel(title if idx == 0 else "", fontsize=10, fontweight='bold')
        ax.legend(loc='upper right', frameon=True, fontsize=8.0)

    plt.suptitle(f"Score Distributions & Youden's J Cutoffs ({dataset_label}: {n_pairs} Pairs)", fontsize=14, fontweight='bold', y=1.03)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "classification_score_distributions.svg"), format='svg', bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, "classification_score_distributions.png"), format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # ------------------------------------------------------------------
    # 3. Score Density Histograms
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 4, figsize=(20, 9.5), dpi=300)
    axes_flat = axes.flatten()

    hist_metrics = [
        ("AF3 ipTM", df['AF3_ipTM'], "[0, 1] Bounded", "#4C72B0"),
        ("AF3 Inv PAE (1/PAE)", pd.Series(inv_pae), "Inv Å (Higher is better)", "#55A868"),
        ("AF3 Score (ipTM / (1+PAE))", df['AF3_Score'], "[0, 1] Bounded", "#D55E00"),
        ("GNINA CNNscore", df['Gnina_CNNscore'], "[0, 1] Bounded", "#8172B0"),
        ("GNINA CNNaffinity", df['Gnina_CNNaffinity'], "pK_d (Higher is better)", "#CCB974"),
        ("GNINA VS (CNNscore * pK_d)", df['Gnina_CNN_VS'], "Product Score", "#64B5CD"),
        ("Consensus (AF3 * GNINA_VS)", df['Consensus_Score'], "Composite Score", "#009E73")
    ]

    for idx, (title, series_val, scale_desc, color) in enumerate(hist_metrics):
        ax = axes_flat[idx]
        pos_data = series_val[df['Is_True_Positive'] == 1]
        neg_data = series_val[df['Is_True_Positive'] == 0]

        sns.histplot(pos_data, kde=True, ax=ax, color=color, label="True Positive", stat="density", common_norm=False, alpha=0.45, bins=25)
        sns.histplot(neg_data, kde=True, ax=ax, color="#444444", label="Decoy Negative", stat="density", common_norm=False, alpha=0.35, bins=25)

        pos_med = pos_data.median()
        neg_med = neg_data.median()
        skew_val = series_val.skew()

        ax.set_title(f"{title}\n({scale_desc} | Skew: {skew_val:+.2f})", fontsize=11, fontweight='bold', pad=8)
        ax.set_xlabel(f"{title} Value", fontsize=10, fontweight='bold')
        ax.set_ylabel("Density", fontsize=10, fontweight='bold')
        ax.legend(loc="upper right", frameon=True, fontsize=8.5)
        ax.annotate(f"Pos Med: {pos_med:.2f}\nNeg Med: {neg_med:.2f}", xy=(0.05, 0.75), xycoords='axes fraction',
                    fontsize=8.5, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85))

    fig.delaxes(axes_flat[7])
    plt.suptitle(f"Score Distribution Histograms & KDE Density ({dataset_label}: {n_pairs} Pairs)", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(out_dir, "score_distribution_histograms.svg"), format='svg', bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, "score_distribution_histograms.png"), format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # ------------------------------------------------------------------
    # 4. Spearman Rank Correlation Matrix
    # ------------------------------------------------------------------
    fig_corr, ax_corr = plt.subplots(figsize=(9.5, 8.0), dpi=300)

    corr_df = pd.DataFrame({
        'Ground Truth Label': df['Is_True_Positive'].values,
        'AF3 ipTM': df['AF3_ipTM'].values,
        'AF3 Inv PAE': inv_pae,
        'AF3 Score': df['AF3_Score'].values,
        'GNINA CNNscore': df['Gnina_CNNscore'].values,
        'GNINA CNNaffinity': df['Gnina_CNNaffinity'].values,
        'GNINA VS Score': df['Gnina_CNN_VS'].values,
        'Consensus Score': df['Consensus_Score'].values
    })

    corr_matrix = corr_df.corr(method='spearman')
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0.0, vmax=1.0,
                ax=ax_corr, cbar_kws={'label': 'Spearman Rank Correlation (ρ)'},
                linewidths=0.8, linecolor='white')
    ax_corr.set_title(f"Spearman Rank Correlation Matrix – {dataset_label} ({n_pairs} Pairs)", fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    fig_corr.savefig(os.path.join(out_dir, "score_correlation_matrix.svg"), format='svg', bbox_inches='tight')
    fig_corr.savefig(os.path.join(out_dir, "score_correlation_matrix.png"), format='png', dpi=300, bbox_inches='tight')
    plt.close(fig_corr)

    print(f"Generated complete plot suite for '{dataset_label}' in '{out_dir}'.")


def print_summary_table(df, dataset_name):
    """Prints terminal evaluation metrics summary table."""
    y_true = df['Is_True_Positive'].values
    n_pos = int(np.sum(y_true))
    n_neg = len(y_true) - n_pos

    inv_pae = 1.0 / np.maximum(df['AF3_PAE_min'].values, 0.01)

    metrics = {
        "AF3 ipTM": df['AF3_ipTM'].values,
        "AF3 PAE_min (Inv: 1/PAE)": inv_pae,
        "AF3 Alone (ipTM / (1+PAE))": df['AF3_Score'].values,
        "GNINA CNNscore": df['Gnina_CNNscore'].values,
        "GNINA CNNaffinity (pK_d)": df['Gnina_CNNaffinity'].values,
        "GNINA VS Score (CNNscore * CNNaffinity)": df['Gnina_CNN_VS'].values,
        "Consensus Score (AF3 * GNINA_VS)": df['Consensus_Score'].values
    }

    print("\n" + "="*115)
    print(f"EVALUATION SUMMARY: {dataset_name} ({len(df)} total pairs | {n_pos} Positives, {n_neg} Negatives)")
    print("="*115)
    print(f"{'Scoring Method':<45} | {'ROC AUC':<9} | {'PR AUC':<9} | {'Opt Cutoff':<10} | {'Sens':<6} | {'Spec':<6} | {'Bal Acc':<8}")
    print("-"*115)

    for name, y_scores in metrics.items():
        # compute_optimal_threshold now returns roc_auc inside the metrics dict,
        # eliminating the separate roc_curve call that was here before.
        pr_auc_val = average_precision_score(y_true, y_scores)
        opt_t, opt_m = compute_optimal_threshold(y_true, y_scores)
        print(f"{name:<45} | {opt_m['roc_auc']:<9.4f} | {pr_auc_val:<9.4f} | {opt_t:<10.4f} | "
              f"{opt_m['sensitivity']:<6.2f} | {opt_m['specificity']:<6.2f} | {opt_m['balanced_accuracy']:<8.4f}")
    print("="*115 + "\n")


def _run_track(pred_dirs, gnina_csvs, out_dir, label, gt_map):
    """Executes one evaluation track: ranking → plots → summary table."""
    report_csv = os.path.join(out_dir, 'ranked_pairings_report.csv')
    rows, val_data = process_and_rank_dataset(pred_dirs, gnina_csvs, gt_map, report_csv)
    df = pd.DataFrame(rows)
    print_summary_table(df, label)
    generate_svg_slopegraph(val_data, os.path.join(out_dir, 'true_positive_comparison.svg'), title_suffix=f"({label})")
    plot_dataset_diagnostics(df, out_dir, label)


def main():
    parser = argparse.ArgumentParser(
        description="Run virtual screening evaluation. "
                    "With no arguments, runs all three standard tracks (Dataset 1, Dataset 2, Combined). "
                    "Supply --pred-dirs etc. to evaluate a custom dataset (e.g. Dataset 3)."
    )
    parser.add_argument('--pred-dirs', nargs='+', metavar='DIR',
                        help="Prediction ZIP directories for a custom single-track run.")
    parser.add_argument('--gnina-csvs', nargs='+', metavar='CSV',
                        help="GNINA score CSVs for the custom track.")
    parser.add_argument('--out-dir', metavar='DIR',
                        help="Output directory for the custom track.")
    parser.add_argument('--label', default='Custom Dataset',
                        help="Dataset label shown in plots and summary (default: 'Custom Dataset').")
    args = parser.parse_args()

    gt_map = load_ground_truth_map()
    print(f"Loaded ground truth mapping for {len(gt_map)} TF-ligand pairs.")

    if args.pred_dirs:
        # Custom single-track mode
        if not args.gnina_csvs or not args.out_dir:
            parser.error("--pred-dirs requires --gnina-csvs and --out-dir.")
        print(f"\n{'#'*80}\n# RUNNING CUSTOM EVALUATION: {args.label}\n{'#'*80}")
        _run_track(args.pred_dirs, args.gnina_csvs, args.out_dir, args.label, gt_map)
        print(f"\n{'='*80}\nEVALUATION COMPLETE → {args.out_dir}/\n{'='*80}\n")
        return

    # Default: run all three standard tracks
    print("\n" + "#"*80)
    print("# RUNNING EVALUATION ON DATASET 1 ONLY (First Set - 268 Pairs)")
    print("#"*80)
    _run_track(
        ['alphafold3_predictions_dataset1'],
        ['data/processed/gnina_scores_dataset1.csv', 'data/processed/gnina_scores.csv'],
        'results/dataset1',
        'Dataset 1 Only (First Set)',
        gt_map
    )

    print("\n" + "#"*80)
    print("# RUNNING EVALUATION ON DATASET 2 ONLY (Score2 Set - 204 Pairs)")
    print("#"*80)
    _run_track(
        ['alphafold3_predictions_score2'],
        ['data/processed/gnina_scores_score2.csv'],
        'results/dataset2_score2',
        'Dataset 2 Only (Score2 Set)',
        gt_map
    )

    print("\n" + "#"*80)
    print("# RUNNING EVALUATION ON COMBINED DATASET (Dataset 1 + Dataset 2 - 472 Pairs Total)")
    print("#"*80)
    _run_track(
        ['alphafold3_predictions_dataset1', 'alphafold3_predictions_score2'],
        ['data/processed/gnina_scores_dataset1.csv', 'data/processed/gnina_scores_score2.csv', 'data/processed/gnina_scores.csv'],
        'results/combined',
        'Combined Dataset (Set 1 + Set 2)',
        gt_map
    )

    print("\n" + "="*80)
    print("ALL 3 ANALYSIS RUNS PASSED SUCCESSFULLY!")
    print("Clean outputs generated:")
    print("  1. Dataset 1 (First Set): results/dataset1/")
    print("  2. Dataset 2 (Score2):    results/dataset2_score2/")
    print("  3. Combined Set:         results/combined/")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
