#!/usr/bin/env python3
"""
Correlate structural scores (AF3, GNINA, Consensus) with the database-derived
likelihood score from consensus_rank_interval.csv.

The consensus_rank file provides a continuous prior score for TF–metabolite pairs
derived from literature/database evidence.  Only pairs present in BOTH the
consensus_rank file AND the Dataset 3 ranked results are included.

Usage:
    python scripts/plot_consensus_rank_correlation.py \
        --ranked   results/dataset3/ranked_pairings_report.csv \
        --cr-file  data/raw/consensus_rank_interval.csv \
        --out-dir  results/consensus_rank_correlation
"""

import argparse
import ast
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from pipeline_utils import sanitize_key

# ── palette (single series: blue; regression: amber; muted fill: same blue @ 15 %) ──
BLUE    = "#2166ac"
AMBER   = "#d6604d"
MUTED   = "#abd9e9"
SURFACE = "#ffffff"


def load_data(ranked_csv, cr_csv):
    # Load consensus-rank map
    cr_map = {}
    with open(cr_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tf, met = ast.literal_eval(row["tf_met_pair"])
            k = sanitize_key(f"{tf}_{met}")
            cr_map[k] = {
                "cr_score":      float(row["score"]),
                "likelihood_sum": float(row["likelihood_sum"]),
                "consensus_count": float(row["consensus_count"]),
            }

    # Join with structural scores
    rows = []
    with open(ranked_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            k = sanitize_key(row["TF_Ligand"])
            if k not in cr_map:
                continue
            cr = cr_map[k]
            gnina = float(row["Gnina_CNN_VS"])
            rows.append({
                "TF_Ligand":       row["TF_Ligand"],
                "cr_score":        cr["cr_score"],
                "likelihood_sum":  cr["likelihood_sum"],
                "consensus_count": cr["consensus_count"],
                "AF3_Score":       float(row["AF3_Score"]),
                "Gnina_CNN_VS":    gnina,
                "Consensus_Score": float(row["Consensus_Score"]),
                "has_gnina":       gnina > 0,
            })
    return pd.DataFrame(rows)


def corr_stats(x, y):
    """Returns Pearson r, Spearman r, and p-values for both."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)
    return pr, pp, sr, sp, len(x)


def pval_str(p):
    if p < 0.001: return "p<0.001"
    if p < 0.01:  return f"p={p:.3f}"
    return f"p={p:.2f}"


def plot_scatter_grid(df, x_col, x_label, out_path_base, title_suffix=""):
    """3-panel scatter (AF3 / GNINA VS / Consensus) vs one x variable."""
    structural = [
        ("AF3_Score",       "AF3 Score\n(ipTM / (1+PAE))"),
        ("Gnina_CNN_VS",    "GNINA VS Score\n(CNNscore × CNNaffinity)"),
        ("Consensus_Score", "Consensus Score\n(AF3 × GNINA VS)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    fig.patch.set_facecolor(SURFACE)
    fig.suptitle(
        f"Structural Score vs. Consensus-Rank Prior{title_suffix}",
        fontsize=12, fontweight="bold", y=1.01,
    )

    for ax, (y_col, y_label) in zip(axes, structural):
        # Use only rows with non-zero GNINA when plotting GNINA/Consensus
        if y_col in ("Gnina_CNN_VS", "Consensus_Score"):
            sub = df[df["has_gnina"]].copy()
            note = f"(GNINA-scored pairs only, n={len(sub)})"
        else:
            sub = df.copy()
            note = f"(n={len(sub)})"

        x = sub[x_col].values
        y = sub[y_col].values

        pr, pp, sr, sp, n = corr_stats(x, y)

        # Regression line
        slope, intercept, *_ = stats.linregress(x, y)
        x_line = np.linspace(x.min(), x.max(), 200)

        ax.set_facecolor(SURFACE)
        ax.scatter(x, y, s=18, alpha=0.25, color=BLUE,
                   linewidths=0, rasterized=True)
        ax.plot(x_line, slope * x_line + intercept,
                color=AMBER, linewidth=2, zorder=5)

        # Correlation annotation
        ax.text(0.04, 0.95,
                f"Pearson  r={pr:+.3f}  {pval_str(pp)}\n"
                f"Spearman r={sr:+.3f}  {pval_str(sp)}\n{note}",
                transform=ax.transAxes, fontsize=8,
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="0.8", alpha=0.9))

        ax.set_xlabel(x_label, fontsize=9)
        ax.set_ylabel(y_label, fontsize=9)
        ax.grid(True, linewidth=0.4, alpha=0.4, color="0.7")
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    for ext in ("png", "svg"):
        p = f"{out_path_base}.{ext}"
        fig.savefig(p, dpi=150 if ext == "png" else None, bbox_inches="tight")
        print(f"Saved {p}")
    plt.close(fig)


def plot_binned(df, x_col, x_label, out_path_base, n_bins=5):
    """Box plots: structural scores per quintile of the CR variable."""
    structural = [
        ("AF3_Score",       "AF3 Score"),
        ("Gnina_CNN_VS",    "GNINA VS Score"),
        ("Consensus_Score", "Consensus Score"),
    ]

    sub_all  = df.copy()
    sub_gnina = df[df["has_gnina"]].copy()

    bins = pd.qcut(df[x_col], q=n_bins, duplicates="drop")
    bin_labels = [f"Q{i+1}\n[{iv.left:.2f},{iv.right:.2f}]"
                  for i, iv in enumerate(sorted(bins.cat.categories))]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    fig.patch.set_facecolor(SURFACE)
    fig.suptitle(
        f"Structural Scores by {x_label} Quintile",
        fontsize=12, fontweight="bold", y=1.01,
    )

    for ax, (y_col, y_label) in zip(axes, structural):
        use = sub_gnina if y_col in ("Gnina_CNN_VS", "Consensus_Score") else sub_all
        bin_groups = pd.qcut(use[x_col], q=n_bins, duplicates="drop")
        group_data = [use.loc[bin_groups == cat, y_col].values
                      for cat in sorted(bin_groups.cat.categories)]
        bin_lbls = [f"Q{i+1}\n[{iv.left:.2f},{iv.right:.2f}]"
                    for i, iv in enumerate(sorted(bin_groups.cat.categories))]

        bp = ax.boxplot(group_data, patch_artist=True,
                        medianprops=dict(color="black", linewidth=1.5),
                        whiskerprops=dict(linewidth=0.8),
                        capprops=dict(linewidth=0.8),
                        flierprops=dict(marker="o", markersize=3,
                                        markerfacecolor=BLUE, alpha=0.4,
                                        linewidth=0))
        for patch in bp["boxes"]:
            patch.set(facecolor=MUTED, alpha=0.7)

        # n counts below
        for i, gd in enumerate(group_data):
            ax.text(i + 1, ax.get_ylim()[0],
                    f"n={len(gd)}", ha="center", va="top", fontsize=7, color="0.45")

        ax.set_xticks(range(1, len(bin_lbls) + 1))
        ax.set_xticklabels(bin_lbls, fontsize=7.5)
        ax.set_xlabel(x_label, fontsize=9)
        ax.set_ylabel(y_label, fontsize=9)
        ax.grid(axis="y", linewidth=0.4, alpha=0.4, color="0.7")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_facecolor(SURFACE)

    fig.tight_layout()
    for ext in ("png", "svg"):
        p = f"{out_path_base}.{ext}"
        fig.savefig(p, dpi=150 if ext == "png" else None, bbox_inches="tight")
        print(f"Saved {p}")
    plt.close(fig)


def print_summary(df):
    print("\n" + "=" * 85)
    print(f"CONSENSUS-RANK × STRUCTURAL SCORE CORRELATION  (n={len(df)} matched pairs, "
          f"{df['has_gnina'].sum()} with GNINA)")
    print("=" * 85)
    hdr = f"  {'Structural score':<32} {'X variable':<18} {'Pearson r':>9} {'Spearman r':>11}  p(Pearson)"
    print(hdr)
    print("-" * 85)
    for y_col, y_label in [
        ("AF3_Score",       "AF3 Score"),
        ("Gnina_CNN_VS",    "GNINA VS Score"),
        ("Consensus_Score", "Consensus Score"),
    ]:
        use = df[df["has_gnina"]] if y_col != "AF3_Score" else df
        for x_col, x_label in [("cr_score", "cr_score"), ("likelihood_sum", "likelihood_sum")]:
            pr, pp, sr, sp, n = corr_stats(
                use[x_col].values.astype(float),
                use[y_col].values.astype(float),
            )
            print(f"  {y_label:<32} {x_label:<18} {pr:>+9.4f} {sr:>+11.4f}  {pval_str(pp)}")
    print("=" * 85 + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranked",  default="results/dataset3/ranked_pairings_report.csv")
    parser.add_argument("--cr-file", default="data/raw/consensus_rank_interval.csv")
    parser.add_argument("--out-dir", default="results/consensus_rank_correlation")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = load_data(args.ranked, args.cr_file)
    print(f"Loaded {len(df)} matched pairs "
          f"({df['has_gnina'].sum()} with non-zero GNINA score)")

    print_summary(df)

    plot_scatter_grid(
        df, "cr_score", "Consensus-Rank Score (prior)",
        os.path.join(args.out_dir, "scatter_cr_score"),
    )
    plot_scatter_grid(
        df, "likelihood_sum", "Likelihood Sum (prior)",
        os.path.join(args.out_dir, "scatter_likelihood_sum"),
    )
    plot_binned(
        df, "cr_score", "Consensus-Rank Score",
        os.path.join(args.out_dir, "binned_cr_score"),
    )
    plot_binned(
        df, "likelihood_sum", "Likelihood Sum",
        os.path.join(args.out_dir, "binned_likelihood_sum"),
    )

    print(f"All outputs written to: {args.out_dir}/")


if __name__ == "__main__":
    main()
