#!/usr/bin/env python3
"""
Correlate structural scores (AF3, GNINA, Consensus) with evidence confidence
levels from the curated TF–effector ground truth CSV.

Only pairs that appear in both a ranked-results CSV and the curated GT file are
included; unmatched pairs are silently excluded.  Ions and PTMs are excluded
because AF3 / GNINA are not designed to score them.

Usage:
    python scripts/plot_evidence_correlation.py \
        --ranked  results/dataset1/ranked_pairings_report.csv \
                  results/dataset2_score2/ranked_pairings_report.csv \
        --gt      data/raw/tf_effectors_curated_2607.csv \
        --out-dir results/evidence_correlation
"""

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from pipeline_utils import sanitize_key


EVIDENCE_ORDER = ["Strong\nPositive", "Weak\nPositive", "Positive\n(no evid.)",
                  "Strong\nNegative", "Weak\nNegative", "Negative\n(no evid.)"]

PALETTE = {
    "Strong\nPositive":    "#2166ac",
    "Weak\nPositive":      "#74add1",
    "Positive\n(no evid.)":"#abd9e9",
    "Strong\nNegative":    "#d73027",
    "Weak\nNegative":      "#f4a582",
    "Negative\n(no evid.)":"#fee090",
}

SCORE_COLS = [
    ("AF3_Score",       "AF3 Alone\n(ipTM / (1+PAE))"),
    ("Gnina_CNN_VS",    "GNINA VS Score\n(CNNscore × CNNaffinity)"),
    ("Consensus_Score", "Consensus Score\n(AF3 × GNINA_VS)"),
]


def load_gt_map(gt_csv):
    """Return dict sanitize_key(TF_lig) → {conf, ftype, smtype}."""
    gt = {}
    with open(gt_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tf   = row["Transcription factor"].strip()
            lig  = row["effector_name"].strip()
            kegg = row.get("kegg_id", "").strip()
            conf  = row.get("Evidence confidence level", "").strip()
            ftype = row.get("Effector functional type", "").strip().lower()
            smtype = row.get("Effector type", "").strip().lower()
            # exclude ions and PTMs (AF3/GNINA not designed for them)
            if smtype not in ("small molecule", ""):
                continue
            entry = {"conf": conf, "ftype": ftype}
            for id_field in [lig, kegg]:
                if tf and id_field:
                    gt.setdefault(sanitize_key(f"{tf}_{id_field}"), entry)
    return gt


def evidence_group(conf, ftype):
    """Map (confidence_level, functional_type) → group label."""
    if ftype == "positive":
        if conf == "Strong":
            return "Strong\nPositive"
        elif conf == "Weak":
            return "Weak\nPositive"
        else:
            return "Positive\n(no evid.)"
    else:  # negative or unknown
        if conf == "Strong":
            return "Strong\nNegative"
        elif conf == "Weak":
            return "Weak\nNegative"
        else:
            return "Negative\n(no evid.)"


def load_ranked(paths, gt_map):
    """Merge multiple ranked-result CSVs; keep only GT-matched small-molecule pairs."""
    rows = []
    seen = set()
    for p in paths:
        if not os.path.exists(p):
            print(f"Warning: {p} not found — skipping.")
            continue
        with open(p, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                tf  = row.get("TF_Name", "").strip()
                lig = row.get("Ligand_Name", "").strip()
                key = sanitize_key(f"{tf}_{lig}")
                if key in gt_map and key not in seen:
                    seen.add(key)
                    entry = gt_map[key]
                    rows.append({
                        "TF_Ligand":       row["TF_Ligand"],
                        "AF3_Score":       float(row["AF3_Score"]),
                        "Gnina_CNN_VS":    float(row["Gnina_CNN_VS"]),
                        "Consensus_Score": float(row["Consensus_Score"]),
                        "AF3_ipTM":        float(row["AF3_ipTM"]),
                        "AF3_PAE_min":     float(row["AF3_PAE_min"]),
                        "conf":            entry["conf"],
                        "ftype":           entry["ftype"],
                        "group":           evidence_group(entry["conf"], entry["ftype"]),
                    })
    return pd.DataFrame(rows)


def mann_whitney(a, b):
    """Two-sided Mann-Whitney U; return (U, p)."""
    if len(a) < 2 or len(b) < 2:
        return None, None
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return u, p


def sig_label(p):
    if p is None: return "n.d."
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."


def add_bracket(ax, x1, x2, y, p):
    """Draw a significance bracket between x1 and x2 at height y."""
    label = sig_label(p)
    h = 0.015 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.8, color="0.3")
    ax.text((x1 + x2) / 2, y + h * 1.1, label, ha="center", va="bottom", fontsize=8)


def plot_violin(df, out_dir):
    """Main violin/strip figure: 3 score columns × evidence groups."""
    present_groups = [g for g in EVIDENCE_ORDER if g in df["group"].values]
    n_groups = len(present_groups)

    fig, axes = plt.subplots(1, 3, figsize=(14, 6), sharey=False)
    fig.suptitle(
        "Structural Scores vs. Evidence Confidence Level\n"
        f"(n={len(df)} GT-matched small-molecule pairs)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    for ax, (col, label) in zip(axes, SCORE_COLS):
        group_data = [df.loc[df["group"] == g, col].values for g in present_groups]
        colors     = [PALETTE[g] for g in present_groups]

        positions = range(n_groups)
        parts = ax.violinplot(
            [d if len(d) > 0 else [0] for d in group_data],
            positions=positions,
            showmedians=True,
            showextrema=True,
        )
        for pc, c in zip(parts["bodies"], colors):
            pc.set_facecolor(c)
            pc.set_alpha(0.65)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.5)

        # Jitter strip
        rng = np.random.default_rng(42)
        for i, (d, c) in enumerate(zip(group_data, colors)):
            if len(d) == 0:
                continue
            jx = i + rng.uniform(-0.12, 0.12, len(d))
            ax.scatter(jx, d, color=c, edgecolors="0.3", linewidths=0.4,
                       s=22, alpha=0.8, zorder=3)

        ax.set_xticks(range(n_groups))
        ax.set_xticklabels(present_groups, fontsize=7.5)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(axis="y", linewidth=0.4, alpha=0.5)

        # Counts below x-axis
        for i, (g, d) in enumerate(zip(present_groups, group_data)):
            ax.text(i, ax.get_ylim()[0] - 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                    f"n={len(d)}", ha="center", va="top", fontsize=7, color="0.4")

        # Significance: Strong Pos vs Weak Pos, Strong Pos vs Strong Neg
        sp_idx = present_groups.index("Strong\nPositive") if "Strong\nPositive" in present_groups else None
        wp_idx = present_groups.index("Weak\nPositive")  if "Weak\nPositive" in present_groups else None
        sn_idx = present_groups.index("Strong\nNegative") if "Strong\nNegative" in present_groups else None

        ymax = ax.get_ylim()[1]
        dy   = 0.08 * (ax.get_ylim()[1] - ax.get_ylim()[0])

        if sp_idx is not None and wp_idx is not None:
            _, p = mann_whitney(group_data[sp_idx], group_data[wp_idx])
            add_bracket(ax, sp_idx, wp_idx, ymax + 0.5 * dy, p)
        if sp_idx is not None and sn_idx is not None:
            _, p = mann_whitney(group_data[sp_idx], group_data[sn_idx])
            add_bracket(ax, sp_idx, sn_idx, ymax + 1.4 * dy, p)

    fig.tight_layout()
    for ext in ("png", "svg"):
        path = os.path.join(out_dir, f"evidence_vs_scores.{ext}")
        dpi = 150 if ext == "png" else None
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"Saved {path}")
    plt.close(fig)


def plot_mean_heatmap(df, out_dir):
    """Heatmap of mean scores per evidence group."""
    present_groups = [g for g in EVIDENCE_ORDER if g in df["group"].values]
    score_labels   = [s[1].replace("\n", " ") for s in SCORE_COLS]
    score_cols     = [s[0] for s in SCORE_COLS]

    means = np.array([
        [df.loc[df["group"] == g, col].mean() for col in score_cols]
        for g in present_groups
    ])

    fig, ax = plt.subplots(figsize=(7, max(3, 0.6 * len(present_groups))))
    im = ax.imshow(means, aspect="auto", cmap="RdYlBu", vmin=0)
    ax.set_xticks(range(len(score_cols)))
    ax.set_xticklabels(score_labels, fontsize=9)
    ax.set_yticks(range(len(present_groups)))
    ax.set_yticklabels(present_groups, fontsize=9)
    for (i, j), val in np.ndenumerate(means):
        ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9,
                color="white" if val < means.max() * 0.4 else "black")
    plt.colorbar(im, ax=ax, label="Mean score")
    ax.set_title("Mean Structural Score by Evidence Group", fontsize=11, fontweight="bold")
    fig.tight_layout()
    for ext in ("png", "svg"):
        path = os.path.join(out_dir, f"evidence_mean_heatmap.{ext}")
        dpi = 150 if ext == "png" else None
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"Saved {path}")
    plt.close(fig)


def print_summary(df):
    print("\n" + "=" * 90)
    print(f"EVIDENCE–SCORE CORRELATION SUMMARY  ({len(df)} GT-matched small-molecule pairs)")
    print("=" * 90)
    groups = [g for g in EVIDENCE_ORDER if g in df["group"].values]
    hdr = f"{'Group':<25} {'n':>4}  {'AF3_Score':>10}  {'GNINA_VS':>10}  {'Consensus':>10}"
    print(hdr)
    print("-" * 70)
    for g in groups:
        sub = df[df["group"] == g]
        print(f"{g.replace(chr(10), ' '):<25} {len(sub):>4}  "
              f"{sub['AF3_Score'].mean():>10.4f}  "
              f"{sub['Gnina_CNN_VS'].mean():>10.4f}  "
              f"{sub['Consensus_Score'].mean():>10.4f}")
    print()

    # Spearman: encode confidence × label as ordinal
    label_map = {
        "Strong\nPositive": 5, "Weak\nPositive": 4, "Positive\n(no evid.)": 3,
        "Negative\n(no evid.)": 2, "Weak\nNegative": 1, "Strong\nNegative": 0,
    }
    ordinal = df["group"].map(label_map).dropna()
    if len(ordinal) > 5:
        for col, label in SCORE_COLS:
            scores = df.loc[ordinal.index, col]
            r, p = stats.spearmanr(ordinal, scores)
            print(f"  Spearman r({label.split(chr(10))[0].strip()}): {r:+.3f}  p={p:.4f}")
    print("=" * 90 + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranked", nargs="+", default=[
        "results/dataset1/ranked_pairings_report.csv",
        "results/dataset2_score2/ranked_pairings_report.csv",
    ], help="Ranked-results CSVs (default: D1 + D2)")
    parser.add_argument("--gt", default="data/raw/tf_effectors_curated_2607.csv")
    parser.add_argument("--out-dir", default="results/evidence_correlation")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    gt_map = load_gt_map(args.gt)
    print(f"Loaded GT map: {len(gt_map)} keys (small-molecule pairs only)")

    df = load_ranked(args.ranked, gt_map)
    print(f"Matched {len(df)} GT-annotated pairs from {len(args.ranked)} ranked file(s)")

    if df.empty:
        print("No matching pairs found — exiting.")
        sys.exit(0)

    print_summary(df)
    plot_violin(df, args.out_dir)
    plot_mean_heatmap(df, args.out_dir)
    print(f"\nAll outputs written to: {args.out_dir}/")


if __name__ == "__main__":
    main()
