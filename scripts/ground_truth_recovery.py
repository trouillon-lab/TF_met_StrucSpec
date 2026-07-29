#!/usr/bin/env python3
"""
Ground Truth Recovery Analysis

D1 experimental pairs are the only true ground-truth positives in this study.
Datasets D2, D3, D4 contain predictions for many TF–metabolite combinations, but
without experimental confirmation.  This script asks:

  "For each TF that has an experimentally-confirmed D1 positive, if we rank all
   predictions for that TF (across D2/D3/D4) by structural score, does the
   experimentally-confirmed metabolite land at rank 1 or 2?"

Matching approach
-----------------
D1 and D2/D3/D4 use different metabolite ID systems (KEGG vs BiGG).  We match
on a stripped canonical SMILES string: lowercase, sorted atoms, stripped of
spaces.  This is not a true canonical SMILES, but it handles the common case
where the same SMILES string was used across datasets (they all sourced from
PubChem).  When the SMILES differ for the same compound, the pair will be noted
as unmatched.

Outputs
-------
  results/ground_truth_recovery/recovery_report.csv
  results/ground_truth_recovery/recovery_by_tf.png
  results/ground_truth_recovery/recovery_summary.png
"""

import argparse
import csv
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# ── configuration ──────────────────────────────────────────────────────────────

D1_CSVS = [
    'data/processed/pairings_subset_20.csv',
    'data/processed/pairings_remaining_248.csv',
    'data/processed/pairings_score2_benchmark.csv',
]

DATASET_RANKED = {
    'D2': 'results/dataset2_score2/ranked_pairings_report.csv',
    'D3': 'results/dataset3/ranked_pairings_report.csv',
    'D4': 'results/dataset4/ranked_pairings_report.csv',
}

# Pairings CSVs that carry SMILES (used to join onto ranked reports)
DATASET_PAIRINGS = {
    'D2': 'data/processed/pairings_score2_benchmark.csv',
    'D3': 'data/processed/pairings_dataset3_weekend.csv',
    'D4': 'data/processed/pairings_dataset4_stratified.csv',
}

OUT_DIR = 'results/ground_truth_recovery'
SCORE_COL = 'Consensus_Score'   # primary ranking; fall back to AF3_Score


def strip_smiles(s):
    """Normalize SMILES for matching: lowercase, remove spaces and charges."""
    return re.sub(r'\s', '', str(s).strip().lower())


def load_d1_positives():
    """Return dict: tf_name → set of stripped SMILES for confirmed positives."""
    positives = defaultdict(set)
    for path in D1_CSVS:
        if not os.path.exists(path):
            continue
        with open(path, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if row.get('Label', '').strip().lower() == 'positive':
                    tf  = row['TF_Name'].strip()
                    smi = strip_smiles(row.get('Ligand_SMILES', ''))
                    if smi:
                        positives[tf].add(smi)
    return positives


def build_smiles_lookup(pairings_path):
    """Build {clean_filename(TF)_bigg_id → stripped_SMILES} from a pairings CSV."""
    lookup = {}
    if not os.path.exists(pairings_path):
        return lookup
    with open(pairings_path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            tf   = row.get('TF_Name', '').strip()
            bigg = row.get('KEGG_ID', '').strip()
            smi  = strip_smiles(row.get('Ligand_SMILES', ''))
            if tf and bigg and smi:
                key = re.sub(r'[^a-zA-Z0-9]', '_', tf) + '_' + bigg
                lookup[key] = smi
    return lookup


def load_ranked_report(path, score_col=SCORE_COL, smiles_lookup=None):
    """Return list of dicts from ranked_pairings_report.csv, highest score first."""
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            sc = float(row.get(score_col) or row.get('AF3_Score') or 0)
            tfl = row.get('TF_Ligand', '').strip()
            smi = (smiles_lookup or {}).get(tfl, '')
            rows.append({
                'TF_Name':       row['TF_Name'].strip(),
                'Ligand_Name':   row.get('Ligand_Name', '').strip(),
                'TF_Ligand':     tfl,
                'AF3_Score':     float(row.get('AF3_Score', 0) or 0),
                'Score':         sc,
                'SMILES':        smi,
            })
    rows.sort(key=lambda r: r['Score'], reverse=True)
    return rows


def build_ranked_by_tf(ranked_rows):
    """Group ranked rows by TF, preserving score-order within each TF."""
    by_tf = defaultdict(list)
    for r in ranked_rows:
        by_tf[r['TF_Name']].append(r)
    return by_tf


# ── main analysis ──────────────────────────────────────────────────────────────

def run_analysis(out_dir, score_col=SCORE_COL):
    os.makedirs(out_dir, exist_ok=True)

    d1_pos = load_d1_positives()
    print(f"D1 ground-truth positives: {len(d1_pos)} TFs, "
          f"{sum(len(v) for v in d1_pos.values())} pairs")

    # Collect ranked predictions across datasets
    all_by_dataset = {}
    for ds, path in DATASET_RANKED.items():
        smi_lookup = build_smiles_lookup(DATASET_PAIRINGS.get(ds, ''))
        rows = load_ranked_report(path, score_col=score_col, smiles_lookup=smi_lookup)
        if rows:
            all_by_dataset[ds] = build_ranked_by_tf(rows)
            print(f"  {ds}: {len(rows)} predictions, {len(all_by_dataset[ds])} TFs")

    if not all_by_dataset:
        print("No ranked reports found — run run_full_evaluation.py first.")
        return

    # Per-TF recovery results
    records = []

    for tf, d1_smiles_set in sorted(d1_pos.items()):
        for ds, by_tf in all_by_dataset.items():
            if tf not in by_tf:
                continue
            tf_rows = by_tf[tf]  # sorted highest→lowest
            n_candidates = len(tf_rows)

            # assign ranks (1-indexed)
            for rank, row in enumerate(tf_rows, 1):
                row['_rank'] = rank

            # check if any D1-positive SMILES appears
            matched_rows = [r for r in tf_rows if r['SMILES'] in d1_smiles_set]
            unmatched_d1 = len(d1_smiles_set) - len(matched_rows)

            for r in matched_rows:
                records.append({
                    'TF':            tf,
                    'Dataset':       ds,
                    'Ligand':        r['TF_Ligand'],
                    'Score':         round(r['Score'], 4),
                    'AF3_Score':     round(r['AF3_Score'], 4),
                    'Rank':          r['_rank'],
                    'N_Candidates':  n_candidates,
                    'Percentile':    round(100 * (1 - (r['_rank'] - 1) / max(n_candidates - 1, 1)), 1),
                    'Recovered':     r['_rank'] <= 2,
                    'D1_Unmatched':  unmatched_d1,
                })

            if not matched_rows and unmatched_d1 > 0:
                records.append({
                    'TF':            tf,
                    'Dataset':       ds,
                    'Ligand':        'NO_SMILES_MATCH',
                    'Score':         None,
                    'AF3_Score':     None,
                    'Rank':          None,
                    'N_Candidates':  n_candidates,
                    'Percentile':    None,
                    'Recovered':     False,
                    'D1_Unmatched':  unmatched_d1,
                })

    # Write CSV
    csv_path = os.path.join(out_dir, 'recovery_report.csv')
    fieldnames = ['TF', 'Dataset', 'Ligand', 'Score', 'AF3_Score', 'Rank',
                  'N_Candidates', 'Percentile', 'Recovered', 'D1_Unmatched']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(records)
    print(f"\nWrote {len(records)} recovery records to {csv_path}")

    # Summary statistics
    matched = [r for r in records if r['Rank'] is not None]
    if not matched:
        print("No SMILES-matched pairs found.")
        return

    recovered_top1  = sum(1 for r in matched if r['Rank'] == 1)
    recovered_top2  = sum(1 for r in matched if r['Rank'] <= 2)
    recovered_top5  = sum(1 for r in matched if r['Rank'] <= 5)
    recovered_top10 = sum(1 for r in matched if r['Rank'] <= 10)
    total = len(matched)

    print(f"\nRecovery summary (SMILES-matched D1 positives vs D2/D3/D4 ranking):")
    print(f"  Total matched positive pairs: {total}")
    print(f"  Rank 1:    {recovered_top1}/{total}  ({100*recovered_top1/total:.1f}%)")
    print(f"  Rank ≤ 2:  {recovered_top2}/{total}  ({100*recovered_top2/total:.1f}%)")
    print(f"  Rank ≤ 5:  {recovered_top5}/{total}  ({100*recovered_top5/total:.1f}%)")
    print(f"  Rank ≤ 10: {recovered_top10}/{total}  ({100*recovered_top10/total:.1f}%)")

    _plot_recovery(matched, out_dir)

    return records


def _plot_recovery(matched, out_dir):
    # ── Figure 1: rank distribution histogram ─────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('D1 Ground-Truth Recovery in D2/D3/D4 Predictions', fontsize=14)

    datasets = sorted({r['Dataset'] for r in matched})
    colors = {'D2': '#2563EB', 'D3': '#16A34A', 'D4': '#9333EA'}

    ax = axes[0]
    max_rank_plot = 20
    for ds in datasets:
        ranks = [r['Rank'] for r in matched if r['Dataset'] == ds and r['Rank'] is not None]
        counts = [sum(1 for rank in ranks if rank <= k) for k in range(1, max_rank_plot + 1)]
        ax.plot(range(1, max_rank_plot + 1), counts,
                label=f'{ds} (n={len(ranks)})', color=colors.get(ds, 'grey'), lw=2, marker='o', ms=4)
    ax.set_xlabel('Rank cutoff', fontsize=11)
    ax.set_ylabel('D1 positives recovered', fontsize=11)
    ax.set_title('Cumulative recovery by rank', fontsize=12)
    ax.legend()
    ax.set_xlim(1, max_rank_plot)
    ax.set_xticks([1, 2, 5, 10, 15, 20])
    ax.grid(True, alpha=0.3)

    # ── Figure 2: percentile scatter ──────────────────────────────────────────
    ax2 = axes[1]
    for ds in datasets:
        rows_ds = [r for r in matched if r['Dataset'] == ds and r['Percentile'] is not None]
        ax2.scatter([r['Score'] for r in rows_ds],
                    [r['Percentile'] for r in rows_ds],
                    label=ds, color=colors.get(ds, 'grey'), alpha=0.7, s=60, edgecolors='white', lw=0.5)
        for r in rows_ds:
            if r['Rank'] <= 5:
                ax2.annotate(r['TF'], (r['Score'], r['Percentile']),
                             fontsize=7, ha='left', va='bottom',
                             xytext=(3, 3), textcoords='offset points')
    ax2.axhline(y=80, color='orange', ls='--', lw=1, label='80th percentile')
    ax2.set_xlabel('Consensus Score', fontsize=11)
    ax2.set_ylabel('Percentile rank of D1 positive', fontsize=11)
    ax2.set_title('Score vs. percentile for D1 positives', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, 'recovery_summary.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved recovery_summary.png")

    # ── Figure 3: per-TF bar chart ─────────────────────────────────────────────
    tfs = sorted({r['TF'] for r in matched})
    fig, ax = plt.subplots(figsize=(max(10, len(tfs) * 0.4), 5))
    x = np.arange(len(tfs))
    bar_h = 0.25
    for i, ds in enumerate(datasets):
        ds_ranks = []
        for tf in tfs:
            rows = [r['Rank'] for r in matched if r['TF'] == tf and r['Dataset'] == ds]
            ds_ranks.append(min(rows) if rows else None)
        vals = [r if r is not None else 0 for r in ds_ranks]
        bars = ax.barh(x + i * bar_h, vals, bar_h,
                       label=ds, color=colors.get(ds, 'grey'), alpha=0.8)
    ax.axvline(x=1, color='red', ls='--', lw=1, label='Rank 1')
    ax.axvline(x=2, color='orange', ls='--', lw=1, label='Rank 2')
    ax.set_yticks(x + bar_h)
    ax.set_yticklabels(tfs, fontsize=8)
    ax.set_xlabel('Rank of D1 experimental positive', fontsize=11)
    ax.set_title('Per-TF rank of experimental positive across datasets\n(lower = better; rank 0 = no prediction)', fontsize=11)
    ax.legend(loc='lower right')
    ax.set_xlim(0, max(20, max(r['Rank'] for r in matched if r['Rank']) + 1))
    ax.invert_yaxis()
    plt.tight_layout()
    path2 = os.path.join(out_dir, 'recovery_by_tf.png')
    fig.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved recovery_by_tf.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out-dir', default=OUT_DIR,
                    help=f'Output directory (default: {OUT_DIR})')
    ap.add_argument('--score-col', default=SCORE_COL,
                    choices=['Consensus_Score', 'AF3_Score'],
                    help='Score column to use for ranking (default: Consensus_Score)')
    args = ap.parse_args()

    run_analysis(args.out_dir, score_col=args.score_col)


if __name__ == '__main__':
    main()
