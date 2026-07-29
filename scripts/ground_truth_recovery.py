#!/usr/bin/env python3
"""
Ground Truth Recovery Analysis

D1 contains experimentally validated TF–metabolite interactions — the only true
positives in this study.  D2/D3/D4 contain structural predictions for many
metabolite candidates per TF, without experimental confirmation.

The key question: for each TF with a D1-confirmed binder, if we rank ALL
structurally-scored candidates for that TF (pooled across D2+D3+D4, deduplicated
by SMILES) by AF3 score, does the experimentally confirmed metabolite land at
rank 1 or 2?

This tests whether AF3 structural scoring alone can identify the real binder
from the pool of candidates we ran predictions on.

Matching approach
-----------------
D1 uses KEGG/CHEBI IDs; D2/D3/D4 use BiGG IDs.  We join on SMILES strings
sourced from the pairings CSVs (both pulled from PubChem, so strings usually
match).  When SMILES differ for the same compound the pair is logged as
unmatched in the coverage column.

Outputs
-------
  results/ground_truth_recovery/recovery_report.csv   — per-TF ranked results
  results/ground_truth_recovery/recovery_summary.png  — cumulative + percentile
  results/ground_truth_recovery/recovery_by_tf.png    — per-TF rank bar chart
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

# Ranked scores already computed by run_full_evaluation.py
DATASET_RANKED = {
    'D2': 'results/dataset2_score2/ranked_pairings_report.csv',
    'D3': 'results/dataset3/ranked_pairings_report.csv',
    'D4': 'results/dataset4/ranked_pairings_report.csv',
}

# Pairings CSVs carry the SMILES (ranked reports do not)
DATASET_PAIRINGS = {
    'D2': 'data/processed/pairings_score2_benchmark.csv',
    'D3': 'data/processed/pairings_dataset3_weekend.csv',
    'D4': 'data/processed/pairings_dataset4_stratified.csv',
}

OUT_DIR = 'results/ground_truth_recovery'


def strip_smiles(s):
    """Normalise SMILES for matching: lowercase, remove whitespace."""
    return re.sub(r'\s', '', str(s).strip().lower())


def load_d1_positives():
    """Return {tf_name: set(stripped_smiles)} for experimentally confirmed D1 pairs."""
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
    """Return {TF_Ligand_key: stripped_smiles} from a pairings CSV."""
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


def load_all_predictions(score_col='AF3_Score'):
    """
    Pool every scored pair from all available datasets.

    Returns {tf_name: [{smi, score, af3_score, tfl, dataset}, ...]}
    where each unique SMILES is kept only once per TF (highest score wins
    when the same compound appears in multiple datasets).
    """
    by_tf = defaultdict(dict)   # tf → {smi: best_row}

    for ds, ranked_path in DATASET_RANKED.items():
        if not os.path.exists(ranked_path):
            continue
        smi_lookup = build_smiles_lookup(DATASET_PAIRINGS.get(ds, ''))

        with open(ranked_path, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                tf  = row['TF_Name'].strip()
                tfl = row.get('TF_Ligand', '').strip()
                smi = smi_lookup.get(tfl, '')
                af3 = float(row.get('AF3_Score', 0) or 0)
                con = float(row.get('Consensus_Score', 0) or row.get('AF3_Score', 0) or 0)
                sc  = af3 if score_col == 'AF3_Score' else con

                if not smi:
                    continue

                # Keep the highest-scored entry when the same SMILES appears
                # in multiple datasets (deduplication)
                existing = by_tf[tf].get(smi)
                if existing is None or sc > existing['score']:
                    by_tf[tf][smi] = {
                        'smi':      smi,
                        'score':    sc,
                        'af3':      af3,
                        'con':      con,
                        'tfl':      tfl,
                        'dataset':  ds,
                    }

    # Convert to sorted lists
    pooled = {}
    for tf, smi_dict in by_tf.items():
        rows = sorted(smi_dict.values(), key=lambda r: r['score'], reverse=True)
        for rank, r in enumerate(rows, 1):
            r['rank'] = rank
        pooled[tf] = rows
    return pooled


# ── main analysis ──────────────────────────────────────────────────────────────

def run_analysis(out_dir, score_col='AF3_Score'):
    os.makedirs(out_dir, exist_ok=True)

    d1_pos = load_d1_positives()
    print(f"D1 ground-truth positives: {len(d1_pos)} TFs, "
          f"{sum(len(v) for v in d1_pos.values())} pairs")

    pooled = load_all_predictions(score_col=score_col)
    loaded_ds = [ds for ds, p in DATASET_RANKED.items() if os.path.exists(p)]
    print(f"Datasets loaded: {loaded_ds}")
    print(f"TFs with any prediction: {len(pooled)}")
    total_scored = sum(len(v) for v in pooled.values())
    print(f"Total unique scored pairs (deduplicated): {total_scored}")

    records = []
    no_predictions = []
    no_smiles_match = []

    for tf, d1_smis in sorted(d1_pos.items()):
        if tf not in pooled:
            no_predictions.append(tf)
            continue

        tf_rows = pooled[tf]
        n_cand  = len(tf_rows)
        tf_smis = {r['smi'] for r in tf_rows}

        for d1_smi in d1_smis:
            if d1_smi not in tf_smis:
                no_smiles_match.append((tf, d1_smi[:40]))
                continue

            match = next(r for r in tf_rows if r['smi'] == d1_smi)
            records.append({
                'TF':           tf,
                'TF_Ligand':    match['tfl'],
                'Dataset':      match['dataset'],
                'AF3_Score':    round(match['af3'],  4),
                'Score':        round(match['score'], 4),
                'Rank':         match['rank'],
                'N_Candidates': n_cand,
                'Percentile':   round(100 * (1 - (match['rank'] - 1) / max(n_cand - 1, 1)), 1),
                'Top1':         match['rank'] == 1,
                'Top2':         match['rank'] <= 2,
                'Top5':         match['rank'] <= 5,
                'Top10':        match['rank'] <= 10,
            })

    # ── write CSV ────────────────────────────────────────────────────────────
    csv_path = os.path.join(out_dir, 'recovery_report.csv')
    fieldnames = ['TF', 'TF_Ligand', 'Dataset', 'AF3_Score', 'Score',
                  'Rank', 'N_Candidates', 'Percentile', 'Top1', 'Top2', 'Top5', 'Top10']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(records)

    # ── print summary ─────────────────────────────────────────────────────────
    n = len(records)
    if n == 0:
        print("No SMILES-matched D1 positives found in any scored dataset.")
        return

    n1  = sum(r['Top1']  for r in records)
    n2  = sum(r['Top2']  for r in records)
    n5  = sum(r['Top5']  for r in records)
    n10 = sum(r['Top10'] for r in records)
    med_rank = float(np.median([r['Rank'] for r in records]))
    med_pct  = float(np.median([r['Percentile'] for r in records]))
    avg_cand = float(np.mean([r['N_Candidates'] for r in records]))

    print(f"\n{'─'*55}")
    print(f" Ground-truth recovery  (scoring: {score_col})")
    print(f"{'─'*55}")
    print(f" Matched D1 positives:  {n}  (across {len({r['TF'] for r in records})} TFs)")
    print(f" Avg candidates per TF: {avg_cand:.1f}")
    print(f" Rank 1:    {n1:3d}/{n}  ({100*n1/n:.1f}%)   vs random {100/avg_cand:.1f}%")
    print(f" Rank ≤ 2:  {n2:3d}/{n}  ({100*n2/n:.1f}%)")
    print(f" Rank ≤ 5:  {n5:3d}/{n}  ({100*n5/n:.1f}%)")
    print(f" Rank ≤ 10: {n10:3d}/{n}  ({100*n10/n:.1f}%)")
    print(f" Median rank:           {med_rank:.0f}  (median percentile: {med_pct:.0f}th)")
    print(f"{'─'*55}")
    print(f"\n TFs with no structural predictions: {len(no_predictions)}")
    print(f" D1 positives with no SMILES match:  {len(no_smiles_match)}")
    if no_smiles_match[:5]:
        print(f"   e.g. {no_smiles_match[:5]}")

    _plot(records, out_dir, score_col, avg_cand)
    print(f"\nWrote {n} records → {csv_path}")


def _plot(records, out_dir, score_col, avg_cand):
    # ── Figure 1: cumulative recovery + random baseline ───────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f'D1 Ground-Truth Recovery  |  combined D2+D3+D4 pool  |  ranked by {score_col}',
        fontsize=12
    )

    ax = axes[0]
    max_k = 20
    ks = list(range(1, max_k + 1))
    actual  = [sum(1 for r in records if r['Rank'] <= k) for k in ks]
    random_ = [min(len(records), k / avg_cand * len(records)) for k in ks]

    ax.plot(ks, actual,  color='#16A34A', lw=2.5, marker='o', ms=4, label='AF3 score')
    ax.plot(ks, random_, color='#9CA3AF', lw=1.5, ls='--', label=f'Random baseline (1/{avg_cand:.0f})')
    ax.fill_between(ks, random_, actual, alpha=0.15, color='#16A34A')
    ax.set_xlabel('Rank cutoff k', fontsize=11)
    ax.set_ylabel('D1 positives recovered', fontsize=11)
    ax.set_title('Cumulative recovery vs random', fontsize=11)
    ax.set_xlim(1, max_k)
    ax.set_xticks([1, 2, 5, 10, 15, 20])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Enrichment factor labels at k=1,2,5
    for k in [1, 2, 5]:
        ef = actual[k-1] / max(random_[k-1], 0.001)
        ax.annotate(f'EF={ef:.1f}×', (k, actual[k-1]),
                    fontsize=8, ha='left', va='bottom',
                    xytext=(3, 4), textcoords='offset points', color='#16A34A')

    # ── Figure 2: rank distribution ───────────────────────────────────────────
    ax2 = axes[1]
    bins = [0.5, 1.5, 2.5, 5.5, 10.5, 20.5, max(r['Rank'] for r in records) + 0.5]
    labels = ['1', '2', '3–5', '6–10', '11–20', f'>{20}']
    counts = [
        sum(1 for r in records if r['Rank'] == 1),
        sum(1 for r in records if r['Rank'] == 2),
        sum(1 for r in records if 3 <= r['Rank'] <= 5),
        sum(1 for r in records if 6 <= r['Rank'] <= 10),
        sum(1 for r in records if 11 <= r['Rank'] <= 20),
        sum(1 for r in records if r['Rank'] > 20),
    ]
    colors = ['#16A34A', '#65A30D', '#EAB308', '#F97316', '#DC2626', '#6B7280']
    bars = ax2.bar(labels, counts, color=colors, edgecolor='white', lw=0.5)
    for bar, cnt in zip(bars, counts):
        if cnt:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     str(cnt), ha='center', va='bottom', fontsize=9)
    ax2.set_xlabel('Rank of D1 experimental metabolite\n(in pooled D2+D3+D4 candidate list)', fontsize=10)
    ax2.set_ylabel('Number of TF–metabolite pairs', fontsize=10)
    ax2.set_title('Rank distribution of D1 positives', fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    p = os.path.join(out_dir, 'recovery_summary.png')
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {p}")

    # ── Figure 3: per-TF rank ─────────────────────────────────────────────────
    tfs = sorted({r['TF'] for r in records})
    # For TFs with multiple D1 positives show the best (lowest) rank
    tf_best_rank = {tf: min(r['Rank'] for r in records if r['TF'] == tf) for tf in tfs}
    tfs_sorted   = sorted(tfs, key=lambda t: tf_best_rank[t])

    fig2, ax3 = plt.subplots(figsize=(max(10, len(tfs_sorted) * 0.35), 5))
    xpos = np.arange(len(tfs_sorted))
    bar_colors = ['#16A34A' if tf_best_rank[t] <= 2 else
                  '#EAB308' if tf_best_rank[t] <= 5 else
                  '#DC2626'
                  for t in tfs_sorted]
    ax3.bar(xpos, [tf_best_rank[t] for t in tfs_sorted],
            color=bar_colors, edgecolor='white', lw=0.3)
    ax3.axhline(y=1, color='#16A34A', ls='--', lw=1, label='Rank 1')
    ax3.axhline(y=2, color='#EAB308', ls='--', lw=1, label='Rank 2')
    ax3.set_xticks(xpos)
    ax3.set_xticklabels(tfs_sorted, rotation=75, ha='right', fontsize=7)
    ax3.set_ylabel('Best rank of D1 experimental metabolite\n(lower = better)', fontsize=10)
    ax3.set_title('Per-TF recovery rank  (sorted best→worst)  |  pooled D2+D3+D4', fontsize=11)
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    p2 = os.path.join(out_dir, 'recovery_by_tf.png')
    fig2.savefig(p2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved {p2}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out-dir', default=OUT_DIR)
    ap.add_argument('--score-col', default='AF3_Score',
                    choices=['AF3_Score', 'Consensus_Score'],
                    help='Score to rank by (default: AF3_Score)')
    args = ap.parse_args()
    run_analysis(args.out_dir, score_col=args.score_col)


if __name__ == '__main__':
    main()
