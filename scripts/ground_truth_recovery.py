#!/usr/bin/env python3
"""
Ground Truth Recovery Analysis

D1 contains experimentally validated TF–metabolite pairs — the only true
positives in this study.  D1 pairs have their own AF3 structural scores stored
in results/dataset1/ranked_pairings_report.csv.

D2/D3/D4 are all computational candidate sets: none are experimentally
validated.  Their AF3 scores are in their respective ranked_pairings_report.csv
files.

The key question: for each TF with a D1-confirmed binder, if we pool that TF's
D1 positive pair(s) (by AF3 score) together with ALL D2/D3/D4 predictions for
that same TF and rank the entire combined pool by AF3 score — does the
experimentally confirmed metabolite land at rank 1 (or 2)?

This tests whether AF3 structural scoring can distinguish the real binder from
all computational candidates across datasets.

No SMILES matching is required: D1 pairs are identified directly by their
TF_Ligand key in the D1 ranked report.  If the same metabolite appears in both
D1 and a candidate dataset under different names, it simply appears twice in the
pool; this does not affect whether the D1 pair ranks first.

Outputs
-------
  results/ground_truth_recovery/recovery_report.csv    per-pair results
  results/ground_truth_recovery/recovery_summary.png   cumulative recovery + rank distribution
  results/ground_truth_recovery/recovery_by_tf.png     per-TF best rank bar chart
"""

import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# ── paths ──────────────────────────────────────────────────────────────────────

D1_RANKED = 'results/dataset1/ranked_pairings_report.csv'

CANDIDATE_RANKED = {
    'D2': 'results/dataset2_score2/ranked_pairings_report.csv',
    'D3': 'results/dataset3/ranked_pairings_report.csv',
    'D4': 'results/dataset4/ranked_pairings_report.csv',
}

OUT_DIR = 'results/ground_truth_recovery'


# ── data loading ───────────────────────────────────────────────────────────────

def load_d1_positives(score_col):
    """Load experimentally confirmed D1 pairs (Is_True_Positive=True) with scores.

    Returns {tf_name: [row_dict, ...]} where each row has keys:
        tfl, ligand, af3, consensus, score, dataset
    """
    result = defaultdict(list)
    with open(D1_RANKED, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row.get('Is_True_Positive', '').strip() != 'True':
                continue
            tf  = row['TF_Name'].strip()
            af3 = float(row.get('AF3_Score', 0) or 0)
            con = float(row.get('Consensus_Score', 0) or row.get('AF3_Score', 0) or 0)
            result[tf].append({
                'tfl':       row['TF_Ligand'].strip(),
                'ligand':    row.get('Ligand_Name', '').strip(),
                'af3':       af3,
                'consensus': con,
                'score':     af3 if score_col == 'AF3_Score' else con,
                'dataset':   'D1',
            })
    return result


def load_candidates(score_col):
    """Load all scored pairs from available candidate datasets (D2/D3/D4).

    Returns {tf_name: [row_dict, ...]} sorted descending by score within each TF.
    """
    by_tf = defaultdict(list)
    for ds, path in CANDIDATE_RANKED.items():
        if not os.path.exists(path):
            continue
        with open(path, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                tf  = row['TF_Name'].strip()
                af3 = float(row.get('AF3_Score', 0) or 0)
                con = float(row.get('Consensus_Score', 0) or row.get('AF3_Score', 0) or 0)
                by_tf[tf].append({
                    'tfl':       row['TF_Ligand'].strip(),
                    'ligand':    row.get('Ligand_Name', '').strip(),
                    'af3':       af3,
                    'consensus': con,
                    'score':     af3 if score_col == 'AF3_Score' else con,
                    'dataset':   ds,
                })
    return dict(by_tf)


# ── main analysis ──────────────────────────────────────────────────────────────

def run_analysis(out_dir, score_col='AF3_Score'):
    os.makedirs(out_dir, exist_ok=True)

    d1_pos     = load_d1_positives(score_col)
    candidates = load_candidates(score_col)

    loaded_ds  = [ds for ds, p in CANDIDATE_RANKED.items() if os.path.exists(p)]
    n_d1_pairs = sum(len(v) for v in d1_pos.values())

    print(f"D1 experimentally confirmed positives: {len(d1_pos)} TFs, {n_d1_pairs} pairs")
    print(f"Candidate datasets loaded: {loaded_ds}")
    total_cand = sum(len(v) for v in candidates.values())
    print(f"Total candidate predictions: {total_cand} across {len(candidates)} TFs")

    records        = []
    no_candidates  = []

    for tf, d1_rows in sorted(d1_pos.items()):
        cands = candidates.get(tf, [])
        if not cands:
            no_candidates.append(tf)

        # Pool: D1 positive pair(s) + all D2/D3/D4 candidates for this TF
        pool = d1_rows + cands
        pool.sort(key=lambda r: r['score'], reverse=True)
        n_pool = len(pool)

        for rank, row in enumerate(pool, 1):
            row['_rank'] = rank

        datasets_in_pool = sorted({r['dataset'] for r in pool} - {'D1'}) or ['(none)']

        for d1_row in d1_rows:
            rank = d1_row['_rank']
            records.append({
                'TF':                   tf,
                'D1_Pair':              d1_row['tfl'],
                'D1_Ligand':            d1_row['ligand'],
                'D1_AF3_Score':         round(d1_row['af3'],       4),
                'D1_Consensus_Score':   round(d1_row['consensus'], 4),
                'Rank_in_Combined':     rank,
                'Pool_Size':            n_pool,
                'N_Candidates':         len(cands),
                'Percentile':           round(100 * (1 - (rank - 1) / max(n_pool - 1, 1)), 1),
                'Candidate_Datasets':   '+'.join(datasets_in_pool),
                'Top1':                 rank == 1,
                'Top2':                 rank <= 2,
                'Top5':                 rank <= 5,
                'Top10':                rank <= 10,
            })

    # ── CSV output ───────────────────────────────────────────────────────────
    csv_path = os.path.join(out_dir, 'recovery_report.csv')
    fieldnames = [
        'TF', 'D1_Pair', 'D1_Ligand', 'D1_AF3_Score', 'D1_Consensus_Score',
        'Rank_in_Combined', 'Pool_Size', 'N_Candidates', 'Percentile',
        'Candidate_Datasets', 'Top1', 'Top2', 'Top5', 'Top10',
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(sorted(records, key=lambda r: r['Rank_in_Combined']))

    # ── summary stats ────────────────────────────────────────────────────────
    n = len(records)
    if n == 0:
        print("No D1 positives found in ranked report.")
        return

    n1   = sum(1 for r in records if r['Top1'])
    n2   = sum(1 for r in records if r['Top2'])
    n5   = sum(1 for r in records if r['Top5'])
    n10  = sum(1 for r in records if r['Top10'])
    ranks = [r['Rank_in_Combined'] for r in records]
    avg_cand = float(np.mean([r['N_Candidates'] for r in records]))

    print(f"\n{'─'*55}")
    print(f" Ground-truth recovery  (ranking by: {score_col})")
    print(f" Candidate datasets:     {loaded_ds}")
    print(f"{'─'*55}")
    print(f" D1 pairs evaluated:     {n}  ({len({r['TF'] for r in records})} TFs)")
    print(f" Avg competitors / TF:   {avg_cand:.1f}  (D2+D3+D4 only)")
    print(f" Rank 1:    {n1:3d}/{n}  ({100*n1/n:.1f}%)   random baseline {100/max(avg_cand+1,1):.1f}%")
    print(f" Rank ≤ 2:  {n2:3d}/{n}  ({100*n2/n:.1f}%)")
    print(f" Rank ≤ 5:  {n5:3d}/{n}  ({100*n5/n:.1f}%)")
    print(f" Rank ≤ 10: {n10:3d}/{n}  ({100*n10/n:.1f}%)")
    print(f" Median rank:            {float(np.median(ranks)):.0f}")
    print(f"{'─'*55}")
    print(f" TFs with D1 positive but no candidate predictions: {len(no_candidates)}")
    if no_candidates:
        print(f"   {sorted(no_candidates)}")

    _plot(records, out_dir, score_col, avg_cand)
    print(f"\nWrote {n} records → {csv_path}")


def _plot(records, out_dir, score_col, avg_cand):
    # ── Figure 1: cumulative recovery + rank distribution ─────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f'D1 Experimental Ground-Truth Recovery\n'
        f'Combined pool: D1 positive + D2+D3+D4 candidates  |  ranked by {score_col}',
        fontsize=11,
    )

    ax = axes[0]
    max_k = 20
    ks = list(range(1, max_k + 1))
    n  = len(records)
    actual   = [sum(1 for r in records if r['Rank_in_Combined'] <= k) for k in ks]
    # Random baseline: 1 D1 pair among (avg_cand + 1) total → expected recovered at k
    pool_avg = avg_cand + 1  # +1 for the D1 pair itself
    random_  = [min(n, n * k / pool_avg) for k in ks]

    ax.plot(ks, actual,  color='#16A34A', lw=2.5, marker='o', ms=4, label=f'{score_col}')
    ax.plot(ks, random_, color='#9CA3AF', lw=1.5, ls='--', label=f'Random baseline (1/{pool_avg:.0f})')
    ax.fill_between(ks, random_, actual, alpha=0.15, color='#16A34A')
    ax.set_xlabel('Rank cutoff k', fontsize=11)
    ax.set_ylabel('D1 positives recovered', fontsize=11)
    ax.set_title('Cumulative recovery vs random', fontsize=11)
    ax.set_xlim(1, max_k)
    ax.set_xticks([1, 2, 5, 10, 15, 20])
    ax.legend()
    ax.grid(True, alpha=0.3)

    for k in [1, 2, 5]:
        ef = actual[k-1] / max(random_[k-1], 0.001)
        ax.annotate(f'EF={ef:.1f}×', (k, actual[k-1]),
                    fontsize=8, ha='left', va='bottom',
                    xytext=(3, 4), textcoords='offset points', color='#16A34A')

    # ── rank distribution bar chart ──────────────────────────────────────────
    ax2 = axes[1]
    labels = ['1', '2', '3–5', '6–10', '11–20', f'>20']
    counts = [
        sum(1 for r in records if r['Rank_in_Combined'] == 1),
        sum(1 for r in records if r['Rank_in_Combined'] == 2),
        sum(1 for r in records if 3 <= r['Rank_in_Combined'] <= 5),
        sum(1 for r in records if 6 <= r['Rank_in_Combined'] <= 10),
        sum(1 for r in records if 11 <= r['Rank_in_Combined'] <= 20),
        sum(1 for r in records if r['Rank_in_Combined'] > 20),
    ]
    colors = ['#16A34A', '#65A30D', '#EAB308', '#F97316', '#DC2626', '#6B7280']
    bars = ax2.bar(labels, counts, color=colors, edgecolor='white', lw=0.5)
    for bar, cnt in zip(bars, counts):
        if cnt:
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     str(cnt), ha='center', va='bottom', fontsize=9)
    ax2.set_xlabel('Rank of D1 experimental pair\n(in pooled D1+D2+D3+D4 list per TF)', fontsize=10)
    ax2.set_ylabel('Number of D1 pairs', fontsize=10)
    ax2.set_title('Rank distribution of D1 positives', fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    p = os.path.join(out_dir, 'recovery_summary.png')
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {p}")

    # ── Figure 2: per-TF best rank ────────────────────────────────────────────
    tfs = sorted({r['TF'] for r in records})
    tf_best = {tf: min(r['Rank_in_Combined'] for r in records if r['TF'] == tf) for tf in tfs}
    tfs_sorted = sorted(tfs, key=lambda t: tf_best[t])

    fig2, ax3 = plt.subplots(figsize=(max(10, len(tfs_sorted) * 0.35), 5))
    xpos = np.arange(len(tfs_sorted))
    bar_colors = [
        '#16A34A' if tf_best[t] == 1 else
        '#65A30D' if tf_best[t] == 2 else
        '#EAB308' if tf_best[t] <= 5 else
        '#F97316' if tf_best[t] <= 10 else
        '#DC2626'
        for t in tfs_sorted
    ]
    ax3.bar(xpos, [tf_best[t] for t in tfs_sorted],
            color=bar_colors, edgecolor='white', lw=0.3)
    ax3.axhline(y=1, color='#16A34A', ls='--', lw=1, label='Rank 1')
    ax3.axhline(y=2, color='#65A30D', ls='--', lw=1, label='Rank 2')
    ax3.set_xticks(xpos)
    ax3.set_xticklabels(tfs_sorted, rotation=75, ha='right', fontsize=7)
    ax3.set_ylabel('Best rank of D1 pair in combined pool\n(lower = better)', fontsize=10)
    ax3.set_title(
        'Per-TF recovery rank (sorted best→worst)\n'
        'D1 pair ranked among all D2+D3+D4 candidates for same TF',
        fontsize=10,
    )
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    p2 = os.path.join(out_dir, 'recovery_by_tf.png')
    fig2.savefig(p2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved {p2}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--out-dir',   default=OUT_DIR)
    ap.add_argument('--score-col', default='AF3_Score',
                    choices=['AF3_Score', 'Consensus_Score'],
                    help='Score column to rank by (default: AF3_Score)')
    args = ap.parse_args()
    run_analysis(args.out_dir, score_col=args.score_col)


if __name__ == '__main__':
    main()
