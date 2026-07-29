#!/usr/bin/env python3
"""
Dataset 4 (Stratified CR-Range) Generator & AF3 Input JSON Creator

Samples TF–metabolite pairs uniformly across all CR-score quintiles from
consensus_rank_interval.csv, excluding pairs already tested in Datasets 1–3.
The resulting dataset spans the full CR-score distribution, eliminating the
range restriction that biases correlation estimates in Dataset 3.

Design rationale
----------------
Dataset 3 positives cluster at CR score > 1.5 (top 20 %), which artificially
attenuates Pearson/Spearman r between structural and CR scores.  A stratified
sample of ~PAIRS_PER_QUINTILE * 5 pairs spanning all five quintiles gives
adequate power to estimate the true incremental predictive value of structural
scores beyond the CR prior.

Outputs
-------
  data/processed/pairings_dataset4_stratified.csv  — pairings table
  alphafold3_jsons_dataset4/                        — AF3 input JSON files

Usage
-----
  python scripts/prepare_dataset4_stratified.py [--pairs-per-quintile 150] \
      [--seed 42] [--json-dir alphafold3_jsons_dataset4] \
      [--out-csv data/processed/pairings_dataset4_stratified.csv]
"""

import argparse
import ast
import csv
import json
import os
import re
import sys
import urllib.parse
import urllib.request

import numpy as np

CACHE_FILE = 'data/processed/cache_sequences_smiles.json'

# TFs whose sequences could not be resolved in Dataset 3 (multi-subunit complexes
# whose combined name does not map to a single UniProt entry).
EXCLUDED_TFS = {
    'Dan', 'FhlA', 'FlhDC', 'GadRcs', 'HU', 'HipAB',
    'HyfR', 'HypT', 'MazEF', 'NfeR', 'PtrR', 'RcsB-BglJ',
}

# Source files
CR_FILE  = 'data/raw/consensus_rank_interval.csv'
D3_CSV   = 'data/processed/pairings_dataset3_weekend.csv'
D12_CSVS = [
    'data/processed/pairings_subset_20.csv',
    'data/processed/pairings_remaining_248.csv',
    'data/processed/pairings_score2_benchmark.csv',
]


# ── utilities ────────────────────────────────────────────────────────────────

def clean_filename(name):
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', str(name))


def fetch_url(url, timeout=15):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode('utf-8'))


def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {'uniprot': {}, 'kegg_smiles': {}, 'gene_uniprot': {}, 'bigg_smiles': {}}


def save_cache(cache):
    os.makedirs(os.path.dirname(os.path.abspath(CACHE_FILE)), exist_ok=True)
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2)


def resolve_smiles_by_bigg(bigg_id, cache):
    bigg_cache = cache.setdefault('bigg_smiles', {})
    if bigg_id in bigg_cache:
        return bigg_cache[bigg_id]

    clean_id = bigg_id.rsplit('_', 1)[0] if bigg_id.endswith(('_c', '_e', '_p')) else bigg_id
    url = f"http://bigg.ucsd.edu/api/v2/universal/metabolites/{clean_id}"

    name, smiles = clean_id, None
    try:
        data = fetch_url(url)
        name = data.get('name', clean_id) or clean_id
        db_links = data.get('database_links', {})

        # Try PubChem CID first
        if 'PubChem' in db_links:
            for link in db_links['PubChem']:
                cid = link.get('id', '')
                if cid:
                    try:
                        pc = fetch_url(
                            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
                            f"{cid}/property/CanonicalSMILES,SMILES/JSON"
                        )
                        props = pc.get('PropertyTable', {}).get('Properties', [{}])[0]
                        smiles = props.get('CanonicalSMILES') or props.get('SMILES')
                        if smiles:
                            break
                    except Exception:
                        pass

        # Fallback: InChI Key
        if not smiles and 'InChI Key' in db_links:
            for link in db_links['InChI Key']:
                inchikey = link.get('id', '')
                if inchikey:
                    try:
                        pc = fetch_url(
                            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/"
                            f"{inchikey}/property/CanonicalSMILES,SMILES/JSON"
                        )
                        props = pc.get('PropertyTable', {}).get('Properties', [{}])[0]
                        smiles = props.get('CanonicalSMILES') or props.get('SMILES')
                        if smiles:
                            break
                    except Exception:
                        pass

        # Fallback: name lookup
        if not smiles and name:
            try:
                pc = fetch_url(
                    f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
                    f"{urllib.parse.quote(name)}/property/CanonicalSMILES,SMILES/JSON"
                )
                props = pc.get('PropertyTable', {}).get('Properties', [{}])[0]
                smiles = props.get('CanonicalSMILES') or props.get('SMILES')
            except Exception:
                pass

    except Exception as e:
        print(f"  Warning: BiGG lookup failed for '{bigg_id}': {e}", file=sys.stderr)

    result = (name, smiles)
    bigg_cache[bigg_id] = result
    return result


# ── data loading ─────────────────────────────────────────────────────────────

def load_excluded_pairs():
    """Return set of (TF, bigg_id) already tested in D1–D3."""
    excluded = set()
    for path in D12_CSVS + [D3_CSV]:
        if not os.path.exists(path):
            continue
        with open(path, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                tf   = row.get('TF_Name', '').strip()
                bigg = row.get('KEGG_ID', row.get('kegg_id', '')).strip()
                if tf and bigg:
                    excluded.add((tf, bigg))
    return excluded


def load_tf_sequences():
    """Return dict tf_name → protein_sequence from D3 pairings (valid sequences only)."""
    seqs = {}
    if not os.path.exists(D3_CSV):
        return seqs
    with open(D3_CSV, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            tf  = row['TF_Name'].strip()
            seq = row['TF_Sequence'].strip()
            if tf not in seqs and len(seq) > 5:
                seqs[tf] = seq
    return seqs


def load_d3_smiles():
    """Return dict bigg_id → smiles from D3 pairings."""
    smiles = {}
    if not os.path.exists(D3_CSV):
        return smiles
    with open(D3_CSV, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            bigg = row['KEGG_ID'].strip()
            s    = row['Ligand_SMILES'].strip()
            if bigg and s:
                smiles[bigg] = s
    return smiles


def load_cr_pairs():
    """Return list of (tf, bigg_id, cr_score) from CR file."""
    pairs = []
    with open(CR_FILE, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            tf, met = ast.literal_eval(row['tf_met_pair'])
            pairs.append((tf.strip(), met.strip(), float(row['score'])))
    return pairs


# ── stratified sampling ──────────────────────────────────────────────────────

def stratified_sample(pairs, n_per_quintile, seed):
    """
    Assign quintile labels by CR score and sample n_per_quintile from each.
    Returns sampled list and per-quintile summary dict.
    """
    scores = np.array([s for _, _, s in pairs])
    boundaries = np.percentile(scores, [20, 40, 60, 80])

    def quintile(s):
        if s <= boundaries[0]: return 1
        if s <= boundaries[1]: return 2
        if s <= boundaries[2]: return 3
        if s <= boundaries[3]: return 4
        return 5

    from collections import defaultdict
    buckets = defaultdict(list)
    for tf, met, s in pairs:
        buckets[quintile(s)].append((tf, met, s))

    rng = np.random.default_rng(seed)
    sampled = []
    summary = {}
    for q in range(1, 6):
        pool = buckets[q]
        k    = min(n_per_quintile, len(pool))
        idx  = rng.choice(len(pool), size=k, replace=False)
        chosen = [pool[i] for i in sorted(idx)]
        sampled.extend(chosen)
        summary[q] = {
            'pool_size': len(pool),
            'sampled':   k,
            'score_min': min(s for _, _, s in chosen),
            'score_max': max(s for _, _, s in chosen),
        }
    return sampled, summary


# ── JSON generation ──────────────────────────────────────────────────────────

def write_af3_json(pair_name, tf_seq, smiles, out_dir):
    data = {
        "dialect": "alphafold3",
        "version": 2,
        "name":    pair_name,
        "sequences": [
            {"protein": {"id": "A", "sequence": tf_seq}},
            {"ligand":  {"id": "B", "smiles":   smiles}},
        ],
        "modelSeeds": [1],
    }
    path = os.path.join(out_dir, f"{pair_name}.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    return path


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pairs-per-quintile', type=int, default=150,
                    help='Pairs to sample from each CR-score quintile (default: 150)')
    ap.add_argument('--seed', type=int, default=42,
                    help='Random seed for reproducibility (default: 42)')
    ap.add_argument('--json-dir', default='alphafold3_jsons_dataset4',
                    help='Output directory for AF3 JSON files')
    ap.add_argument('--out-csv', default='data/processed/pairings_dataset4_stratified.csv',
                    help='Output pairings CSV')
    args = ap.parse_args()

    os.makedirs(args.json_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)

    # ── load reference data ──────────────────────────────────────────────────
    print("Loading reference data...")
    excluded   = load_excluded_pairs()
    tf_seqs    = load_tf_sequences()
    known_smiles = load_d3_smiles()
    cache      = load_cache()

    print(f"  Excluded pairs (D1–D3): {len(excluded)}")
    print(f"  TFs with known sequences: {len(tf_seqs)}")
    print(f"  Metabolites with known SMILES: {len(known_smiles)}")

    # ── filter CR pairs ──────────────────────────────────────────────────────
    print("\nFiltering CR pairs...")
    cr_pairs = load_cr_pairs()
    print(f"  Total CR pairs: {len(cr_pairs)}")

    valid = []
    for tf, met, score in cr_pairs:
        if (tf, met) in excluded:
            continue
        if tf in EXCLUDED_TFS:
            continue
        if tf not in tf_seqs:
            continue
        valid.append((tf, met, score))

    print(f"  Valid (not in D1–D3, known TF seq): {len(valid)}")

    # ── stratified sampling ──────────────────────────────────────────────────
    print(f"\nStratified sampling ({args.pairs_per_quintile}/quintile, seed={args.seed})...")
    sampled, summary = stratified_sample(valid, args.pairs_per_quintile, args.seed)
    print(f"  Total sampled: {len(sampled)}")
    for q, s in summary.items():
        print(f"  Q{q}: {s['sampled']}/{s['pool_size']} pairs, "
              f"CR score {s['score_min']:.3f}–{s['score_max']:.3f}")

    # ── resolve SMILES ───────────────────────────────────────────────────────
    unique_mets = {met for _, met, _ in sampled if met not in known_smiles}
    if unique_mets:
        print(f"\nFetching SMILES for {len(unique_mets)} metabolites not in D3 cache...")
    resolved = 0
    failed   = []
    for met in sorted(unique_mets):
        name, smiles = resolve_smiles_by_bigg(met, cache)
        if smiles:
            known_smiles[met] = smiles
            resolved += 1
        else:
            failed.append(met)
        if (resolved + len(failed)) % 20 == 0:
            save_cache(cache)
            print(f"  Fetched {resolved + len(failed)}/{len(unique_mets)} "
                  f"({resolved} OK, {len(failed)} failed)")
    if unique_mets:
        save_cache(cache)
        print(f"  SMILES resolved: {resolved}, failed: {len(failed)}")
        if failed:
            print(f"  Failed metabolites: {failed[:10]}" +
                  (f" ... and {len(failed)-10} more" if len(failed) > 10 else ""))

    # ── generate outputs ─────────────────────────────────────────────────────
    print(f"\nGenerating outputs...")
    rows  = []
    skipped_smiles = 0
    json_written   = 0

    for tf, met, score in sampled:
        smiles = known_smiles.get(met)
        if not smiles:
            skipped_smiles += 1
            continue
        seq  = tf_seqs[tf]
        # name from BiGG cache (first element of tuple) or met
        cached_entry = cache.get('bigg_smiles', {}).get(met)
        ligand_name = (cached_entry[0] if cached_entry else met) or met
        pair_name = clean_filename(f"{tf}_{met}")

        write_af3_json(pair_name, seq, smiles, args.json_dir)
        json_written += 1

        rows.append({
            'TF_Name':      tf,
            'TF_Sequence':  seq,
            'Ligand_Name':  ligand_name,
            'KEGG_ID':      met,
            'Ligand_SMILES': smiles,
            'CR_Score':     score,
            'Label':        'unknown',
        })

    # write CSV
    fieldnames = ['TF_Name', 'TF_Sequence', 'Ligand_Name', 'KEGG_ID',
                  'Ligand_SMILES', 'CR_Score', 'Label']
    with open(args.out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"  JSONs written:  {json_written}  → {args.json_dir}/")
    print(f"  Pairs CSV:      {args.out_csv}")
    if skipped_smiles:
        print(f"  Skipped (no SMILES): {skipped_smiles}")
    print("\nDone.")


if __name__ == '__main__':
    main()
