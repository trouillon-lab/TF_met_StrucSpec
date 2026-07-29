#!/usr/bin/env python3
"""
Regenerate AF3 input JSONs for multi-subunit TFs excluded from Dataset 3.

The 263 pairs moved to alphafold3_jsons_excluded/ had placeholder sequence "M"
because their TF names (e.g., FlhDC, HipAB) represent protein complexes whose
combined names did not resolve to a single UniProt entry.

This script:
  1. Looks up the UniProt accessions for each subunit from a curated table.
  2. Fetches protein sequences via the UniProt REST API (cached locally).
  3. Generates AF3 JSON files with one protein chain per subunit plus the ligand
     as the final chain.  The ligand-last convention is required by pipeline_utils
     parse_af3_summary, which treats the last chain as the ligand for scoring.
  4. Writes corrected JSONs to alphafold3_jsons/ (= alphafold3_jsons_dataset3/)
     and removes the placeholder from alphafold3_jsons_excluded/.

After running this script, the new JSONs will appear as "missing ZIPs" for the
next batch-infer run.  The af3io data-fill step requires MSA data for each
protein sequence; run:

    batch-infer start alphafold3_datafill_msas

BEFORE submitting predictions, since the subunit sequences are not yet in the
MSA data index.

Usage:
    python scripts/prepare_multichain_jsons.py [--dry-run] [--json-dir DIR]
"""

import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.request
import urllib.parse

# ── curated TF → subunit UniProt accession mapping ───────────────────────────
#
# Each entry is a list of (display_name, uniprot_accession) tuples.
# Single-element lists = single-chain TFs whose names failed automatic lookup.
# Multi-element lists = heterodimers / composite TFs.
#
# NfeR and PtrR are absent — no UniProt reviewed entry exists for E. coli K-12.
#
MULTICHAIN_COMPOSITION = {
    # True heterodimers / composite TFs
    'FlhDC':     [('FlhD', 'P0A8S9'), ('FlhC', 'P0ABY7')],
    'HipAB':     [('HipA', 'P23874'), ('HipB', 'P23873')],
    'MazEF':     [('MazF', 'P0AE70'), ('MazE', 'P0AE72')],
    'HU':        [('HupA', 'P0ACF0'), ('HupB', 'P0ACF4')],
    'RcsB-BglJ': [('RcsB', 'P0DMC7'), ('BglJ', 'P39404')],
    # GadRcs = GadE (acid-resistance activator) + RcsB (Rcs phosphorelay)
    'GadRcs':    [('GadE', 'P63204'), ('RcsB', 'P0DMC7')],
    # Single-chain TFs whose compound-style names foiled name-based lookup
    'Dan':  [('Dan',  'P76034')],   # yciT gene product
    'FhlA': [('FhlA', 'P19323')],   # fhlA gene product
    'HyfR': [('HyfR', 'P71229')],   # hyfR gene product
    'HypT': [('HypT', 'P28911')],   # yhhH gene product
    # NfeR, PtrR: not present — skip silently
}

# Unresolvable TFs (no UniProt entry in E. coli K-12)
UNRESOLVABLE_TFS = {'NfeR', 'PtrR'}

CACHE_FILE  = 'data/processed/cache_sequences_smiles.json'
ECOLI_TAXON = 83333


# ── utilities ─────────────────────────────────────────────────────────────────

def clean_filename(s):
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', str(s))


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


def fetch_uniprot_sequence(accession, cache, retries=3):
    """Fetch canonical protein sequence from UniProt by accession. Uses cache."""
    acc_cache = cache.setdefault('uniprot', {})
    if accession in acc_cache:
        return acc_cache[accession]

    url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"
    seq = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=15) as r:
                lines = r.read().decode('utf-8').splitlines()
            seq = ''.join(l.strip() for l in lines if not l.startswith('>'))
            break
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  Warning: UniProt fetch failed for {accession}: {e}", file=sys.stderr)

    acc_cache[accession] = seq
    return seq


def get_subunit_sequences(tf_name, cache):
    """
    Return list of (name, sequence) for all subunits of tf_name.
    Returns None if the TF is unresolvable or if any sequence fetch fails.
    """
    if tf_name in UNRESOLVABLE_TFS:
        return None
    if tf_name not in MULTICHAIN_COMPOSITION:
        return None

    subunits = []
    for display_name, acc in MULTICHAIN_COMPOSITION[tf_name]:
        seq = fetch_uniprot_sequence(acc, cache)
        if not seq:
            print(f"  Warning: Could not fetch sequence for {display_name} ({acc})", file=sys.stderr)
            return None
        subunits.append((display_name, seq))
    return subunits


def write_af3_json(pair_name, subunit_seqs, smiles, out_dir, dry_run=False):
    """
    Write an AF3 input JSON.

    subunit_seqs : list of (name, sequence) — one entry per protein chain.
    Protein chains are assigned IDs A, B, C, …; the ligand is assigned the
    next letter (last chain), matching the convention assumed by parse_af3_summary.
    """
    n_prot = len(subunit_seqs)
    ligand_chain_id = chr(ord('A') + n_prot)  # A→B for 1 prot, B→C for 2, etc.

    sequences = []
    for i, (_, seq) in enumerate(subunit_seqs):
        sequences.append({"protein": {"id": chr(ord('A') + i), "sequence": seq}})
    sequences.append({"ligand": {"id": ligand_chain_id, "smiles": smiles}})

    data = {
        "dialect":    "alphafold3",
        "version":    2,
        "name":       pair_name,
        "sequences":  sequences,
        "modelSeeds": [1],
    }

    path = os.path.join(out_dir, f"{pair_name}.json")
    if not dry_run:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    return path


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dry-run', action='store_true',
                    help='Print what would happen without writing any files.')
    ap.add_argument('--json-dir', default='alphafold3_jsons',
                    help='Destination for regenerated JSONs (default: alphafold3_jsons)')
    ap.add_argument('--excluded-dir', default='alphafold3_jsons_excluded',
                    help='Source dir for the malformed placeholder JSONs')
    ap.add_argument('--pairings-csv', default='data/processed/pairings_dataset3_weekend.csv',
                    help='D3 pairings CSV (for SMILES lookup)')
    args = ap.parse_args()

    # ── collect SMILES from D3 pairings ─────────────────────────────────────
    bigg_to_smiles = {}
    if os.path.exists(args.pairings_csv):
        with open(args.pairings_csv, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                bigg = row['KEGG_ID'].strip()
                s    = row['Ligand_SMILES'].strip()
                if bigg and s:
                    bigg_to_smiles[bigg] = s
    print(f"Loaded {len(bigg_to_smiles)} SMILES from {args.pairings_csv}")

    # ── find placeholder JSONs ───────────────────────────────────────────────
    if not os.path.isdir(args.excluded_dir):
        print(f"Excluded dir not found: {args.excluded_dir}", file=sys.stderr)
        sys.exit(1)

    excluded_jsons = sorted(f for f in os.listdir(args.excluded_dir) if f.endswith('.json'))
    print(f"Found {len(excluded_jsons)} placeholder JSONs in {args.excluded_dir}/")

    # ── load / populate cache ────────────────────────────────────────────────
    cache = load_cache()

    # ── process each excluded pair ───────────────────────────────────────────
    os.makedirs(args.json_dir, exist_ok=True)

    stats = {'ok': 0, 'no_smiles': 0, 'unresolvable': 0, 'seq_fail': 0, 'already_exists': 0}

    for fname in excluded_jsons:
        # pair name = filename without .json
        pair_name = fname[:-5]  # strip .json
        src_path  = os.path.join(args.excluded_dir, fname)
        dst_path  = os.path.join(args.json_dir, fname)

        # Parse TF name and BiGG ID from pair name (convention: TF_bigg_id)
        # Split only on the first underscore to get TF name
        # e.g. "FlhDC_acald_c" → tf="FlhDC", bigg="acald_c"
        underscore_idx = pair_name.index('_')
        tf_name  = pair_name[:underscore_idx]
        bigg_id  = pair_name[underscore_idx + 1:]

        # Skip if target already exists
        if os.path.exists(dst_path):
            stats['already_exists'] += 1
            continue

        # SMILES lookup
        smiles = bigg_to_smiles.get(bigg_id)
        if not smiles:
            print(f"  Skip {pair_name}: no SMILES for '{bigg_id}'")
            stats['no_smiles'] += 1
            continue

        # Sequence lookup
        if tf_name in UNRESOLVABLE_TFS:
            stats['unresolvable'] += 1
            continue

        subunit_seqs = get_subunit_sequences(tf_name, cache)
        if not subunit_seqs:
            print(f"  Skip {pair_name}: could not resolve sequences for '{tf_name}'")
            stats['seq_fail'] += 1
            continue

        # Write JSON
        n_chains_str = f"{len(subunit_seqs)}+1"
        if dry_run := args.dry_run:
            print(f"  [dry-run] Would write {pair_name}.json "
                  f"({n_chains_str} chains: {[s[0] for s in subunit_seqs]} + ligand)")
        else:
            write_af3_json(pair_name, subunit_seqs, smiles, args.json_dir)
        stats['ok'] += 1

    # save updated cache
    if not args.dry_run:
        save_cache(cache)

    # ── summary ──────────────────────────────────────────────────────────────
    print(f"\nSummary:")
    print(f"  Generated:      {stats['ok']}")
    print(f"  Already exists: {stats['already_exists']}")
    print(f"  No SMILES:      {stats['no_smiles']}")
    print(f"  Unresolvable TF (NfeR/PtrR): {stats['unresolvable']}")
    print(f"  Seq fetch fail: {stats['seq_fail']}")
    total_skipped = stats['no_smiles'] + stats['unresolvable'] + stats['seq_fail']
    print(f"\n  → {stats['ok']} new JSONs ready in {args.json_dir}/")
    print(f"  → {total_skipped} pairs permanently excluded (NfeR/PtrR or missing SMILES)")
    if not args.dry_run and stats['ok'] > 0:
        print(
            "\nNext steps on Euler:\n"
            "  1. git pull\n"
            "  2. python scripts/prepare_multichain_jsons.py  # (already done if running remotely)\n"
            "  3. source software/batch-infer/.venv/bin/activate\n"
            "  4. batch-infer start alphafold3_datafill_msas   # compute MSAs for new subunit seqs\n"
            "  5. batch-infer start alphafold3_datafill_predictions  # run AF3 on new JSONs"
        )


if __name__ == '__main__':
    main()
