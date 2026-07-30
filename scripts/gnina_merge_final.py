#!/usr/bin/env python3
"""
Final merge for one GNINA scoring pool: unions an optional pre-existing base
CSV (pairs already scored before this pool's gap/retry work started) with
every gnina_chunk_*.csv matched by the given glob patterns, deduplicating by
TF_Ligand (first occurrence wins), and writes the combined CSV.
"""
import argparse
import csv
import glob
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-csv", default=None, help="Pre-existing scores CSV to include, if any")
    ap.add_argument("--score-glob", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    fieldnames = ["TF_Ligand", "CNNscore", "CNNaffinity", "Gnina_Mode"]
    rows = {}

    if args.base_csv and os.path.exists(args.base_csv):
        with open(args.base_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("TF_Ligand"):
                    rows.setdefault(row["TF_Ligand"], row)

    chunk_files = []
    for pattern in args.score_glob:
        for d in glob.glob(pattern):
            if os.path.isdir(d):
                chunk_files.extend(
                    os.path.join(d, fn)
                    for fn in os.listdir(d)
                    if fn.startswith("gnina_chunk_") and fn.endswith(".csv")
                )

    for cf in sorted(chunk_files):
        with open(cf, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("TF_Ligand"):
                    rows.setdefault(row["TF_Ligand"], row)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for pair in sorted(rows):
            writer.writerow({k: rows[pair].get(k, "") for k in fieldnames})

    print(f"Merged {len(chunk_files)} chunk files (+base {'yes' if args.base_csv else 'no'}) "
          f"into '{args.output}': {len(rows)} total pairs.")


if __name__ == "__main__":
    main()
