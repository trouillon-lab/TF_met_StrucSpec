#!/usr/bin/env python3
"""
Decision logic for the GNINA auto-retry watchdog.

Given a pool of pairs that must be GNINA-scored and a set of directories
already holding scored chunk CSVs, computes which pairs are still missing,
escalates each missing pair to the next walltime tier (2h, then 4h — the
gpu.4h partition's ceiling), and caps retries there: a pair still missing
after the 4h attempt is logged as a chronic failure and never retried again,
so the watchdog can't loop forever on a pathological pair.

Writes a retry state TSV (pair -> highest tier already attempted) so tier
escalation is remembered across repeated invocations, and a JSON plan for the
calling shell script to act on (submit retry array(s), or finalize).
"""
import argparse
import csv
import glob
import json
import os

MAX_TIER_HOURS = 4
TIER_SEQUENCE = {0: 2, 2: 4}  # last-attempted-tier -> next-tier


def scored_pairs(score_globs):
    scored = set()
    for pattern in score_globs:
        for d in glob.glob(pattern):
            if not os.path.isdir(d):
                continue
            for fn in os.listdir(d):
                if fn.startswith("gnina_chunk_") and fn.endswith(".csv"):
                    with open(os.path.join(d, fn), newline="", encoding="utf-8") as f:
                        for row in csv.DictReader(f):
                            if row.get("TF_Ligand"):
                                scored.add(row["TF_Ligand"])
    return scored


def load_state(path):
    state = {}
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                pair, tier = line.split("\t")
                state[pair] = int(tier)
    return state


def write_state(path, state):
    with open(path, "w", encoding="utf-8") as f:
        for pair in sorted(state):
            f.write(f"{pair}\t{state[pair]}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--pool-dir", required=True)
    ap.add_argument("--retry-state", required=True)
    ap.add_argument("--chronic-file", required=True)
    ap.add_argument("--score-glob", nargs="+", required=True)
    ap.add_argument("--plan-out", required=True)
    args = ap.parse_args()

    full_pool = sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(args.pool_dir)
        if f.endswith(".zip")
    )
    scored = scored_pairs(args.score_glob)
    missing = [p for p in full_pool if p not in scored]

    state = load_state(args.retry_state)
    existing_chronic = set()
    if os.path.exists(args.chronic_file):
        with open(args.chronic_file, encoding="utf-8") as f:
            existing_chronic = set(l.strip() for l in f if l.strip())

    tier_groups = {}
    new_chronic = []
    for p in missing:
        if p in existing_chronic:
            continue  # already given up on this one; don't resurrect it
        last_tier = state.get(p, 0)
        next_tier = TIER_SEQUENCE.get(last_tier)
        if next_tier is None or next_tier > MAX_TIER_HOURS:
            new_chronic.append(p)
            continue
        tier_groups.setdefault(next_tier, []).append(p)
        state[p] = next_tier

    write_state(args.retry_state, state)
    if new_chronic:
        with open(args.chronic_file, "a", encoding="utf-8") as f:
            for p in new_chronic:
                f.write(p + "\n")

    plan = {
        "label": args.label,
        "pool_size": len(full_pool),
        "scored_count": len(scored & set(full_pool)),
        "missing_count": len(missing),
        "new_chronic_count": len(new_chronic),
        "total_chronic_count": len(existing_chronic) + len(new_chronic),
        "tier_groups": {str(t): pairs for t, pairs in tier_groups.items()},
        "status": "retry" if tier_groups else "done",
    }
    with open(args.plan_out, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)

    print(
        f"[{args.label}] pool={plan['pool_size']} scored={plan['scored_count']} "
        f"missing={plan['missing_count']} new_chronic={plan['new_chronic_count']} "
        f"status={plan['status']}"
    )
    for t, pairs in tier_groups.items():
        print(f"[{args.label}]   tier {t}h: {len(pairs)} pairs to (re)submit")
    if new_chronic:
        print(f"[{args.label}]   {len(new_chronic)} pairs gave up at {MAX_TIER_HOURS}h ceiling: {new_chronic[:10]}{' ...' if len(new_chronic) > 10 else ''}")


if __name__ == "__main__":
    main()
