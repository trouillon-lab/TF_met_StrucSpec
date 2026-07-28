#!/usr/bin/env python3
"""
Shared utilities for the AF3 + GNINA virtual screening pipeline.

Centralises parse_af3_summary, load_gnina_scores, generate_svg_slopegraph,
sanitize_key, and the canonical GNINA fallback so that rank_candidates.py
and run_full_evaluation.py stay in sync.
"""

import os
import re
import csv
import json
import zipfile

# Canonical fallback for pairs with no GNINA rescoring result.
# Zero is the deliberate policy: unrescored pairs must not receive a
# synthetic positive signal in the consensus score.
GNINA_FALLBACK = {'CNNscore': 0.0, 'CNNaffinity': 0.0, 'CNN_VS': 0.0}


def sanitize_key(s):
    """Strips all non-alphanumeric characters and lowercases for key matching."""
    return re.sub(r'[^a-zA-Z0-9]', '', str(s)).lower()


def parse_af3_summary(zip_path):
    """Parses chain_pair_iptm, chain_pair_pae_min, and has_clash from an AF3 zip.

    Returns a dict {'iptm', 'pae_min', 'has_clash'} on success, or None on any
    failure. Logs a warning to stdout so failures are visible in SLURM logs.
    Silent value substitution is intentionally avoided: a bad ZIP should be
    excluded from the benchmark rather than entered with synthetic scores.
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            conf_files = [f for f in zip_ref.namelist() if f.endswith('summary_confidences.json')]
            if not conf_files:
                print(f"Warning: No summary_confidences.json in {zip_path} — skipping pair.")
                return None

            conf_text = zip_ref.read(conf_files[0]).decode('utf-8')
            data = json.loads(conf_text)

            iptm = data.get('chain_pair_iptm')
            pae_min = data.get('chain_pair_pae_min')
            has_clash = data.get('has_clash', False)

            # Extract [0][1]: protein chain A × ligand chain B interaction.
            val_iptm = None
            if iptm and isinstance(iptm, list):
                if len(iptm) > 0 and isinstance(iptm[0], list) and len(iptm[0]) > 1:
                    val_iptm = iptm[0][1]
                elif len(iptm) > 1:
                    val_iptm = iptm[0]

            val_pae = None
            if pae_min and isinstance(pae_min, list):
                if len(pae_min) > 0 and isinstance(pae_min[0], list) and len(pae_min[0]) > 1:
                    val_pae = pae_min[0][1]
                elif len(pae_min) > 1:
                    val_pae = pae_min[0]

            if val_iptm is None:
                print(f"Warning: Could not extract ipTM from {zip_path} — skipping pair.")
                return None
            if val_pae is None:
                print(f"Warning: Could not extract PAE_min from {zip_path} — skipping pair.")
                return None

            return {
                "iptm": float(val_iptm),
                "pae_min": float(val_pae),
                "has_clash": bool(has_clash)
            }
    except Exception as e:
        print(f"Warning: Failed to parse AF3 summary from {zip_path}: {e}")
        return None


def load_gnina_scores(gnina_csvs):
    """Loads GNINA scores from a list of CSV file paths.

    Returns a dict keyed by sanitized TF_Ligand name. Callers should use
    GNINA_FALLBACK for pairs not present in this dict.
    """
    gnina_data = {}
    for gcsv in gnina_csvs:
        if not os.path.exists(gcsv):
            continue
        with open(gcsv, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                pair = row['TF_Ligand'].strip()
                s_key = sanitize_key(pair)
                cnn_s = float(row['CNNscore'])
                cnn_a = float(row['CNNaffinity'])
                gnina_data[s_key] = {
                    'CNNscore': cnn_s,
                    'CNNaffinity': cnn_a,
                    'CNN_VS': cnn_s * cnn_a,
                    'TF_Ligand': pair
                }
    return gnina_data


def generate_svg_slopegraph(validation_data, output_path, title_suffix=""):
    """Generates a pure-Python SVG slopegraph comparing AF3 vs Consensus ranks.

    Green lines = consensus improved the rank; red = declined; grey dashed = unchanged.
    """
    width, height = 600, 500
    padding_top, padding_bottom = 80, 50
    padding_left, padding_right = 120, 120

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%">',
        '<rect width="100%" height="100%" fill="#111827"/>',
        '<style>',
        '  .title { font-family: "Inter", sans-serif; font-size: 17px; fill: #F3F4F6; font-weight: bold; text-anchor: middle; }',
        '  .axis-label { font-family: "Inter", sans-serif; font-size: 14px; fill: #9CA3AF; font-weight: 600; text-anchor: middle; }',
        '  .rank-val { font-family: monospace; font-size: 11px; fill: #D1D5DB; font-weight: bold; }',
        '  .line-active { stroke: #10B981; stroke-width: 3; stroke-linecap: round; }',
        '  .line-inactive { stroke: #EF4444; stroke-width: 3; stroke-linecap: round; }',
        '  .line-neutral { stroke: #6B7280; stroke-width: 2; stroke-dasharray: 4; stroke-linecap: round; }',
        '  .node-dot { stroke-width: 2; fill: #1F2937; }',
        '  .label-text { font-family: sans-serif; font-size: 10px; fill: #E5E7EB; }',
        '</style>',
        f'<text x="{width/2}" y="35" class="title">True Positive Rank Validation {title_suffix}</text>',
        f'<text x="{width/2}" y="55" font-family="sans-serif" font-size="12" fill="#9CA3AF" text-anchor="middle">'
        'Lower rank is better (Rank 1 = Top Candidate)</text>',
        f'<text x="{padding_left}" y="{padding_top - 15}" class="axis-label">AF3 Alone</text>',
        f'<line x1="{padding_left}" y1="{padding_top}" x2="{padding_left}" y2="{height - padding_bottom}" stroke="#4B5563" stroke-width="2"/>',
        f'<text x="{width - padding_right}" y="{padding_top - 15}" class="axis-label">Consensus (AF3+Gnina)</text>',
        f'<line x1="{width - padding_right}" y1="{padding_top}" x2="{width - padding_right}" y2="{height - padding_bottom}" stroke="#4B5563" stroke-width="2"/>',
    ]

    if not validation_data:
        svg.append(
            f'<text x="{width/2}" y="{height/2}" font-family="sans-serif" font-size="14" '
            'fill="#9CA3AF" text-anchor="middle">No True Positive data available</text>'
        )
    else:
        max_rank = max(max(item['af3_rank'], item['consensus_rank'], 5) for item in validation_data)
        y_start, y_end = padding_top, height - padding_bottom
        y_span = y_end - y_start

        def get_y(r):
            return y_start + ((r - 1) / max(max_rank - 1, 1)) * y_span

        for item in validation_data:
            af3_r, cons_r = item['af3_rank'], item['consensus_rank']
            label = item['TF_Ligand']
            y_af3, y_cons = get_y(af3_r), get_y(cons_r)

            if cons_r < af3_r:
                line_class, dot_color = "line-active", "#10B981"
            elif cons_r > af3_r:
                line_class, dot_color = "line-inactive", "#EF4444"
            else:
                line_class, dot_color = "line-neutral", "#9CA3AF"

            svg.append(f'  <line x1="{padding_left}" y1="{y_af3}" x2="{width - padding_right}" y2="{y_cons}" class="{line_class}"/>')
            svg.append(f'  <circle cx="{padding_left}" cy="{y_af3}" r="5" class="node-dot" stroke="{dot_color}"/>')
            svg.append(f'  <text x="{padding_left - 12}" y="{y_af3 + 4}" class="rank-val" text-anchor="end">#{af3_r}</text>')
            svg.append(f'  <text x="{padding_left - 35}" y="{y_af3 + 4}" class="label-text" text-anchor="end">{label}</text>')
            svg.append(f'  <circle cx="{width - padding_right}" cy="{y_cons}" r="5" class="node-dot" stroke="{dot_color}"/>')
            svg.append(f'  <text x="{width - padding_right + 12}" y="{y_cons + 4}" class="rank-val" text-anchor="start">#{cons_r}</text>')
            svg.append(f'  <text x="{width - padding_right + 35}" y="{y_cons + 4}" class="label-text" text-anchor="start">{label}</text>')

    svg.append('</svg>')
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(svg))
    print(f"Generated slopegraph SVG at '{output_path}'.")
