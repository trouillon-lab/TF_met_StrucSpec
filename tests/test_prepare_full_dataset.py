import os
import csv
import pytest
from scripts.prepare_full_dataset import build_full_dataset

def test_build_full_dataset(tmp_path):
    raw_csv = tmp_path / "tf_effectors_curated_test.csv"
    sub_csv = tmp_path / "pairings_subset_20.csv"
    out_csv = tmp_path / "pairings_remaining_248.csv"
    out_json_dir = tmp_path / "alphafold3_jsons_remaining"
    qc_svg = tmp_path / "full_dataset_qc.svg"
    qc_png = tmp_path / "full_dataset_qc.png"
    
    # Write mock raw CSV
    raw_fieldnames = ["Transcription factor", "Uniprot ID", "effector_name", "kegg_id", "Effector type"]
    raw_rows = [
        {"Transcription factor": "AraC", "Uniprot ID": "P0A9E0", "effector_name": "arabinose", "kegg_id": "C02604", "Effector type": "small molecule"},
        {"Transcription factor": "AraC", "Uniprot ID": "P0A9E0", "effector_name": "D-fucose", "kegg_id": "C02095", "Effector type": "small molecule"},
        {"Transcription factor": "AcrR", "Uniprot ID": "P0ACS9", "effector_name": "ethidium", "kegg_id": "C11161", "Effector type": "small molecule"},
        {"Transcription factor": "TyrR", "Uniprot ID": "P07604", "effector_name": "L-tryptophan", "kegg_id": "C00078", "Effector type": "small molecule"}
    ]
    with open(raw_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=raw_fieldnames)
        w.writeheader()
        w.writerows(raw_rows)
        
    # Write mock sub CSV (exclude AraC - arabinose)
    sub_fieldnames = ["TF_Name", "Ligand_Name", "Label"]
    sub_rows = [
        {"TF_Name": "AraC", "Ligand_Name": "arabinose", "Label": "positive"}
    ]
    with open(sub_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=sub_fieldnames)
        w.writeheader()
        w.writerows(sub_rows)
        
    # Run build_full_dataset
    pairs = build_full_dataset(
        raw_csv=str(raw_csv),
        subset_csv=str(sub_csv),
        output_csv=str(out_csv),
        output_json_dir=str(out_json_dir),
        qc_svg=str(qc_svg),
        qc_png=str(qc_png),
        seed=42
    )
    
    assert len(pairs) > 0
    assert out_csv.exists()
    assert out_json_dir.exists()
    assert qc_svg.exists()
    assert qc_png.exists()
    
    # Check AraC - arabinose is excluded
    tf_ligs = [(p['TF_Name'], p['Ligand_Name']) for p in pairs]
    assert ('AraC', 'arabinose') not in tf_ligs
