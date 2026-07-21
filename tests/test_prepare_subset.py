import os
import csv
import json
import pytest
from scripts.prepare_test_subset import build_subset_pairs

def test_build_subset_pairs(tmp_path):
    csv_out = tmp_path / "pairings_subset_20.csv"
    json_dir = tmp_path / "alphafold3_jsons"
    
    pairs = build_subset_pairs(output_csv=str(csv_out), output_json_dir=str(json_dir))
    
    assert len(pairs) == 20, "Should generate exactly 20 pairs (10 pos + 10 neg)"
    pos_count = sum(1 for p in pairs if p['Label'] == 'positive')
    neg_count = sum(1 for p in pairs if p['Label'] == 'negative')
    assert pos_count == 10, "Should have 10 positive controls"
    assert neg_count == 10, "Should have 10 negative controls"
    
    # Check MSA re-use property: unique TFs should be small (<= 5)
    unique_tfs = set(p['TF_Name'] for p in pairs)
    assert len(unique_tfs) <= 5, f"Should re-use TFs (found {len(unique_tfs)} unique TFs)"
    
    # Verify CSV creation
    assert csv_out.exists()
    with open(csv_out, 'r') as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 20
        
    # Verify JSON files
    json_files = list(json_dir.glob("*.json"))
    assert len(json_files) == 20
    for jf in json_files:
        with open(jf, 'r') as f:
            data = json.load(f)
            assert data['dialect'] == 'alphafold3'
            assert len(data['sequences']) == 2
