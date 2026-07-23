import pytest
import os
import json
import pandas as pd
from scripts.prepare_score2_dataset import process_consensus_dataset

def test_process_consensus_dataset(tmp_path):
    raw_csv = tmp_path / "test_consensus.csv"
    out_csv = tmp_path / "benchmark.csv"
    out_json_dir = tmp_path / "af3_jsons"
    
    # Create dummy raw CSV matching consensus_rank_interval.csv
    data = [
        [0, "('AraC', 'arab__L_c')", 120.0, 5, 1.0, 1.0, 2.0],
        [1, "('AraC', 'rbl__L_c')", 120.0, 5, 1.0, 1.0, 2.0],
        [2, "('ArcA', 'crnDcoa_c')", 100.0, 4, 0.8, 0.8, 1.5]
    ]
    df = pd.DataFrame(data, columns=['', 'tf_met_pair', 'likelihood_sum', 'consensus_count', 'scaled_likelihood_sum', 'scaled_consensus_count', 'score'])
    df.to_csv(raw_csv, index=False)
    
    csv_res, count_pos, count_neg = process_consensus_dataset(
        raw_csv=str(raw_csv),
        out_csv=str(out_csv),
        out_json_dir=str(out_json_dir),
        random_seed=42
    )
    
    assert os.path.exists(out_csv)
    assert count_pos == 2
    assert count_neg == 2
    
    res_df = pd.read_csv(out_csv)
    assert len(res_df) == 4
    assert set(res_df['Label']) == {'positive', 'negative'}
    
    # Check generated JSONs
    json_files = list(out_json_dir.glob("*.json"))
    assert len(json_files) == 4
    
    with open(json_files[0], 'r') as f:
        af3_json = json.load(f)
        assert af3_json['dialect'] == 'alphafold3'
        assert len(af3_json['sequences']) == 2
