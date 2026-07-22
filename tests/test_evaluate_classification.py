import os
import csv
import pytest
import numpy as np
from scripts.evaluate_classification import (
    load_ground_truth_positives,
    compute_optimal_threshold,
    evaluate_classification_metrics,
    evaluate_multibinder_ranking,
    run_classification_analysis
)

def test_load_ground_truth_positives(tmp_path):
    raw_csv = tmp_path / "raw.csv"
    fieldnames = ["Transcription factor", "effector_name", "Effector type"]
    rows = [
        {"Transcription factor": "AraC", "effector_name": "arabinose", "Effector type": "small molecule"},
        {"Transcription factor": "AraC", "effector_name": "D-fucose", "Effector type": "small molecule"},
        {"Transcription factor": "AcrR", "effector_name": "ethidium", "Effector type": "small molecule"},
        {"Transcription factor": "NhaR", "effector_name": "Na", "Effector type": "ion"}  # should be skipped
    ]
    with open(raw_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
        
    positives = load_ground_truth_positives(str(raw_csv))
    assert ("AraC", "arabinose") in positives
    assert ("AraC", "D-fucose") in positives
    assert ("AcrR", "ethidium") in positives
    assert ("NhaR", "Na") not in positives

def test_compute_optimal_threshold():
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    
    thresh, metrics = compute_optimal_threshold(y_true, y_scores)
    assert 0.3 <= thresh <= 0.7
    assert metrics['sensitivity'] == 1.0
    assert metrics['specificity'] == 1.0
    assert metrics['f1'] == 1.0

def test_evaluate_multibinder_ranking():
    rows = [
        {"TF_Name": "AraC", "TF_Ligand": "AraC_arabinose", "Consensus_Score": "2.8", "is_tp": 1},
        {"TF_Name": "AraC", "TF_Ligand": "AraC_D-fucose", "Consensus_Score": "2.5", "is_tp": 1},
        {"TF_Name": "AraC", "TF_Ligand": "AraC_tryptophan", "Consensus_Score": "0.2", "is_tp": 0},
        {"TF_Name": "TyrR", "TF_Ligand": "TyrR_phenylalanine", "Consensus_Score": "1.8", "is_tp": 1},
        {"TF_Name": "TyrR", "TF_Ligand": "TyrR_serine", "Consensus_Score": "0.1", "is_tp": 0}
    ]
    res = evaluate_multibinder_ranking(rows, score_col='Consensus_Score')
    assert "AraC" in res
    assert res["AraC"]["recall_at_1"] == 0.5  # 1 of 2 top 1
    assert res["AraC"]["recall_at_2"] == 1.0  # 2 of 2 top 2

def test_run_classification_analysis(tmp_path):
    report_csv = tmp_path / "ranked_pairings_report.csv"
    raw_csv = tmp_path / "raw.csv"
    out_svg = tmp_path / "dist.svg"
    out_png = tmp_path / "dist.png"
    
    with open(raw_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=["Transcription factor", "effector_name", "Effector type"])
        w.writeheader()
        w.writerows([{"Transcription factor": "AraC", "effector_name": "arabinose", "Effector type": "small molecule"}])
        
    with open(report_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ["TF_Name", "Ligand_Name", "TF_Ligand", "AF3_ipTM", "AF3_Score", "Gnina_CNNscore", "Gnina_CNN_VS", "Consensus_Score"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows([
            {"TF_Name": "AraC", "Ligand_Name": "arabinose", "TF_Ligand": "AraC_arabinose", "AF3_ipTM": "0.8", "AF3_Score": "0.4", "Gnina_CNNscore": "0.9", "Gnina_CNN_VS": "6.3", "Consensus_Score": "2.52"},
            {"TF_Name": "AraC", "Ligand_Name": "tryptophan", "TF_Ligand": "AraC_tryptophan", "AF3_ipTM": "0.2", "AF3_Score": "0.02", "Gnina_CNNscore": "0.1", "Gnina_CNN_VS": "0.2", "Consensus_Score": "0.004"}
        ])
        
    summary = run_classification_analysis(
        report_csv=str(report_csv),
        raw_csv=str(raw_csv),
        out_dist_svg=str(out_svg),
        out_dist_png=str(out_png)
    )
    assert summary is not None
    assert out_svg.exists()
    assert out_png.exists()
