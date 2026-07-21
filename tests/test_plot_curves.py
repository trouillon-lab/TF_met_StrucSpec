import os
import csv
import pytest
import numpy as np
from scripts.plot_curves import load_data, compute_roc_pr, generate_plots

def test_compute_roc_pr():
    # Ground truth: 2 positives, 2 negatives
    y_true = np.array([1, 1, 0, 0])
    y_scores = np.array([0.9, 0.8, 0.4, 0.1])
    
    fpr, tpr, roc_auc, recall, precision, pr_auc = compute_roc_pr(y_true, y_scores)
    
    assert roc_auc == 1.0, f"Expected perfect ROC AUC 1.0, got {roc_auc}"
    assert pr_auc == 1.0, f"Expected perfect PR AUC 1.0, got {pr_auc}"
    assert len(fpr) > 0
    assert len(tpr) > 0
    assert len(precision) > 0
    assert len(recall) > 0

def test_generate_plots(tmp_path):
    report_csv = tmp_path / "ranked_pairings_report.csv"
    svg_out = tmp_path / "roc_pr_curves.svg"
    png_out = tmp_path / "roc_pr_curves.png"
    
    fieldnames = [
        "TF_Name", "Ligand_Name", "TF_Ligand", "AF3_ipTM", "AF3_PAE_min", 
        "AF3_Has_Clash", "AF3_Score", "AF3_Rank", "Gnina_CNNscore", 
        "Gnina_CNNaffinity", "Gnina_CNN_VS", "Consensus_Score", "Consensus_Rank", "Is_True_Positive"
    ]
    
    rows = [
        {"TF_Name": "AraC", "Ligand_Name": "arabinose", "TF_Ligand": "AraC_arabinose", "AF3_ipTM": "0.8", "AF3_PAE_min": "2.0", "AF3_Has_Clash": "False", "AF3_Score": "0.4", "AF3_Rank": "1", "Gnina_CNNscore": "0.9", "Gnina_CNNaffinity": "7.0", "Gnina_CNN_VS": "6.3", "Consensus_Score": "2.52", "Consensus_Rank": "1", "Is_True_Positive": "True"},
        {"TF_Name": "AraC", "Ligand_Name": "ethidium", "TF_Ligand": "AraC_ethidium", "AF3_ipTM": "0.3", "AF3_PAE_min": "10.0", "AF3_Has_Clash": "False", "AF3_Score": "0.03", "AF3_Rank": "2", "Gnina_CNNscore": "0.2", "Gnina_CNNaffinity": "3.0", "Gnina_CNN_VS": "0.6", "Consensus_Score": "0.018", "Consensus_Rank": "2", "Is_True_Positive": "False"}
    ]
    
    with open(report_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        
    y_true, metrics_data = load_data(str(report_csv))
    assert len(y_true) == 2
    assert "Consensus (AF3 + GNINA)" in metrics_data
    
    generate_plots(y_true, metrics_data, svg_path=str(svg_out), png_path=str(png_out))
    assert svg_out.exists()
    assert png_out.exists()
