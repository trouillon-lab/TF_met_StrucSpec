import pytest
import numpy as np
import pandas as pd
from scripts.plot_all_distributions import generate_distribution_diagnostics

def test_generate_distribution_diagnostics(tmp_path):
    df = pd.DataFrame({
        'TF_Name': ['AraC', 'AraC', 'AcrR', 'AcrR'],
        'Ligand_Name': ['arabinose', 'fucose', 'ethidium', 'proflavin'],
        'Is_True_Positive': [True, False, True, False],
        'AF3_ipTM': [0.8, 0.4, 0.9, 0.3],
        'AF3_PAE_min': [1.2, 5.0, 0.8, 6.0],
        'AF3_Score': [0.66, 0.08, 1.125, 0.05],
        'Gnina_CNNscore': [0.95, 0.2, 0.88, 0.15],
        'Gnina_CNNaffinity': [5.8, 3.2, 6.1, 2.8],
        'Gnina_CNN_VS': [5.51, 0.64, 5.368, 0.42],
        'Consensus_Score': [3.636, 0.051, 6.039, 0.021]
    })
    
    csv_file = tmp_path / "test_report.csv"
    df.to_csv(csv_file, index=False)
    
    out_hist_svg = tmp_path / "hist.svg"
    out_hist_png = tmp_path / "hist.png"
    out_corr_svg = tmp_path / "corr.svg"
    out_corr_png = tmp_path / "corr.png"
    
    generate_distribution_diagnostics(
        report_csv=str(csv_file),
        out_hist_svg=str(out_hist_svg),
        out_hist_png=str(out_hist_png),
        out_corr_svg=str(out_corr_svg),
        out_corr_png=str(out_corr_png)
    )
    
    assert out_hist_svg.exists()
    assert out_corr_svg.exists()
