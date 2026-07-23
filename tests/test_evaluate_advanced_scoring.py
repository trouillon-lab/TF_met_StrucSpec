import pytest
import numpy as np
import pandas as pd
from scripts.evaluate_advanced_scoring import (
    engineer_shifted_features,
    compute_metrics_table
)

def test_engineer_shifted_features():
    df = pd.DataFrame({
        'ipTM': [0.8, 0.5],
        'PAE_min': [1.0, 5.0],
        'CNNscore': [0.9, 0.4],
        'CNNaffinity': [5.0, 3.0],
        'AF3_Score': [0.8, 0.1],
        'Gnina_CNN_VS': [4.5, 1.2],
        'Consensus_Score': [3.6, 0.12]
    })
    df_feat = engineer_shifted_features(df, eps_values=[0.1, 0.2, 0.3, 0.5])
    
    # Check Shifted Consensus for eps=0.2: (0.8 / (1.0 + 0.2)) * (0.9 * 5.0) = (0.8 / 1.2) * 4.5 = 3.0
    expected_eps_02 = (0.8 / 1.2) * (0.9 * 5.0)
    assert np.isclose(df_feat['S_Shifted_eps_0.2'].iloc[0], expected_eps_02)

def test_compute_metrics_table():
    y_true = np.array([1, 1, 1, 0, 0, 0])
    scores_dict = {
        "Test Score": np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    }
    table = compute_metrics_table(y_true, scores_dict)
    assert len(table) == 1
    assert table.iloc[0]['ROC AUC'] == 1.0
    assert table.iloc[0]['PR AUC'] == 1.0
    assert table.iloc[0]['Max F1'] == 1.0
