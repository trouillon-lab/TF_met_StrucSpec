import pytest
import numpy as np
import pandas as pd
from scripts.evaluate_advanced_scoring import (
    load_and_preprocess_report,
    compute_metrics_table
)

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
