from __future__ import annotations

import os
from typing import Dict, List

import pandas as pd


def verified_correction_rate(review_results: List[Dict], ground_truth_path: str | None):
    """Compare accepted corrections with manual annotations.

    CSV columns: row_index,column,expected_value. Returns None when no manual
    ground truth exists, preventing mock acceptance from being reported as
    correction accuracy.
    """
    if not ground_truth_path or not os.path.exists(ground_truth_path):
        return None
    truth = pd.read_csv(ground_truth_path, dtype=str)
    required = {"row_index", "column", "expected_value"}
    if not required.issubset(truth.columns):
        raise ValueError(f"Ground-truth CSV must contain {sorted(required)}")
    expected = {(str(r.row_index), r.column): str(r.expected_value) for r in truth.itertuples()}
    if not expected:
        return None
    correct = 0
    for result in review_results:
        key = (str(result["row_index"]), result["column"])
        if result.get("accepted") and key in expected:
            correct += str(result.get("suggested_value")) == expected[key]
    return 100.0 * correct / len(expected)
