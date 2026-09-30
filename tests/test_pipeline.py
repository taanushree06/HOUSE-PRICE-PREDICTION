"""
tests/test_pipeline.py
------------------------
Basic unit tests so the CI pipeline catches breakage before it ships.
Run with: pytest tests/
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import numpy as np
from feature_engineering import engineer_features, transform_target, inverse_transform_target, remove_outliers


def test_engineer_features_creates_totalsf():
    df = pd.DataFrame({
        "TotalBsmtSF": [500], "1stFlrSF": [800], "2ndFlrSF": [400],
        "YrSold": [2020], "YearBuilt": [2000], "YearRemodAdd": [2010],
        "FullBath": [2], "HalfBath": [1], "BsmtFullBath": [0], "BsmtHalfBath": [0],
        "GarageArea": [200], "PoolArea": [0], "LotArea": [5000], "GrLivArea": [1200]
    })
    result = engineer_features(df)
    assert result["TotalSF"].iloc[0] == 1700
    assert result["HouseAge"].iloc[0] == 20
    assert result["HasGarage"].iloc[0] == 1
    assert result["HasPool"].iloc[0] == 0


def test_target_transform_is_invertible():
    y = pd.Series([100000, 250000, 500000])
    y_log = transform_target(y)
    y_back = inverse_transform_target(y_log.values)
    assert np.allclose(y.values, y_back, atol=1)


def test_remove_outliers_reduces_or_keeps_rows():
    df = pd.DataFrame({"GrLivArea": [1000, 1100, 1200, 50000], "LotArea": [5000, 5200, 4800, 100000]})
    cleaned = remove_outliers(df, columns=["GrLivArea", "LotArea"])
    assert cleaned.shape[0] <= df.shape[0]
