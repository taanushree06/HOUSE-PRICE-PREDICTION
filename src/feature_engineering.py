"""
feature_engineering.py
------------------------
Step 7-8 of the pipeline: Outlier removal + creating new engineered features.
"""

import numpy as np
import pandas as pd


def remove_outliers(df: pd.DataFrame, columns: list, factor: float = 3.0) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        before = df.shape[0]
        df = df[(df[col] >= lower) & (df[col] <= upper)]
        removed = before - df.shape[0]
        if removed > 0:
            print(f"[INFO] Removed {removed} outlier rows based on '{col}'")
    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if {"TotalBsmtSF", "1stFlrSF", "2ndFlrSF"}.issubset(df.columns):
        df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]

    if {"YrSold", "YearBuilt"}.issubset(df.columns):
        df["HouseAge"] = df["YrSold"] - df["YearBuilt"]

    if {"YrSold", "YearRemodAdd"}.issubset(df.columns):
        df["YearsSinceRemodel"] = df["YrSold"] - df["YearRemodAdd"]

    bath_cols = ["FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"]
    if all(c in df.columns for c in bath_cols):
        df["TotalBathrooms"] = (
            df["FullBath"] + 0.5 * df["HalfBath"] +
            df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"]
        )

    if "GarageArea" in df.columns:
        df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
    if "PoolArea" in df.columns:
        df["HasPool"] = (df["PoolArea"] > 0).astype(int)
    if "2ndFlrSF" in df.columns:
        df["HasSecondFloor"] = (df["2ndFlrSF"] > 0).astype(int)

    if {"OverallQual", "GrLivArea"}.issubset(df.columns):
        df["Qual_x_LivArea"] = df["OverallQual"] * df["GrLivArea"]
    if {"OverallQual", "OverallCond"}.issubset(df.columns):
        df["OverallScore"] = df["OverallQual"] * df["OverallCond"]
    if {"OverallQual", "TotalSF"}.issubset(df.columns):
        df["Qual_x_TotalSF"] = df["OverallQual"] * df["TotalSF"]

    skewed_candidates = [
        "LotArea", "GrLivArea", "TotalSF", "1stFlrSF",
        "TotalBsmtSF", "GarageArea"
    ]
    for col in skewed_candidates:
        if col in df.columns:
            df[f"{col}_log"] = np.log1p(df[col])

    return df


def fit_target_encoding(train_column: pd.Series, train_target: pd.Series) -> dict:
    """Learn a category -> average target value mapping, using ONLY training data."""
    encoding_map = train_target.groupby(train_column).mean().to_dict()
    overall_mean = float(train_target.mean())
    return {"map": encoding_map, "overall_mean": overall_mean}


def apply_target_encoding(df: pd.DataFrame, column: str, encoding: dict) -> pd.DataFrame:
    """Apply a previously-fitted target encoding to df."""
    df = df.copy()
    if column not in df.columns:
        return df
    df[f"{column}_encoded"] = df[column].map(encoding["map"]).fillna(encoding["overall_mean"])
    return df


def add_target_encoding(df_train: pd.DataFrame, df_valid: pd.DataFrame,
                         column: str, target: pd.Series) -> tuple:
    """Fits encoding on df_train, applies to both df_train and df_valid (no leakage)."""
    encoding = fit_target_encoding(df_train[column], target)
    df_train = apply_target_encoding(df_train, column, encoding)
    df_valid = apply_target_encoding(df_valid, column, encoding)
    return df_train, df_valid, encoding


def ensure_expected_columns(df: pd.DataFrame, expected_columns: list) -> pd.DataFrame:
    """
    Guarantee every column the trained pipeline expects is present in df,
    even if a real-world request is missing a field. Missing columns are
    added as NaN - the pipeline's own SimpleImputer (fit on TRAINING data
    medians/modes) fills them correctly.
    """
    df = df.copy()
    for col in expected_columns:
        if col not in df.columns:
            df[col] = np.nan
    return df


def transform_target(y: pd.Series) -> pd.Series:
    """Log-transform target (SalePrice) to reduce right-skew."""
    return np.log1p(y)


def inverse_transform_target(y_log: np.ndarray) -> np.ndarray:
    """Convert log-scale predictions back to real price scale."""
    return np.expm1(y_log)