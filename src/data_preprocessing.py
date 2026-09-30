"""
data_preprocessing.py
----------------------
Step 3-6 of the pipeline: Load data, clean it, and treat missing values.

Enna pannuthu (short):
- load_data()          -> CSV read pannum
- clean_data()         -> duplicates, wrong dtypes, id column remove pannum
- treat_missing_values() -> numeric=median, categorical(meaningful)="None",
                            categorical(random)=mode, high-missing cols=drop
"""

import pandas as pd
import numpy as np


def load_data(path: str) -> pd.DataFrame:
    """Load raw CSV into a DataFrame."""
    df = pd.read_csv(path)
    print(f"[INFO] Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates, drop useless columns, fix obvious dtype issues."""
    df = df.copy()

    # 1. Remove duplicate rows
    before = df.shape[0]
    df.drop_duplicates(inplace=True)
    print(f"[INFO] Removed {before - df.shape[0]} duplicate rows")

    # 2. Drop identifier column (no predictive value)
    if "Id" in df.columns:
        df.drop(columns=["Id"], inplace=True)

    # 3. Strip whitespace / standardize casing for object columns
    obj_cols = df.select_dtypes(include="object").columns
    for col in obj_cols:
        df[col] = df[col].astype(str).str.strip()

    return df


def treat_missing_values(df: pd.DataFrame, drop_threshold: float = 0.80) -> pd.DataFrame:
    """
    Handle missing values with different strategies:
    - Columns with > drop_threshold missing -> dropped entirely
    - Categorical columns where NaN is meaningful (e.g. no garage) -> filled with 'None'
    - Remaining categorical -> filled with mode
    - Numerical -> filled with median
    """
    df = df.copy()

    # 1. Drop columns with too many missing values
    missing_pct = df.isnull().mean()
    cols_to_drop = missing_pct[missing_pct > drop_threshold].index.tolist()
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
        print(f"[INFO] Dropped high-missing columns: {cols_to_drop}")

    # 2. Columns where NaN genuinely means "feature absent" (Kaggle housing dataset)
    none_means_absent = [
        "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
        "GarageType", "GarageFinish", "GarageQual", "GarageCond",
        "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
        "BsmtFinType2", "MasVnrType"
    ]
    for col in none_means_absent:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    # 3. Numeric columns -> median fill
    num_cols = df.select_dtypes(include=np.number).columns
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    # 4. Remaining categorical -> mode fill
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    remaining_missing = df.isnull().sum().sum()
    print(f"[INFO] Missing values remaining after treatment: {remaining_missing}")
    return df
