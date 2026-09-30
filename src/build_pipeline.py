"""
build_pipeline.py
-------------------
Step 9-10 of the pipeline: Encoding + Scaling, wrapped in a single
sklearn Pipeline so preprocessing + model always travel together
(no train/test mismatch, no manual re-coding at prediction time).

Enna pannuthu (short):
- numeric features    -> median impute (safety net) + StandardScaler
- categorical features -> most_frequent impute (safety net) + OneHotEncoder
- ColumnTransformer combines both, then attaches to the chosen model
"""

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


def build_preprocessor(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    """Build the preprocessing ColumnTransformer."""

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    return preprocessor


def build_full_pipeline(preprocessor: ColumnTransformer, model) -> Pipeline:
    """Attach a regressor to the preprocessing pipeline."""
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])
