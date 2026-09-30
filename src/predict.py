"""
predict.py
-----------
Step 18: Load the saved model and predict prices for new properties.

Usage:
    python src/predict.py --input new_properties.csv
"""

import argparse
import os
import joblib
import pandas as pd

from data_preprocessing import clean_data
from feature_engineering import (
    engineer_features, inverse_transform_target,
    apply_target_encoding, ensure_expected_columns
)


def predict_new_properties(input_csv: str, model_path: str = "models/house_price_model.pkl") -> pd.DataFrame:
    model = joblib.load(model_path)

    df = pd.read_csv(input_csv)

    # NOTE: we intentionally do NOT call treat_missing_values() here.
    # That function looks at "% missing in this batch" to decide what to
    # drop/fill - correct for a full training set, but wrong for scoring
    # (a batch of 1-5 new properties can look "100% missing" in a column
    # just by chance, causing it to be dropped and breaking the model).
    # Instead: light cleaning, then let the trained pipeline's own
    # imputer (fit on training data stats) handle any missing values.
    df_clean = clean_data(df)
    df_clean = engineer_features(df_clean)

    # Apply the same Neighborhood target-encoding learned during training
    encoding_path = "models/neighborhood_encoding.pkl"
    if os.path.exists(encoding_path) and "Neighborhood" in df_clean.columns:
        neighborhood_encoding = joblib.load(encoding_path)
        df_clean = apply_target_encoding(df_clean, "Neighborhood", neighborhood_encoding)

    # Guarantee every column the trained model expects exists (as NaN if
    # missing), so the pipeline's imputer can fill it instead of erroring.
    schema_path = "models/feature_schema.pkl"
    if os.path.exists(schema_path):
        schema = joblib.load(schema_path)
        expected_columns = schema["numeric"] + schema["categorical"]
        df_clean = ensure_expected_columns(df_clean, expected_columns)

    log_preds = model.predict(df_clean)
    preds = inverse_transform_target(log_preds)

    df["PredictedSalePrice"] = preds
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to CSV with new property data")
    parser.add_argument("--output", default="predictions.csv", help="Where to save predictions")
    args = parser.parse_args()

    result = predict_new_properties(args.input)
    result.to_csv(args.output, index=False)
    print(f"[INFO] Predictions saved to {args.output}")