"""
tests/validate_model_quality.py
---------------------------------
MLOps quality gate: after training, check the model actually meets a
minimum performance bar BEFORE it's allowed to be built into a Docker
image and deployed. If RMSE is worse than the threshold, fail the
pipeline (exit code 1) so a bad model never reaches production.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from data_preprocessing import load_data, clean_data, treat_missing_values
from feature_engineering import remove_outliers, engineer_features, transform_target, inverse_transform_target

RMSE_THRESHOLD = 35000  # max acceptable error in dollars; tune based on business need

def main():
    df = load_data("data/train.csv")
    df = clean_data(df)
    df = treat_missing_values(df)
    df = remove_outliers(df, columns=["GrLivArea", "LotArea"])
    df = engineer_features(df)

    y = transform_target(df["SalePrice"])
    X = df.drop(columns=["SalePrice"])

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = joblib.load("models/house_price_model.pkl")
    preds_log = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(
        inverse_transform_target(y_test),
        inverse_transform_target(preds_log)
    ))

    print(f"[VALIDATION] Model RMSE on holdout: {rmse:,.2f} (threshold: {RMSE_THRESHOLD:,.2f})")

    if rmse > RMSE_THRESHOLD:
        print("[FAIL] Model did not meet quality threshold. Blocking deployment.")
        sys.exit(1)

    print("[PASS] Model meets quality bar. Safe to deploy.")


if __name__ == "__main__":
    main()
