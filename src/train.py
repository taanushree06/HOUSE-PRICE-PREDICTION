"""
train.py
---------
Step 11-17 of the pipeline: Train/test split, baseline + advanced models,
cross-validation, hyperparameter tuning, evaluation, and saving the best model.

Run this file directly:
    python src/train.py
"""

import os
import yaml
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from data_preprocessing import load_data, clean_data, treat_missing_values
from feature_engineering import (
    remove_outliers, engineer_features, transform_target,
    inverse_transform_target, add_target_encoding
)
from build_pipeline import build_preprocessor, build_full_pipeline


def load_config(path="config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def strip_prefix(params: dict) -> dict:
    """GridSearchCV best_params_ come back as {'model__alpha': 1.0, ...}.
    Strip the 'model__' prefix so the value can be passed straight into
    a fresh (unfitted) instance of that model, e.g. Ridge(alpha=1.0)."""
    return {k.split("__", 1)[1]: v for k, v in params.items()}


def evaluate_model(name, y_true_log, y_pred_log):
    """Convert back from log scale and compute RMSE, MAE, R2 on real price scale."""
    y_true = inverse_transform_target(y_true_log)
    y_pred = inverse_transform_target(y_pred_log)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n[RESULTS - {name}]")
    print(f"  RMSE : {rmse:,.2f}")
    print(f"  MAE  : {mae:,.2f}")
    print(f"  R2   : {r2:.4f}")
    return {"model": name, "rmse": rmse, "mae": mae, "r2": r2}


def plot_residuals(y_true_log, y_pred_log, save_path):
    y_true = inverse_transform_target(y_true_log)
    y_pred = inverse_transform_target(y_pred_log)
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(y_pred, y_true, alpha=0.4)
    axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    axes[0].set_xlabel("Predicted Price")
    axes[0].set_ylabel("Actual Price")
    axes[0].set_title("Actual vs Predicted")

    axes[1].scatter(y_pred, residuals, alpha=0.4)
    axes[1].axhline(0, color='r', linestyle='--')
    axes[1].set_xlabel("Predicted Price")
    axes[1].set_ylabel("Residual")
    axes[1].set_title("Residual Plot")

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[INFO] Saved residual plot to {save_path}")


def main():
    cfg = load_config()

    # ---------- Step 3-6: Load, clean, treat missing values ----------
    df = load_data(cfg["data"]["raw_path"])
    df = clean_data(df)
    df = treat_missing_values(df)

    # ---------- Step 7-8: Outlier removal + feature engineering ----------
    outlier_factor = cfg["training"].get("outlier_factor", 3.0)
    df = remove_outliers(df, columns=["GrLivArea", "LotArea"], factor=outlier_factor)
    df = engineer_features(df)

    target_col = cfg["data"]["target_column"]
    y_raw = df[target_col]                 # original scale, needed for target encoding
    y = transform_target(y_raw)            # log scale, used for modeling
    X = df.drop(columns=[target_col])

    # ---------- Step 11: Train/test split (raw target carried alongside for encoding) ----------
    X_train, X_test, y_train, y_test, y_train_raw, y_test_raw = train_test_split(
        X, y, y_raw,
        test_size=cfg["data"]["test_size"],
        random_state=cfg["data"]["random_state"]
    )
    print(f"[INFO] Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # ---------- Optimization 1: Target-encode Neighborhood ----------
    # Learned ONLY from training data + training target, then applied to test
    # data, so there is no leakage from test set into the encoding.
    neighborhood_encoding = None
    if "Neighborhood" in X_train.columns:
        X_train, X_test, neighborhood_encoding = add_target_encoding(
            X_train, X_test, "Neighborhood", y_train_raw
        )
        print("[INFO] Added target-encoded 'Neighborhood_encoded' feature")

    numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_train.select_dtypes(include="object").columns.tolist()

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    results = []
    fitted_pipelines = {}

    # ---------- Step 12: Baseline model ----------
    baseline_pipeline = build_full_pipeline(preprocessor, LinearRegression())
    baseline_pipeline.fit(X_train, y_train)
    pred = baseline_pipeline.predict(X_test)
    results.append(evaluate_model("Linear Regression (baseline)", y_test, pred))
    fitted_pipelines["Linear Regression"] = baseline_pipeline

    # ---------- Optimization: Ridge Regression (regularized linear model) ----------
    # Penalizes large coefficients, which helps when features are correlated
    # (this dataset has several overlapping size features).
    ridge_pipeline = build_full_pipeline(preprocessor, Ridge(random_state=42))
    ridge_grid = {
        "model__alpha": cfg["training"]["ridge"]["alpha"]
    }
    ridge_search = GridSearchCV(
        ridge_pipeline, ridge_grid,
        cv=cfg["training"]["cv_folds"],
        scoring=cfg["training"]["scoring"],
        n_jobs=-1
    )
    ridge_search.fit(X_train, y_train)
    best_ridge = ridge_search.best_estimator_
    print(f"[INFO] Best Ridge params: {ridge_search.best_params_}")
    pred = best_ridge.predict(X_test)
    results.append(evaluate_model("Ridge Regression (tuned)", y_test, pred))
    fitted_pipelines["Ridge Regression"] = best_ridge

    # ---------- Step 12-14: Random Forest + tuning ----------
    rf_pipeline = build_full_pipeline(preprocessor, RandomForestRegressor(random_state=42))
    rf_grid = {
        "model__n_estimators": cfg["training"]["random_forest"]["n_estimators"],
        "model__max_depth": cfg["training"]["random_forest"]["max_depth"],
        "model__min_samples_split": cfg["training"]["random_forest"]["min_samples_split"],
        "model__max_features": cfg["training"]["random_forest"]["max_features"],
    }
    rf_search = GridSearchCV(
        rf_pipeline, rf_grid,
        cv=cfg["training"]["cv_folds"],
        scoring=cfg["training"]["scoring"],
        n_jobs=-1
    )
    rf_search.fit(X_train, y_train)
    best_rf = rf_search.best_estimator_
    print(f"[INFO] Best RF params: {rf_search.best_params_}")
    pred = best_rf.predict(X_test)
    results.append(evaluate_model("Random Forest (tuned)", y_test, pred))
    fitted_pipelines["Random Forest"] = best_rf

    # ---------- Step 12-14: XGBoost + tuning ----------
    xgb_pipeline = build_full_pipeline(preprocessor, XGBRegressor(random_state=42, objective="reg:squarederror"))
    xgb_grid = {
        "model__n_estimators": cfg["training"]["xgboost"]["n_estimators"],
        "model__max_depth": cfg["training"]["xgboost"]["max_depth"],
        "model__learning_rate": cfg["training"]["xgboost"]["learning_rate"],
    }
    xgb_search = GridSearchCV(
        xgb_pipeline, xgb_grid,
        cv=cfg["training"]["cv_folds"],
        scoring=cfg["training"]["scoring"],
        n_jobs=-1
    )
    xgb_search.fit(X_train, y_train)
    best_xgb = xgb_search.best_estimator_
    print(f"[INFO] Best XGB params: {xgb_search.best_params_}")
    pred = best_xgb.predict(X_test)
    results.append(evaluate_model("XGBoost (tuned)", y_test, pred))
    fitted_pipelines["XGBoost"] = best_xgb

    # ---------- Optimization 2: CatBoost (handles categoricals natively, +
    # this dataset is categorical-heavy) ----------
    cat_pipeline = build_full_pipeline(preprocessor, CatBoostRegressor(random_state=42, verbose=0))
    cat_grid = {
        "model__iterations": cfg["training"]["catboost"]["iterations"],
        "model__depth": cfg["training"]["catboost"]["depth"],
        "model__learning_rate": cfg["training"]["catboost"]["learning_rate"],
    }
    cat_search = GridSearchCV(
        cat_pipeline, cat_grid,
        cv=cfg["training"]["cv_folds"],
        scoring=cfg["training"]["scoring"],
        n_jobs=-1
    )
    cat_search.fit(X_train, y_train)
    best_cat = cat_search.best_estimator_
    print(f"[INFO] Best CatBoost params: {cat_search.best_params_}")
    pred = best_cat.predict(X_test)
    results.append(evaluate_model("CatBoost (tuned)", y_test, pred))
    fitted_pipelines["CatBoost"] = best_cat

    # ---------- Optimization: Stacking Ensemble (combines all 5 models) ----------
    # Each base model is re-created (unfitted) with ITS OWN best hyperparameters
    # found above, so the ensemble isn't starting from defaults. StackingRegressor
    # trains each base model internally via cross-validation to produce
    # out-of-fold predictions (avoiding leakage), then trains a final Ridge
    # "meta-model" that learns how to best weight/combine those predictions.
    print("\n[INFO] Training Stacking Ensemble (combining all 5 models)...")
    stacking_model = StackingRegressor(
        estimators=[
            ("linear", LinearRegression()),
            ("ridge", Ridge(random_state=42, **strip_prefix(ridge_search.best_params_))),
            ("rf", RandomForestRegressor(random_state=42, **strip_prefix(rf_search.best_params_))),
            ("xgb", XGBRegressor(random_state=42, objective="reg:squarederror",
                                  **strip_prefix(xgb_search.best_params_))),
            ("cat", CatBoostRegressor(random_state=42, verbose=0, **strip_prefix(cat_search.best_params_))),
        ],
        final_estimator=Ridge(alpha=1.0),
        n_jobs=-1
    )
    stacking_pipeline = build_full_pipeline(preprocessor, stacking_model)
    stacking_pipeline.fit(X_train, y_train)
    pred = stacking_pipeline.predict(X_test)
    results.append(evaluate_model("Stacking Ensemble (all 5)", y_test, pred))
    fitted_pipelines["Stacking Ensemble"] = stacking_pipeline

    # ---------- Step 16: Compare & select best model ----------
    results_df = pd.DataFrame(results).sort_values("rmse")
    print("\n===== MODEL COMPARISON (sorted by RMSE, lower is better) =====")
    print(results_df.to_string(index=False))

    best_model_name = results_df.iloc[0]["model"].split(" (")[0]
    # map back to fitted_pipelines key
    key_map = {"Linear Regression": "Linear Regression", "Random Forest": "Random Forest", "XGBoost": "XGBoost"}
    best_key = next(k for k in fitted_pipelines if k in best_model_name)
    best_pipeline = fitted_pipelines[best_key]
    print(f"\n[INFO] Best model selected: {best_key}")

    # ---------- Optimization 3: Cross-validation stability check ----------
    # A single train/test split can be a "lucky" or "unlucky" split. This
    # re-checks the winning model across 5 different splits of the FULL
    # training data, so we know the RMSE we saw isn't a fluke.
    print("\n[INFO] Running 5-fold cross-validation on the best model for stability check...")
    cv_scores = cross_val_score(
        best_pipeline, X_train, y_train,
        cv=cfg["training"]["cv_folds"],
        scoring=cfg["training"]["scoring"],
        n_jobs=-1
    )
    cv_rmse = -cv_scores  # note: y_train is log-scale, so this is log-RMSE
    print(f"[INFO] CV RMSE (log-scale) across folds: {np.round(cv_rmse, 4)}")
    print(f"[INFO] CV RMSE mean: {cv_rmse.mean():.4f}, std: {cv_rmse.std():.4f}")
    if cv_rmse.std() / cv_rmse.mean() > 0.15:
        print("[WARNING] High variance across folds - model may be unstable. Consider more data or regularization.")
    else:
        print("[INFO] Low variance across folds - model performance looks stable.")

    # Save diagnostic residual plot for the best model
    os.makedirs("models", exist_ok=True)
    best_pred = best_pipeline.predict(X_test)
    plot_residuals(y_test, best_pred, "models/residual_plot.png")

    # ---------- Step 17: Save the best model ----------
    joblib.dump(best_pipeline, cfg["model"]["save_path"])
    print(f"[INFO] Saved best model to {cfg['model']['save_path']}")

    # Save the feature list used, for validation at inference time
    joblib.dump({"numeric": numeric_features, "categorical": categorical_features},
                "models/feature_schema.pkl")

    # Save the target-encoding map so predict.py / the API can apply the
    # EXACT same Neighborhood -> average price mapping learned here.
    if neighborhood_encoding is not None:
        joblib.dump(neighborhood_encoding, "models/neighborhood_encoding.pkl")
        print("[INFO] Saved Neighborhood target-encoding map to models/neighborhood_encoding.pkl")


if __name__ == "__main__":
    main()
