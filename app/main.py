"""
app/main.py
------------
Step 18: Deployment. Serves the trained model as a REST API using FastAPI.

Run locally:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Then POST property data to:  http://localhost:8000/predict
Interactive docs available at: http://localhost:8000/docs
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from feature_engineering import (
    engineer_features, inverse_transform_target,
    apply_target_encoding, ensure_expected_columns
)

app = FastAPI(
    title="House Price Prediction API",
    description="Predicts residential property sale price from structural, quality, and location features.",
    version="1.0.0"
)

# Allow the frontend (opened as a local file, or hosted separately) to call
# this API from the browser. Restrict allow_origins to your real frontend's
# domain once deployed - "*" is fine for local testing.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "models/house_price_model.pkl"
ENCODING_PATH = "models/neighborhood_encoding.pkl"
SCHEMA_PATH = "models/feature_schema.pkl"
model = None
neighborhood_encoding = None
feature_schema = None


@app.on_event("startup")
def load_model():
    """Load the trained pipeline + encoding map + expected feature schema once, at startup."""
    global model, neighborhood_encoding, feature_schema
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("[INFO] Model loaded successfully.")
    else:
        print("[WARNING] Model file not found. Train the model first (python src/train.py).")
    if os.path.exists(ENCODING_PATH):
        neighborhood_encoding = joblib.load(ENCODING_PATH)
        print("[INFO] Neighborhood encoding map loaded.")
    if os.path.exists(SCHEMA_PATH):
        feature_schema = joblib.load(SCHEMA_PATH)
        print("[INFO] Feature schema loaded.")


class PropertyFeatures(BaseModel):
    """
    Minimal example schema - extend with every column your trained model expects.
    Only the fields that matter most to price are shown here for brevity.
    """
    OverallQual: int
    GrLivArea: float
    TotalBsmtSF: float
    FirstFlrSF: float
    SecondFlrSF: float = 0
    GarageArea: float = 0
    GarageCars: int = 0
    YearBuilt: int
    YearRemodAdd: int
    YrSold: int
    FullBath: int = 1
    HalfBath: int = 0
    BsmtFullBath: int = 0
    BsmtHalfBath: int = 0
    LotArea: float
    PoolArea: float = 0
    Neighborhood: str
    HouseStyle: Optional[str] = "1Story"
    KitchenQual: Optional[str] = "TA"


@app.get("/")
def root():
    return {"status": "ok", "message": "House Price Prediction API is running."}


@app.get("/health")
def health_check():
    return {"model_loaded": model is not None}


@app.post("/predict")
def predict_price(features: PropertyFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train the model first.")

    data = features.dict()
    data["1stFlrSF"] = data.pop("FirstFlrSF")
    data["2ndFlrSF"] = data.pop("SecondFlrSF")

    df = pd.DataFrame([data])
    df = engineer_features(df)

    if neighborhood_encoding is not None and "Neighborhood" in df.columns:
        df = apply_target_encoding(df, "Neighborhood", neighborhood_encoding)

    # Any column the model expects that wasn't in this request gets added
    # as NaN, and the pipeline's own trained imputer fills it - instead of
    # crashing with "columns are missing".
    if feature_schema is not None:
        expected_columns = feature_schema["numeric"] + feature_schema["categorical"]
        df = ensure_expected_columns(df, expected_columns)

    try:
        log_pred = model.predict(df)
        price = float(inverse_transform_target(log_pred)[0])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

    return {
        "predicted_sale_price": round(price, 2),
        "currency": "USD"
    }