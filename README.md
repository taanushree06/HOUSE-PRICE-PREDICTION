# House Price Prediction — End-to-End ML System

Regression system that predicts residential property sale price from
structural, quality, and location features. Covers the full lifecycle:
data cleaning → EDA → feature engineering → training → tuning →
evaluation → deployment → MLOps automation.

## Project Structure

```
house-price-ml/
├── data/
│   └── train.csv                # place your Kaggle-style dataset here
├── src/
│   ├── data_preprocessing.py    # load, clean, missing-value treatment
│   ├── feature_engineering.py   # outliers, new features, log transform
│   ├── build_pipeline.py        # ColumnTransformer + sklearn Pipeline
│   ├── train.py                 # trains & tunes 3 models, saves the best
│   └── predict.py               # batch prediction on new properties (CLI)
├── app/
│   └── main.py                  # FastAPI serving app (REST API)
├── tests/
│   ├── test_pipeline.py         # unit tests
│   └── validate_model_quality.py # CI quality gate
├── models/                      # trained model + preprocessor land here
├── config.yaml                  # all paths & hyperparameters, in one place
├── requirements.txt
├── Dockerfile
└── .github/workflows/ci-cd.yaml # automated MLOps pipeline
```

## How to Run (Local)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place your dataset at data/train.csv
#    (Kaggle "House Prices - Advanced Regression Techniques" schema)

# 3. Train the model (cleans data, engineers features, tunes hyperparameters,
#    evaluates, and saves the best model to models/house_price_model.pkl)
python src/train.py

# 4. Predict on new properties from a CSV
python src/predict.py --input data/new_properties.csv --output predictions.csv

# 5. Serve the model as an API
uvicorn app.main:app --reload --port 8000
# then open http://localhost:8000/docs to test /predict interactively
```

## How to Run (Docker)

```bash
docker build -t house-price-api .
docker run -p 8000:8000 house-price-api
```

---

## MLOps Pipeline — What Happens Automatically

Idhu than "code work pannuradhu" and "production-ready system" oda difference.
MLOps na, model training + deployment ah manual steps ah illama, automated,
repeatable, and monitored ah pannuradhu.

### 1. Version Control (Git)
All code, config, and pipeline definitions live in Git. Every change is
tracked — if a model regresses, you can trace exactly which code/data
change caused it.

### 2. Continuous Integration (CI) — `.github/workflows/ci-cd.yaml`
On every push:
- **`test` job** — runs `pytest tests/` to catch broken preprocessing or
  feature logic before it ever touches training.
- **`train_and_validate` job** — retrains the model fresh, then runs
  `validate_model_quality.py`, a **quality gate**: if RMSE is worse than
  a set threshold, the pipeline **fails on purpose** — a bad model can
  never reach production silently.
- **`build_and_push_image` job** — only runs if the model passed
  validation. Builds the Docker image and (in a real setup) pushes it
  to a registry (ECR/GCR/Docker Hub) and triggers deployment.

### 3. Model & Config Versioning
- `config.yaml` centralizes every hyperparameter and path — no
  hardcoded magic numbers buried in code.
- `models/feature_schema.pkl` stores exactly which features the model
  expects, so serving code can validate incoming data and fail loudly
  instead of silently mispredicting.
- For larger teams, this is where **MLflow Tracking** (already in
  `requirements.txt`) or DVC would log every run's metrics, parameters,
  and model artifact for comparison and rollback.

### 4. Serving Layer — `app/main.py`
- FastAPI loads the model **once** at startup (not per-request — that
  would be slow and wasteful).
- `/predict` validates input shape via Pydantic before it ever reaches
  the model — bad requests get a clean 400 error, not a silent crash.
- `/health` lets a load balancer or Kubernetes liveness probe check the
  service is actually ready.

### 5. Containerization — `Dockerfile`
Packages code + trained model + dependencies into one image, so
"works on my machine" stops being a problem. The same image runs
identically in dev, staging, and production.

### 6. Monitoring & Retraining (next steps in a mature setup)
Not yet wired in this scaffold, but the natural next additions:
- **Data drift monitoring** — compare incoming request feature
  distributions to training data distributions over time.
- **Prediction logging** — store every prediction + eventual actual
  sale price (once known) to measure real-world model decay.
- **Scheduled retraining** — trigger `train.py` on a cron/Airflow
  schedule when drift or decay crosses a threshold, re-running the same
  CI quality gate before any new model replaces the live one.

---

## Metrics Used
- **RMSE** (Root Mean Squared Error) — penalizes large errors heavily;
  primary metric for tuning.
- **MAE** (Mean Absolute Error) — average error in actual dollars,
  easier to explain to business stakeholders.
- **R²** — how much price variance the model explains overall.

---

## Optimization Pass (v2) — What Changed

This version adds several optimizations on top of the baseline pipeline:

1. **Target Encoding for `Neighborhood`** — encodes each neighborhood as its
   average training-set sale price (`Neighborhood_encoded`), learned only
   from training data to avoid leakage. The mapping is saved to
   `models/neighborhood_encoding.pkl` and reused by `predict.py` and the API.
2. **More log-transformed features** — `1stFlrSF`, `TotalBsmtSF`,
   `GarageArea` added to the skew-correction list (previously only
   `LotArea`, `GrLivArea`, `TotalSF`).
3. **Interaction features** — `Qual_x_LivArea`, `OverallScore`,
   `Qual_x_TotalSF` capture that quality and size together predict price
   better than either alone.
4. **Stricter outlier removal** — IQR factor tightened from 3.0 to 2.5
   (configurable in `config.yaml` under `training.outlier_factor`).
5. **Wider Random Forest grid** — added `max_features` (`sqrt`/`log2`),
   which was missing before and likely caused RF to underperform relative
   to XGBoost.
6. **CatBoost added** as a 4th candidate model — handles categorical
   columns natively, which suits this dataset's many categorical features.
7. **Ridge Regression added** as a 5th candidate model — Linear Regression
   with L2 regularization, which helps when features are correlated
   (this dataset has several overlapping size features like `TotalSF`,
   `GrLivArea`, `TotalBsmtSF`). Often competitive with tree-based models
   on this kind of tabular data.
8. **5-fold cross-validation stability check** on the winning model, run
   after model selection, to confirm the reported RMSE isn't a lucky split.
9. **Stacking Ensemble** — combines all 5 tuned models (Linear Regression,
   Ridge, Random Forest, XGBoost, CatBoost) into one. Each base model runs
   internally via cross-validation to produce out-of-fold predictions
   (no leakage), and a Ridge "meta-model" learns how to best weight and
   combine their outputs. This is evaluated as a 6th candidate alongside
   the 5 individual models — it wins when the models' errors are
   different enough from each other to be complementary.

Re-run `python src/train.py` after pulling this version to retrain with
all of the above. Training now fits 6 candidates (5 tuned models + the
stacking ensemble), so it takes noticeably longer than the original
3-model version — expect several minutes depending on your machine.
