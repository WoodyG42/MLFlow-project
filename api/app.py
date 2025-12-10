import json
import os
import time   # NEW

import pandas as pd
import mlflow
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from typing import List


# ---------- Load historical feature stats ----------
STATS_PATH = os.path.join("data", "feature_stats.json")
if not os.path.exists(STATS_PATH):
    raise RuntimeError(
        f"Feature stats file not found at {STATS_PATH}. "
        "Run train.py first so it can generate data/feature_stats.json."
    )

with open(STATS_PATH, "r") as f:
    FEATURE_STATS = json.load(f)

# These are the original training column names (with spaces)
TRAINING_FEATURES = list(FEATURE_STATS.keys())
N_FEATURES = len(TRAINING_FEATURES)

# Map API field names (snake_case) -> training column names (with spaces)
API_TO_TRAINING = {
    "mean_radius": "mean radius",
    "mean_texture": "mean texture",
    "mean_perimeter": "mean perimeter",
    "mean_area": "mean area",
    "mean_smoothness": "mean smoothness",
    "mean_compactness": "mean compactness",
    "mean_concavity": "mean concavity",
    "mean_concave_points": "mean concave points",
    "mean_symmetry": "mean symmetry",
    "mean_fractal_dimension": "mean fractal dimension",
    "radius_error": "radius error",
    "texture_error": "texture error",
    "perimeter_error": "perimeter error",
    "area_error": "area error",
    "smoothness_error": "smoothness error",
    "compactness_error": "compactness error",
    "concavity_error": "concavity error",
    "concave_points_error": "concave points error",
    "symmetry_error": "symmetry error",
    "fractal_dimension_error": "fractal dimension error",
    "worst_radius": "worst radius",
    "worst_texture": "worst texture",
    "worst_perimeter": "worst perimeter",
    "worst_area": "worst area",
    "worst_smoothness": "worst smoothness",
    "worst_compactness": "worst compactness",
    "worst_concavity": "worst concavity",
    "worst_concave_points": "worst concave points",
    "worst_symmetry": "worst symmetry",
    "worst_fractal_dimension": "worst fractal dimension",
}

# Sanity check: make sure mapping covers all training features
missing = set(TRAINING_FEATURES) - set(API_TO_TRAINING.values())
if missing:
    raise RuntimeError(f"Mapping missing training features: {missing}")

# ----------Latency Monitoring -----------------
MLFLOW_MONITORING_EXPERIMENT = "mlflow_monitoring"
mlflow.set_experiment(MLFLOW_MONITORING_EXPERIMENT)

# ---------- Pydantic models with named fields ----------
class FeatureVector(BaseModel):
    mean_radius: float
    mean_texture: float
    mean_perimeter: float
    mean_area: float
    mean_smoothness: float
    mean_compactness: float
    mean_concavity: float
    mean_concave_points: float
    mean_symmetry: float
    mean_fractal_dimension: float
    radius_error: float
    texture_error: float
    perimeter_error: float
    area_error: float
    smoothness_error: float
    compactness_error: float
    concavity_error: float
    concave_points_error: float
    symmetry_error: float
    fractal_dimension_error: float
    worst_radius: float
    worst_texture: float
    worst_perimeter: float
    worst_area: float
    worst_smoothness: float
    worst_compactness: float
    worst_concavity: float
    worst_concave_points: float
    worst_symmetry: float
    worst_fractal_dimension: float

    # Pydantic v2-style validator: use field_validator + info
    @field_validator("*", mode="before")
    @classmethod
    def no_nones(cls, v, info):
        if v is None:
            raise ValueError(f"Missing value for field '{info.field_name}'")
        return v


class Features(BaseModel):
    instances: List[FeatureVector]

    @field_validator("instances")
    @classmethod
    def non_empty(cls, v):
        if not v:
            raise ValueError("At least one instance is required")
        return v

def log_inference_metrics(
    batch_size: int,
    latency_ms: float,
    hard_violation_count: int,
    soft_warning_count: int,
    success: bool,
):
    """
    Logs per-request inference metrics to MLflow.
    One run per request is fine for this assignment.
    """
    with mlflow.start_run(run_name="inference", nested=False):
        mlflow.log_metric("latency_ms", latency_ms)
        mlflow.log_metric("batch_size", batch_size)
        mlflow.log_metric("hard_violation_count", hard_violation_count)
        mlflow.log_metric("soft_warning_count", soft_warning_count)
        mlflow.log_param("success", int(success))  # 1 or 0

# ---------- Load model from MLflow ----------
import mlflow.pyfunc

MODEL_URI = "chosen_model"  # relative to /app inside container
model = mlflow.pyfunc.load_model(MODEL_URI)

#MODEL_NAME = "LR_Cancer_Model"
#MODEL_URI = f"models:/{MODEL_NAME}@champion"  # using alias

app = FastAPI(title="CancerClassificationAPI")

try:
    model = mlflow.pyfunc.load_model(MODEL_URI)
except Exception as e:
    raise RuntimeError(f"Failed to load model from '{MODEL_URI}': {e}")


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_uri": MODEL_URI,
        "n_features": N_FEATURES,
    }


@app.get("/feature_stats")
def feature_stats():
    """
    Optional: expose the historical means/std/min/max for transparency.
    """
    return FEATURE_STATS


@app.post("/predict")
def predict(features: Features):
    start_time = time.time()

    # Convert Pydantic models -> list of dicts
    api_rows = [inst.model_dump() for inst in features.instances]
    batch_size = len(api_rows)

    # Map API keys (snake_case) to training feature names (with spaces)
    converted_rows = []
    for row in api_rows:
        new_row = {}
        for api_name, value in row.items():
            training_name = API_TO_TRAINING[api_name]
            new_row[training_name] = value
        converted_rows.append(new_row)

    # Create DataFrame with training feature names and correct column order
    df = pd.DataFrame(converted_rows)
    df = df[TRAINING_FEATURES]  # enforce order

    # ---- Range checks using historical stats ----
    hard_violations = []
    soft_warnings = []

    for col in TRAINING_FEATURES:
        stats = FEATURE_STATS[col]
        col_min = stats["min"]
        col_max = stats["max"]
        mean = stats["mean"]
        std = stats["std"] or 1e-9  # avoid divide-by-zero

        # Hard: outside historical min/max
        mask_hard = (df[col] < col_min) | (df[col] > col_max)
        out_idx = df.index[mask_hard].tolist()
        if out_idx:
            hard_violations.append(
                {
                    "feature": col,
                    "rows": out_idx,
                    "min_allowed": col_min,
                    "max_allowed": col_max,
                    "values": df.loc[out_idx, col].tolist(),
                }
            )

        # Soft: >3 std dev away from mean
        z_scores = (df[col] - mean) / std
        mask_soft = z_scores.abs() > 3
        warn_idx = df.index[mask_soft].tolist()
        if warn_idx:
            soft_warnings.append(
                {
                    "feature": col,
                    "rows": warn_idx,
                    "mean": mean,
                    "std": std,
                    "threshold_std": 3,
                    "values": df.loc[warn_idx, col].tolist(),
                }
            )

    # If there are hard violations, log and reject
    if hard_violations:
        latency_ms = (time.time() - start_time) * 1000.0
        log_inference_metrics(
            batch_size=batch_size,
            latency_ms=latency_ms,
            hard_violation_count=len(hard_violations),
            soft_warning_count=len(soft_warnings),
            success=False,
        )
        raise HTTPException(
            status_code=422,
            detail={
                "message": "One or more inputs are outside the historical min/max range",
                "violations": hard_violations,
            },
        )

    # ---- No hard violations -> run model ----
    try:
        preds = model.predict(df)
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000.0
        log_inference_metrics(
            batch_size=batch_size,
            latency_ms=latency_ms,
            hard_violation_count=len(hard_violations),
            soft_warning_count=len(soft_warnings),
            success=False,
        )
        raise HTTPException(
            status_code=400,
            detail=f"Error while generating predictions: {str(e)}",
        )

    # Successful path: log metrics & return
    latency_ms = (time.time() - start_time) * 1000.0
    log_inference_metrics(
        batch_size=batch_size,
        latency_ms=latency_ms,
        hard_violation_count=len(hard_violations),
        soft_warning_count=len(soft_warnings),
        success=True,
    )

    return {
        "n_rows": len(df),
        "predictions": preds.tolist(),
        "warnings": soft_warnings,
        "latency_ms": latency_ms,
    }

