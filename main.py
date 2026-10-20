"""
MEQYAS AI - FastAPI Anomaly Detection Service
============================================
FastAPI service to detect anomalies in machine energy telemetry data
using the trained IsolationForest pipeline loaded from the `model/` directory.

Run directly:
    python main.py

Or via Uvicorn CLI:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Paths and Constants
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

# Status code mapping matching training notebook & predict.py
STATUS_MAP = {
    0: "idle",
    1: "manual",
    2: "automatic",
    3: "alarm or interrupted state",
}

COLUMNS_TO_DROP = ["ts", "status_time", "cycle_time"]

# Baseline min and max decision_function scores computed on training dataset
# Used to convert raw decision_function scores into a normalized 0-100% anomaly score:
#   • 100% = most anomalous (lowest raw score)
#   •   0% = most normal    (highest raw score)
BASELINE_MIN_SCORE = -0.06802
BASELINE_MAX_SCORE = 0.23254


# ---------------------------------------------------------------------------
# Global Model Storage & Loader
# ---------------------------------------------------------------------------
class ModelContainer:
    model: Any = None
    model_path: Optional[Path] = None


model_container = ModelContainer()


def load_model_from_dir(directory: Path) -> Any:
    """Find and load the trained model (.joblib) from the model directory."""
    if not directory.exists():
        raise FileNotFoundError(f"Model directory not found: {directory}")

    joblib_files = sorted(directory.glob("*.joblib"))
    if not joblib_files:
        raise FileNotFoundError(f"No .joblib model files found in: {directory}")

    # Prioritize 'IsolationForest_Model.joblib' if present, otherwise take the first
    target_file = next(
        (f for f in joblib_files if f.name == "IsolationForest_Model.joblib"),
        joblib_files[0],
    )

    t0 = time.perf_counter()
    loaded_model = joblib.load(target_file)
    elapsed = time.perf_counter() - t0
    print(f"[INFO] Loaded model '{target_file.name}' in {elapsed:.3f}s from {target_file}")
    model_container.model = loaded_model
    model_container.model_path = target_file
    return loaded_model


# ---------------------------------------------------------------------------
# Lifespan Handler (Model loading at startup)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML model into memory on startup and clean up on shutdown."""
    print("[INFO] Starting FastAPI Anomaly Detection Service...")
    try:
        load_model_from_dir(MODEL_DIR)
    except Exception as exc:
        print(f"[ERROR] Failed to load model from {MODEL_DIR}: {exc}", file=sys.stderr)
        # Service can still run, endpoints will check model availability
    yield
    print("[INFO] Shutting down FastAPI Anomaly Detection Service...")


# ---------------------------------------------------------------------------
# FastAPI App Definition
# ---------------------------------------------------------------------------
app = FastAPI(
    title="MEQYAS AI - Anomaly Detection API",
    description=(
        "Industrial telemetry anomaly detection service for MEQYAS. "
        "Predicts whether machine energy and production telemetry is anomalous, "
        "with percentage-based anomaly scores."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend / external service access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request & Response Schemas
# ---------------------------------------------------------------------------
class TelemetryReading(BaseModel):
    """Input telemetry reading for anomaly detection."""
    asset: Union[int, str] = Field(
        ...,
        description="Machine identifier: numeric (0, 1, 2) or string ('machine_1', 'machine_2', 'machine_3')",
        examples=[0],
    )
    items: float = Field(
        ...,
        description="Number of items produced in the interval",
        examples=[4.0],
    )
    status: Union[int, float, str] = Field(
        ...,
        description="Machine status: numeric (0: idle, 1: manual, 2: automatic, 3: alarm) or category string",
        examples=[2],
    )
    power_avg: float = Field(
        ...,
        description="Average power consumed in kW",
        examples=[2.0],
    )
    alarm: int = Field(
        ...,
        description="Alarm flag (0 = no alarm, 1 = alarm active)",
        examples=[0],
    )
    product: Union[int, str] = Field(
        ...,
        description="Product identifier: numeric (0–13) or string ('product_1' – 'product_14')",
        examples=[0],
    )
    ts: Optional[str] = Field(
        None,
        description="Optional timestamp (YYYY-MM-DD HH:mm:ss+TZ), ignored during inference",
        examples=["2022-08-31 22:00:00+00:00"],
    )
    status_time: Optional[float] = Field(
        None,
        description="Optional time spent in status (seconds), ignored during inference",
        examples=[43.0],
    )
    cycle_time: Optional[float] = Field(
        None,
        description="Optional cycle time (seconds), ignored during inference",
        examples=[0.0],
    )


class SinglePredictionResponse(BaseModel):
    """Output prediction result for a single telemetry reading."""
    is_anomaly: bool = Field(
        ...,
        description="Whether this reading is classified as an anomaly (True) or normal (False)",
    )
    prediction: str = Field(
        ...,
        description="Class label: 'anomaly' or 'normal'",
    )
    anomaly_percentage: float = Field(
        ...,
        description="Anomaly score as a percentage (0% = completely normal, 100% = most anomalous)",
    )
    raw_score: float = Field(
        ...,
        description="Raw IsolationForest decision_function score (negative = anomaly, positive = normal)",
    )
    message: str = Field(
        ...,
        description="Human-readable assessment of the reading",
    )


class BatchPredictionRequest(BaseModel):
    """Batch prediction request containing multiple telemetry readings."""
    readings: List[TelemetryReading] = Field(
        ...,
        description="List of telemetry readings to evaluate",
    )


class BatchPredictionResponse(BaseModel):
    """Output for batch prediction requests."""
    total_readings: int
    anomalies_count: int
    normal_count: int
    anomaly_rate_pct: float
    average_anomaly_percentage: float
    results: List[SinglePredictionResponse]


# ---------------------------------------------------------------------------
# Data Preprocessing & Scoring Helpers
# ---------------------------------------------------------------------------
def normalize_asset(val: Union[int, str]) -> str:
    """Ensure machine asset is in format 'machine_X'."""
    val_str = str(val).strip()
    if val_str.startswith("machine_"):
        return val_str
    try:
        # Numeric index 0-based -> machine_1, machine_2...
        return f"machine_{int(float(val_str)) + 1}"
    except (ValueError, TypeError):
        return val_str


def normalize_product(val: Union[int, str]) -> str:
    """Ensure product is in format 'product_X'."""
    val_str = str(val).strip()
    if val_str.startswith("product_"):
        return val_str
    try:
        # Numeric index 0-based -> product_1, product_2...
        return f"product_{int(float(val_str)) + 1}"
    except (ValueError, TypeError):
        return val_str


def normalize_status(val: Union[int, float, str]) -> str:
    """Convert status code to human-readable label if numeric."""
    val_str = str(val).strip()
    try:
        int_code = int(float(val_str))
        if int_code in STATUS_MAP:
            return STATUS_MAP[int_code]
    except (ValueError, TypeError):
        pass
    return val_str.lower()


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same transformations used during model training."""
    df = df.copy()

    # Normalize categorical columns
    if "asset" in df.columns:
        df["asset"] = df["asset"].apply(normalize_asset)
    if "product" in df.columns:
        df["product"] = df["product"].apply(normalize_product)
    if "status" in df.columns:
        df["status"] = df["status"].apply(normalize_status)

    # Drop columns not used by the trained model
    drop_cols = [c for c in COLUMNS_TO_DROP if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # Ensure required numeric columns are present and numeric
    for col in ["items", "power_avg", "alarm"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df


def calculate_anomaly_percentage(
    raw_scores: np.ndarray,
    min_score: float = BASELINE_MIN_SCORE,
    max_score: float = BASELINE_MAX_SCORE,
) -> np.ndarray:
    """
    Convert IsolationForest decision_function scores to a 0–100 % anomaly percentage.
    Lower raw scores indicate higher anomaly severity:
        • 100 % = highest anomaly
        •   0 % = perfectly normal
    """
    if max_score == min_score:
        return np.zeros_like(raw_scores, dtype=float)

    pct = (max_score - raw_scores) / (max_score - min_score) * 100.0
    # Clip between 0.0% and 100.0%
    return np.round(np.clip(pct, 0.0, 100.0), 2)


def get_loaded_model():
    """Retrieve the loaded model or raise HTTP 503 if unavailable."""
    if model_container.model is None:
        try:
            load_model_from_dir(MODEL_DIR)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model could not be loaded from '{MODEL_DIR}': {e}",
            )
    return model_container.model


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------
@app.get("/", tags=["Info"])
def root():
    """Service overview, status, and loaded model metadata."""
    model_loaded = model_container.model is not None
    model_name = model_container.model_path.name if model_container.model_path else "None"
    return {
        "service": "MEQYAS AI - Anomaly Detection API",
        "status": "online",
        "model_loaded": model_loaded,
        "model_name": model_name,
        "docs_url": "/docs",
        "endpoints": {
            "health": "/health",
            "predict_single": "POST /predict",
            "predict_batch": "POST /predict/batch",
            "predict_csv": "POST /predict/csv",
        },
    }


@app.get("/health", tags=["Health"])
def health_check():
    """Health check endpoint for container and orchestration monitoring."""
    model_loaded = model_container.model is not None
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "model_file": str(model_container.model_path) if model_container.model_path else None,
    }


@app.post("/predict", response_model=SinglePredictionResponse, tags=["Inference"])
def predict_single(reading: TelemetryReading):
    """
    Predict whether a single telemetry reading is an anomaly or normal,
    along with its anomaly percentage.
    """
    model = get_loaded_model()

    # Convert Pydantic object to DataFrame and preprocess
    input_df = pd.DataFrame([reading.model_dump()])
    X = prepare_dataframe(input_df)

    try:
        raw_pred = model.predict(X)[0]               # 1 = normal, -1 = anomaly
        raw_score = float(model.decision_function(X)[0])
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Inference error: {exc}",
        )

    is_anomaly = bool(raw_pred == -1)
    anomaly_pct = float(calculate_anomaly_percentage(np.array([raw_score]))[0])
    pred_label = "anomaly" if is_anomaly else "normal"

    message = (
        f"Anomaly detected with {anomaly_pct:.2f}% anomaly score."
        if is_anomaly
        else f"Reading is normal with {anomaly_pct:.2f}% anomaly score."
    )

    return SinglePredictionResponse(
        is_anomaly=is_anomaly,
        prediction=pred_label,
        anomaly_percentage=anomaly_pct,
        raw_score=round(raw_score, 6),
        message=message,
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch(batch: BatchPredictionRequest):
    """
    Predict anomaly status and percentages for multiple telemetry readings.
    """
    if not batch.readings:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The readings list cannot be empty.",
        )

    model = get_loaded_model()

    input_df = pd.DataFrame([r.model_dump() for r in batch.readings])
    X = prepare_dataframe(input_df)

    try:
        raw_preds = model.predict(X)
        raw_scores = model.decision_function(X)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Inference error during batch execution: {exc}",
        )

    anomaly_pcts = calculate_anomaly_percentage(raw_scores)

    results: List[SinglePredictionResponse] = []
    anomalies_count = 0

    for pred, score, pct in zip(raw_preds, raw_scores, anomaly_pcts):
        is_anomaly = bool(pred == -1)
        if is_anomaly:
            anomalies_count += 1
        label = "anomaly" if is_anomaly else "normal"
        msg = (
            f"Anomaly detected with {pct:.2f}% anomaly score."
            if is_anomaly
            else f"Reading is normal with {pct:.2f}% anomaly score."
        )
        results.append(
            SinglePredictionResponse(
                is_anomaly=is_anomaly,
                prediction=label,
                anomaly_percentage=float(pct),
                raw_score=round(float(score), 6),
                message=msg,
            )
        )

    total = len(results)
    normal_count = total - anomalies_count
    rate = round((anomalies_count / total) * 100.0, 2)
    avg_pct = round(float(np.mean(anomaly_pcts)), 2)

    return BatchPredictionResponse(
        total_readings=total,
        anomalies_count=anomalies_count,
        normal_count=normal_count,
        anomaly_rate_pct=rate,
        average_anomaly_percentage=avg_pct,
        results=results,
    )


@app.post("/predict/csv", tags=["Inference"])
async def predict_from_csv(file: UploadFile = File(...)):
    """
    Upload a CSV file of telemetry readings to get anomaly predictions and summary stats.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only CSV files (.csv) are supported.",
        )

    model = get_loaded_model()

    try:
        raw_df = pd.read_csv(file.file)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not parse CSV file: {exc}",
        )

    required_cols = {"asset", "items", "status", "power_avg", "alarm", "product"}
    missing = required_cols - set(raw_df.columns)
    if missing:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Missing required columns in CSV: {sorted(missing)}",
        )

    X = prepare_dataframe(raw_df)

    try:
        preds = model.predict(X)
        scores = model.decision_function(X)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Inference error on CSV data: {exc}",
        )

    pcts = calculate_anomaly_percentage(scores)
    total = len(preds)
    anomalies_count = int((preds == -1).sum())
    normal_count = total - anomalies_count

    sample_results = []
    for i in range(min(total, 50)):  # return up to first 50 row details
        is_anom = bool(preds[i] == -1)
        sample_results.append({
            "row_index": i,
            "is_anomaly": is_anom,
            "prediction": "anomaly" if is_anom else "normal",
            "anomaly_percentage": float(pcts[i]),
            "raw_score": round(float(scores[i]), 6),
        })

    return {
        "filename": file.filename,
        "total_rows": total,
        "anomalies_count": anomalies_count,
        "normal_count": normal_count,
        "anomaly_rate_pct": round((anomalies_count / total) * 100.0, 2),
        "avg_anomaly_score_pct": round(float(np.mean(pcts)), 2),
        "max_anomaly_score_pct": round(float(np.max(pcts)), 2),
        "first_50_predictions": sample_results,
    }


# ---------------------------------------------------------------------------
# Server Startup (Uvicorn)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    print(f"[INFO] Starting Uvicorn server on http://{host}:{port}")
    uvicorn.run("main:app", host=host, port=port, reload=True)
