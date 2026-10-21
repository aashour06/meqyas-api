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
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
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
# Used to convert raw decision_function scores into a calibrated 0-100% anomaly score:
#   • raw_score >= 0 (Normal inlier): mapped to 0% - 50%
#   • raw_score < 0  (Anomaly outlier): mapped to 50% - 100%
# The IsolationForest decision boundary (raw_score == 0) aligns with 50.0%.
BASELINE_MIN_SCORE = -0.06802
BASELINE_MAX_SCORE = 0.23254
DEFAULT_ANOMALY_THRESHOLD = 50.0  # Anomaly score percentage threshold
MAX_RATED_POWER_KW = 10.0         # Maximum rated machine power consumption in dataset (kW)


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
        "with percentage-based anomaly scores and multi-criteria diagnostic evaluation."
    ),
    version="1.1.0",
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
        description="Machine status: numeric (0: idle, 1: manual, 2: automatic, 3: alarm) or category string ('automatic', 'manual', 'alarm', 'idle')",
        examples=[2],
    )
    power_avg: float = Field(
        ...,
        description="Average power consumed in kW",
        examples=[2.0],
    )
    alarm: Union[int, float, str, bool] = Field(
        ...,
        description="Alarm flag (0 = no alarm, 1 = alarm active, or boolean true/false)",
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
    threshold: Optional[float] = Field(
        default=DEFAULT_ANOMALY_THRESHOLD,
        description="Anomaly score percentage threshold (0-100%) above which reading is classified as an anomaly. Default is 50.0%.",
        examples=[50.0],
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
        description="Raw IsolationForest decision_function score (negative = outlier, positive = inlier)",
    )
    message: str = Field(
        ...,
        description="Human-readable assessment of the reading",
    )
    anomaly_reasons: Optional[List[str]] = Field(
        default=None,
        description="List of detected conditions triggering anomaly classification",
    )


class BatchPredictionRequest(BaseModel):
    """Batch prediction request containing multiple telemetry readings."""
    readings: List[TelemetryReading] = Field(
        ...,
        description="List of telemetry readings to evaluate",
    )
    threshold: Optional[float] = Field(
        default=DEFAULT_ANOMALY_THRESHOLD,
        description="Anomaly percentage threshold (0-100%) for batch evaluation (default: 50.0%)",
        examples=[50.0],
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
# Data Preprocessing & Normalization Helpers
# ---------------------------------------------------------------------------
def normalize_asset(val: Union[int, float, str]) -> str:
    """Ensure machine asset is in format 'machine_X'."""
    val_str = str(val).strip().lower()
    if val_str.startswith("machine_"):
        return val_str
    if val_str.startswith("machine"):
        num_part = val_str.replace("machine", "").replace("_", "").replace(" ", "").strip()
        if num_part:
            return f"machine_{num_part}"
    try:
        num = int(float(val_str))
        # Handle 0-based indexing (0, 1, 2) -> machine_1, machine_2, machine_3
        # Handle 1-based indexing (1, 2, 3): 3 -> machine_3
        if num == 3:
            return "machine_3"
        elif num in (0, 1, 2):
            return f"machine_{num + 1}"
        else:
            return f"machine_{num}"
    except (ValueError, TypeError):
        return val_str


def normalize_product(val: Union[int, float, str]) -> str:
    """Ensure product is in format 'product_X'."""
    val_str = str(val).strip().lower()
    if val_str.startswith("product_"):
        return val_str
    if val_str.startswith("product"):
        num_part = val_str.replace("product", "").replace("_", "").replace(" ", "").strip()
        if num_part:
            return f"product_{num_part}"
    try:
        num = int(float(val_str))
        # Handle 0-based indexing 0..13 -> product_1..product_14
        if num == 14:
            return "product_14"
        elif 0 <= num <= 13:
            return f"product_{num + 1}"
        else:
            return f"product_{num}"
    except (ValueError, TypeError):
        return val_str


def normalize_status(val: Union[int, float, str]) -> str:
    """
    Convert status code or string to one of the exact categories used during training:
      - 'alarm or interrupted state'
      - 'automatic'
      - 'manual'
      - 'idle'
    """
    val_str = str(val).strip().lower()

    # Direct keyword matching for alarm / interrupted conditions
    if any(k in val_str for k in ("alarm", "interrupted", "fault", "error", "alert", "breakdown", "warning")):
        return "alarm or interrupted state"
    if any(k in val_str for k in ("auto", "automatic", "running", "normal")):
        return "automatic"
    if "manual" in val_str:
        return "manual"
    if any(k in val_str for k in ("idle", "stopped", "standby", "off")):
        return "idle"

    # Numeric code lookup (0: idle, 1: manual, 2: automatic, 3: alarm or interrupted state)
    try:
        int_code = int(float(val_str))
        if int_code in STATUS_MAP:
            return STATUS_MAP[int_code]
    except (ValueError, TypeError):
        pass

    return val_str


def normalize_alarm(val: Any) -> int:
    """Ensure alarm flag is strictly integer 0 or 1."""
    if isinstance(val, bool):
        return 1 if val else 0
    val_str = str(val).strip().lower()
    if val_str in ("1", "1.0", "true", "yes", "alarm", "active"):
        return 1
    return 0


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

    # Normalize alarm column
    if "alarm" in df.columns:
        df["alarm"] = df["alarm"].apply(normalize_alarm)

    # Drop columns not used by the trained model
    drop_cols = [c for c in COLUMNS_TO_DROP if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # Ensure required numeric columns are present and numeric
    for col in ["items", "power_avg", "alarm"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df


# ---------------------------------------------------------------------------
# Anomaly Scoring & Evaluation
# ---------------------------------------------------------------------------
def calculate_anomaly_percentage(
    raw_scores: Union[np.ndarray, List[float], float],
    min_score: float = BASELINE_MIN_SCORE,
    max_score: float = BASELINE_MAX_SCORE,
) -> np.ndarray:
    """
    Convert IsolationForest decision_function scores to a calibrated 0–100% anomaly score:
      • raw_score >= 0 (Normal / inlier): mapped smoothly to 0% - 50%
        (max_score -> 0.0%, 0.114 -> ~25.5%, 0.0 -> 50.0%)
      • raw_score < 0  (Anomaly / outlier): mapped smoothly to 50% - 100%
        (0.0 -> 50.0%, min_score -> 100.0%)
    This guarantees that the decision boundary (raw_score = 0) perfectly aligns with 50.0%.
    """
    scores = np.asarray(raw_scores, dtype=float)
    pcts = np.zeros_like(scores, dtype=float)

    # Inlier region (scores >= 0): 0% to 50%
    inlier_mask = scores >= 0
    if max_score > 0:
        pcts[inlier_mask] = (max_score - scores[inlier_mask]) / max_score * 50.0
    else:
        pcts[inlier_mask] = 0.0

    # Outlier region (scores < 0): 50% to 100%
    outlier_mask = scores < 0
    if min_score < 0:
        pcts[outlier_mask] = 50.0 + ((-scores[outlier_mask]) / (-min_score)) * 50.0
    else:
        pcts[outlier_mask] = 100.0

    return np.round(np.clip(pcts, 0.0, 100.0), 2)


def evaluate_anomaly(
    raw_pred: int,
    raw_score: float,
    anomaly_pct: float,
    alarm: int,
    status_str: str,
    power_avg: float,
    items: float,
    threshold: float = DEFAULT_ANOMALY_THRESHOLD,
) -> tuple[bool, str, float, str, List[str]]:
    """
    Comprehensive multi-criteria anomaly evaluation:
      1. Active alarm signal (alarm == 1)
      2. Alarm or interrupted machine state (status in alarm/interrupted)
      3. Severe operational out-of-bound conditions (power surge > 10 kW, negative readings)
      4. Isolation Forest model prediction (raw_pred == -1, i.e., raw_score < 0)
      5. Anomaly percentage exceeding threshold (anomaly_pct >= threshold)
    """
    reasons = []

    # Check 1: Active alarm flag
    if alarm == 1:
        reasons.append("Active machine alarm signal (alarm=1)")

    # Check 2: Alarm or interrupted machine status
    if status_str == "alarm or interrupted state":
        reasons.append("Machine in alarm or interrupted state (status=3)")

    # Check 3: Power consumption exceeds rated machine maximum
    if power_avg > MAX_RATED_POWER_KW:
        reasons.append(f"Power consumption exceeds rated limit ({power_avg:.2f} kW > {MAX_RATED_POWER_KW:.1f} kW)")

    # Check 4: Sensor telemetry invalid negative readings
    if power_avg < 0:
        reasons.append(f"Invalid negative power consumption ({power_avg:.2f} kW)")
    if items < 0:
        reasons.append(f"Invalid negative items produced ({items:.2f})")

    # Check 5: Isolation Forest model prediction
    if raw_pred == -1:
        reasons.append(f"Isolation Forest classified reading as outlier (raw_score={raw_score:.6f} < 0)")

    # Check 6: Anomaly score meets or exceeds threshold
    if anomaly_pct >= threshold:
        reasons.append(f"Anomaly score ({anomaly_pct:.2f}%) meets or exceeds threshold ({threshold:.2f}%)")

    is_anomaly = len(reasons) > 0

    # Ensure final score reflects anomaly severity when explicit alarm/conditions trigger
    final_pct = anomaly_pct
    if alarm == 1:
        final_pct = max(final_pct, 85.0)
    elif status_str == "alarm or interrupted state":
        final_pct = max(final_pct, 80.0)
    elif power_avg > MAX_RATED_POWER_KW:
        final_pct = max(final_pct, 85.0)
    elif power_avg < 0 or items < 0:
        final_pct = max(final_pct, 90.0)

    pred_label = "anomaly" if is_anomaly else "normal"

    if is_anomaly:
        primary_reason = reasons[0]
        message = f"Anomaly detected: {primary_reason} with {final_pct:.2f}% anomaly score."
    else:
        message = f"Reading is normal with {final_pct:.2f}% anomaly score."

    return is_anomaly, pred_label, float(final_pct), message, reasons


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
        "default_threshold_pct": DEFAULT_ANOMALY_THRESHOLD,
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
    evaluating both ML Isolation Forest scores and operational conditions (alarms, status, power extremes).
    """
    model = get_loaded_model()

    # Pre-extract input fields for operational checks
    alarm_val = normalize_alarm(reading.alarm)
    status_str = normalize_status(reading.status)
    power_val = float(reading.power_avg)
    items_val = float(reading.items)
    threshold_val = float(reading.threshold) if reading.threshold is not None else DEFAULT_ANOMALY_THRESHOLD

    # Convert Pydantic object to DataFrame and preprocess for ML pipeline
    input_dict = reading.model_dump()
    input_dict["alarm"] = alarm_val
    input_df = pd.DataFrame([input_dict])
    X = prepare_dataframe(input_df)

    try:
        raw_pred = int(model.predict(X)[0])               # 1 = inlier, -1 = outlier
        raw_score = float(model.decision_function(X)[0])
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Inference error: {exc}",
        )

    base_pct = float(calculate_anomaly_percentage(np.array([raw_score]))[0])

    is_anomaly, pred_label, final_pct, message, reasons = evaluate_anomaly(
        raw_pred=raw_pred,
        raw_score=raw_score,
        anomaly_pct=base_pct,
        alarm=alarm_val,
        status_str=status_str,
        power_avg=power_val,
        items=items_val,
        threshold=threshold_val,
    )

    return SinglePredictionResponse(
        is_anomaly=is_anomaly,
        prediction=pred_label,
        anomaly_percentage=final_pct,
        raw_score=round(raw_score, 6),
        message=message,
        anomaly_reasons=reasons if is_anomaly else None,
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
    threshold_val = float(batch.threshold) if batch.threshold is not None else DEFAULT_ANOMALY_THRESHOLD

    prepared_records = []
    metadata_list = []
    for r in batch.readings:
        alarm_val = normalize_alarm(r.alarm)
        status_str = normalize_status(r.status)
        power_val = float(r.power_avg)
        items_val = float(r.items)
        r_thresh = float(r.threshold) if r.threshold is not None else threshold_val

        r_dict = r.model_dump()
        r_dict["alarm"] = alarm_val
        prepared_records.append(r_dict)
        metadata_list.append({
            "alarm": alarm_val,
            "status_str": status_str,
            "power_avg": power_val,
            "items": items_val,
            "threshold": r_thresh,
        })

    input_df = pd.DataFrame(prepared_records)
    X = prepare_dataframe(input_df)

    try:
        raw_preds = model.predict(X)
        raw_scores = model.decision_function(X)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Inference error during batch execution: {exc}",
        )

    base_pcts = calculate_anomaly_percentage(raw_scores)

    results: List[SinglePredictionResponse] = []
    anomalies_count = 0
    all_final_pcts = []

    for pred, score, base_pct, meta in zip(raw_preds, raw_scores, base_pcts, metadata_list):
        is_anomaly, pred_label, final_pct, message, reasons = evaluate_anomaly(
            raw_pred=int(pred),
            raw_score=float(score),
            anomaly_pct=float(base_pct),
            alarm=meta["alarm"],
            status_str=meta["status_str"],
            power_avg=meta["power_avg"],
            items=meta["items"],
            threshold=meta["threshold"],
        )
        if is_anomaly:
            anomalies_count += 1
        all_final_pcts.append(final_pct)

        results.append(
            SinglePredictionResponse(
                is_anomaly=is_anomaly,
                prediction=pred_label,
                anomaly_percentage=final_pct,
                raw_score=round(float(score), 6),
                message=message,
                anomaly_reasons=reasons if is_anomaly else None,
            )
        )

    total = len(results)
    normal_count = total - anomalies_count
    rate = round((anomalies_count / total) * 100.0, 2)
    avg_pct = round(float(np.mean(all_final_pcts)), 2)

    return BatchPredictionResponse(
        total_readings=total,
        anomalies_count=anomalies_count,
        normal_count=normal_count,
        anomaly_rate_pct=rate,
        average_anomaly_percentage=avg_pct,
        results=results,
    )


@app.post("/predict/csv", tags=["Inference"])
async def predict_from_csv(
    file: UploadFile = File(...),
    threshold: float = Query(DEFAULT_ANOMALY_THRESHOLD, description="Anomaly percentage threshold (0-100%)"),
):
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
        raw_preds = model.predict(X)
        raw_scores = model.decision_function(X)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Inference error on CSV data: {exc}",
        )

    base_pcts = calculate_anomaly_percentage(raw_scores)
    total = len(raw_preds)

    anomalies_count = 0
    final_pcts = []
    sample_results = []

    for i in range(total):
        alarm_val = normalize_alarm(raw_df["alarm"].iloc[i]) if "alarm" in raw_df.columns else 0
        status_str = normalize_status(raw_df["status"].iloc[i]) if "status" in raw_df.columns else "automatic"
        power_val = float(raw_df["power_avg"].iloc[i]) if "power_avg" in raw_df.columns else 0.0
        items_val = float(raw_df["items"].iloc[i]) if "items" in raw_df.columns else 0.0

        is_anom, label, final_pct, msg, reasons = evaluate_anomaly(
            raw_pred=int(raw_preds[i]),
            raw_score=float(raw_scores[i]),
            anomaly_pct=float(base_pcts[i]),
            alarm=alarm_val,
            status_str=status_str,
            power_avg=power_val,
            items=items_val,
            threshold=threshold,
        )
        if is_anom:
            anomalies_count += 1
        final_pcts.append(final_pct)

        if i < 50:  # Include detailed sample for first 50 rows
            sample_results.append({
                "row_index": i,
                "is_anomaly": is_anom,
                "prediction": label,
                "anomaly_percentage": final_pct,
                "raw_score": round(float(raw_scores[i]), 6),
                "reasons": reasons if is_anom else None,
            })

    normal_count = total - anomalies_count

    return {
        "filename": file.filename,
        "total_rows": total,
        "anomalies_count": anomalies_count,
        "normal_count": normal_count,
        "anomaly_rate_pct": round((anomalies_count / total) * 100.0, 2),
        "avg_anomaly_score_pct": round(float(np.mean(final_pcts)), 2) if final_pcts else 0.0,
        "max_anomaly_score_pct": round(float(np.max(final_pcts)), 2) if final_pcts else 0.0,
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
