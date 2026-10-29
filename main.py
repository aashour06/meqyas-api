"""
MEQYAS AI — Unified Industrial Energy Intelligence & Anomaly Detection API
===========================================================================
Single FastAPI application integrating two complementary machine learning models:
  1. Machine Telemetry Anomaly Detection: Unsupervised Isolation Forest model
     monitoring machine operation states, alarms, production volume, and power surges.
  2. Predictive Power Factor Risk Model: Supervised XGBoost pipeline forecasting
     the risk of Low Power Factor (< 0.90) 3 hours in advance from 3-phase telemetry.

Run directly:
    python main.py

Or with Uvicorn CLI:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import io
import os
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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

ISOLATION_FOREST_FILE = MODEL_DIR / "IsolationForest_Model.joblib"
POWER_MODEL_FILE = MODEL_DIR / "final_model.pkl"

# Model 1 (Isolation Forest) Constants
STATUS_MAP = {
    0: "idle",
    1: "manual",
    2: "automatic",
    3: "alarm or interrupted state",
}
COLUMNS_TO_DROP = ["ts", "status_time", "cycle_time"]
BASELINE_MIN_SCORE = -0.06802
BASELINE_MAX_SCORE = 0.23254
DEFAULT_MACHINE_THRESHOLD = 50.0  # Anomaly score percentage threshold
DEFAULT_ANOMALY_THRESHOLD = DEFAULT_MACHINE_THRESHOLD  # Backward compatibility alias
MAX_RATED_POWER_KW = 10.0

# Model 2 (XGBoost Power Factor) Constants
DEFAULT_POWER_THRESHOLD = 0.50
POWER_REQUIRED_FEATURES = [
    "hour",
    "volt_A",
    "volt_B",
    "volt_C",
    "I_A",
    "I_B",
    "I_C",
    "Total_Active_Power",
    "Total_Reactive_Power",
    "Total_Power_Factor",
    "Freq",
    "Q_P_ratio",
]


# ---------------------------------------------------------------------------
# Global Model Storage & Loaders
# ---------------------------------------------------------------------------
class ModelContainer:
    """Singleton container storing loaded models in memory."""
    # Model 1: Isolation Forest
    model: Any = None                     # Alias for backward compatibility with test_api.py
    isolation_forest: Any = None
    isolation_forest_path: Optional[Path] = None
    isolation_forest_error: Optional[str] = None
    model_path: Optional[Path] = None     # Alias for backward compatibility

    # Model 2: XGBoost Power Factor Pipeline
    power_model: Any = None
    power_model_path: Optional[Path] = None
    power_model_error: Optional[str] = None


model_container = ModelContainer()


def load_isolation_forest(path: Path = ISOLATION_FOREST_FILE) -> Any:
    """Load the trained Isolation Forest pipeline from model directory."""
    if not path.exists():
        err = f"Isolation Forest model file not found: {path}"
        model_container.isolation_forest_error = err
        raise FileNotFoundError(err)

    t0 = time.perf_counter()
    loaded = joblib.load(path)
    elapsed = time.perf_counter() - t0
    print(f"[INFO] Loaded Isolation Forest model '{path.name}' in {elapsed:.3f}s from {path}")

    model_container.isolation_forest = loaded
    model_container.isolation_forest_path = path
    model_container.isolation_forest_error = None
    model_container.model = loaded
    model_container.model_path = path
    return loaded


def load_power_model(path: Path = POWER_MODEL_FILE) -> Any:
    """Load the trained XGBoost Power Factor pipeline from model directory."""
    if not path.exists():
        err = f"Power Factor model file not found: {path}"
        model_container.power_model_error = err
        raise FileNotFoundError(err)

    t0 = time.perf_counter()
    loaded_obj = joblib.load(path)
    elapsed = time.perf_counter() - t0

    pipeline = loaded_obj.get("pipeline", loaded_obj) if isinstance(loaded_obj, dict) else loaded_obj
    print(f"[INFO] Loaded XGBoost Power Factor model '{path.name}' in {elapsed:.3f}s from {path}")

    model_container.power_model = pipeline
    model_container.power_model_path = path
    model_container.power_model_error = None
    return pipeline


def load_model_from_dir(directory: Path = MODEL_DIR) -> Any:
    """Backward compatibility loader for test_api.py that ensures all models are loaded."""
    load_all_models()
    return model_container.isolation_forest


def load_all_models() -> Tuple[bool, bool]:
    """Load both models during startup."""
    m1_ok = False
    m2_ok = False

    try:
        load_isolation_forest(ISOLATION_FOREST_FILE)
        m1_ok = True
    except Exception as exc:
        print(f"[ERROR] Failed to load Isolation Forest: {exc}", file=sys.stderr)

    try:
        load_power_model(POWER_MODEL_FILE)
        m2_ok = True
    except Exception as exc:
        print(f"[ERROR] Failed to load Power Factor model: {exc}", file=sys.stderr)

    return m1_ok, m2_ok


def get_isolation_forest_model() -> Any:
    """Retrieve loaded Isolation Forest model or attempt on-demand load."""
    if model_container.isolation_forest is None:
        try:
            return load_isolation_forest()
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Isolation Forest model is unavailable: {exc}",
            )
    return model_container.isolation_forest


def get_loaded_model() -> Any:
    """Backward compatibility helper for Isolation Forest model."""
    return get_isolation_forest_model()


def get_power_factor_model() -> Any:
    """Retrieve loaded XGBoost Power Factor model or attempt on-demand load."""
    if model_container.power_model is None:
        try:
            return load_power_model()
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Power Factor prediction model is unavailable: {exc}",
            )
    return model_container.power_model


# ---------------------------------------------------------------------------
# Lifespan Handler (Model Loading on Server Startup)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load both ML models into memory on server startup and release on shutdown."""
    print("[INFO] Starting MEQYAS AI Unified Machine Learning Service...")
    load_all_models()
    yield
    print("[INFO] Shutting down MEQYAS AI Unified Machine Learning Service...")


# ---------------------------------------------------------------------------
# FastAPI Application Definition
# ---------------------------------------------------------------------------
tags_metadata = [
    {
        "name": "System & Health",
        "description": "API status, operational readiness, and model metadata.",
    },
    {
        "name": "Machine Anomaly Detection (Isolation Forest)",
        "description": "Unsupervised telemetry anomaly detection for machine operations, alarms, and power spikes.",
    },
    {
        "name": "Power Factor Prediction (XGBoost)",
        "description": "Predictive forecasting of Low Power Factor (< 0.90) risk 3 hours into the future.",
    },
]

app = FastAPI(
    title="MEQYAS AI — Unified Industrial Energy & Anomaly Intelligence API",
    description=(
        "Production-grade unified AI service for MEQYAS smart industrial monitoring. "
        "Integrates unsupervised machine-level anomaly detection (Isolation Forest) "
        "and predictive 3-phase power quality forecasting (XGBoost)."
    ),
    version="2.0.0",
    lifespan=lifespan,
    openapi_tags=tags_metadata,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas: Model 1 (Machine Telemetry Anomaly Detection)
# ---------------------------------------------------------------------------
class TelemetryReading(BaseModel):
    """Input telemetry reading for machine-level anomaly detection."""
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
        default=DEFAULT_MACHINE_THRESHOLD,
        description="Anomaly score percentage threshold (0-100%) above which reading is classified as an anomaly. Default is 50.0%.",
        examples=[50.0],
    )


class SinglePredictionResponse(BaseModel):
    """Output prediction result for a single machine telemetry reading."""
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
        description="Continuously scaled anomaly percentage (0% = completely normal, 100% = most anomalous)",
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
    """Batch prediction request containing multiple machine telemetry readings."""
    readings: List[TelemetryReading] = Field(
        ...,
        min_length=1,
        description="List of telemetry readings to evaluate",
    )
    threshold: Optional[float] = Field(
        default=DEFAULT_MACHINE_THRESHOLD,
        description="Anomaly percentage threshold (0-100%) for batch evaluation (default: 50.0%)",
        examples=[50.0],
    )


class BatchPredictionResponse(BaseModel):
    """Output summary for machine batch prediction requests."""
    total_readings: int
    anomalies_count: int
    normal_count: int
    anomaly_rate_pct: float
    average_anomaly_percentage: float
    results: List[SinglePredictionResponse]


# Aliases for explicit naming
MachineTelemetryReading = TelemetryReading
MachinePredictionResponse = SinglePredictionResponse
MachineBatchRequest = BatchPredictionRequest
MachineBatchResponse = BatchPredictionResponse


# ---------------------------------------------------------------------------
# Schemas: Model 2 (Predictive Power Factor Risk)
# ---------------------------------------------------------------------------
class PowerReading(BaseModel):
    """Input 3-phase telemetry reading for Power Factor risk forecasting."""
    volt_A: float = Field(..., description="Phase A Voltage (V)", examples=[126.8])
    volt_B: float = Field(..., description="Phase B Voltage (V)", examples=[125.4])
    volt_C: float = Field(..., description="Phase C Voltage (V)", examples=[126.2])
    I_A: float = Field(..., description="Phase A Current (A)", examples=[0.35])
    I_B: float = Field(..., description="Phase B Current (A)", examples=[0.42])
    I_C: float = Field(..., description="Phase C Current (A)", examples=[0.21])
    Total_Active_Power: Optional[float] = Field(None, description="Active Power P (kW)", examples=[0.087])
    Total_Reactive_Power: Optional[float] = Field(None, description="Reactive Power Q (kVAR)", examples=[0.015])
    Total_Power_Factor: Optional[float] = Field(None, description="Power Factor (0.0-1.0 or meter 1.0-2.0)", examples=[0.95])
    Freq: Optional[float] = Field(None, description="Grid Frequency (Hz)", examples=[60.0])

    # Hardware meter aliases (from raw industrial power monitors)
    P_T: Optional[float] = Field(None, description="Hardware alias for Total_Active_Power")
    Q_T: Optional[float] = Field(None, description="Hardware alias for Total_Reactive_Power")
    FP_T: Optional[float] = Field(None, description="Hardware alias for Total_Power_Factor")
    Frec: Optional[float] = Field(None, description="Hardware alias for Freq")

    hour: Optional[int] = Field(None, description="Hour of day (0-23). Auto-derived if omitted", examples=[14])
    Timestamp: Optional[str] = Field(None, description="ISO timestamp string (e.g. '2024-08-06 14:00:00')", examples=["2024-08-06 14:00:00"])
    Q_P_ratio: Optional[float] = Field(None, description="Reactive/Active power ratio (auto-computed if omitted)")
    threshold: Optional[float] = Field(DEFAULT_POWER_THRESHOLD, description="Decision threshold (default 0.50)", examples=[0.50])


class PowerPredictionResponse(BaseModel):
    """Output prediction for 3-hour Power Factor risk."""
    prediction: str = Field(..., description="Classification label: 'Normal Power Factor' or 'Low Power Factor Risk (Anomaly)'")
    is_low_pf_risk: bool = Field(..., description="True if risk probability >= threshold")
    risk_probability: float = Field(..., description="Predicted probability (0.0 - 1.0)")
    risk_percentage: float = Field(..., description="Predicted risk percentage (0.0% - 100.0%)")
    threshold_used: float = Field(..., description="Probability threshold applied")
    actual_power_factor: float = Field(..., description="Decoded physical power factor (0.0 - 1.0)")
    computed_q_p_ratio: float = Field(..., description="Calculated Q/P ratio")
    message: str = Field(..., description="Human-readable operational advisory")


class PowerBatchRequest(BaseModel):
    """Batch prediction request for multiple power telemetry readings."""
    readings: List[PowerReading] = Field(..., min_length=1, description="List of power telemetry readings")
    threshold: Optional[float] = Field(DEFAULT_POWER_THRESHOLD, description="Decision threshold applied across batch")


class PowerBatchResponse(BaseModel):
    """Output summary for power batch predictions."""
    total: int = Field(..., description="Total readings evaluated")
    low_pf_risk_count: int = Field(..., description="Count of low PF risk readings")
    normal_count: int = Field(..., description="Count of normal PF readings")
    risk_rate_pct: float = Field(..., description="Percentage of readings at risk")
    results: List[PowerPredictionResponse] = Field(..., description="Detailed prediction list")


class PowerFileUploadResponse(BaseModel):
    """Output summary for CSV or Excel file prediction."""
    filename: str
    total_rows: int
    low_pf_risk_count: int
    normal_count: int
    risk_rate_pct: float
    average_risk_probability_pct: float
    first_50_predictions: List[Dict[str, Any]]


# Aliases
PowerTelemetryReading = PowerReading


# ---------------------------------------------------------------------------
# Preprocessing Helpers: Model 1 (Isolation Forest)
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
        if num == 14:
            return "product_14"
        elif 0 <= num <= 13:
            return f"product_{num + 1}"
        else:
            return f"product_{num}"
    except (ValueError, TypeError):
        return val_str


def normalize_status(val: Union[int, float, str]) -> str:
    """Convert status code or string to canonical training categories."""
    val_str = str(val).strip().lower()

    if any(k in val_str for k in ("alarm", "interrupted", "fault", "error", "alert", "breakdown", "warning")):
        return "alarm or interrupted state"
    if any(k in val_str for k in ("auto", "automatic", "running", "normal")):
        return "automatic"
    if "manual" in val_str:
        return "manual"
    if any(k in val_str for k in ("idle", "stopped", "standby", "off")):
        return "idle"

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

    if "asset" in df.columns:
        df["asset"] = df["asset"].apply(normalize_asset)
    if "product" in df.columns:
        df["product"] = df["product"].apply(normalize_product)
    if "status" in df.columns:
        df["status"] = df["status"].apply(normalize_status)
    if "alarm" in df.columns:
        df["alarm"] = df["alarm"].apply(normalize_alarm)

    drop_cols = [c for c in COLUMNS_TO_DROP if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    for col in ["items", "power_avg", "alarm"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df


def calculate_ml_percentage(
    raw_scores: Union[np.ndarray, List[float], float],
    min_score: float = BASELINE_MIN_SCORE,
    max_score: float = BASELINE_MAX_SCORE,
) -> np.ndarray:
    """Convert raw IsolationForest scores to a calibrated 0-100% anomaly score."""
    scores = np.asarray(raw_scores, dtype=float)
    pcts = np.zeros_like(scores, dtype=float)

    inlier_mask = scores >= 0
    if max_score > 0:
        pcts[inlier_mask] = (max_score - scores[inlier_mask]) / max_score * 50.0

    outlier_mask = scores < 0
    if min_score < 0:
        pcts[outlier_mask] = 50.0 + ((-scores[outlier_mask]) / (-min_score)) * 50.0
    else:
        pcts[outlier_mask] = 100.0

    return np.round(np.clip(pcts, 0.0, 100.0), 2)


def calculate_anomaly_percentage(
    raw_scores: Union[np.ndarray, List[float], float],
    min_score: float = BASELINE_MIN_SCORE,
    max_score: float = BASELINE_MAX_SCORE,
) -> np.ndarray:
    """Backward compatibility alias."""
    return calculate_ml_percentage(raw_scores, min_score, max_score)


def compute_operational_risk(
    alarm: int,
    status_str: str,
    power_avg: float,
    items: float,
) -> Tuple[float, List[str]]:
    """Compute physical telemetry risk factors."""
    reasons = []
    risk_factors = []

    if alarm == 1:
        reasons.append("Active machine alarm signal (alarm=1)")
        alarm_risk = 35.0 + min(15.0, max(0.0, power_avg - 2.0) * 2.0)
        risk_factors.append(alarm_risk)

    if status_str == "alarm or interrupted state":
        reasons.append("Machine in alarm or interrupted state (status=3)")
        status_risk = 25.0 + min(15.0, max(0.0, power_avg) * 1.5)
        risk_factors.append(status_risk)
    elif status_str == "idle":
        if power_avg > 1.0:
            reasons.append(f"Excess power draw while machine is idle ({power_avg:.2f} kW)")
            risk_factors.append(min(35.0, power_avg * 8.0))
        if items > 0:
            reasons.append(f"Items reported produced while machine is idle ({items:.1f} items)")
            risk_factors.append(min(30.0, 10.0 + items * 2.0))

    if power_avg < 0:
        reasons.append(f"Invalid negative power consumption ({power_avg:.2f} kW)")
        risk_factors.append(55.0)
    elif power_avg > MAX_RATED_POWER_KW:
        excess = power_avg - MAX_RATED_POWER_KW
        power_risk = 30.0 + min(40.0, (excess / 10.0) * 20.0)
        reasons.append(f"Power consumption exceeds rated limit ({power_avg:.2f} kW > {MAX_RATED_POWER_KW:.1f} kW)")
        risk_factors.append(power_risk)
    elif power_avg > 4.0:
        elevated_risk = ((power_avg - 4.0) / 6.0) * 12.0
        risk_factors.append(elevated_risk)

    if items < 0:
        reasons.append(f"Invalid negative items produced ({items:.2f})")
        risk_factors.append(55.0)
    elif items > 15.0:
        excess_items = min(1.0, (items - 15.0) / 35.0)
        item_risk = 15.0 + excess_items * 20.0
        reasons.append(f"Abnormally high item production ({items:.1f} items in interval)")
        risk_factors.append(item_risk)

    if not risk_factors:
        return 0.0, reasons

    rem = 1.0
    for rf in risk_factors:
        rem *= (1.0 - min(100.0, rf) / 100.0)
    combined_risk = (1.0 - rem) * 100.0

    return round(combined_risk, 2), reasons


def evaluate_anomaly(
    raw_pred: int,
    raw_score: float,
    ml_pct: float,
    alarm: int,
    status_str: str,
    power_avg: float,
    items: float,
    threshold: float = DEFAULT_MACHINE_THRESHOLD,
) -> Tuple[bool, str, float, str, List[str]]:
    """Multi-criteria anomaly evaluation."""
    op_risk, op_reasons = compute_operational_risk(alarm, status_str, power_avg, items)
    reasons = list(op_reasons)

    if raw_pred == -1:
        reasons.append(f"Isolation Forest classified reading as outlier (raw_score={raw_score:.6f} < 0)")

    if op_risk > 0:
        combined_pct = 100.0 - ((100.0 - ml_pct) * (100.0 - op_risk) / 100.0)
        has_hard_fault = (
            alarm == 1
            or status_str == "alarm or interrupted state"
            or power_avg > MAX_RATED_POWER_KW
            or power_avg < 0
            or items < 0
        )
        if has_hard_fault:
            final_pct = 50.0 + (combined_pct / 100.0) * 48.0
        else:
            final_pct = combined_pct
    else:
        final_pct = ml_pct

    final_pct = round(float(np.clip(final_pct, 0.0, 100.0)), 2)

    if final_pct >= threshold and not any("threshold" in r for r in reasons):
        if not op_reasons and raw_pred != -1:
            reasons.append(f"Anomaly score ({final_pct:.2f}%) meets or exceeds threshold ({threshold:.2f}%)")

    is_anomaly = len(reasons) > 0 or final_pct >= threshold or raw_pred == -1
    pred_label = "anomaly" if is_anomaly else "normal"

    if is_anomaly:
        primary_reason = reasons[0] if reasons else f"Score {final_pct:.2f}% >= {threshold:.2f}%"
        message = f"Anomaly detected: {primary_reason} with {final_pct:.2f}% anomaly score."
    else:
        message = f"Reading is normal with {final_pct:.2f}% anomaly score."

    return is_anomaly, pred_label, final_pct, message, reasons


# ---------------------------------------------------------------------------
# Preprocessing Helpers: Model 2 (XGBoost Power Factor)
# ---------------------------------------------------------------------------
def preprocess_power_reading(reading: PowerReading) -> Tuple[pd.DataFrame, float, float]:
    """
    Transforms a single PowerReading into the exact 12-feature DataFrame
    expected by the final_model pipeline. Returns (df, actual_pf, qp_ratio).
    """
    # 1. Resolve Active Power
    active_power = reading.Total_Active_Power if reading.Total_Active_Power is not None else reading.P_T
    if active_power is None:
        raise HTTPException(
            status_code=422,
            detail="Active power required: provide Total_Active_Power or P_T",
        )

    # 2. Resolve Reactive Power
    reactive_power = reading.Total_Reactive_Power if reading.Total_Reactive_Power is not None else reading.Q_T
    if reactive_power is None:
        raise HTTPException(
            status_code=422,
            detail="Reactive power required: provide Total_Reactive_Power or Q_T",
        )

    # 3. Resolve Power Factor & Handle Meter Offset (+1)
    raw_pf = reading.Total_Power_Factor if reading.Total_Power_Factor is not None else reading.FP_T
    if raw_pf is None:
        raise HTTPException(
            status_code=422,
            detail="Power factor required: provide Total_Power_Factor or FP_T",
        )

    # Training data encoded PF with +1 offset (meter scale [1.0, 2.0])
    if raw_pf <= 1.0:
        model_pf = raw_pf + 1.0
        actual_pf = round(raw_pf, 4)
    else:
        model_pf = raw_pf
        actual_pf = round(raw_pf - 1.0, 4)

    # 4. Resolve Frequency
    freq = reading.Freq if reading.Freq is not None else reading.Frec
    if freq is None:
        freq = 60.0

    # 5. Resolve Hour
    if reading.hour is not None:
        hour = int(reading.hour)
    elif reading.Timestamp:
        try:
            hour = pd.to_datetime(reading.Timestamp).hour
        except Exception:
            hour = datetime.now().hour
    else:
        hour = datetime.now().hour

    # 6. Resolve Q_P_ratio
    if reading.Q_P_ratio is not None:
        qp_ratio = float(reading.Q_P_ratio)
    else:
        qp_ratio = float(reactive_power / (abs(active_power) + 1e-9))

    df = pd.DataFrame([{
        "hour": hour,
        "volt_A": float(reading.volt_A),
        "volt_B": float(reading.volt_B),
        "volt_C": float(reading.volt_C),
        "I_A": float(reading.I_A),
        "I_B": float(reading.I_B),
        "I_C": float(reading.I_C),
        "Total_Active_Power": float(active_power),
        "Total_Reactive_Power": float(reactive_power),
        "Total_Power_Factor": float(model_pf),
        "Freq": float(freq),
        "Q_P_ratio": round(qp_ratio, 6),
    }], columns=POWER_REQUIRED_FEATURES)

    return df, actual_pf, qp_ratio


def prepare_power_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[float], List[float]]:
    """
    Standardizes a batch DataFrame of power readings (from CSV or Excel),
    handling hardware column renames, hour derivation, Q/P ratio, and PF offset.
    Returns (model_ready_df, actual_pfs, qp_ratios).
    """
    df = df.copy()

    # Hardware meter column aliases mapping
    alias_map = {
        "P_T": "Total_Active_Power",
        "Q_T": "Total_Reactive_Power",
        "FP_T": "Total_Power_Factor",
        "Frec": "Freq",
    }
    for old_col, new_col in alias_map.items():
        if old_col in df.columns and new_col not in df.columns:
            df[new_col] = df[old_col]

    # Validate essential columns
    required_cols = {"volt_A", "volt_B", "volt_C", "I_A", "I_B", "I_C", "Total_Active_Power", "Total_Reactive_Power", "Total_Power_Factor"}
    missing = required_cols - set(df.columns)
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing required power columns: {sorted(missing)}",
        )

    if "Freq" not in df.columns:
        df["Freq"] = 60.0

    # Derive hour
    if "hour" not in df.columns or df["hour"].isna().all():
        if "Timestamp" in df.columns:
            try:
                df["hour"] = pd.to_datetime(df["Timestamp"]).dt.hour.fillna(datetime.now().hour).astype(int)
            except Exception:
                df["hour"] = datetime.now().hour
        else:
            df["hour"] = datetime.now().hour

    # Calculate Q_P_ratio
    if "Q_P_ratio" not in df.columns:
        df["Q_P_ratio"] = df["Total_Reactive_Power"] / (df["Total_Active_Power"].abs() + 1e-9)

    # Decode and normalize Power Factor
    raw_pfs = df["Total_Power_Factor"].astype(float).values
    actual_pfs = []
    model_pfs = []
    for pf in raw_pfs:
        if pf <= 1.0:
            model_pfs.append(pf + 1.0)
            actual_pfs.append(round(pf, 4))
        else:
            model_pfs.append(pf)
            actual_pfs.append(round(pf - 1.0, 4))

    df["Total_Power_Factor"] = model_pfs
    qp_ratios = [round(float(x), 4) for x in df["Q_P_ratio"].values]

    # Select and order exact 12 features
    model_df = df[POWER_REQUIRED_FEATURES].copy()
    for col in POWER_REQUIRED_FEATURES:
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce").fillna(0.0)

    return model_df, actual_pfs, qp_ratios


# ---------------------------------------------------------------------------
# Endpoints: System & Health
# ---------------------------------------------------------------------------
@app.get("/", tags=["System & Health"])
def root() -> Dict[str, Any]:
    """Root endpoint with service overview, model status, and route directory."""
    m1_ready = model_container.isolation_forest is not None
    m2_ready = model_container.power_model is not None

    return {
        "service": "MEQYAS AI — Unified Industrial Intelligence API",
        "version": "2.0.0",
        "status": "online",
        "models": {
            "model_1_machine_anomaly": {
                "name": "IsolationForest_Model.joblib",
                "framework": "Scikit-Learn",
                "purpose": "Machine operational telemetry anomaly detection",
                "loaded": m1_ready,
                "default_threshold_pct": DEFAULT_MACHINE_THRESHOLD,
            },
            "model_2_power_factor": {
                "name": "final_model.pkl",
                "framework": "XGBoost + Scikit-Learn Pipeline",
                "purpose": "Predictive Low Power Factor (< 0.90) risk forecasting (3 hours ahead)",
                "loaded": m2_ready,
                "default_threshold": DEFAULT_POWER_THRESHOLD,
            },
        },
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "endpoints": {
            "health": "GET /health",
            "machine_anomaly": {
                "predict_single": "POST /machine/predict",
                "predict_batch": "POST /machine/predict/batch",
                "predict_csv": "POST /machine/predict/csv",
                "legacy_predict": "POST /predict",
            },
            "power_factor": {
                "predict_single": "POST /power/predict",
                "predict_batch": "POST /power/predict/batch",
                "predict_file": "POST /power/predict/csv",
            },
        },
    }


@app.get("/health", tags=["System & Health"])
def health_check() -> Dict[str, Any]:
    """Readiness and health probe reporting status for both models."""
    m1_ready = model_container.isolation_forest is not None
    m2_ready = model_container.power_model is not None

    if m1_ready and m2_ready:
        overall_status = "healthy"
    elif m1_ready or m2_ready:
        overall_status = "degraded"
    else:
        overall_status = "unhealthy"

    return {
        "status": overall_status,
        "model_loaded": m1_ready,  # Backward compatibility field
        "model_file": str(model_container.isolation_forest_path) if model_container.isolation_forest_path else None,
        "models": {
            "machine_anomaly_model": {
                "loaded": m1_ready,
                "file": model_container.isolation_forest_path.name if model_container.isolation_forest_path else None,
                "error": model_container.isolation_forest_error,
            },
            "power_factor_model": {
                "loaded": m2_ready,
                "file": model_container.power_model_path.name if model_container.power_model_path else None,
                "error": model_container.power_model_error,
                "features_count": len(POWER_REQUIRED_FEATURES),
            },
        },
    }


# ---------------------------------------------------------------------------
# Endpoints: Model 1 — Machine Telemetry Anomaly Detection (Isolation Forest)
# ---------------------------------------------------------------------------
@app.post(
    "/machine/predict",
    response_model=SinglePredictionResponse,
    tags=["Machine Anomaly Detection (Isolation Forest)"],
)
@app.post(
    "/predict",
    response_model=SinglePredictionResponse,
    tags=["Machine Anomaly Detection (Isolation Forest)"],
    include_in_schema=True,
)
def predict_single(reading: TelemetryReading):
    """
    Predict whether a single machine telemetry reading is an anomaly or normal,
    returning a continuous dynamic anomaly percentage (0-100%).
    """
    model = get_isolation_forest_model()

    alarm_val = normalize_alarm(reading.alarm)
    status_str = normalize_status(reading.status)
    power_val = float(reading.power_avg)
    items_val = float(reading.items)
    threshold_val = float(reading.threshold) if reading.threshold is not None else DEFAULT_MACHINE_THRESHOLD

    input_dict = reading.model_dump()
    input_dict["alarm"] = alarm_val
    input_df = pd.DataFrame([input_dict])
    X = prepare_dataframe(input_df)

    try:
        raw_pred = int(model.predict(X)[0])
        raw_score = float(model.decision_function(X)[0])
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Machine telemetry inference error: {exc}",
        )

    base_ml_pct = float(calculate_ml_percentage(np.array([raw_score]))[0])

    is_anomaly, pred_label, final_pct, message, reasons = evaluate_anomaly(
        raw_pred=raw_pred,
        raw_score=raw_score,
        ml_pct=base_ml_pct,
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


@app.post(
    "/machine/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Machine Anomaly Detection (Isolation Forest)"],
)
@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Machine Anomaly Detection (Isolation Forest)"],
    include_in_schema=True,
)
def predict_batch(batch: BatchPredictionRequest):
    """
    Predict anomaly status and continuous percentages for a batch of machine readings.
    """
    if not batch.readings:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The readings list cannot be empty.",
        )

    model = get_isolation_forest_model()
    threshold_val = float(batch.threshold) if batch.threshold is not None else DEFAULT_MACHINE_THRESHOLD

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
            status_code=422,
            detail=f"Inference error during batch execution: {exc}",
        )

    base_ml_pcts = calculate_ml_percentage(raw_scores)

    results: List[SinglePredictionResponse] = []
    anomalies_count = 0
    all_final_pcts = []

    for pred, score, ml_pct, meta in zip(raw_preds, raw_scores, base_ml_pcts, metadata_list):
        is_anomaly, pred_label, final_pct, message, reasons = evaluate_anomaly(
            raw_pred=int(pred),
            raw_score=float(score),
            ml_pct=float(ml_pct),
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


@app.post(
    "/machine/predict/csv",
    tags=["Machine Anomaly Detection (Isolation Forest)"],
)
@app.post(
    "/predict/csv",
    tags=["Machine Anomaly Detection (Isolation Forest)"],
    include_in_schema=True,
)
async def predict_machine_from_csv(
    file: UploadFile = File(...),
    threshold: float = Query(DEFAULT_MACHINE_THRESHOLD, description="Anomaly percentage threshold (0-100%)"),
):
    """
    Upload a CSV file of machine telemetry readings to run bulk anomaly scoring.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only CSV files (.csv) are supported for machine anomaly detection.",
        )

    model = get_isolation_forest_model()

    try:
        content = await file.read()
        raw_df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not parse CSV file: {exc}",
        )

    required_cols = {"asset", "items", "status", "power_avg", "alarm", "product"}
    missing = required_cols - set(raw_df.columns)
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing required columns in CSV: {sorted(missing)}",
        )

    X = prepare_dataframe(raw_df)

    try:
        raw_preds = model.predict(X)
        raw_scores = model.decision_function(X)
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Inference error on CSV data: {exc}",
        )

    base_ml_pcts = calculate_ml_percentage(raw_scores)
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
            ml_pct=float(base_ml_pcts[i]),
            alarm=alarm_val,
            status_str=status_str,
            power_avg=power_val,
            items=items_val,
            threshold=threshold,
        )
        if is_anom:
            anomalies_count += 1
        final_pcts.append(final_pct)

        if i < 50:
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
# Endpoints: Model 2 — Predictive Power Factor Risk Model (XGBoost)
# ---------------------------------------------------------------------------
@app.post(
    "/power/predict",
    response_model=PowerPredictionResponse,
    tags=["Power Factor Prediction (XGBoost)"],
)
def predict_power_single(reading: PowerReading):
    """
    Predict whether 3-phase power factor will drop below 0.90 within the next 3 hours.
    """
    model = get_power_factor_model()
    df, actual_pf, qp_ratio = preprocess_power_reading(reading)
    threshold = reading.threshold if reading.threshold is not None else DEFAULT_POWER_THRESHOLD

    try:
        prob = float(model.predict_proba(df)[0][1])
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Power Factor model inference failed: {exc}",
        )

    is_risk = prob >= threshold
    prob_pct = round(prob * 100.0, 2)

    if is_risk:
        message = (
            f"ALERT: High risk ({prob_pct}%) of Power Factor dropping below 0.90 "
            f"within 3 hours (Threshold: {threshold:.2f}). Corrective capacitor dispatch recommended."
        )
        label = "Low Power Factor Risk (Anomaly)"
    else:
        message = (
            f"NORMAL: Power factor risk is low ({prob_pct}%). "
            f"Operating stably above 0.90 (Current PF: {actual_pf:.2f})."
        )
        label = "Normal Power Factor"

    return PowerPredictionResponse(
        prediction=label,
        is_low_pf_risk=is_risk,
        risk_probability=round(prob, 4),
        risk_percentage=prob_pct,
        threshold_used=threshold,
        actual_power_factor=actual_pf,
        computed_q_p_ratio=round(qp_ratio, 4),
        message=message,
    )


@app.post(
    "/power/predict/batch",
    response_model=PowerBatchResponse,
    tags=["Power Factor Prediction (XGBoost)"],
)
def predict_power_batch(batch: PowerBatchRequest):
    """
    Predict Low Power Factor risk (< 0.90) for a batch of 3-phase electrical readings.
    """
    if not batch.readings:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The readings list cannot be empty.",
        )

    model = get_power_factor_model()
    threshold = batch.threshold if batch.threshold is not None else DEFAULT_POWER_THRESHOLD

    dfs = []
    meta = []
    for r in batch.readings:
        df_row, actual_pf, qp_ratio = preprocess_power_reading(r)
        dfs.append(df_row)
        meta.append((actual_pf, qp_ratio))

    df_combined = pd.concat(dfs, ignore_index=True)

    try:
        probs = model.predict_proba(df_combined)[:, 1]
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Power batch inference failed: {exc}",
        )

    results: List[PowerPredictionResponse] = []
    low_pf_count = 0

    for prob_val, (actual_pf, qp_ratio) in zip(probs, meta):
        p_float = float(prob_val)
        is_risk = p_float >= threshold
        p_pct = round(p_float * 100.0, 2)

        if is_risk:
            low_pf_count += 1
            label = "Low Power Factor Risk (Anomaly)"
            msg = (
                f"ALERT: High risk ({p_pct}%) of Power Factor dropping below 0.90 "
                f"within 3 hours (Threshold: {threshold:.2f})."
            )
        else:
            label = "Normal Power Factor"
            msg = f"NORMAL: Power factor risk is low ({p_pct}%). Current PF: {actual_pf:.2f}."

        results.append(
            PowerPredictionResponse(
                prediction=label,
                is_low_pf_risk=is_risk,
                risk_probability=round(p_float, 4),
                risk_percentage=p_pct,
                threshold_used=threshold,
                actual_power_factor=actual_pf,
                computed_q_p_ratio=round(qp_ratio, 4),
                message=msg,
            )
        )

    total = len(batch.readings)
    normal_count = total - low_pf_count
    risk_rate = round((low_pf_count / total) * 100.0, 2)

    return PowerBatchResponse(
        total=total,
        low_pf_risk_count=low_pf_count,
        normal_count=normal_count,
        risk_rate_pct=risk_rate,
        results=results,
    )


@app.post(
    "/power/predict/csv",
    response_model=PowerFileUploadResponse,
    tags=["Power Factor Prediction (XGBoost)"],
)
async def predict_power_from_file(
    file: UploadFile = File(...),
    threshold: float = Query(DEFAULT_POWER_THRESHOLD, description="Decision probability threshold (0.0 - 1.0)"),
):
    """
    Upload a power monitoring file (.csv or .xlsx) to evaluate Low Power Factor risk across all records.
    """
    fname = file.filename.lower()
    if not (fname.endswith(".csv") or fname.endswith(".xlsx") or fname.endswith(".xls")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Supported formats: .csv or .xlsx",
        )

    model = get_power_factor_model()

    try:
        content = await file.read()
        if fname.endswith(".csv"):
            raw_df = pd.read_csv(io.BytesIO(content))
        else:
            raw_df = pd.read_excel(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not parse uploaded file: {exc}",
        )

    model_df, actual_pfs, qp_ratios = prepare_power_dataframe(raw_df)

    try:
        probs = model.predict_proba(model_df)[:, 1]
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error on file data: {exc}",
        )

    total = len(probs)
    low_risk_count = 0
    all_prob_pcts = []
    sample_results = []

    for i in range(total):
        p_val = float(probs[i])
        is_risk = p_val >= threshold
        p_pct = round(p_val * 100.0, 2)
        all_prob_pcts.append(p_pct)

        if is_risk:
            low_risk_count += 1
            label = "Low Power Factor Risk (Anomaly)"
        else:
            label = "Normal Power Factor"

        if i < 50:
            sample_results.append({
                "row_index": i,
                "is_low_pf_risk": is_risk,
                "prediction": label,
                "risk_probability": round(p_val, 4),
                "risk_percentage": p_pct,
                "actual_power_factor": actual_pfs[i],
                "computed_q_p_ratio": qp_ratios[i],
            })

    normal_count = total - low_risk_count
    risk_rate = round((low_risk_count / total) * 100.0, 2)
    avg_prob = round(float(np.mean(all_prob_pcts)), 2) if all_prob_pcts else 0.0

    return PowerFileUploadResponse(
        filename=file.filename,
        total_rows=total,
        low_pf_risk_count=low_risk_count,
        normal_count=normal_count,
        risk_rate_pct=risk_rate,
        average_risk_probability_pct=avg_prob,
        first_50_predictions=sample_results,
    )


# ---------------------------------------------------------------------------
# Server Startup (Uvicorn)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    print(f"[INFO] Starting MEQYAS AI Unified Server on http://{host}:{port}")
    uvicorn.run("main:app", host=host, port=port, reload=True)
