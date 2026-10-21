"""
Fast AI Predictor — loads the IsolationForest pipeline from model/ and
produces anomaly predictions with dynamic continuous percentage-based anomaly scores
and operational diagnosis.

Usage:
    python predict.py                              # predict on default CSV
    python predict.py --input data.csv             # predict on custom CSV
    python predict.py --output results.csv         # save results to file
    python predict.py --top 20                     # show top 20 anomalies
    python predict.py --threshold 50.0             # anomaly threshold pct
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, List, Union

import joblib
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paths and Constants
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
DEFAULT_DATA = BASE_DIR / "data" / "raw" / "company_A_labelled_subset.csv"

STATUS_MAP = {
    0: "idle",
    1: "manual",
    2: "automatic",
    3: "alarm or interrupted state",
}

COLUMNS_TO_DROP = ["ts", "status_time", "cycle_time"]

BASELINE_MIN_SCORE = -0.06802
BASELINE_MAX_SCORE = 0.23254
DEFAULT_ANOMALY_THRESHOLD = 50.0
MAX_RATED_POWER_KW = 10.0


# ---------------------------------------------------------------------------
# Preprocessing Normalizers
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


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply data transformations matching the trained pipeline."""
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


# ---------------------------------------------------------------------------
# Dynamic Continuous Score Conversion
# ---------------------------------------------------------------------------
def calculate_ml_percentage(
    raw_scores: np.ndarray,
    min_score: float = BASELINE_MIN_SCORE,
    max_score: float = BASELINE_MAX_SCORE,
) -> np.ndarray:
    """
    Convert IsolationForest decision_function scores to a calibrated 0–100% anomaly score:
      • raw_score >= 0: mapped smoothly to 0% - 50%
      • raw_score < 0: mapped smoothly to 50% - 100%
    """
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


def compute_operational_risk(
    alarm: int,
    status_str: str,
    power_avg: float,
    items: float,
) -> tuple[float, List[str]]:
    """Compute continuous operational risk (0-100%) and reasons."""
    reasons = []
    risk_factors = []

    if alarm == 1:
        reasons.append("Active alarm (alarm=1)")
        alarm_risk = 35.0 + min(15.0, max(0.0, power_avg - 2.0) * 2.0)
        risk_factors.append(alarm_risk)

    if status_str == "alarm or interrupted state":
        reasons.append("Alarm/interrupted state")
        status_risk = 25.0 + min(15.0, max(0.0, power_avg) * 1.5)
        risk_factors.append(status_risk)
    elif status_str == "idle":
        if power_avg > 1.0:
            reasons.append(f"Idle power draw ({power_avg:.1f} kW)")
            risk_factors.append(min(35.0, power_avg * 8.0))
        if items > 0:
            reasons.append(f"Idle production ({items:.0f} items)")
            risk_factors.append(min(30.0, 10.0 + items * 2.0))

    if power_avg < 0:
        reasons.append("Negative power reading")
        risk_factors.append(55.0)
    elif power_avg > MAX_RATED_POWER_KW:
        excess = power_avg - MAX_RATED_POWER_KW
        power_risk = 30.0 + min(40.0, (excess / 10.0) * 20.0)
        reasons.append(f"Power surge ({power_avg:.1f} kW > {MAX_RATED_POWER_KW} kW)")
        risk_factors.append(power_risk)
    elif power_avg > 4.0:
        elevated_risk = ((power_avg - 4.0) / 6.0) * 12.0
        risk_factors.append(elevated_risk)

    if items < 0:
        reasons.append("Negative items produced")
        risk_factors.append(55.0)
    elif items > 15.0:
        excess_items = min(1.0, (items - 15.0) / 35.0)
        item_risk = 15.0 + excess_items * 20.0
        reasons.append(f"High production rate ({items:.0f} items)")
        risk_factors.append(item_risk)

    if not risk_factors:
        return 0.0, reasons

    rem = 1.0
    for rf in risk_factors:
        rem *= (1.0 - min(100.0, rf) / 100.0)
    combined_risk = (1.0 - rem) * 100.0

    return round(combined_risk, 2), reasons


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(model_dir: Path):
    """Load the trained .joblib model found in model_dir."""
    model_files = sorted(model_dir.glob("*.joblib"))
    if not model_files:
        print(f"[ERROR] No .joblib model files found in {model_dir}")
        sys.exit(1)

    target_file = next(
        (f for f in model_files if f.name == "IsolationForest_Model.joblib"),
        model_files[0],
    )
    print(f"[INFO]  Loading model from: {target_file}")
    t0 = time.perf_counter()
    model = joblib.load(target_file)
    elapsed = time.perf_counter() - t0
    print(f"[INFO]  Model loaded in {elapsed:.3f}s")
    return model


# ---------------------------------------------------------------------------
# Main prediction pipeline
# ---------------------------------------------------------------------------
def predict(
    input_path: str | Path,
    output_path: str | Path | None = None,
    top_n: int | None = None,
    threshold: float = DEFAULT_ANOMALY_THRESHOLD,
):
    # 1. Load model
    model = load_model(MODEL_DIR)

    # 2. Read data
    print(f"[INFO]  Reading data from: {input_path}")
    t0 = time.perf_counter()
    raw_df = pd.read_csv(input_path)
    print(f"[INFO]  Loaded {len(raw_df):,} rows in {time.perf_counter() - t0:.3f}s")

    # 3. Prepare features
    X = prepare_data(raw_df)

    # 4. Predict + score
    print("[INFO]  Running predictions …")
    t0 = time.perf_counter()
    raw_preds = model.predict(X)
    raw_scores = model.decision_function(X)
    ml_pcts = calculate_ml_percentage(raw_scores)
    elapsed = time.perf_counter() - t0
    print(f"[INFO]  Predictions done in {elapsed:.3f}s")

    # 5. Build results DataFrame with dynamic continuous percentage evaluation
    results = raw_df.copy()
    is_anomaly_list = []
    labels_list = []
    final_pcts = []
    reasons_list = []

    for i in range(len(results)):
        alarm_val = normalize_alarm(raw_df["alarm"].iloc[i]) if "alarm" in raw_df.columns else 0
        status_str = normalize_status(raw_df["status"].iloc[i]) if "status" in raw_df.columns else "automatic"
        power_val = float(raw_df["power_avg"].iloc[i]) if "power_avg" in raw_df.columns else 0.0
        items_val = float(raw_df["items"].iloc[i]) if "items" in raw_df.columns else 0.0

        op_risk, op_reasons = compute_operational_risk(alarm_val, status_str, power_val, items_val)
        reasons = list(op_reasons)

        if raw_preds[i] == -1:
            reasons.append("Isolation Forest outlier")

        ml_p = float(ml_pcts[i])
        if op_risk > 0:
            combined_pct = 100.0 - ((100.0 - ml_p) * (100.0 - op_risk) / 100.0)
            has_hard_fault = (
                alarm_val == 1
                or status_str == "alarm or interrupted state"
                or power_val > MAX_RATED_POWER_KW
                or power_val < 0
                or items_val < 0
            )
            if has_hard_fault:
                r_pct = 50.0 + (combined_pct / 100.0) * 48.0
            else:
                r_pct = combined_pct
        else:
            r_pct = ml_p

        r_pct = round(float(np.clip(r_pct, 0.0, 100.0)), 2)

        if r_pct >= threshold and not any("threshold" in r for r in reasons):
            if not op_reasons and raw_preds[i] != -1:
                reasons.append(f"Score >= {threshold}%")

        is_anom = len(reasons) > 0 or r_pct >= threshold or raw_preds[i] == -1
        is_anomaly_list.append(is_anom)
        labels_list.append("anomaly" if is_anom else "normal")
        final_pcts.append(r_pct)
        reasons_list.append("; ".join(reasons) if is_anom else "Normal")

    results["is_anomaly"] = is_anomaly_list
    results["prediction"] = labels_list
    results["raw_score"] = np.round(raw_scores, 6)
    results["anomaly_score_pct"] = final_pcts
    results["anomaly_reasons"] = reasons_list

    # 6. Summary statistics
    n_anomalies = sum(is_anomaly_list)
    n_normal = len(results) - n_anomalies
    print()
    print("=" * 60)
    print("  PREDICTION SUMMARY")
    print("=" * 60)
    print(f"  Total rows        : {len(results):>8,}")
    print(f"  Normal             : {n_normal:>8,}  ({n_normal / len(results) * 100:.2f}%)")
    print(f"  Anomalies          : {n_anomalies:>8,}  ({n_anomalies / len(results) * 100:.2f}%)")
    print(f"  Avg anomaly score  : {np.mean(final_pcts):>8.2f}%")
    print(f"  Max anomaly score  : {np.max(final_pcts):>8.2f}%")
    print("=" * 60)
    print()

    # 7. Show top anomalies
    anomalies_df = results[results["is_anomaly"]].sort_values(
        "anomaly_score_pct", ascending=False
    )
    show_n = top_n or 10
    if not anomalies_df.empty:
        print(f"  Top {min(show_n, len(anomalies_df))} anomalies (by anomaly_score_pct):")
        cols_to_show = [c for c in ["asset", "items", "status", "power_avg", "alarm", "product", "prediction", "anomaly_score_pct", "anomaly_reasons"] if c in anomalies_df.columns]
        print(anomalies_df[cols_to_show].head(show_n).to_string(index=False))
    else:
        print("  No anomalies detected.")
    print()

    # 8. Optionally save
    if output_path:
        results.to_csv(output_path, index=False)
        print(f"[INFO]  Results saved to: {output_path}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Fast anomaly detection using the trained IsolationForest model."
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=str(DEFAULT_DATA),
        help="Path to input CSV file (default: data/raw/company_A_labelled_subset.csv)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to save results CSV (optional).",
    )
    parser.add_argument(
        "--top", "-t",
        type=int,
        default=10,
        help="Number of top anomalies to display (default: 10).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_ANOMALY_THRESHOLD,
        help="Anomaly percentage threshold (0-100, default: 50.0).",
    )
    args = parser.parse_args()

    predict(
        input_path=args.input,
        output_path=args.output,
        top_n=args.top,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
