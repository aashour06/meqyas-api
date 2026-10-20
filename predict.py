"""
Fast AI Predictor — loads the IsolationForest pipeline from model/ and
produces anomaly predictions with percentage-based anomaly scores.

Usage:
    python predict.py                              # predict on default CSV
    python predict.py --input data.csv             # predict on custom CSV
    python predict.py --output results.csv         # save results to file
    python predict.py --top 20                     # show top 20 anomalies

The percentage score is derived from `estimator.decision_function(X)`:
    • 100% = most anomalous (lowest raw score)
    •   0% = most normal   (highest raw score)
"""

import argparse
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
DEFAULT_DATA = BASE_DIR / "data" / "raw" / "company_A_labelled_subset.csv"


# ---------------------------------------------------------------------------
# Data preparation — mirrors notebook preprocessing exactly
# ---------------------------------------------------------------------------
STATUS_MAP = {
    0: "idle",
    1: "manual",
    2: "automatic",
    3: "alarm or interrupted state",
}

COLUMNS_TO_DROP = ["ts", "status_time", "cycle_time"]


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same transformations used during training."""
    df = df.copy()

    # Convert asset & product to categorical strings (same as notebook)
    df["asset"] = "machine_" + (df["asset"] + 1).astype(str)
    df["product"] = "product_" + (df["product"] + 1).astype(str)
    df["status"] = df["status"].map(STATUS_MAP)

    # Drop columns not used by the model
    df = df.drop(columns=[c for c in COLUMNS_TO_DROP if c in df.columns])
    return df


# ---------------------------------------------------------------------------
# Score conversion: raw decision_function → percentage (0–100 %)
# ---------------------------------------------------------------------------
def raw_scores_to_percentage(raw_scores: np.ndarray) -> np.ndarray:
    """
    Convert IsolationForest `decision_function` scores to a 0–100 %
    anomaly percentage.

    IsolationForest.decision_function returns values where:
        • negative → anomaly
        • positive → normal
        • more negative → more anomalous

    We invert and normalise so that:
        100 % = most anomalous (lowest raw score)
          0 % = most normal   (highest raw score)
    """
    min_score = raw_scores.min()
    max_score = raw_scores.max()

    if max_score == min_score:
        # All scores identical — return 0 % (all equally normal/anomalous)
        return np.zeros_like(raw_scores)

    # Invert: low raw score → high anomaly %
    pct = (max_score - raw_scores) / (max_score - min_score) * 100.0
    return np.round(pct, 2)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(model_dir: Path):
    """Load the first .joblib model found in *model_dir*."""
    model_files = sorted(model_dir.glob("*.joblib"))
    if not model_files:
        print(f"[ERROR] No .joblib model files found in {model_dir}")
        sys.exit(1)

    model_path = model_files[0]
    print(f"[INFO]  Loading model from: {model_path}")
    t0 = time.perf_counter()
    model = joblib.load(model_path)
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
):
    # 1. Load model
    model = load_model(MODEL_DIR)

    # 2. Read data
    print(f"[INFO]  Reading data from: {input_path}")
    t0 = time.perf_counter()
    raw_df = pd.read_csv(input_path)
    print(f"[INFO]  Loaded {len(raw_df):,} rows in {time.perf_counter() - t0:.3f}s")

    # 3. Prepare features (same transforms as training notebook)
    X = prepare_data(raw_df)

    # 4. Predict + score
    print("[INFO]  Running predictions …")
    t0 = time.perf_counter()
    predictions = model.predict(X)                  # 1 = normal, -1 = anomaly
    raw_scores = model.decision_function(X)          # raw anomaly scores
    anomaly_pct = raw_scores_to_percentage(raw_scores)
    elapsed = time.perf_counter() - t0
    print(f"[INFO]  Predictions done in {elapsed:.3f}s")

    # 5. Build results DataFrame
    results = raw_df.copy()
    results["prediction"] = np.where(predictions == -1, "anomaly", "normal")
    results["raw_score"] = np.round(raw_scores, 6)
    results["anomaly_score_pct"] = anomaly_pct

    # 6. Summary statistics
    n_anomalies = (predictions == -1).sum()
    n_normal = (predictions == 1).sum()
    print()
    print("=" * 60)
    print("  PREDICTION SUMMARY")
    print("=" * 60)
    print(f"  Total rows        : {len(results):>8,}")
    print(f"  Normal             : {n_normal:>8,}  ({n_normal / len(results) * 100:.2f}%)")
    print(f"  Anomalies          : {n_anomalies:>8,}  ({n_anomalies / len(results) * 100:.2f}%)")
    print(f"  Avg anomaly score  : {anomaly_pct.mean():>8.2f}%")
    print(f"  Max anomaly score  : {anomaly_pct.max():>8.2f}%")
    print("=" * 60)
    print()

    # 7. Show top anomalies
    anomalies_df = results[results["prediction"] == "anomaly"].sort_values(
        "anomaly_score_pct", ascending=False
    )
    show_n = top_n or 10
    if not anomalies_df.empty:
        print(f"  Top {min(show_n, len(anomalies_df))} anomalies (by anomaly_score_pct):")
        print(anomalies_df.head(show_n).to_string(index=False))
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
    args = parser.parse_args()

    predict(
        input_path=args.input,
        output_path=args.output,
        top_n=args.top,
    )


if __name__ == "__main__":
    main()
