"""
Test script for MEQYAS AI Anomaly Detection API
Verifies normal, alarm, status, power surge, and batch conditions.
Also verifies that anomaly_percentage changes dynamically and is NOT stuck at 85%.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Import main components
from main import (
    app,
    TelemetryReading,
    SinglePredictionResponse,
    BatchPredictionRequest,
    predict_single,
    predict_batch,
    load_model_from_dir,
    MODEL_DIR,
    model_container,
)
from fastapi.testclient import TestClient

print("=" * 60)
print("1. Loading Model...")
load_model_from_dir(MODEL_DIR)
print(f"Model loaded: {model_container.model is not None}")
assert model_container.model is not None, "Failed to load model"

client = TestClient(app)

print("\n" + "=" * 60)
print("2. Testing Health & Root Endpoints...")
r_root = client.get("/")
print("Root response:", r_root.json())
assert r_root.status_code == 200

r_health = client.get("/health")
print("Health response:", r_health.json())
assert r_health.json()["status"] == "healthy"

print("\n" + "=" * 60)
print("3. Testing Normal Reading...")
normal_payload = {
    "asset": 0,
    "items": 4.0,
    "status": 2,
    "power_avg": 2.0,
    "alarm": 0,
    "product": 0,
}
r_norm = client.post("/predict", json=normal_payload)
print("Normal Response:", r_norm.json())
res_norm = r_norm.json()
assert r_norm.status_code == 200
assert res_norm["is_anomaly"] is False, f"Expected False, got {res_norm['is_anomaly']}"
assert res_norm["prediction"] == "normal", f"Expected 'normal', got {res_norm['prediction']}"
assert res_norm["anomaly_percentage"] < 50.0, f"Expected < 50%, got {res_norm['anomaly_percentage']}"
print(">>> Normal reading test PASSED")

print("\n" + "=" * 60)
print("4. Testing Alarm Flag Condition (alarm=1)...")
alarm_payload = {
    "asset": 0,
    "items": 4.0,
    "status": 2,
    "power_avg": 2.0,
    "alarm": 1,
    "product": 0,
}
r_alarm = client.post("/predict", json=alarm_payload)
print("Alarm Response:", r_alarm.json())
res_alarm = r_alarm.json()
assert r_alarm.status_code == 200
assert res_alarm["is_anomaly"] is True, f"Expected True, got {res_alarm['is_anomaly']}"
assert res_alarm["prediction"] == "anomaly", f"Expected 'anomaly', got {res_alarm['prediction']}"
assert res_alarm["anomaly_percentage"] >= 50.0, f"Expected >= 50%, got {res_alarm['anomaly_percentage']}"
assert res_alarm["anomaly_reasons"] is not None and len(res_alarm["anomaly_reasons"]) > 0
print(">>> Alarm flag test PASSED")

print("\n" + "=" * 60)
print("5. Testing Status='alarm' string condition...")
status_alarm_payload = {
    "asset": 0,
    "items": 0.0,
    "status": "alarm",
    "power_avg": 0.0,
    "alarm": 0,
    "product": 0,
}
r_status_alarm = client.post("/predict", json=status_alarm_payload)
print("Status='alarm' Response:", r_status_alarm.json())
res_status_alarm = r_status_alarm.json()
assert r_status_alarm.status_code == 200
assert res_status_alarm["is_anomaly"] is True, f"Expected True, got {res_status_alarm['is_anomaly']}"
assert res_status_alarm["prediction"] == "anomaly", f"Expected 'anomaly', got {res_status_alarm['prediction']}"
print(">>> Status='alarm' test PASSED")

print("\n" + "=" * 60)
print("6. Testing Status=3 (alarm or interrupted state) condition...")
status_3_payload = {
    "asset": 2,
    "items": 0.0,
    "status": 3,
    "power_avg": 0.0,
    "alarm": 0,
    "product": 1,
}
r_status_3 = client.post("/predict", json=status_3_payload)
print("Status=3 Response:", r_status_3.json())
res_status_3 = r_status_3.json()
assert r_status_3.status_code == 200
assert res_status_3["is_anomaly"] is True, f"Expected True, got {res_status_3['is_anomaly']}"
assert res_status_3["prediction"] == "anomaly", f"Expected 'anomaly', got {res_status_3['prediction']}"
print(">>> Status=3 test PASSED")

print("\n" + "=" * 60)
print("7. Testing Power Surge Condition (power_avg=20 kW > 10 kW)...")
power_surge_payload = {
    "asset": 1,
    "items": 4.0,
    "status": 2,
    "power_avg": 20.0,
    "alarm": 0,
    "product": 1,
}
r_surge = client.post("/predict", json=power_surge_payload)
print("Power Surge (20 kW) Response:", r_surge.json())
res_surge = r_surge.json()
assert r_surge.status_code == 200
assert res_surge["is_anomaly"] is True, f"Expected True, got {res_surge['is_anomaly']}"
assert res_surge["prediction"] == "anomaly", f"Expected 'anomaly', got {res_surge['prediction']}"

# Test higher power surge to confirm dynamic scaling
power_surge_high_payload = {
    "asset": 1,
    "items": 4.0,
    "status": 2,
    "power_avg": 35.0,
    "alarm": 0,
    "product": 1,
}
r_surge_high = client.post("/predict", json=power_surge_high_payload)
print("Higher Power Surge (35 kW) Response:", r_surge_high.json())
res_surge_high = r_surge_high.json()
assert res_surge_high["anomaly_percentage"] > res_surge["anomaly_percentage"], (
    f"Expected higher surge {res_surge_high['anomaly_percentage']}% > {res_surge['anomaly_percentage']}%"
)
print(">>> Power surge dynamic scaling test PASSED")

print("\n" + "=" * 60)
print("8. Verifying Dynamic Anomaly Percentage (Scores must differ, not all 85%)...")
anomaly_scores = [
    res_alarm["anomaly_percentage"],
    res_status_alarm["anomaly_percentage"],
    res_status_3["anomaly_percentage"],
    res_surge["anomaly_percentage"],
    res_surge_high["anomaly_percentage"],
]
print("Collected anomaly scores for different conditions:", anomaly_scores)
unique_scores = set(anomaly_scores)
print(f"Number of distinct scores: {len(unique_scores)} out of {len(anomaly_scores)}")
assert len(unique_scores) >= 4, f"Scores must vary dynamically! Got: {unique_scores}"
assert not all(s == 85.0 for s in anomaly_scores), "Scores should NOT all be 85.0%!"
print(">>> Dynamic anomaly percentage verification PASSED")

print("\n" + "=" * 60)
print("9. Testing Batch Prediction (2 normal, 2 anomaly)...")
batch_payload = {
    "readings": [
        normal_payload,
        alarm_payload,
        status_alarm_payload,
        {
            "asset": "machine_2",
            "items": 8.0,
            "status": "automatic",
            "power_avg": 1.0,
            "alarm": 0,
            "product": "product_2",
        },
    ]
}
r_batch = client.post("/predict/batch", json=batch_payload)
print("Batch Response Summary:")
batch_data = r_batch.json()
print(f"Total: {batch_data['total_readings']}, Anomalies: {batch_data['anomalies_count']}, Normal: {batch_data['normal_count']}")
assert r_batch.status_code == 200
assert batch_data["total_readings"] == 4
assert batch_data["anomalies_count"] == 2, f"Expected 2 anomalies, got {batch_data['anomalies_count']}"
assert batch_data["normal_count"] == 2, f"Expected 2 normal, got {batch_data['normal_count']}"
print(">>> Batch prediction test PASSED")

print("\n" + "=" * 60)
print("10. Testing CSV Prediction on subset data...")
csv_path = Path("data/raw/company_A_labelled_subset.csv")
if csv_path.exists():
    with open(csv_path, "rb") as f:
        r_csv = client.post("/predict/csv", files={"file": ("company_A_labelled_subset.csv", f, "text/csv")})
    assert r_csv.status_code == 200
    csv_data = r_csv.json()
    print("CSV Prediction Summary:")
    print(f"Total rows: {csv_data['total_rows']:,}")
    print(f"Anomalies: {csv_data['anomalies_count']:,} ({csv_data['anomaly_rate_pct']}%)")
    print(f"Normal: {csv_data['normal_count']:,}")
    print(f"Avg anomaly score: {csv_data['avg_anomaly_score_pct']}%")
    assert csv_data["anomalies_count"] > 0, "Expected anomalies in CSV dataset"
    print(">>> CSV prediction test PASSED")

print("\n" + "=" * 60)
print("ALL TESTS COMPLETED SUCCESSFULLY! ANOMALY PERCENTAGE IS DYNAMIC!")
print("=" * 60)
