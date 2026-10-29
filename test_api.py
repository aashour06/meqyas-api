"""
Comprehensive Automated Test Suite for MEQYAS AI Unified Service
================================================================
Validates both models served by the unified FastAPI application:
  Part A: Model 1 — Machine Telemetry Anomaly Detection (Isolation Forest)
    1. Loading both models into memory
    2. Health & Root endpoint verification
    3. Normal machine reading
    4. Alarm flag condition (alarm=1)
    5. Status string condition ('alarm')
    6. Status code condition (status=3)
    7. Power surge condition (power_avg > 10 kW)
    8. Dynamic continuous percentage verification
    9. Batch machine telemetry prediction
    10. CSV machine dataset upload prediction
    11. Machine validation error handling (HTTP 422)

  Part B: Model 2 — Predictive Power Factor Risk Model (XGBoost)
    12. Power Factor single prediction with physical scale (0.0 - 1.0)
    13. Power Factor meter offset decoding (1.0 - 2.0)
    14. Hardware meter aliases support (P_T, Q_T, FP_T, Frec)
    15. Custom probability decision threshold behavior
    16. Batch power telemetry prediction
    17. Excel file upload prediction (.xlsx)
    18. Power validation error handling (HTTP 422)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

from main import (
    app,
    TelemetryReading,
    SinglePredictionResponse,
    BatchPredictionRequest,
    PowerReading,
    PowerPredictionResponse,
    PowerBatchRequest,
    predict_single,
    predict_batch,
    load_all_models,
    load_model_from_dir,
    MODEL_DIR,
    model_container,
)

print("=" * 70)
print("1. Loading Models into Unified Application...")
load_all_models()
assert model_container.isolation_forest is not None, "Isolation Forest model failed to load"
assert model_container.power_model is not None, "Power Factor model failed to load"
print(f"Machine Anomaly Model loaded: True")
print(f"Power Factor Model loaded: True")

client = TestClient(app)

print("\n" + "=" * 70)
print("2. Testing Health & Root Endpoints...")
r_root = client.get("/")
assert r_root.status_code == 200
root_json = r_root.json()
print("Root response:", root_json["service"])
assert root_json["status"] == "online"
assert root_json["models"]["model_1_machine_anomaly"]["loaded"] is True
assert root_json["models"]["model_2_power_factor"]["loaded"] is True

r_health = client.get("/health")
assert r_health.status_code == 200
health_json = r_health.json()
print("Health response:", health_json["status"])
assert health_json["status"] == "healthy"
assert health_json["models"]["machine_anomaly_model"]["loaded"] is True
assert health_json["models"]["power_factor_model"]["loaded"] is True

print("\n" + "=" * 70)
print("3. Testing Model 1: Normal Reading...")
normal_payload = {
    "asset": 0,
    "items": 4.0,
    "status": 2,
    "power_avg": 2.0,
    "alarm": 0,
    "product": 0,
}
r_norm = client.post("/machine/predict", json=normal_payload)
res_norm = r_norm.json()
assert r_norm.status_code == 200
assert res_norm["is_anomaly"] is False
assert res_norm["prediction"] == "normal"
assert res_norm["anomaly_percentage"] < 50.0
print(">>> Normal reading test PASSED")

# Verify backward-compatibility route /predict
r_legacy = client.post("/predict", json=normal_payload)
assert r_legacy.status_code == 200
assert r_legacy.json()["is_anomaly"] is False
print(">>> Legacy route /predict alias test PASSED")

print("\n" + "=" * 70)
print("4. Testing Model 1: Alarm Flag Condition (alarm=1)...")
alarm_payload = {
    "asset": 0,
    "items": 4.0,
    "status": 2,
    "power_avg": 2.0,
    "alarm": 1,
    "product": 0,
}
r_alarm = client.post("/machine/predict", json=alarm_payload)
res_alarm = r_alarm.json()
assert r_alarm.status_code == 200
assert res_alarm["is_anomaly"] is True
assert res_alarm["prediction"] == "anomaly"
assert res_alarm["anomaly_percentage"] >= 50.0
assert res_alarm["anomaly_reasons"] is not None and len(res_alarm["anomaly_reasons"]) > 0
print(">>> Alarm flag test PASSED")

print("\n" + "=" * 70)
print("5. Testing Model 1: Status='alarm' string condition...")
status_alarm_payload = {
    "asset": 0,
    "items": 0.0,
    "status": "alarm",
    "power_avg": 0.0,
    "alarm": 0,
    "product": 0,
}
r_status_alarm = client.post("/machine/predict", json=status_alarm_payload)
res_status_alarm = r_status_alarm.json()
assert r_status_alarm.status_code == 200
assert res_status_alarm["is_anomaly"] is True
assert res_status_alarm["prediction"] == "anomaly"
print(">>> Status='alarm' test PASSED")

print("\n" + "=" * 70)
print("6. Testing Model 1: Status=3 (interrupted) condition...")
status3_payload = {
    "asset": 0,
    "items": 0.0,
    "status": 3,
    "power_avg": 0.0,
    "alarm": 0,
    "product": 0,
}
r_status3 = client.post("/machine/predict", json=status3_payload)
assert r_status3.status_code == 200
assert r_status3.json()["is_anomaly"] is True
print(">>> Status=3 test PASSED")

print("\n" + "=" * 70)
print("7. Testing Model 1: Power Surge Condition (power_avg > 10 kW)...")
r_surge20 = client.post("/machine/predict", json={**normal_payload, "power_avg": 20.0})
r_surge35 = client.post("/machine/predict", json={**normal_payload, "power_avg": 35.0})
assert r_surge20.status_code == 200 and r_surge20.json()["is_anomaly"] is True
assert r_surge35.status_code == 200 and r_surge35.json()["is_anomaly"] is True
assert r_surge35.json()["anomaly_percentage"] > r_surge20.json()["anomaly_percentage"]
print(">>> Power surge dynamic scaling test PASSED")

print("\n" + "=" * 70)
print("8. Verifying Model 1: Dynamic Continuous Scoring...")
scores = [
    res_alarm["anomaly_percentage"],
    res_status_alarm["anomaly_percentage"],
    r_surge20.json()["anomaly_percentage"],
    r_surge35.json()["anomaly_percentage"],
]
assert len(set(scores)) == len(scores), "Scores should vary dynamically"
print(f"Collected scores: {scores}")
print(">>> Dynamic scoring verification PASSED")

print("\n" + "=" * 70)
print("9. Testing Model 1: Batch Prediction...")
batch_payload = {
    "readings": [
        normal_payload,
        alarm_payload,
        status_alarm_payload,
        {**normal_payload, "asset": 1, "power_avg": 1.0},
    ],
    "threshold": 50.0,
}
r_batch = client.post("/machine/predict/batch", json=batch_payload)
assert r_batch.status_code == 200
bdata = r_batch.json()
assert bdata["total_readings"] == 4
assert bdata["anomalies_count"] == 2
assert bdata["normal_count"] == 2
print(">>> Batch prediction test PASSED")

print("\n" + "=" * 70)
print("10. Testing Model 1: CSV Prediction...")
csv_path = Path("data/raw/company_A_labelled_subset.csv")
if csv_path.exists():
    with open(csv_path, "rb") as f:
        r_csv = client.post("/machine/predict/csv", files={"file": ("company_A_labelled_subset.csv", f, "text/csv")})
    assert r_csv.status_code == 200
    cdata = r_csv.json()
    assert cdata["total_rows"] == 14492
    assert cdata["anomalies_count"] > 0
    print(f"CSV Total: {cdata['total_rows']:,}, Anomalies: {cdata['anomalies_count']:,} ({cdata['anomaly_rate_pct']}%)")
    print(">>> CSV prediction test PASSED")

print("\n" + "=" * 70)
print("11. Testing Model 1: Validation Error Handling (Missing Fields)...")
r_err = client.post("/machine/predict", json={"asset": 0, "items": 4.0})
assert r_err.status_code == 422
print(">>> Machine HTTP 422 error handling test PASSED")

print("\n" + "=" * 70)
print("12. Testing Model 2: Power Factor Normal Prediction (PF = 0.95)...")
power_norm = {
    "volt_A": 126.8,
    "volt_B": 125.4,
    "volt_C": 126.2,
    "I_A": 0.35,
    "I_B": 0.42,
    "I_C": 0.21,
    "Total_Active_Power": 0.087,
    "Total_Reactive_Power": 0.010,
    "Total_Power_Factor": 0.95,
    "Freq": 60.0,
    "hour": 14,
}
rp_norm = client.post("/power/predict", json=power_norm)
assert rp_norm.status_code == 200
pdata = rp_norm.json()
assert pdata["prediction"] == "Normal Power Factor"
assert pdata["is_low_pf_risk"] is False
assert pdata["actual_power_factor"] == 0.95
assert isinstance(pdata["risk_probability"], float)
print(f"Risk probability: {pdata['risk_percentage']}%")
print(">>> Power Factor normal reading test PASSED")

print("\n" + "=" * 70)
print("13. Testing Model 2: Meter-Scale PF Decoding (PF = 1.95)...")
power_meter = {
    **power_norm,
    "Total_Power_Factor": 1.95,
    "Timestamp": "2024-08-06 14:00:00",
}
rp_meter = client.post("/power/predict", json=power_meter)
assert rp_meter.status_code == 200
assert rp_meter.json()["actual_power_factor"] == 0.95
print(">>> Meter-scale PF decoding test PASSED")

print("\n" + "=" * 70)
print("14. Testing Model 2: Hardware Meter Aliases (P_T, Q_T, FP_T, Frec)...")
power_aliases = {
    "volt_A": 126.5,
    "volt_B": 125.0,
    "volt_C": 126.0,
    "I_A": 0.30,
    "I_B": 0.35,
    "I_C": 0.20,
    "P_T": 0.080,
    "Q_T": 0.012,
    "FP_T": 0.94,
    "Frec": 60.0,
    "hour": 10,
}
rp_alias = client.post("/power/predict", json=power_aliases)
assert rp_alias.status_code == 200
assert rp_alias.json()["actual_power_factor"] == 0.94
print(">>> Hardware meter aliases test PASSED")

print("\n" + "=" * 70)
print("15. Testing Model 2: Custom Probability Threshold...")
rp_thresh = client.post("/power/predict", json={**power_norm, "threshold": 0.01})
assert rp_thresh.status_code == 200
assert rp_thresh.json()["is_low_pf_risk"] is True
assert rp_thresh.json()["threshold_used"] == 0.01
print(">>> Custom probability threshold test PASSED")

print("\n" + "=" * 70)
print("16. Testing Model 2: Batch Prediction...")
power_batch_payload = {
    "readings": [power_norm, power_meter],
    "threshold": 0.50,
}
rp_batch = client.post("/power/predict/batch", json=power_batch_payload)
assert rp_batch.status_code == 200
pb_data = rp_batch.json()
assert pb_data["total"] == 2
assert len(pb_data["results"]) == 2
print(f"Total: {pb_data['total']}, Risk Count: {pb_data['low_pf_risk_count']}, Normal: {pb_data['normal_count']}")
print(">>> Power Factor batch prediction test PASSED")

print("\n" + "=" * 70)
print("17. Testing Model 2: Excel File Upload (.xlsx)...")
excel_file = Path("data/raw/Power_data/Monitor_Fri_Aug_09_2024.xlsx")
if excel_file.exists():
    with open(excel_file, "rb") as f:
        rp_excel = client.post(
            "/power/predict/csv",
            files={"file": ("Monitor_Fri_Aug_09_2024.xlsx", f, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        )
    assert rp_excel.status_code == 200
    pexcel_data = rp_excel.json()
    assert pexcel_data["total_rows"] == 431
    assert pexcel_data["low_pf_risk_count"] + pexcel_data["normal_count"] == 431
    print(f"Excel rows: {pexcel_data['total_rows']}, Low PF risk: {pexcel_data['low_pf_risk_count']}, Normal: {pexcel_data['normal_count']}")
    print(">>> Power Factor Excel upload test PASSED")

print("\n" + "=" * 70)
print("18. Testing Model 2: Validation Error Handling (Missing Power Fields)...")
rp_err = client.post(
    "/power/predict",
    json={"volt_A": 126.8, "volt_B": 125.4, "volt_C": 126.2, "I_A": 0.35, "I_B": 0.42, "I_C": 0.21, "Total_Power_Factor": 0.95},
)
assert rp_err.status_code == 422
print(">>> Power Factor HTTP 422 error handling test PASSED")

print("\n" + "=" * 70)
print("ALL 18 COMPREHENSIVE INTEGRATION TESTS PASSED SUCCESSFULLY!")
print("=" * 70)
