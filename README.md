# MEQYAS AI

The AI and data analysis module for the **MEQYAS** platform — an energy monitoring system for industrial manufacturing environments.

## Overview

This module performs **anomaly detection on machine energy consumption** data using an Isolation Forest model. It identifies abnormal patterns in power usage, production output, and machine status to surface potential equipment issues, energy waste, or process deviations.

## Project Structure

```
AI/
├── data/
│   ├── raw/                        # Original unprocessed datasets
│   │   └── company_A_labelled_subset.csv
│   └── processed/                  # Cleaned / transformed data (generated)
├── model/
│   └── IsolationForest_Model.joblib  # Trained anomaly detection model
├── notebook/
│   └── meqyas.ipynb                # End-to-end analysis & training notebook
├── .env                            # Environment variables (git-ignored)
├── .gitignore
├── main.py                         # FastAPI service for real-time anomaly detection
├── predict.py                      # CLI script for batch anomaly prediction
├── requirements.txt
└── README.md
```

## Dataset

The raw dataset (`company_A_labelled_subset.csv`) contains **14,492 rows** of 5-minute interval telemetry from **3 machines** producing **14 product types**, spanning 2022-08-31 to 2022-09-21.

| Column        | Type    | Description                                                                                         |
|---------------|---------|-----------------------------------------------------------------------------------------------------|
| `ts`          | string  | Timestamp in `YYYY-MM-DD HH:mm:ss+TZ` format                                                       |
| `asset`       | int     | Machine identifier (0–2)                                                                            |
| `items`       | float   | Number of items produced in the time interval                                                       |
| `status`      | float   | Machine state — `0`: idle, `1`: manual, `2`: automatic, `3`: alarm / interrupted                    |
| `status_time` | float   | Time (seconds) the machine spent in the given status                                                |
| `power_avg`   | float   | Average power consumed (kW)                                                                         |
| `cycle_time`  | float   | Total production cycle time (seconds)                                                               |
| `alarm`       | int     | Alarm flag (`0` = no alarm, `1` = alarm)                                                            |
| `product`     | int     | Product identifier (0–13)                                                                           |

## ML Pipeline

The analysis notebook follows this workflow:

```
Data Reading → EDA → Feature Engineering → Preprocessing → Model Training → Evaluation → Export
```

### 1. Exploratory Data Analysis (EDA)

- Correlation heatmaps across numeric features
- Power consumption distribution per machine and product
- Machine status breakdowns and alarm frequency
- Cycle-time vs. power and items-produced scatter plots

### 2. Feature Engineering

- `asset` → prefixed labels (`machine_1`, `machine_2`, `machine_3`)
- `product` → prefixed labels (`product_1` … `product_14`)
- `status` → human-readable categories (`idle`, `manual`, `automatic`, `alarm or interrupted state`)
- Dropped low-signal columns: `ts`, `status_time`, `cycle_time`

### 3. Preprocessing

- **Categorical features** (`asset`, `status`, `product`): One-Hot Encoding
- **Numerical features** (`items`, `power_avg`, `alarm`): Standard Scaling
- Pipeline assembled with `ColumnTransformer`

### 4. Model — Isolation Forest

- **Algorithm**: `sklearn.ensemble.IsolationForest`
- **Hyperparameter tuning**: `GridSearchCV` (3-fold CV) over:
  - `n_estimators`: [100, 200, 300]
  - `max_samples`: ["auto", 256, 512]
  - `contamination`: ["auto", 0.01, 0.05, 0.1]
- **Scoring**: Custom anomaly scoring function based on decision-function separation
- **Train/Test split**: 70 / 30 (random_state=42)

### 5. Multi-Criteria Anomaly Detection

A reading is identified as an anomaly if any of the following conditions occur:
1. **Active machine alarm** (`alarm == 1`)
2. **Alarm or interrupted machine state** (`status == 3` or `"alarm or interrupted state"`)
3. **Power surge exceeding rated machine limit** (`power_avg > 10.0 kW`)
4. **Invalid sensor telemetry** (negative power or items)
5. **Isolation Forest outlier detection** (`raw_score < 0`)
6. **Anomaly score percentage meeting or exceeding threshold** (default 50.0%)

## Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### Running the FastAPI Server (Uvicorn)

You can launch the FastAPI server directly with Python or using the Uvicorn CLI:

```bash
# Direct python run
python main.py

# Or via Uvicorn CLI with hot reload
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Once running, interactive API docs are accessible at:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

##### Example 1: Normal Reading (`POST /predict`):

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "asset": 0,
       "items": 4.0,
       "status": 2,
       "power_avg": 2.0,
       "alarm": 0,
       "product": 0
     }'
```

```json
{
  "is_anomaly": false,
  "prediction": "normal",
  "anomaly_percentage": 25.47,
  "raw_score": 0.114097,
  "message": "Reading is normal with 25.47% anomaly score.",
  "anomaly_reasons": null
}
```

##### Example 2: Anomaly Condition — Active Alarm (`POST /predict`):

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "asset": 0,
       "items": 4.0,
       "status": 3,
       "power_avg": 2.0,
       "alarm": 1,
       "product": 0
     }'
```

```json
{
  "is_anomaly": true,
  "prediction": "anomaly",
  "anomaly_percentage": 85.0,
  "raw_score": 0.045123,
  "message": "Anomaly detected: Active machine alarm signal (alarm=1) with 85.00% anomaly score.",
  "anomaly_reasons": [
    "Active machine alarm signal (alarm=1)",
    "Machine in alarm or interrupted state (status=3)"
  ]
}
```

##### Example 3: Anomaly Condition — Extreme Power Surge (`POST /predict`):

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "asset": 1,
       "items": 2.0,
       "status": 2,
       "power_avg": 18.5,
       "alarm": 0,
       "product": 1
     }'
```

```json
{
  "is_anomaly": true,
  "prediction": "anomaly",
  "anomaly_percentage": 85.0,
  "raw_score": -0.012345,
  "message": "Anomaly detected: Power consumption exceeds rated limit (18.50 kW > 10.0 kW) with 85.00% anomaly score.",
  "anomaly_reasons": [
    "Power consumption exceeds rated limit (18.50 kW > 10.0 kW)",
    "Isolation Forest classified reading as outlier (raw_score=-0.012345 < 0)",
    "Anomaly score (85.00%) meets or exceeds threshold (50.00%)"
  ]
}
```

#### Running predictions via CLI

Use `predict.py` to run batch anomaly detection from the command line:

```bash
# Predict on the default dataset
python predict.py

# Predict on a custom CSV
python predict.py --input path/to/data.csv

# Save results to a file
python predict.py --output results.csv

# Show top 20 anomalies with 50% threshold
python predict.py --top 20 --threshold 50.0
```

## Key Dependencies

| Package        | Purpose                          |
|----------------|----------------------------------|
| pandas         | Data manipulation                |
| numpy          | Numerical computing              |
| scikit-learn   | ML pipeline & Isolation Forest   |
| matplotlib     | Plotting                         |
| seaborn        | Statistical visualizations       |
| joblib         | Model serialization              |
| fastapi        | Backend API integration          |
| uvicorn        | ASGI web server                  |
