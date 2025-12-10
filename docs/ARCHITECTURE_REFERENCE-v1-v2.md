# Architecture Reference: Real Estate Price Predictor

**Document Created:** 2025-12-09  
**Current State:** V3.1 MLOps Complete  
**Production Model:** V2.5 XGBoost

---

## System Overview

```
+---------------------------------------------------------------------------------+
|                              REAL ESTATE ESTIMATOR                               |
|                                                                                  |
|  +---------------------+   +---------------------+   +---------------------+     |
|  |    GitHub Actions   |   |       MLflow        |   |      FastAPI        |     |
|  |    (CI/CD)          |   |   (Tracking)        |   |      (API)          |     |
|  +---------------------+   +---------------------+   +---------------------+     |
|           |                         |                         |                  |
|           v                         v                         v                  |
|  +-------------------------------------------------------------------------+    |
|  |                          Model Pipeline                                  |    |
|  |   +----------+    +--------------+    +----------------------------+    |    |
|  |   |  Data    |--->| RobustScaler |--->|     XGBRegressor (V2.5)    |    |    |
|  |   |  (43 ft) |    |              |    |  n_est=239, depth=7, lr=.08|    |    |
|  |   +----------+    +--------------+    +----------------------------+    |    |
|  +-------------------------------------------------------------------------+    |
|                                                                                  |
+---------------------------------------------------------------------------------+
```

---

## Directory Structure

```
c:\Experiments\Real-Estate-Estimator\
|
+-- .github/
|   +-- workflows/
|       +-- ci.yml                    # CI pipeline (lint, test, validate)
|       +-- train.yml                 # Training pipeline (manual trigger)
|
+-- data/
|   +-- kc_house_data.csv            # 21,613 training samples (2014-2015)
|   +-- zipcode_demographics.csv      # 70 zipcodes, 26 features
|   +-- future_unseen_examples.csv    # 300 test examples
|
+-- docs/
|   +-- ARCHITECTURE.md              # High-level architecture
|   +-- API.md                       # API documentation
|   +-- CODE_WALKTHROUGH.md          # Interview prep guide
|   +-- EVALUATION.md                # Model evaluation report
|   +-- V2_Detailed_Roadmap.md       # V2 version history
|   +-- V3_Detailed_Roadmap.md       # V3 MLOps roadmap
|   +-- V3.1_Completion_Summary.md   # V3.1 deliverables
|   +-- Changes_from_*.md            # Version change logs
|   +-- V*.Lessons_Learned.md        # Lessons by version
|
+-- logs/
|   +-- master_log.md                # Central index
|   +-- v2.3_grid_search_results.csv # GridSearchCV results
|   +-- v2.4_model_comparison_*.csv  # Model comparison
|   +-- v2.5_robust_evaluation_*.json # Robust eval results
|   +-- *_implementation_log.md      # Implementation logs
|
+-- mlflow/
|   +-- mlflow.db                    # SQLite tracking database
|   +-- artifacts/                   # Model artifacts
|
+-- model/
|   +-- model.joblib                 # Production model (V2.5 XGBoost)
|   +-- model_features.json          # 43 feature names
|   +-- metrics.json                 # Current metrics
|
+-- Reference_Docs/
|   +-- An agentic-friendly copilot-instruc.txt  # Protocol framework
|   +-- King_County_Assessment_data_ALL/         # 2024 data (gitignored)
|   +-- mle-project-challenge-2/                 # Original challenge
|
+-- scripts/
|   +-- test_api.py                  # API testing
|   +-- compare_endpoints.py         # Endpoint comparison
|   +-- compare_endpoints_by_tier.py # Price tier analysis
|   +-- feature_routing_experiment.py # Routing experiments
|   +-- git_v*.ps1                   # Git workflow scripts
|
+-- src/
|   +-- __init__.py
|   +-- config.py                    # Pydantic settings
|   +-- main.py                      # FastAPI entry point
|   +-- models.py                    # Pydantic schemas
|   +-- train.py                     # Basic training
|   +-- train_with_mlflow.py         # MLflow-integrated training
|   +-- tune.py                      # KNN GridSearchCV
|   +-- tune_xgboost.py              # XGBoost Optuna tuning
|   +-- tune_top_models.py           # Multi-model tuning
|   +-- compare_models.py            # Model comparison
|   +-- evaluate.py                  # Basic evaluation
|   +-- robust_evaluate.py           # Robust evaluation (CV, CI)
|   +-- api/
|   |   +-- __init__.py
|   |   +-- prediction.py            # API endpoints
|   +-- services/
|       +-- __init__.py
|       +-- model_service.py         # Model loading
|       +-- feature_service.py       # Demographics lookup
|
+-- tests/
|   +-- __init__.py
|   +-- test_model.py                # 13 tests
|
+-- .gitignore
+-- Dockerfile
+-- docker-compose.yml
+-- mlflow_config.py                 # MLflow configuration
+-- pyproject.toml                   # Tool configurations
+-- README.md
+-- requirements.txt
+-- RESTART_20251208.md              # Session restart protocol
+-- START_HERE.md                    # Project entry point
```

---

## Core Components

### 1. API Layer (`src/api/`)

```
src/api/prediction.py
+-- /api/v1/health          GET   - Health check
+-- /api/v1/predict         POST  - Full prediction (requires zipcode)
+-- /api/v1/predict-minimal POST  - Quick estimate (7 features)
+-- /api/v1/predict-batch   POST  - Batch predictions
+-- /api/v1/predict-full    POST  - All 17 home features, no zipcode
```

**Request Flow:**
```
Client Request
     |
     v
+-------------+
|  FastAPI    | --- Pydantic validation
|  Router     |
+-------------+
     |
     v
+-------------+
|  Feature    | --- Add demographics (if zipcode provided)
|  Service    |     OR use average demographics
+-------------+
     |
     v
+-------------+
|   Model     | --- Load model, make prediction
|  Service    |
+-------------+
     |
     v
Response (JSON)
```

### 2. Model Service (`src/services/model_service.py`)

**Responsibilities:**
- Load model from disk (`model/model.joblib`)
- Auto-detect model version
- Make predictions
- Provide model status

**Key Methods:**
```python
class ModelService:
    def __init__(self, model_path: str)
    def predict_single(self, features: dict) -> float
    def predict_batch(self, features: list[dict]) -> list[float]
    def get_status(self) -> dict
    def reload(self) -> None
```

### 3. Feature Service (`src/services/feature_service.py`)

**Responsibilities:**
- Load demographics data
- Lookup demographics by zipcode
- Provide average demographics (for no-zipcode predictions)
- Apply V2.1 default features

**Key Data:**
```python
V21_DEFAULT_FEATURES = {
    "waterfront": 0,
    "view": 0,
    "condition": 3,
    "grade": 7,
    "yr_built": 1975,
    "yr_renovated": 0,
    "lat": 47.5601,
    "long": -122.2139,
    "sqft_living15": 1986,
    "sqft_lot15": 7620,
}
```

### 4. Training Pipeline

**Scripts by Purpose:**

| Script | Purpose | Tech |
|--------|---------|------|
| `train.py` | Basic training | sklearn |
| `train_with_mlflow.py` | Training with tracking | sklearn + MLflow |
| `tune.py` | KNN hyperparameter tuning | GridSearchCV |
| `tune_xgboost.py` | XGBoost tuning | Optuna |
| `compare_models.py` | Multi-model comparison | sklearn + xgboost |

**Current Production Training:**
```python
# Pipeline structure
pipeline = Pipeline([
    ('scaler', RobustScaler()),
    ('model', XGBRegressor(
        n_estimators=239,
        max_depth=7,
        learning_rate=0.0863,
        subsample=0.7472,
        colsample_bytree=0.8388,
        min_child_weight=6,
        gamma=0.1589,
        reg_alpha=0.2791,
        reg_lambda=1.3826,
        random_state=42
    ))
])
```

### 5. Evaluation Pipeline

| Script | Purpose | Outputs |
|--------|---------|---------|
| `evaluate.py` | Basic metrics | R2, MAE, RMSE |
| `robust_evaluate.py` | Comprehensive | CV, CI, residuals |

**Robust Evaluation Components:**
1. 5-fold cross-validation
2. Bootstrap confidence intervals (500 samples)
3. Log transform experiment
4. Residual analysis by price range

---

## Data Architecture

### Input Features (43 total)

#### Home Features (17)
| Feature | Type | Source |
|---------|------|--------|
| bedrooms | int | User input |
| bathrooms | float | User input |
| sqft_living | int | User input |
| sqft_lot | int | User input |
| floors | float | User input |
| sqft_above | int | User input |
| sqft_basement | int | User input |
| waterfront | binary | User input (V2.1+) |
| view | ordinal 0-4 | User input (V2.1+) |
| condition | ordinal 1-5 | User input (V2.1+) |
| grade | ordinal 1-13 | User input (V2.1+) |
| yr_built | year | User input (V2.1+) |
| yr_renovated | year | User input (V2.1+) |
| lat | float | User input (V2.1+) |
| long | float | User input (V2.1+) |
| sqft_living15 | int | User input (V2.1+) |
| sqft_lot15 | int | User input (V2.1+) |

#### Demographic Features (26)
Looked up by zipcode from `data/zipcode_demographics.csv`:
- Population statistics
- Income metrics
- Education levels
- Housing characteristics
- etc.

### Data Files

| File | Samples | Features | Vintage |
|------|---------|----------|---------|
| kc_house_data.csv | 21,613 | 18 cols | 2014-2015 |
| zipcode_demographics.csv | 70 zips | 26 cols | ~2014 |
| future_unseen_examples.csv | 300 | 18 cols | Test data |

---

## CI/CD Pipeline

### GitHub Actions Workflows

#### ci.yml (Continuous Integration)
```yaml
Trigger: push/PR to main, develop

Jobs:
  1. code-quality:
     - Install ruff, black
     - ruff check .
     - black . --check
     
  2. tests:
     - Install dependencies
     - pytest tests/ -v --cov=src
     
  3. model-validation:
     - Load model
     - Test predictions
     - Start API briefly
     
  4. ci-success:
     - Depends on all above
     - Aggregate status
```

#### train.yml (Training Pipeline)
```yaml
Trigger: manual (workflow_dispatch)

Inputs:
  - n_estimators (default: 239)
  - max_depth (default: 7)
  - learning_rate (default: 0.0863)
  
Steps:
  1. Checkout code
  2. Setup Python
  3. Install dependencies
  4. Run training
  5. Upload MLflow artifacts
```

---

## MLflow Integration

### Configuration (`mlflow_config.py`)

```python
MLFLOW_TRACKING_URI = "sqlite:///mlflow/mlflow.db"
MLFLOW_ARTIFACT_ROOT = Path("mlflow/artifacts").as_uri()
DEFAULT_EXPERIMENT_NAME = "real-estate-predictor"
```

### Usage

```powershell
# Train with tracking
python src/train_with_mlflow.py

# View experiments
mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db
# Open http://localhost:5000
```

### Tracked Items
- Parameters (hyperparameters)
- Metrics (MAE, R2, RMSE)
- Artifacts (model, feature list, metrics JSON)
- Tags (version, timestamp)

---

## API Endpoints

### Base URL
```
http://localhost:8000/api/v1
```

### Endpoints

#### GET /health
```json
Response:
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "v2.5",
  "n_features": 43
}
```

#### POST /predict
```json
Request:
{
  "bedrooms": 3,
  "bathrooms": 2.0,
  "sqft_living": 1800,
  "sqft_lot": 5000,
  "floors": 1.0,
  "sqft_above": 1800,
  "sqft_basement": 0,
  "zipcode": "98103",
  // ... other features
}

Response:
{
  "predicted_price": 642376.25,
  "model_version": "v2.5",
  "prediction_id": "abc123...",
  "timestamp": "2025-12-09T..."
}
```

#### POST /predict-minimal
```json
Request:
{
  "bedrooms": 3,
  "bathrooms": 2.0,
  "sqft_living": 1800,
  "sqft_lot": 5000,
  "floors": 1.0,
  "sqft_above": 1800,
  "sqft_basement": 0
}

Response:
{
  "predicted_price": 518234.50,
  "model_version": "v2.5",
  // Uses average demographics + V21 defaults
}
```

#### POST /predict-batch
```json
Request:
{
  "items": [
    { /* property 1 */ },
    { /* property 2 */ },
    // ...
  ]
}

Response:
{
  "predictions": [
    { "predicted_price": 542000.0, ... },
    { "predicted_price": 681000.0, ... },
    // ...
  ]
}
```

---

## Model Architecture

### Current Production Model (V2.5)

```
Input (43 features)
       |
       v
+--------------+
| RobustScaler |  --- Scale features (robust to outliers)
+--------------+
       |
       v
+------------------------------------------------+
|              XGBRegressor                       |
|  +-------------------------------------------+ |
|  | n_estimators: 239 trees                   | |
|  | max_depth: 7 levels per tree              | |
|  | learning_rate: 0.0863                     | |
|  | subsample: 0.7472 (row sampling)          | |
|  | colsample_bytree: 0.8388 (feature sample) | |
|  | min_child_weight: 6                       | |
|  | gamma: 0.1589 (regularization)            | |
|  | reg_alpha: 0.2791 (L1)                    | |
|  | reg_lambda: 1.3826 (L2)                   | |
|  +-------------------------------------------+ |
+------------------------------------------------+
       |
       v
Output (predicted price)
```

### Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| CV MAE | $63,529 +/- $2,150 | 5-fold |
| CV R2 | 0.8945 +/- 0.0168 | 5-fold |
| 95% CI MAE | [$63,590, $70,971] | Bootstrap |
| Prediction Time | <10ms | Single sample |
| Model Size | ~5 MB | .joblib |

### Error Profile by Price Range

| Price Range | Count | MAE | Typical Error |
|-------------|-------|-----|---------------|
| <$300k | 888 | $37k | 12-15% |
| $300k-$500k | 1,555 | $41k | 10-12% |
| $500k-$750k | 1,130 | $60k | 10-12% |
| $750k-$1M | 424 | $108k | 12-15% |
| >$1M | 326 | $241k | 15-20% |

---

## Testing Architecture

### Test Suite (`tests/test_model.py`)

```
tests/
+-- test_model.py
    +-- TestModelLoading (4 tests)
    |   +-- test_model_file_exists
    |   +-- test_model_is_pipeline
    |   +-- test_model_has_predict_method
    |   +-- test_model_has_expected_steps
    |
    +-- TestPrediction (4 tests)
    |   +-- test_prediction_returns_array
    |   +-- test_prediction_correct_shape
    |   +-- test_prediction_positive
    |   +-- test_prediction_reasonable_range
    |
    +-- TestInputValidation (2 tests)
    |   +-- test_single_sample_input
    |   +-- test_multiple_samples_input
    |
    +-- TestPerformance (1 test)
    |   +-- test_prediction_speed
    |
    +-- TestMLflowConfig (2 tests)
        +-- test_mlflow_config_imports
        +-- test_mlflow_config_setup
```

### Running Tests

```powershell
# All tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src

# Specific test class
pytest tests/test_model.py::TestPrediction -v
```

---

## Docker Architecture

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY model/ ./model/
COPY data/ ./data/

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./model:/app/model
      - ./data:/app/data
    environment:
      - ENVIRONMENT=production
```

### Commands

```powershell
# Build image
docker build -t real-estate-predictor .

# Run container
docker run -p 8000:8000 real-estate-predictor

# Using compose
docker-compose up -d
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| ENVIRONMENT | development | Runtime environment |
| MODEL_PATH | model/model.joblib | Model file location |
| DATA_PATH | data/ | Data directory |
| LOG_LEVEL | INFO | Logging verbosity |

### Pydantic Settings (`src/config.py`)

```python
class Settings(BaseSettings):
    environment: str = "development"
    model_path: str = "model/model.joblib"
    demographics_path: str = "data/zipcode_demographics.csv"
    log_level: str = "INFO"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )
```

---

## Commands Reference

### Development

```powershell
# Start API (development)
uvicorn src.main:app --reload --port 8000

# Start API (production)
uvicorn src.main:app --host 0.0.0.0 --port 8000

# Run tests
pytest tests/ -v

# Lint
python -m ruff check .
python -m black . --check

# Format
python -m black .
python -m ruff check . --fix
```

### Training

```powershell
# Basic training
python src/train.py

# Training with MLflow
python src/train_with_mlflow.py

# Model comparison
python src/compare_models.py

# Robust evaluation
python src/robust_evaluate.py
```

### MLflow

```powershell
# View UI
mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db

# Access at http://localhost:5000
```

### Docker

```powershell
# Build
docker build -t real-estate-predictor .

# Run
docker run -p 8000:8000 real-estate-predictor

# Compose
docker-compose up -d
docker-compose down
```

---

## Future Architecture (V3.2+)

### Planned Components

```
+-----------------------------------------------------------------------+
|                           V3.2+ Architecture                           |
|                                                                        |
|  +----------------+   +----------------+   +----------------+          |
|  |  Assessment    |   |   Transform    |   |   Training     |          |
|  |   Data ETL     |-->|   Pipeline     |-->|    Data        |          |
|  |   (TODO)       |   |   (TODO)       |   |   (TODO)       |          |
|  +----------------+   +----------------+   +----------------+          |
|                                                 |                      |
|                                                 v                      |
|  +----------------------------------------------------------------+   |
|  |                     Model Registry (V3.3)                       |   |
|  |   Staging ----------------------------------------> Production  |   |
|  +----------------------------------------------------------------+   |
|                                                 |                      |
|                                                 v                      |
|  +----------------------------------------------------------------+   |
|  |                     Deploy Pipeline (V3.4)                      |   |
|  |   Build --> Push --> Deploy Staging --> Deploy Production       |   |
|  +----------------------------------------------------------------+   |
|                                                                        |
+-----------------------------------------------------------------------+
```

### Data Pipeline Files (Planned for V3.2)
- `src/data/load_assessment_data.py`
- `src/data/transform_assessment_data.py`
- `src/data/validate_schema.py`

---

**Document Version:** 1.0  
**Last Updated:** 2025-12-09
