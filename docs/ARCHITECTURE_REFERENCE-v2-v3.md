# Architecture Reference: Real-Estate-Estimator

This document provides a complete reference of the project architecture, file structure, and component responsibilities at each version.

---

## Current Architecture (V3.1)

```
Real-Estate-Estimator/
├── .github/
│   └── workflows/
│       ├── ci.yml                 # CI pipeline (lint, test, validate)
│       └── train.yml              # Training pipeline (manual trigger)
│
├── data/
│   ├── kc_house_data.csv          # Original 2014-2015 housing data
│   └── zipcode_demographics.csv   # Demographics by zipcode
│
├── docs/
│   ├── V2_Detailed_Roadmap.md     # V2 series roadmap
│   ├── V3_Detailed_Roadmap.md     # V3 series roadmap
│   ├── V3.1_Completion_Summary.md # V3.1 completion details
│   ├── SESSION_LOG_20251208-09.md # This session's detailed log
│   ├── MODEL_VERSION_EVOLUTION.md # Model version history
│   └── ARCHITECTURE_REFERENCE.md  # This file
│
├── logs/
│   ├── human_in_the_loop_corrections.md
│   └── v2.5_robust_evaluation_*.json
│
├── mlflow/
│   ├── mlflow.db                  # SQLite tracking database
│   └── artifacts/                 # Model and artifact storage
│
├── model/
│   ├── model.pkl                  # Production model (V2.5 XGBoost)
│   ├── model.joblib               # Alternative format
│   ├── model_features.json        # Feature names (43 features)
│   └── metrics.json               # Performance metrics
│
├── src/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── prediction.py          # API routes (/predict, /health, etc.)
│   ├── services/
│   │   ├── __init__.py
│   │   ├── feature_service.py     # Demographics lookup
│   │   └── model_service.py       # Model loading/prediction
│   ├── config.py                  # Application settings
│   ├── main.py                    # FastAPI entry point
│   ├── models.py                  # Pydantic schemas
│   ├── train.py                   # Basic training script
│   ├── train_with_mlflow.py       # MLflow-integrated training
│   ├── evaluate.py                # Basic evaluation
│   ├── robust_evaluate.py         # Comprehensive evaluation (V2.5)
│   ├── tune.py                    # KNN hyperparameter tuning
│   ├── tune_xgboost.py            # XGBoost Optuna tuning
│   ├── tune_top_models.py         # Fine-tune best models
│   └── compare_models.py          # Model comparison
│
├── tests/
│   ├── __init__.py
│   └── test_model.py              # Model tests (13 tests)
│
├── scripts/
│   ├── test_api.py                # API testing script
│   └── download_reference_docs.py # Data download helper
│
├── Reference_Docs/                # Local only (gitignored)
│   └── King_County_Assessment_data_ALL/
│
├── .gitignore
├── mlflow_config.py               # MLflow configuration
├── pyproject.toml                 # Tool configurations
├── requirements.txt               # Python dependencies
├── README.md
└── RESTART_20251208.md            # Session state document
```

---

## Component Responsibilities

### 1. Entry Point
**File:** `src/main.py`

**Purpose:** FastAPI application setup and lifecycle management

**Responsibilities:**
- Create FastAPI app instance
- Configure CORS middleware
- Mount API routers (prefix: `/api/v1`)
- Manage startup/shutdown lifecycle
- Load model and demographics at startup

**Key Code:**
```python
app = FastAPI(title="Real Estate Price Predictor", ...)
app.include_router(prediction_router, prefix="/api/v1")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model and demographics
    model_service = get_model_service()
    feature_service = get_feature_service()
    yield
    # Shutdown: Cleanup
```

---

### 2. Business Logic
**File:** `src/services/feature_service.py`

**Purpose:** Feature enrichment and demographics lookup

**Responsibilities:**
- Load demographics data from CSV
- Validate zipcodes against King County
- Enrich home features with demographics
- Provide average demographics for minimal predictions
- Provide V2.1 default values for missing features

**Key Code:**
```python
class FeatureService:
    def enrich_features(self, home_features: dict, zipcode: str) -> dict:
        """Add demographics to home features."""
        demographics = self.get_demographics(zipcode)
        return {**home_features, **demographics}
    
    def enrich_features_with_average(self, home_features: dict) -> dict:
        """Use average demographics when no zipcode provided."""
        enriched = V21_DEFAULT_FEATURES.copy()
        enriched.update(home_features)
        enriched.update(self.average_demographics)
        return enriched
```

---

### 3. Training Engine
**File:** `src/train_with_mlflow.py`

**Purpose:** Model training with full MLflow experiment tracking

**Responsibilities:**
- Load and prepare training data
- Train XGBoost model with given parameters
- Perform cross-validation
- Log parameters, metrics, artifacts to MLflow
- Support CLI for custom runs

**Key Code:**
```python
def train_with_tracking(run_name, params, test_size, register_model):
    setup_mlflow()
    X, y, feature_names = load_data()
    
    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_param("n_estimators", params["n_estimators"])
        
        # Train and evaluate
        pipeline = train_model(X_train, y_train, params)
        metrics = evaluate_model(pipeline, X_test, y_test)
        
        # Log metrics
        mlflow.log_metric("test_mae", metrics["mae"])
        
        # Log model artifact
        mlflow.sklearn.log_model(pipeline, "model")
```

---

### 4. Honest Auditor
**File:** `src/robust_evaluate.py`

**Purpose:** Comprehensive, statistically rigorous model evaluation

**Responsibilities:**
- K-fold cross-validation with confidence intervals
- Bootstrap confidence intervals (500 samples)
- Log transform experiments
- Residual analysis by price range
- Save detailed evaluation reports

**Key Code:**
```python
def run_kfold_cv(model, X, y, k=5):
    """5-fold cross-validation with detailed metrics."""
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    mae_scores = []
    r2_scores = []
    
    for train_idx, val_idx in kfold.split(X):
        model_clone = clone(model)
        model_clone.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_pred = model_clone.predict(X.iloc[val_idx])
        mae_scores.append(mean_absolute_error(y.iloc[val_idx], y_pred))
        r2_scores.append(r2_score(y.iloc[val_idx], y_pred))
    
    return {
        "mae_mean": np.mean(mae_scores),
        "mae_std": np.std(mae_scores),
        "r2_mean": np.mean(r2_scores),
        "r2_std": np.std(r2_scores)
    }
```

---

### 5. Optimizer
**File:** `src/tune_xgboost.py`

**Purpose:** Hyperparameter optimization using Optuna

**Responsibilities:**
- Define hyperparameter search space
- Run Optuna trials with cross-validation
- Find optimal XGBoost parameters
- Save best parameters

**Key Code:**
```python
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 1),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 2)
    }
    
    model = XGBRegressor(**params)
    pipeline = Pipeline([('scaler', RobustScaler()), ('model', model)])
    
    # Cross-validation score
    scores = cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_absolute_error')
    return -scores.mean()
```

---

### 6. ETL (Planned for V3.2)
**File:** `src/data/transform_assessment_data.py` (TBD)

**Purpose:** Transform King County assessment data to model schema

**Planned Responsibilities:**
- Load EXTR_RPSale.csv (sales data)
- Load EXTR_ResBldg.csv (building data)
- Load EXTR_Parcel.csv (parcel data)
- Map columns to existing 17 home features
- Handle data quality issues
- Output training-ready dataset

---

## API Endpoints

**Base URL:** `http://localhost:8000`

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Root - redirect info |
| GET | `/docs` | Swagger UI |
| GET | `/api/v1/health` | Health check |
| POST | `/api/v1/predict` | Full prediction (zipcode required) |
| POST | `/api/v1/predict-minimal` | Minimal prediction (no zipcode) |
| POST | `/api/v1/predict-batch` | Batch predictions |

### Predict Request Schema
```json
{
  "bedrooms": 3,
  "bathrooms": 2,
  "sqft_living": 1800,
  "sqft_lot": 5000,
  "floors": 1,
  "waterfront": 0,
  "view": 0,
  "condition": 3,
  "grade": 7,
  "sqft_above": 1800,
  "sqft_basement": 0,
  "yr_built": 1990,
  "yr_renovated": 0,
  "zipcode": "98103",
  "lat": 47.6,
  "long": -122.3,
  "sqft_living15": 1800,
  "sqft_lot15": 5000
}
```

### Predict Response Schema
```json
{
  "predicted_price": 642376.25,
  "prediction_id": "pred-20251209-032036-5babc445",
  "model_version": "v2.5",
  "confidence_note": "Prediction based on King County 2014-2015 data.",
  "data_vintage_warning": "Model trained on 2014-2015 data.",
  "timestamp": "2025-12-09T03:20:36.687940"
}
```

---

## Configuration Files

### `mlflow_config.py`
```python
MLFLOW_TRACKING_URI = "sqlite:///mlflow/mlflow.db"
MLFLOW_ARTIFACT_LOCATION = (MLFLOW_DIR / "artifacts").as_uri()
MLFLOW_EXPERIMENT_NAME = "real-estate-predictor"
MLFLOW_MODEL_NAME = "real-estate-xgboost"
```

### `pyproject.toml`
```toml
[project]
name = "real-estate-estimator"
version = "3.1.0"

[tool.ruff]
target-version = "py310"
line-length = 100
select = ["E", "W", "F", "I", "B", "C4", "UP"]
ignore = ["E501", "E402", "B008", "B023", "B904", "B905", "W291", "W293"]

[tool.black]
line-length = 100
target-version = ["py310", "py311"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = ["-v", "--tb=short", "-ra"]
```

---

## Dependencies (`requirements.txt`)

```
# Core ML
scikit-learn>=1.0
xgboost>=1.7
pandas>=1.5
numpy>=1.21

# API
fastapi>=0.100
uvicorn>=0.23
pydantic>=2.0

# MLOps
mlflow>=2.0
optuna>=3.0

# Testing
pytest>=8.0
pytest-cov>=4.0

# Code Quality
ruff>=0.1
black>=23.0

# Utilities
scipy>=1.10
joblib>=1.2
```

---

## Data Flow

```
┌─────────────────┐
│   API Request   │
│  /api/v1/predict│
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│ Feature Service │────▶│  Demographics   │
│  (enrichment)   │     │  zipcode_demos  │
└────────┬────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│  Model Service  │────▶│  model.pkl      │
│   (predict)     │     │  (XGBoost)      │
└────────┬────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐
│  API Response   │
│ predicted_price │
└─────────────────┘
```

---

## MLflow Tracking Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    train_with_mlflow.py                      │
├─────────────────────────────────────────────────────────────┤
│  1. setup_mlflow()                                          │
│  2. load_data()                                             │
│  3. mlflow.start_run()                                      │
│     ├── mlflow.log_param(...)     → Parameters              │
│     ├── train_model(...)                                    │
│     ├── evaluate_model(...)                                 │
│     ├── mlflow.log_metric(...)    → Metrics                 │
│     └── mlflow.sklearn.log_model  → Artifacts               │
│  4. mlflow.end_run()                                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       mlflow/                                │
├─────────────────────────────────────────────────────────────┤
│  mlflow.db (SQLite)                                         │
│  ├── experiments                                            │
│  ├── runs                                                   │
│  ├── params                                                 │
│  └── metrics                                                │
│                                                             │
│  artifacts/                                                 │
│  └── <run_id>/                                              │
│      ├── model/                                             │
│      ├── feature_info.json                                  │
│      └── evaluation_report.json                             │
└─────────────────────────────────────────────────────────────┘
```

---

## CI/CD Flow

```
┌──────────────┐
│  git push    │
│  to develop  │
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│                    GitHub Actions CI                          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────┐    ┌─────────┐    ┌──────────┐    ┌──────────┐ │
│  │  lint   │───▶│  test   │───▶│ validate │───▶│ success  │ │
│  │ (ruff/  │    │ (pytest │    │  (model  │    │ (report) │ │
│  │ black)  │    │  13 tx) │    │  loads)  │    │          │ │
│  └─────────┘    └─────────┘    └──────────┘    └──────────┘ │
│                                                              │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────┐
│  PR Merge    │
│  to develop  │
└──────────────┘
```

---

**Document Version:** 1.0  
**Last Updated:** 2025-12-09
