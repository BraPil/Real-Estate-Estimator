# Architecture Reference: Real Estate Estimator
## Complete Technical Documentation

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CLIENT REQUEST                               │
│                    (Web App / Mobile / API Consumer)                │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         FastAPI Application                          │
│                         (src/main.py)                                │
├─────────────────────────────────────────────────────────────────────┤
│  • Lifespan context manager loads model once at startup             │
│  • Pydantic validation on all inputs                                │
│  • Router-based endpoint organization                               │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
┌───────────────────────────┐   ┌───────────────────────────────────┐
│      Feature Service      │   │         Model Service              │
│  (services/feature_       │   │   (services/model_service.py)     │
│        service.py)        │   │                                   │
├───────────────────────────┤   ├───────────────────────────────────┤
│ • Demographic enrichment  │   │ • Loads sklearn Pipeline          │
│ • Default value imputation│   │ • Holds model in memory           │
│ • Temporal feature calc   │   │ • Executes predictions            │
│ • Zipcode lookup (O(1))   │   │ • MLflow integration (V3.3)       │
└───────────────────────────┘   └───────────────────────────────────┘
           │                                    │
           ▼                                    ▼
┌───────────────────────────┐   ┌───────────────────────────────────┐
│   Demographics Cache      │   │        Model Artifacts            │
│   (In-memory DataFrame)   │   │     (model/model.pkl)             │
├───────────────────────────┤   ├───────────────────────────────────┤
│ • zipcode_demographics.csv│   │ • Pickled sklearn Pipeline        │
│ • 26 features per zipcode │   │ • Includes preprocessor + model   │
│ • Loaded once at startup  │   │ • ~1.8 MB file size               │
└───────────────────────────┘   └───────────────────────────────────┘
```

---

## File-by-File Reference

### 1. Entry Point: `src/main.py`

**Purpose:** FastAPI application initialization and routing.

**Key Components:**

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and demographics once at startup."""
    model_service = get_model_service()
    feature_service = get_feature_service()
    yield
    # Cleanup on shutdown
```

**Why Lifespan Pattern?**
- Model loading takes 500ms+
- Without lifespan, every request would reload model
- With lifespan, prediction time is <50ms

**Endpoints Registered:**
```python
app.include_router(prediction_router, prefix="/api/v1")
```

---

### 2. Business Logic: `src/services/feature_service.py`

**Purpose:** Bridge between user input and model requirements.

**Key Features:**

#### Default Values (V2.1+)
```python
V21_DEFAULT_FEATURES = {
    "bedrooms": 3,
    "bathrooms": 2.0,
    "sqft_living": 1910,  # Median
    "sqft_lot": 7618,     # Median
    "floors": 1.5,
    "waterfront": 0,
    "view": 0,
    "condition": 3,       # Mode
    "grade": 7,           # Mode
    # ... etc
}
```

**Why?** Model needs 43-47 features but user may only know 3 (beds, baths, zip).

#### Demographic Lookup
```python
def get_demographics(self, zipcode: str) -> Dict[str, float]:
    """O(1) lookup from in-memory DataFrame."""
    row = self._demographics_df[self._demographics_df['zipcode'] == zipcode]
    return row.iloc[0].to_dict()
```

#### Temporal Features (V3.3)
```python
def calculate_temporal_features(self, date: Optional[str] = None) -> Dict:
    """Extract sale_year, sale_month, sale_quarter, sale_dow."""
    if date is None:
        date = datetime.now()
    return {
        "sale_year": date.year,
        "sale_month": date.month,
        "sale_quarter": (date.month - 1) // 3 + 1,
        "sale_dow": date.weekday()
    }
```

---

### 3. Training Engine: `src/train_with_mlflow.py`

**Purpose:** Train models with full experiment tracking.

**Key Features:**

#### sklearn Pipeline
```python
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', XGBRegressor(**best_params))
])
```

**Why Pipeline?**
- Prevents training-serving skew
- Same preprocessing applied during train AND serve
- Serialized as single artifact

#### MLflow Integration
```python
with mlflow.start_run(run_name=f"train_{timestamp}"):
    mlflow.log_params(best_params)
    mlflow.log_metrics({"cv_mae": cv_mae, "cv_r2": cv_r2})
    mlflow.sklearn.log_model(pipeline, "model")
```

#### Data Source Selection
```python
@click.option('--data-source', type=click.Choice(['original', 'fresh']), default='fresh')
def train(data_source: str):
    if data_source == 'original':
        df = pd.read_csv('data/kc_house_data.csv')
    else:
        df = pd.read_csv('data/assessment_2020_plus_v4.csv')
```

---

### 4. Honest Auditor: `src/evaluate_fresh.py`

**Purpose:** Rigorous evaluation preventing data leakage.

**Key Features:**

#### GroupKFold Cross-Validation
```python
from sklearn.model_selection import GroupKFold

gkf = GroupKFold(n_splits=5)
for train_idx, test_idx in gkf.split(X, y, groups=df['parcel_id']):
    # Ensures same property never appears in both train and test
```

**Why GroupKFold?**
- Repeat sales (same house sold multiple times) create leakage
- Random split: house in train (2021 sale) + test (2023 sale) = memorization
- GroupKFold: entire property history goes to one fold

#### Residual Analysis
```python
def analyze_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    return {
        "mean_bias": residuals.mean(),  # Should be ~0
        "std": residuals.std(),
        "within_50k": (abs(residuals) < 50000).mean(),
        "within_100k": (abs(residuals) < 100000).mean()
    }
```

---

### 5. Optimizer: `src/tune_v33.py`

**Purpose:** Bayesian hyperparameter optimization.

**Key Features:**

#### Optuna Study
```python
import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 10.0)
    }
    
    cv_scores = cross_val_score(model, X, y, cv=gkf, scoring='neg_mean_absolute_error')
    return -cv_scores.mean()

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)
```

**Why Optuna over GridSearch?**
- Bayesian optimization learns from previous trials
- More efficient exploration of hyperparameter space
- Pruning of unpromising trials

---

### 6. ETL Pipeline: `src/data/transform_assessment_data.py`

**Purpose:** Transform raw King County Assessment data to model format.

**Key Features:**

#### Data Loading
```python
def load_assessment_data():
    sales = pd.read_csv('references/EXTR_RPSale.csv')
    buildings = pd.read_csv('references/EXTR_ResBldg.csv')
    parcels = pd.read_csv('references/parcel_centroids.csv')
    
    # Join on Major/Minor parcel ID
    df = sales.merge(buildings, on=['Major', 'Minor'])
    df = df.merge(parcels, on=['Major', 'Minor'])
```

#### Feature Mapping
```python
COLUMN_MAPPING = {
    'SalePrice': 'price',
    'Bedrooms': 'bedrooms',
    'SqFtTotLiving': 'sqft_living',
    'SqFtLot': 'sqft_lot',
    'Stories': 'floors',
    'BldgGrade': 'grade',
    # ... etc
}
```

#### Bathroom Calculation
```python
df['bathrooms'] = (
    df['BathFullCount'] + 
    0.75 * df['Bath3qtrCount'] + 
    0.5 * df['BathHalfCount']
)
```

#### Quality Filters
```python
# Remove distressed sales (bottom 5% per zipcode)
def filter_distressed(df):
    thresholds = df.groupby('zipcode')['price'].quantile(0.05)
    return df[df['price'] > df['zipcode'].map(thresholds)]

# Cap extreme values
df = df[df['price'] <= 3000000]
```

---

## Data Flow Diagrams

### Training Pipeline (V3.3)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    RAW DATA SOURCES                                  │
├─────────────────────────────────────────────────────────────────────┤
│  EXTR_RPSale.csv       EXTR_ResBldg.csv      parcel_centroids.csv  │
│  (Sales records)       (Building details)    (GIS coordinates)      │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│              transform_assessment_data.py                            │
├─────────────────────────────────────────────────────────────────────┤
│  • Join on Major/Minor                                               │
│  • Map column names                                                  │
│  • Calculate bathrooms                                               │
│  • Filter distressed sales                                           │
│  • Cap at $3M                                                        │
│  • Add temporal features                                             │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│              assessment_2020_plus_v4.csv                             │
│              (143,476 clean records)                                 │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
┌───────────────────────────┐   ┌───────────────────────────────────┐
│       tune_v33.py         │   │     train_with_mlflow.py          │
│    (Optuna tuning)        │   │     (Production training)         │
├───────────────────────────┤   ├───────────────────────────────────┤
│ • 30 trial Bayesian       │   │ • Loads best_params.json          │
│ • GroupKFold CV           │   │ • Trains final model              │
│ • Saves best_params.json  │   │ • Logs to MLflow                  │
└───────────────────────────┘   └───────────────────────────────────┘
                                            │
                                            ▼
                            ┌───────────────────────────────────┐
                            │         model/model.pkl           │
                            │    (sklearn Pipeline artifact)    │
                            └───────────────────────────────────┘
```

### Serving Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                      API REQUEST                                     │
│  POST /api/v1/predict                                                │
│  {"bedrooms": 3, "sqft_living": 2000, "zipcode": "98103", ...}      │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   Pydantic Validation                                │
│                   (src/models.py)                                    │
├─────────────────────────────────────────────────────────────────────┤
│  • Type checking                                                     │
│  • Range validation                                                  │
│  • Required fields                                                   │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   Feature Service                                    │
├─────────────────────────────────────────────────────────────────────┤
│  1. Fill missing with defaults                                       │
│  2. Lookup demographics by zipcode                                   │
│  3. Calculate temporal features                                      │
│  4. Combine into 47-feature vector                                   │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   Model Service                                      │
├─────────────────────────────────────────────────────────────────────┤
│  1. Create DataFrame from feature dict                               │
│  2. Ensure column order matches training                             │
│  3. pipeline.predict(X)                                              │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      API RESPONSE                                    │
│  {"predicted_price": 1291832.25, "model_version": "v3.3", ...}      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Configuration Files

### `src/config.py`

```python
class Settings(BaseSettings):
    model_path: str = "model/model.pkl"
    features_path: str = "model/model_features.json"
    demographics_path: str = "data/zipcode_demographics.csv"
    use_mlflow_model: bool = False
    mlflow_model_name: str = "real-estate-price-predictor"
    mlflow_model_stage: str = "Production"
    
    class Config:
        env_file = ".env"
```

### `model/metrics.json` (V3.3)

```json
{
  "version": "v3.3",
  "data_vintage": "2020-2024",
  "training_date": "2025-12-09",
  "model_type": "XGBRegressor (Pipeline)",
  "training_samples": 143476,
  "n_features": 47,
  "metrics": {
    "cv_mae": 115247,
    "cv_r2": 0.868,
    "cv_method": "GroupKFold (by parcel_id)"
  },
  "hyperparameters": {
    "n_estimators": 1500,
    "max_depth": 8,
    "learning_rate": 0.0285,
    "subsample": 0.858,
    "colsample_bytree": 0.614,
    "reg_alpha": 0.0079,
    "reg_lambda": 6.56
  }
}
```

---

## Docker Architecture

### Multi-Version Demo Setup

```
┌─────────────────────────────────────────────────────────────────────┐
│                    docker-compose.demo.yml                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐      │
│  │   V1 MVP        │  │   V2.5          │  │   V3.3          │      │
│  │   Port 8000     │  │   Port 8001     │  │   Port 8002     │      │
│  ├─────────────────┤  ├─────────────────┤  ├─────────────────┤      │
│  │ Dockerfile.v1   │  │ Dockerfile.v2.5 │  │ Dockerfile.v3.3 │      │
│  │ demo/v1/src/    │  │ demo/v2.5/src/  │  │ demo/v3.3/src/  │      │
│  │ KNN model       │  │ XGBoost model   │  │ XGBoost (Optuna)│      │
│  │ 2014-2015 data  │  │ 2014-2015 data  │  │ 2020-2024 data  │      │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Build Process

```bash
# V1: Simple KNN
docker build -f demo/Dockerfile.v1 -t real-estate-v1 .
# Runs: python src/train.py (KNN, 7 features)

# V2.5: XGBoost
docker build -f demo/Dockerfile.v2.5 -t real-estate-v2.5 .
# Runs: python src/train.py && python src/tune_xgboost.py --n-iter 20

# V3.3: Fresh Data
docker build -f demo/Dockerfile.v3.3 -t real-estate-v3.3 .
# Runs: python src/train_fresh_data.py
```

---

## CI/CD Pipeline (V3.3)

### `.github/workflows/ci.yml`

```yaml
name: CI
on:
  push:
    branches: [main, develop, 'feature/*']
  pull_request:
    branches: [main, develop]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v
```

### `.github/workflows/train.yml`

```yaml
name: Train Model
on:
  workflow_dispatch:
    inputs:
      data_source:
        description: 'Data source'
        required: true
        default: 'fresh'
        type: choice
        options: [original, fresh]

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install -r requirements.txt
      - run: python src/train_with_mlflow.py --data-source ${{ inputs.data_source }}
```

---

## Performance Characteristics

### Latency

| Operation | Time | Notes |
|-----------|------|-------|
| Model load (cold start) | 500-800ms | Once at startup |
| Prediction (warm) | 10-50ms | In-memory model |
| Feature enrichment | 1-5ms | O(1) zipcode lookup |
| Full request cycle | 15-60ms | Including validation |

### Memory

| Component | Memory | Notes |
|-----------|--------|-------|
| Model in memory | 50-100MB | XGBoost + Pipeline |
| Demographics cache | 5-10MB | 83 zipcodes x 26 features |
| FastAPI overhead | 50-100MB | Worker process |
| **Total per worker** | ~200MB | |

### Scalability

- Stateless design allows horizontal scaling
- Each worker holds its own model copy
- Load balance across multiple containers
- Kubernetes-ready with health checks
