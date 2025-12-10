# Model Version Evolution: V1, V2.5, V3.3
## Complete Technical Reference

---

## Version Comparison Matrix

### Core Technology Stack

| Component | V1 MVP | V2.5 Optimized | V3.3 Production |
|-----------|--------|----------------|-----------------|
| **Algorithm** | K-Nearest Neighbors | XGBoost | XGBoost (Optuna Tuned) |
| **Preprocessing** | StandardScaler | StandardScaler | StandardScaler |
| **Pipeline** | sklearn Pipeline | sklearn Pipeline | sklearn Pipeline |
| **Hyperparameter Tuning** | None | RandomizedSearchCV | Optuna (Bayesian) |
| **Tracking** | None | None | MLflow |
| **CI/CD** | None | None | GitHub Actions |

---

### File Structure by Version

#### Entry Point (`src/main.py`)

| Version | File | API Prefix | Key Differences |
|---------|------|------------|-----------------|
| V1 | `demo/v1/src/main.py` | `/` (no prefix) | Minimal endpoints: `/predict`, `/health` |
| V2.5 | `demo/v2.5/src/main.py` | `/api/v1/` | Added experimental endpoints |
| V3.3 | `src/main.py` | `/api/v1/` | MLflow integration, lifespan context |

#### Business Logic (`src/services/feature_service.py`)

| Version | File | Features Handled | Key Capabilities |
|---------|------|------------------|------------------|
| V1 | `demo/v1/src/services/feature_service.py` | 33 (7+26) | Basic demographic lookup |
| V2.5 | `demo/v2.5/src/services/feature_service.py` | 43 (17+26) | Tier routing, adaptive prediction |
| V3.3 | `src/services/feature_service.py` | 47 (17+26+4) | Temporal feature calculation |

#### Training Engine

| Version | File | Algorithm | Key Features |
|---------|------|-----------|--------------|
| V1 | `demo/v1/src/train.py` | KNN (k=5) | 7 home features only |
| V2.5 | `demo/v2.5/src/tune_xgboost.py` | XGBoost | RandomizedSearchCV, 20 iterations |
| V3.3 | `src/train_with_mlflow.py` | XGBoost | MLflow logging, flexible data source |

#### Evaluation (The Honest Auditor)

| Version | File | Method | Key Features |
|---------|------|--------|--------------|
| V1 | None | Train/Test Split | Basic metrics only |
| V2.5 | `demo/v2.5/src/robust_evaluate.py` | 5-Fold CV + Bootstrap | Confidence intervals, residual analysis |
| V3.3 | `src/evaluate_fresh.py` | GroupKFold | Prevents repeat-sale leakage |

#### Optimizer

| Version | File | Method | Trials |
|---------|------|--------|--------|
| V1 | None | N/A | N/A |
| V2.5 | `demo/v2.5/src/tune_xgboost.py` | RandomizedSearchCV | 20 |
| V3.3 | `src/tune_v33.py` | Optuna (Bayesian) | 30 |

#### ETL Pipeline

| Version | File | Data Source |
|---------|------|-------------|
| V1 | None (uses raw CSV) | `kc_house_data.csv` |
| V2.5 | None (uses raw CSV) | `kc_house_data.csv` |
| V3.3 | `src/data/transform_assessment_data.py` | King County Assessment 2020-2024 |

---

### Feature Sets by Version

#### V1 MVP: 33 Features (7 Home + 26 Demographic)

**Home Features (7):**
```
bedrooms, bathrooms, sqft_living, sqft_lot, floors, sqft_above, sqft_basement
```

**Demographic Features (26):** All from `zipcode_demographics.csv`

---

#### V2.5 Optimized: 43 Features (17 Home + 26 Demographic)

**Home Features (17):**
```
bedrooms, bathrooms, sqft_living, sqft_lot, floors, sqft_above, sqft_basement,
waterfront, view, condition, grade, yr_built, yr_renovated, lat, long,
sqft_living15, sqft_lot15
```

**V2.1 Additions (10 new features):**
| Feature | Type | Rationale |
|---------|------|-----------|
| waterfront | Binary | 50-100% premium in Seattle |
| view | Ordinal 0-4 | View quality premium |
| condition | Ordinal 1-5 | Maintenance state |
| grade | Ordinal 1-13 | Construction quality (high predictive value) |
| yr_built | Numeric | Age affects value |
| yr_renovated | Numeric | Renovation recency |
| lat | Numeric | Spatial value beyond zipcode |
| long | Numeric | Spatial value beyond zipcode |
| sqft_living15 | Numeric | Neighborhood context |
| sqft_lot15 | Numeric | Neighborhood context |

---

#### V3.3 Production: 47 Features (17 Home + 26 Demographic + 4 Temporal)

**Temporal Features (4) - NEW in V3.3:**
```
sale_year, sale_month, sale_quarter, sale_dow
```

| Feature | Type | Rationale |
|---------|------|-----------|
| sale_year | Numeric | Captures market appreciation |
| sale_month | Numeric | Seasonal patterns |
| sale_quarter | Numeric | Quarterly trends |
| sale_dow | Numeric | Day-of-week patterns |

---

### Data Vintage Comparison

| Aspect | V1 MVP | V2.5 Optimized | V3.3 Production |
|--------|--------|----------------|-----------------|
| **Data Source** | `kc_house_data.csv` | `kc_house_data.csv` | King County Assessment |
| **Date Range** | 2014-05 to 2015-05 | 2014-05 to 2015-05 | 2020-01 to 2024-12 |
| **Records** | 21,613 | 21,613 | 143,476 |
| **Median Price** | $450,000 | $450,000 | $845,000 |
| **Price Range** | $75K - $7.7M | $75K - $7.7M | $150K - $3M (capped) |
| **Filters Applied** | None | None | Distressed + $3M cap |

---

### Performance Metrics

| Metric | V1 MVP | V2.5 Optimized | V3.3 Production |
|--------|--------|----------------|-----------------|
| **R² Score** | 0.728 | 0.876 | 0.868 |
| **MAE** | $102,045 | $67,041 | $115,247 |
| **MAPE** | ~22% | ~14% | ~14% |
| **CV Method** | Train/Test | 5-Fold CV | GroupKFold |
| **Confidence Interval** | None | 95% CI | Yes |

**Note on V3.3 MAE:** Higher absolute MAE reflects 88% median price inflation (2015 → 2024). MAPE remains comparable to V2.5.

---

### Model Artifacts

| Artifact | V1 MVP | V2.5 Optimized | V3.3 Production |
|----------|--------|----------------|-----------------|
| **Model File** | `model/model.pkl` | `model/model_v2.4.1_xgboost_tuned.pkl` | `model/model.pkl` |
| **Features File** | `model/model_features.json` | `model/model_features.json` | `model/model_features.json` |
| **Metrics File** | `model/metrics.json` | `model/metrics_v2.4.1.json` | `model/metrics.json` |
| **MLflow Registered** | No | No | Yes |
| **Model Registry Name** | N/A | N/A | `real-estate-price-predictor` |

---

### XGBoost Hyperparameters (V2.5 vs V3.3)

| Parameter | V2.5 (RandomizedSearchCV) | V3.3 (Optuna) |
|-----------|---------------------------|---------------|
| n_estimators | 500 | 1500 |
| max_depth | 6 | 8 |
| learning_rate | 0.1 | 0.0285 |
| subsample | 0.8 | 0.858 |
| colsample_bytree | 0.8 | 0.614 |
| reg_alpha | 0 | 0.0079 |
| reg_lambda | 1 | 6.56 |
| min_child_weight | 1 | 3 |

---

### API Endpoints by Version

#### V1 MVP
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/predict` | POST | Price prediction |

#### V2.5 Optimized
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health` | GET | Health check |
| `/api/v1/predict` | POST | Full prediction (17 features) |
| `/api/v1/predict-minimal` | POST | Minimal input (7 features + defaults) |
| `/api/v1/predict-adaptive` | POST | Tier-based routing (experimental) |

#### V3.3 Production
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health` | GET | Health check with MLflow status |
| `/api/v1/predict` | POST | Full prediction (47 features) |
| `/api/v1/predict-minimal` | POST | Minimal input with temporal defaults |
| `/api/v1/model-info` | GET | Model metadata |

---

### Docker Configuration

| Setting | V1 MVP | V2.5 Optimized | V3.3 Production |
|---------|--------|----------------|-----------------|
| **Base Image** | python:3.11-slim | python:3.11-slim | python:3.11-slim |
| **Demo Port** | 8000 | 8001 | 8002 |
| **Train Script** | `train.py` | `train.py` + `tune_xgboost.py` | `train_fresh_data.py` |
| **Health Endpoint** | `/health` | `/api/v1/health` | `/api/v1/health` |

---

### Key Architectural Decisions

#### V1 MVP
- **Why KNN?** Simple baseline that matches original `create_model.py` specification
- **Why 7 features?** Minimum viable product - only the most obvious features
- **Why no /api/v1/ prefix?** Bare minimum MVP, versioning added later

#### V2.5 Optimized
- **Why XGBoost?** Model comparison showed 34% MAE improvement over KNN
- **Why 17 features?** Analysis showed `grade`, `waterfront`, `lat/long` highly predictive
- **Why experimental endpoints?** Explored tier-based routing (ultimately abandoned for +0.17% insufficient ROI)

#### V3.3 Production
- **Why fresh data?** 2014-2015 model predicting 2024 prices is fundamentally flawed
- **Why GroupKFold?** Discovered repeat-sale leakage inflating metrics
- **Why Optuna?** More efficient hyperparameter search than grid/random
- **Why temporal features?** Captures seasonality and market trends in multi-year data
