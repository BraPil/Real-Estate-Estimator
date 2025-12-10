# Model Version Evolution: Real-Estate-Estimator

This document tracks the complete evolution of the model through all versions, including tech stack, key files, data vintage, features, and performance metrics.

---

## Version Matrix

| Version | Model Type | MAE | R² | Features | Status |
|---------|------------|-----|-----|----------|--------|
| V1 | KNN (k=5) | $102,045 | 0.7281 | 7 home | ✅ Complete |
| V2.1 | KNN (k=5) | $89,769 | 0.7682 | 17 home + 26 demo | ✅ Complete |
| V2.3 | KNN (tuned) | $84,494 | 0.7932 | 17 home + 26 demo | ✅ Complete |
| V2.4.1 | XGBoost | $67,041 | 0.8755 | 17 home + 26 demo | ✅ Complete |
| **V2.5** | **XGBoost (tuned)** | **$63,529** | **0.8945** | **43 total** | ✅ **Production** |
| V2.7 | Tiered XGBoost | $65,687 | 0.8821 | 43 total | ⏸️ Archived |

---

## V1: Baseline

**Date:** Initial project start  
**Status:** ✅ Complete  
**Branch:** `main` (initial)

### Tech Stack
| Component | Technology |
|-----------|------------|
| Model | `KNeighborsRegressor` (sklearn) |
| Preprocessing | `RobustScaler` |
| API | FastAPI |
| Serialization | pickle |

### Key Files
| Role | File | Description |
|------|------|-------------|
| Entry Point | `src/main.py` | FastAPI application |
| Business Logic | `src/services/feature_service.py` | Feature enrichment |
| Model Service | `src/services/model_service.py` | Model loading/prediction |
| Training | `src/train.py` | Basic training script |
| Evaluation | `src/evaluate.py` | Model evaluation |
| API Routes | `src/api/prediction.py` | Prediction endpoints |

### Data
| Attribute | Value |
|-----------|-------|
| Data Vintage | 2014-2015 |
| Source | King County housing sales |
| Samples | 21,613 |
| Train/Test Split | 80/20 |

### Features (7 Home Features)
```
bedrooms, bathrooms, sqft_living, sqft_lot, floors, sqft_above, sqft_basement
```

### Model Configuration
```python
{
    "model_type": "KNeighborsRegressor",
    "n_neighbors": 5,
    "weights": "uniform",
    "metric": "minkowski",
    "p": 2
}
```

### Performance Metrics
| Metric | Value |
|--------|-------|
| MAE | $102,045 |
| RMSE | $202,000 |
| R² | 0.7281 |

---

## V2.1: Feature Expansion

**Date:** 2025-12-07  
**Status:** ✅ Complete  
**Branch:** `feature/v2.1-features`

### Tech Stack
| Component | Technology |
|-----------|------------|
| Model | `KNeighborsRegressor` (sklearn) |
| Preprocessing | `RobustScaler` |
| Demographics | zipcode-level census data |

### Changes from V1
- Added 10 additional home features
- Added 26 demographic features from zipcode data
- Created `/predict-full` endpoint

### Features (17 Home + 26 Demographic = 43 Total)

**Home Features (17):**
```
bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view, 
condition, grade, sqft_above, sqft_basement, yr_built, yr_renovated, 
lat, long, sqft_living15, sqft_lot15
```

**Demographic Features (26):**
```
ppltn_qty, urbn_ppltn_qty, sbrbn_ppltn_qty, farm_ppltn_qty, non_farm_qty,
medn_hshld_incm_amt, medn_incm_per_prsn_amt, hous_val_amt,
edctn_less_than_9_qty, edctn_9_12_qty, edctn_high_schl_qty, edctn_some_clg_qty,
edctn_assoc_dgre_qty, edctn_bchlr_dgre_qty, edctn_prfsnl_qty,
per_urbn, per_sbrbn, per_farm, per_non_farm,
per_less_than_9, per_9_to_12, per_hsd, per_some_clg, per_assoc, per_bchlr, per_prfsnl
```

### Performance Metrics
| Metric | Value | Improvement |
|--------|-------|-------------|
| MAE | $89,769 | -12.0% |
| R² | 0.7682 | +5.5% |

---

## V2.3: Hyperparameter Tuning

**Date:** 2025-12-07  
**Status:** ✅ Complete  
**Branch:** `feature/v2.3-tuning`

### Tech Stack
| Component | Technology |
|-----------|------------|
| Model | `KNeighborsRegressor` (tuned) |
| Tuning | `GridSearchCV` (sklearn) |
| Preprocessing | `RobustScaler` |

### New Files
| Role | File | Description |
|------|------|-------------|
| Optimizer | `src/tune.py` | GridSearchCV tuning |

### Model Configuration (Tuned)
```python
{
    "model_type": "KNeighborsRegressor",
    "n_neighbors": 7,          # Changed from 5
    "weights": "distance",     # Changed from uniform
    "metric": "manhattan",     # Changed from minkowski
}
```

### Performance Metrics
| Metric | Value | Improvement |
|--------|-------|-------------|
| MAE | $84,494 | -5.9% from V2.1 |
| R² | 0.7932 | +3.3% |

---

## V2.4.1: XGBoost Model

**Date:** 2025-12-07  
**Status:** ✅ Complete  
**Branch:** `feature/v2.4-model-alternatives`

### Tech Stack
| Component | Technology |
|-----------|------------|
| Model | `XGBRegressor` (xgboost) |
| Tuning | `Optuna` |
| Preprocessing | `RobustScaler` |

### New Files
| Role | File | Description |
|------|------|-------------|
| Model Comparison | `src/compare_models.py` | Compare KNN vs XGBoost vs RF |
| XGBoost Tuning | `src/tune_xgboost.py` | Optuna hyperparameter tuning |
| Top Model Tuning | `src/tune_top_models.py` | Fine-tune best models |

### Model Configuration
```python
{
    "model_type": "XGBRegressor",
    "n_estimators": 239,
    "max_depth": 7,
    "learning_rate": 0.0863,
    "subsample": 0.7472,
    "colsample_bytree": 0.8388,
    "min_child_weight": 6,
    "gamma": 0.1589,
    "reg_alpha": 0.2791,
    "reg_lambda": 1.3826,
    "random_state": 42
}
```

### Performance Metrics
| Metric | Value | Improvement |
|--------|-------|-------------|
| MAE | $67,041 | -20.7% from V2.3 |
| R² | 0.8755 | +10.4% |

---

## V2.5: Robust Evaluation (Production)

**Date:** 2025-12-08  
**Status:** ✅ Complete (Production Model)  
**Branch:** `main`

### Tech Stack
| Component | Technology |
|-----------|------------|
| Model | `XGBRegressor` (tuned) |
| Evaluation | K-Fold CV, Bootstrap CI |
| Statistical | `scipy.stats` |
| Preprocessing | `RobustScaler` |

### New Files
| Role | File | Description |
|------|------|-------------|
| Honest Auditor | `src/robust_evaluate.py` | Comprehensive evaluation |

### Evaluation Components
1. **K-Fold Cross-Validation:** 5-fold CV with mean ± std
2. **Bootstrap Confidence Intervals:** 500-sample bootstrap for 95% CI
3. **Log Transform Experiment:** Tested log(price) vs price
4. **Residual Analysis:** Error distribution by price range

### Model Configuration (Same as V2.4.1)
```python
{
    "model_type": "XGBRegressor",
    "n_estimators": 239,
    "max_depth": 7,
    "learning_rate": 0.0863,
    "subsample": 0.7472,
    "colsample_bytree": 0.8388,
    "min_child_weight": 6,
    "gamma": 0.1589,
    "reg_alpha": 0.2791,
    "reg_lambda": 1.3826
}
```

### Performance Metrics (Cross-Validated)
| Metric | Value | Notes |
|--------|-------|-------|
| CV MAE | $63,529 ± $2,150 | 5-fold cross-validation |
| CV R² | 0.8945 ± 0.0168 | 5-fold cross-validation |
| CV RMSE | $119,038 ± $14,312 | 5-fold cross-validation |
| 95% CI MAE | [$63,590, $70,971] | Bootstrap |
| 95% CI R² | [0.8421, 0.9089] | Bootstrap |

### Residual Analysis
| Metric | Value |
|--------|-------|
| Mean Bias | $735 (essentially unbiased) |
| Median Bias | -$1,930 |
| Std Residual | $135,140 |
| Within $50k | 60.8% |
| Within $100k | 82.6% |
| Heteroscedastic | Yes (variance ratio 28.20) |

### Log Transform Experiment
| Target | MAE |
|--------|-----|
| Normal (price) | $63,529 |
| Log (log(price)) | $64,135 |
| **Winner** | **Normal** |

---

## V2.7: Tiered Models (Archived)

**Date:** 2025-12-08  
**Status:** ⏸️ Archived (Insufficient ROI)  
**Branch:** `archive/v2.7-tiered-models-exploration`

### Tech Stack
| Component | Technology |
|-----------|------------|
| Models | `XGBRegressor` (multiple specialists) |
| Classifier | `GradientBoostingClassifier` |
| Calibration | `CalibratedClassifierCV` |
| Preprocessing | `RobustScaler` |

### Archived Files
| Role | File | Description |
|------|------|-------------|
| Analysis | `src/explore_price_tier_predictors.py` | Tier correlation analysis |
| Two-Tier | `src/tiered_model_system.py` | Basic two-tier system |
| Optimizer | `src/optimize_tier_split.py` | Percentile optimization |
| Hybrid | `src/hybrid_tiered_system.py` | Three-zone system |
| Confidence | `src/confidence_routing_system.py` | Confidence-based routing |
| Low Specialist | `src/low_specialist_optimization.py` | Low-tier only |

### Experiments Conducted

#### Experiment 2.7a: Two-Tier System
```
Routing: 50th percentile split
Classifier: Gradient Boosting (91.3% accuracy)
Result: MAE $75,541 (+14.7% WORSE)
Problem: 8.1% misrouting with $119k average error
```

#### Experiment 2.7b: Three-Zone Hybrid
```
Routing: Bottom 25% → Low model
         Middle 50% → Baseline
         Top 25% → High model
Result: MAE $66,003 (+0.3% worse)
```

#### Experiment 2.7c: Confidence-Based Routing
```
Routing: Use classifier confidence scores
         Low confidence → Baseline model
Result: Still worse than baseline
```

#### Experiment 2.7d: Low-Only Specialist
```
Routing: Only bottom tier uses specialist
         Everything else → Baseline
Result: MAE $65,687 (+0.17% improvement)
```

### Why Abandoned
- **Best improvement:** Only 0.17% (not worth complexity)
- **Key insight:** Misrouting penalty overwhelms segment benefits
- **Decision:** ROI insufficient for production complexity

---

## V3.1: MLOps & CI/CD (Infrastructure)

**Date:** 2025-12-08  
**Status:** ✅ Complete  
**Branch:** `main` (merged)  
**Tag:** `v3.1`

### Tech Stack
| Component | Technology |
|-----------|------------|
| Experiment Tracking | MLflow 2.x |
| CI/CD | GitHub Actions |
| Linting | Ruff |
| Formatting | Black |
| Testing | Pytest |
| Coverage | pytest-cov |

### New Files
| Role | File | Description |
|------|------|-------------|
| MLflow Config | `mlflow_config.py` | Configuration and helpers |
| Training Engine | `src/train_with_mlflow.py` | MLflow-integrated training |
| CI Pipeline | `.github/workflows/ci.yml` | Lint, test, validate |
| Train Pipeline | `.github/workflows/train.yml` | Manual training trigger |
| Project Config | `pyproject.toml` | Tool configurations |
| Test Suite | `tests/test_model.py` | 13 tests |

### MLflow Configuration
```python
MLFLOW_TRACKING_URI = "sqlite:///mlflow/mlflow.db"
MLFLOW_ARTIFACT_LOCATION = (MLFLOW_DIR / "artifacts").as_uri()
MLFLOW_EXPERIMENT_NAME = "real-estate-predictor"
MLFLOW_MODEL_NAME = "real-estate-xgboost"
```

### GitHub Actions Jobs

**CI Pipeline (ci.yml):**
```
Jobs: lint → test → validate-model → ci-success
Triggers: push/PR to main, develop
Python: 3.11
```

**Train Pipeline (train.yml):**
```
Jobs: train → evaluate → notify
Triggers: workflow_dispatch (manual)
Parameters: n_estimators, max_depth, learning_rate, register_model
```

### Test Suite (13 Tests)
```
TestModelLoading: 4 tests
  - test_model_file_exists
  - test_model_is_pipeline
  - test_model_has_predict
  - test_model_has_steps

TestPrediction: 4 tests
  - test_predict_returns_array
  - test_predict_returns_correct_shape
  - test_predict_returns_positive
  - test_predict_reasonable_range

TestInputValidation: 2 tests
  - test_handles_single_sample
  - test_handles_multiple_samples

TestPerformance: 1 test
  - test_prediction_speed

TestMLflowConfig: 2 tests
  - test_mlflow_config_imports
  - test_mlflow_setup
```

### Model Metrics (Unchanged from V2.5)
| Metric | Value |
|--------|-------|
| CV MAE | $63,529 ± $2,150 |
| CV R² | 0.8945 ± 0.0168 |

---

## V3.2: Fresh Data Integration (Planned)

**Date:** 2025-12-09 (Starting)  
**Status:** 🚀 Branch Ready  
**Branch:** `feature/v3.2-fresh-data`

### Planned Tech Stack
| Component | Technology |
|-----------|------------|
| Data Source | King County Assessment Data (2024) |
| ETL | `src/data/transform_assessment_data.py` (TBD) |
| Evaluation | `src/evaluate_fresh.py` (TBD) |

### Planned Files
| Role | File | Description |
|------|------|-------------|
| ETL | `src/data/load_assessment_data.py` | Load assessment files |
| Transform | `src/data/transform_assessment_data.py` | Schema mapping |
| Evaluator | `src/evaluate_fresh.py` | Zero-shot evaluation |

### Data Source
```
Reference_Docs/King_County_Assessment_data_ALL/
├── EXTR_RPSale.csv           # Sales data (606 MB)
├── EXTR_ResBldg.csv          # Building data (146 MB)
├── EXTR_Parcel.csv           # Parcel data (234 MB)
├── EXTR_LookUp.csv           # Code lookups
└── ... (30+ files)
```

### Goals
1. Load and transform 2024 King County assessment data
2. Map schema to existing 17 home features
3. Test V2.5 model on new data (zero-shot)
4. Measure model drift/degradation
5. Plan retraining if needed

---

## Cumulative Improvement

| From → To | MAE Change | R² Change |
|-----------|------------|-----------|
| V1 → V2.1 | -12.0% | +5.5% |
| V2.1 → V2.3 | -5.9% | +3.3% |
| V2.3 → V2.4.1 | -20.7% | +10.4% |
| V2.4.1 → V2.5 | -5.2% (CV) | +2.2% (CV) |
| **V1 → V2.5** | **-37.7%** | **+22.8%** |

---

## Data Vintage Reference

| Version | Training Data | Vintage |
|---------|--------------|---------|
| V1-V2.5 | `data/kc_house_data.csv` | 2014-2015 |
| V3.2+ | King County Assessment | 2024 (planned) |

---

**Document Version:** 1.0  
**Last Updated:** 2025-12-09
