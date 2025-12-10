# Model Version Evolution: Real Estate Price Predictor

**Document Created:** 2025-12-09  
**Current Production Version:** V2.5 XGBoost  
**Current Infrastructure Version:** V3.1 MLOps

---

## Version Timeline

```
Original (buggy) -> V1 -> V2.1 -> V2.3 -> V2.4 -> V2.4.1 -> V2.5 -> V2.7(archived) -> V3.1
     Dec 6          Dec 7   Dec 8   Dec 8   Dec 8    Dec 8     Dec 8      Dec 8         Dec 8
```

---

## Version Summary Table

| Version | Model | MAE | R2 | Key Change | Status |
|---------|-------|-----|-----|------------|--------|
| Original | KNN (k=5) | ~$100k+ | ~0.72 | Buggy script | Replaced |
| **V1** | KNN (k=5) | $102,045 | 0.7281 | Fixed + API | Complete |
| **V2.1** | KNN (k=5) | $89,769 | 0.7682 | +10 features | Complete |
| **V2.3** | KNN (tuned) | $84,494 | 0.7932 | Hyperparameter tuning | Complete |
| **V2.4** | Multiple | varies | varies | Model comparison | Complete |
| **V2.4.1** | XGBoost (default) | $67,041 | 0.8755 | Best model selected | Complete |
| **V2.5** | XGBoost (tuned) | $63,529 (CV) | 0.8945 (CV) | Robust evaluation | **PRODUCTION** |
| V2.7 | Tiered XGBoost | $65,687 | 0.8821 | Price-tier specialists | Archived |
| **V3.1** | (infrastructure) | - | - | MLOps & CI/CD | Complete |

---

## Detailed Version Records

---

### ORIGINAL (Pre-Project)

**Date:** Before 2025-12-06  
**Status:** Buggy - replaced

#### Tech Stack
- **Model:** KNeighborsRegressor (sklearn)
- **Pipeline:** RobustScaler + KNN
- **Framework:** Standalone script

#### Files
| Role | File | Notes |
|------|------|-------|
| Training | `Reference_Docs/mle-project-challenge-2/create_model.py` | Original buggy script |

#### Issues Found
- `DEMOGRAPHICS_PATH` pointed to wrong file (`kc_house_data.csv` instead of `zipcode_demographics.csv`)
- Only 7 features used despite 18 available
- No MLflow tracking
- No API

#### Data
- **Vintage:** 2014-2015 King County
- **Samples:** 21,613
- **Features:** 7 home + 26 demographic = 33 total

---

### V1: Initial Production Release

**Date:** 2025-12-07  
**Tag:** v1.0.0  
**Status:** Complete

#### Tech Stack
- **Model:** KNeighborsRegressor
  - `n_neighbors`: 5 (default)
  - `weights`: uniform (default)
  - `metric`: minkowski (default)
- **Pipeline:** RobustScaler + KNN
- **Framework:** FastAPI
- **Tracking:** MLflow (basic)
- **Container:** Docker

#### Files
| Role | File | Description |
|------|------|-------------|
| Entry Point | `src/main.py` | FastAPI application |
| Business Logic | `src/services/feature_service.py` | Demographics lookup |
| Training | `src/train.py` | Basic training script |
| Evaluation | `src/evaluate.py` | Evaluation script |
| API Routes | `src/api/prediction.py` | /health, /predict, /predict-minimal |

#### Features (V1)
| Category | Count | Features |
|----------|-------|----------|
| Home | 7 | bedrooms, bathrooms, sqft_living, sqft_lot, floors, sqft_above, sqft_basement |
| Demographic | 26 | population, income, education, etc. (from zipcode) |
| **Total** | **33** | |

#### Data
- **Vintage:** 2014-2015 King County
- **Source:** `data/kc_house_data.csv`, `data/zipcode_demographics.csv`
- **Samples:** 21,613 training, 70 zipcodes
- **Split:** 80/20 train/test

#### Performance Metrics
| Metric | Value |
|--------|-------|
| Test R2 | 0.7281 |
| Test MAE | $102,045 |
| Test RMSE | $201,659 |
| Train R2 | 0.8414 |
| Overfitting Gap | 0.1133 |

#### MLflow
- **Experiment:** real-estate-v1
- **Model Name:** real-estate-price-predictor

#### Changes from Original
1. Fixed DEMOGRAPHICS_PATH bug
2. Added MLflow tracking
3. Created FastAPI application
4. Added Docker support
5. Comprehensive documentation

---

### V2.1: Feature Expansion

**Date:** 2025-12-08  
**Branch:** feature/v2-model-improvements  
**Status:** Complete

#### Tech Stack
- **Model:** KNeighborsRegressor (unchanged from V1)
- **Pipeline:** RobustScaler + KNN
- **Framework:** FastAPI

#### Files
| Role | File | Changes |
|------|------|---------|
| Entry Point | `src/main.py` | Updated docstrings |
| Business Logic | `src/services/feature_service.py` | Added V21_DEFAULT_FEATURES |
| Training | `src/train.py` | Expanded SALES_COLUMN_SELECTION |
| Model Service | `src/services/model_service.py` | Version auto-detection |
| Models | `src/models.py` | Added PredictionRequestFullFeatures |
| API Routes | `src/api/prediction.py` | Added /predict-full |

#### Features (V2.1)
| Category | Count | Features |
|----------|-------|----------|
| Home | **17** | V1 features + waterfront, view, condition, grade, yr_built, yr_renovated, lat, long, sqft_living15, sqft_lot15 |
| Demographic | 26 | Unchanged |
| **Total** | **43** | +10 features |

#### New Features Added
| Feature | Type | Rationale |
|---------|------|-----------|
| waterfront | binary | Premium for waterfront property |
| view | ordinal (0-4) | Quality of view |
| condition | ordinal (1-5) | Overall condition |
| grade | ordinal (1-13) | Construction quality |
| yr_built | year | Age of home |
| yr_renovated | year | Renovation status |
| lat | float | Latitude (spatial) |
| long | float | Longitude (spatial) |
| sqft_living15 | sqft | Neighbors' living space |
| sqft_lot15 | sqft | Neighbors' lot size |

#### Performance Metrics
| Metric | V1 | V2.1 | Change |
|--------|-----|------|--------|
| Test R2 | 0.7281 | 0.7682 | **+5.5%** |
| Test MAE | $102,045 | $89,769 | **-12.0%** |
| Test RMSE | $201,659 | $186,207 | -7.7% |
| Overfitting Gap | 0.1133 | 0.0910 | -19.7% |

#### MLflow
- **Experiment:** real-estate-v2
- **Model Name:** real-estate-price-predictor-v2.1

---

### V2.1.1: Full Features Endpoint

**Date:** 2025-12-08  
**Status:** Complete (part of V2.1)

#### Changes
- Added `/predict-full` endpoint
- Accepts all 17 home features without zipcode
- Uses average demographics
- Best strategy for no-zipcode predictions

---

### V2.1.2: Adaptive Routing (Experimental)

**Date:** 2025-12-08  
**Status:** LOW PRIORITY - Deferred

#### Concept
Route predictions based on estimated price tier:
- LOW tier (<$400k): Use /predict-minimal
- HIGH tier (>=$400k): Use /predict-full

#### Finding
- Price-tier pattern confirmed (62% minimal wins LOW, 88% full wins HIGH)
- Routing accuracy only 52% - insufficient to beat single strategy

#### Decision
Defer - `/predict-full` is sufficient for all no-zipcode predictions

---

### V2.3: Hyperparameter Tuning

**Date:** 2025-12-08  
**Branch:** feature/v2.3-hyperparameter-tuning  
**Status:** Complete

#### Tech Stack
- **Model:** KNeighborsRegressor (TUNED)
  - `n_neighbors`: 7 (was 5)
  - `weights`: distance (was uniform)
  - `metric`: manhattan (was minkowski)
  - `p`: 1
- **Tuning:** GridSearchCV with 5-fold CV
- **Search Space:** 126 combinations (7x2x3x3)

#### Files
| Role | File | Changes |
|------|------|---------|
| Optimizer | `src/tune.py` | NEW - GridSearchCV tuning |
| Training | `src/train.py` | Updated for tuned params |
| Evaluation | `src/evaluate.py` | Updated to match tune.py |

#### Performance Metrics
| Metric | V2.1 | V2.3 | Change |
|--------|------|------|--------|
| Test R2 | 0.7682 | 0.7932 | **+3.3%** |
| Test MAE | $89,769 | $84,494 | **-5.9%** |
| CV MAE | N/A | $79,976 | - |

#### Key Insights
1. Distance weighting helps (closer neighbors matter more)
2. Manhattan distance better for 43 dimensions (curse of dimensionality)
3. Slightly more neighbors (7 vs 5) provides smoothing

#### MLflow
- **Experiment:** real-estate-v2.3-tuning
- **Model Name:** real-estate-price-predictor-v2.3

---

### V2.4: Model Alternatives

**Date:** 2025-12-08  
**Branch:** feature/v2.4-model-alternatives  
**Status:** Complete

#### Tech Stack
- **Models Compared:**
  - KNN (baseline from V2.3)
  - Random Forest
  - XGBoost
  - LightGBM
  - Ridge Regression

#### Files
| Role | File | Description |
|------|------|-------------|
| Comparison | `src/compare_models.py` | Multi-model evaluation |

#### Results
| Model | MAE | R2 | Notes |
|-------|-----|-----|-------|
| Ridge | $107,943 | 0.7168 | Linear - worst |
| KNN (V2.3) | $84,494 | 0.7932 | Baseline |
| Random Forest | $70,541 | 0.8594 | Good |
| LightGBM | $67,892 | 0.8731 | Very good |
| **XGBoost** | **$67,041** | **0.8755** | **BEST** |

#### Decision
XGBoost selected for production - 20.7% MAE improvement over KNN

#### Artifacts
- `logs/v2.4_model_comparison_*.csv`
- `logs/v2.4_feature_importance_*.csv`

---

### V2.4.1: XGBoost Tuning

**Date:** 2025-12-08  
**Status:** Complete

#### Tech Stack
- **Model:** XGBRegressor (Optuna tuned)
- **Tuning:** Optuna with 50 trials
- **Parameters:**
  ```python
  n_estimators: 239
  max_depth: 7
  learning_rate: 0.0863
  subsample: 0.7472
  colsample_bytree: 0.8388
  min_child_weight: 6
  gamma: 0.1589
  reg_alpha: 0.2791
  reg_lambda: 1.3826
  ```

#### Files
| Role | File | Description |
|------|------|-------------|
| Optimizer | `src/tune_xgboost.py` | Optuna-based XGBoost tuning |

#### Performance
| Metric | V2.4 (default) | V2.4.1 (tuned) |
|--------|----------------|----------------|
| Test MAE | $67,041 | ~$64,000 |
| Test R2 | 0.8755 | ~0.89 |

#### Artifacts
- `logs/v2.4.1_xgboost_feature_importance.csv`

---

### V2.5: Robust Evaluation (PRODUCTION)

**Date:** 2025-12-08  
**Status:** **CURRENT PRODUCTION MODEL**

#### Tech Stack
- **Model:** XGBRegressor (Optuna tuned, same as V2.4.1)
- **Evaluation:** 
  - 5-fold cross-validation
  - Bootstrap confidence intervals (500 samples)
  - Log transform experiment
  - Residual analysis

#### Files
| Role | File | Description |
|------|------|-------------|
| Evaluation | `src/robust_evaluate.py` | Comprehensive evaluation |
| Model | `model/model.joblib` | Production model |
| Metrics | `model/metrics.json` | Full metrics with CV |

#### Performance Metrics (Robust)
| Metric | Value | Notes |
|--------|-------|-------|
| **5-Fold CV MAE** | $63,529 +/- $2,150 | Primary metric |
| **5-Fold CV R2** | 0.8945 +/- 0.0168 | Primary metric |
| 5-Fold CV RMSE | $119,038 +/- $14,312 | |
| 95% CI MAE | [$63,590, $70,971] | Bootstrap |
| 95% CI R2 | [0.8421, 0.9089] | Bootstrap |

#### Log Transform Experiment
| Approach | CV MAE | High-Value MAE |
|----------|--------|----------------|
| Normal | $63,529 | $231,832 |
| Log Transform | $64,135 | $241,788 |
| **Winner** | **Normal** | **Normal** |

Recommendation: Keep normal target (log transform is 1% worse overall, 4.3% worse for high-value)

#### Residual Analysis
| Metric | Value | Interpretation |
|--------|-------|----------------|
| Mean Bias | $735 | Essentially unbiased |
| Median Bias | -$1,930 | Slight overpredict |
| Within $50k | 60.8% | |
| Within $100k | 82.6% | |
| Variance Ratio | 28.20 | Heteroscedastic |

#### Error by Price Range
| Price Range | Count | MAE | Bias |
|-------------|-------|-----|------|
| <$300k | 888 | $37,001 | Overpredict |
| $300k-$500k | 1,555 | $41,347 | Overpredict |
| $500k-$750k | 1,130 | $60,356 | Underpredict |
| $750k-$1M | 424 | $108,407 | Neutral |
| >$1M | 326 | $241,301 | Underpredict |

#### MLflow
- **Experiment:** real-estate-v2.5
- **Model Name:** real-estate-price-predictor-v2.5
- **Stage:** Production

#### Artifacts
- `logs/v2.5_robust_evaluation_*.json`
- `model/metrics.json` (updated)

---

### V2.7: Price-Tiered Models (ARCHIVED)

**Date:** 2025-12-08  
**Branch:** archive/v2.7-tiered-models-exploration  
**Status:** Archived - Insufficient ROI

#### Concept
Train specialized models for price tiers:
- Low-price specialist (<$400k)
- High-price specialist (>=$400k)
- Route predictions based on initial estimate

#### Results
| Strategy | MAE | vs V2.5 |
|----------|-----|---------|
| V2.5 Single Model | $63,529 | Baseline |
| Tiered Models | $65,687 | +3.4% worse |
| Best Achievable | $63,423 | -0.17% better |

#### Why It Failed
1. **Misrouting is catastrophic** - Wrong tier has huge MAE
2. **High-price specialist hurts** - Too much variance in luxury segment
3. **Low-price specialist helps** - But gains offset by misrouting
4. **ROI too low** - 0.17% improvement doesn't justify complexity

#### Decision
Archived - V2.5 single model is production

---

### V3.1: MLOps & CI/CD Infrastructure

**Date:** 2025-12-08  
**Branch:** feature/v3.1-mlops-cicd -> merged to main  
**Tag:** v3.1  
**Status:** Complete

#### Tech Stack
- **Tracking:** MLflow
- **CI/CD:** GitHub Actions
- **Testing:** pytest (13 tests)
- **Linting:** ruff + black

#### Files
| Role | File | Description |
|------|------|-------------|
| MLflow Config | `mlflow_config.py` | Configuration and helpers |
| Training (MLflow) | `src/train_with_mlflow.py` | Full experiment tracking |
| CI Pipeline | `.github/workflows/ci.yml` | lint, test, validate |
| Train Pipeline | `.github/workflows/train.yml` | Manual trigger training |
| Tests | `tests/test_model.py` | 13 tests |
| Tool Config | `pyproject.toml` | ruff, black, pytest |

#### CI/CD Jobs
| Job | Purpose | Status |
|-----|---------|--------|
| Code Quality | ruff + black | Passing |
| Tests | pytest (13 tests) | Passing |
| Model Validation | Load model, start API | Passing |
| CI Success | Aggregate | Passing |

#### MLflow Commands
```powershell
# Train with tracking
python src/train_with_mlflow.py

# View UI
mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db
```

#### Infrastructure Metrics
- Tests: 13/13 passing
- CI Time: ~2 minutes per run
- Code Coverage: Model loading, prediction, validation

---

## Cumulative Improvement Summary

| Metric | V1 | V2.5 | Total Improvement |
|--------|-----|------|-------------------|
| MAE | $102,045 | $63,529 (CV) | **-37.7%** |
| R2 | 0.7281 | 0.8945 (CV) | **+22.8%** |
| Features | 33 | 43 | +30.3% |
| Model | KNN | XGBoost | Algorithm change |

### Improvement Breakdown by Version
| Version | MAE Change | Cumulative |
|---------|------------|------------|
| V1 | Baseline | $102,045 |
| V2.1 | -12.0% | $89,769 |
| V2.3 | -5.9% | $84,494 |
| V2.4/V2.4.1 | -20.7% | $67,041 |
| V2.5 | -5.2% (CV) | $63,529 |

---

## Data Evolution

All versions use the same base data:

| Attribute | Value |
|-----------|-------|
| Source | King County Assessor |
| Vintage | 2014-2015 |
| Samples | 21,613 |
| Zipcodes | 70 |
| Home File | `data/kc_house_data.csv` |
| Demographics File | `data/zipcode_demographics.csv` |

### V3.2 Planned Data Update
- **Source:** King County Assessment 2024
- **Location:** `Reference_Docs/King_County_Assessment_data_ALL/`
- **Files:** EXTR_RPSale.csv (606 MB), EXTR_ResBldg.csv (146 MB), etc.
- **Status:** Downloaded, awaiting integration

---

## Model Registry (MLflow)

| Model Name | Version | Stage | Notes |
|------------|---------|-------|-------|
| real-estate-price-predictor | v1 | Archived | KNN baseline |
| real-estate-price-predictor-v2.1 | v1 | Archived | Feature expansion |
| real-estate-price-predictor-v2.3 | v1 | Archived | Tuned KNN |
| real-estate-price-predictor-v2.5 | v1 | **Production** | XGBoost tuned |

---

## File Inventory by Version

### Entry Points (src/main.py evolution)
| Version | Changes |
|---------|---------|
| V1 | Created FastAPI app |
| V2.1 | Updated docstrings |
| V3.1 | Black formatting |

### Business Logic (src/services/)
| Version | File | Changes |
|---------|------|---------|
| V1 | feature_service.py | Created |
| V2.1 | feature_service.py | Added V21_DEFAULT_FEATURES |
| V2.1 | model_service.py | Added version auto-detection |
| V3.1 | *.py | Black formatting |

### Training Scripts
| Version | File | Tech |
|---------|------|------|
| V1 | src/train.py | Basic training |
| V2.3 | src/tune.py | GridSearchCV |
| V2.4 | src/compare_models.py | Multi-model |
| V2.4.1 | src/tune_xgboost.py | Optuna |
| V2.5 | src/robust_evaluate.py | CV + Bootstrap |
| V3.1 | src/train_with_mlflow.py | MLflow tracking |

### Evaluation Scripts
| Version | File | Features |
|---------|------|----------|
| V1 | src/evaluate.py | Basic metrics |
| V2.5 | src/robust_evaluate.py | CV, CI, residuals |

---

**Document Version:** 1.0  
**Last Updated:** 2025-12-09
