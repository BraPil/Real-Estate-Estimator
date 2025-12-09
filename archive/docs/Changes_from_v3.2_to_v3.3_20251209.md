# Changes from V3.2 to V3.3 (December 9, 2025)

## Summary

V3.3 introduces data quality filters, temporal features, and Optuna hyperparameter optimization. Critical finding: data leakage from repeat property sales was identified and addressed with GroupKFold evaluation.

---

## Files Added

### `src/tune_v33.py`
- **Purpose:** Optuna hyperparameter optimization for XGBoost
- **Features:**
  - 30-trial Bayesian optimization
  - Cross-validation scoring
  - Saves best params to JSON
  - Saves improved model automatically

### `model/best_params.json`
- **Purpose:** Store best hyperparameters from tuning
- **Content:** XGBoost parameters (n_estimators, max_depth, learning_rate, etc.)

---

## Files Modified

### `src/data/transform_assessment_data.py`
- Added `filter_distressed` parameter (removes bottom 5% per zipcode)
- Added `cap_price` parameter (removes properties above threshold)
- Added temporal feature extraction: `sale_year`, `sale_month`, `sale_quarter`, `sale_dow`

### `src/train_fresh_data.py`
- Added temporal features to `feature_cols` list
- Updated data path to use v4 dataset

### `src/evaluate_fresh.py`
- Added temporal features to feature columns
- Updated data path for v4 dataset

### `src/train_with_mlflow.py`
- Added `--data-source` CLI argument
- Choices: `original` (kc_house_data.csv) or `fresh` (assessment data)
- Default changed to `fresh`

### `src/services/feature_service.py`
- Added temporal feature calculation from `date` field
- Added defaults: current year/month/quarter/dow

### `tests/test_model.py`
- Added temporal features to test fixtures
- Updated feature count expectations

### `.github/workflows/ci.yml`
- Added `feature/*` to branch triggers
- CI now runs on feature branches

### `.github/workflows/train.yml`
- Added `data_source` workflow input
- Default: `fresh`
- Passes `--data-source` to training script

### `.gitignore`
- Added `mlflow.db` patterns
- Added `mlflow/artifacts/` pattern
- Excluded MLflow databases from version control

---

## Data Changes

| Dataset | V3.2 | V3.3 |
|---------|------|------|
| File | assessment_2020_plus_v3.csv | assessment_2020_plus_v4.csv |
| Records | 155,855 | 143,476 |
| Features | 43 | 47 |
| Filters | None | Distressed + $3M cap |
| Temporal | No | Yes |

---

## Configuration Changes

### Model Hyperparameters
| Parameter | V3.2 | V3.3 |
|-----------|------|------|
| n_estimators | 500 | 1500 |
| max_depth | 6 | 8 |
| learning_rate | 0.1 | 0.0285 |
| subsample | 1.0 | 0.858 |
| colsample_bytree | 1.0 | 0.614 |
| reg_alpha | 0 | 0.0079 |
| reg_lambda | 1 | 6.56 |

---

## Breaking Changes

None. API contract unchanged. New temporal features have sensible defaults.

---

## Dependencies

No new dependencies added.

---

**Version:** 3.3.0  
**Tag:** v3.3.0  
**Date:** 2025-12-09
