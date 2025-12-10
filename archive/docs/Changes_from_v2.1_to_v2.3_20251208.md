# Changes from V2.1 to V2.3

**Project:** Real Estate Price Predictor  
**Date:** 2025-12-08  
**Branch:** `feature/v2.3-hyperparameter-tuning`

---

## Overview

V2.3 focuses on hyperparameter tuning using GridSearchCV. No new features were added - instead, we optimized the existing KNN model configuration.

**Key Achievement:** Reduced MAE by $5,275 (5.9%) through optimal hyperparameter selection.

---

## V2.1 Baseline (Reference)

| Metric | V2.1 Value |
|--------|-----------|
| Test R² | 0.7682 |
| Test MAE | $89,769 |
| Test RMSE | $186,207 |
| n_neighbors | 5 (default) |
| weights | uniform (default) |
| metric | minkowski (default) |
| p | 2 (default) |
| Features | 43 |

---

## Complete List of Changes

### 1. Hyperparameter Changes

| Parameter | V2.1 | V2.3 | Why |
|-----------|------|------|-----|
| `n_neighbors` | 5 | **7** | Smoother predictions |
| `weights` | uniform | **distance** | Closer neighbors weighted more |
| `metric` | minkowski | **manhattan** | Better for high-dimensional data |
| `p` | 2 | **1** | L1 norm (manhattan) |

### 2. New Training Script

**Created:** `src/tune.py`

```python
# Key components:
- GridSearchCV with 5-fold CV
- 126 parameter combinations tested
- Scoring: neg_mean_absolute_error
- Parallel execution (n_jobs=-1)
- MLflow logging
```

### 3. Model Artifacts

| Artifact | V2.1 | V2.3 |
|----------|------|------|
| `model/model.pkl` | Default KNN | **Tuned KNN** |
| `model/metrics.json` | V2.1 metrics | **V2.3 metrics** |
| `model/tuning_results.json` | N/A | **NEW** |
| `model/model_v2.3_tuned.pkl` | N/A | **NEW** |

---

## Performance Comparison

### Test Set Metrics

| Metric | V2.1 | V2.3 | Change | % Change |
|--------|------|------|--------|----------|
| R² | 0.7682 | 0.7932 | +0.0250 | **+3.3%** |
| MAE | $89,769 | $84,494 | -$5,275 | **-5.9%** |
| RMSE | $186,207 | $176,796 | -$9,411 | **-5.1%** |

### Overfitting Comparison

| Metric | V2.1 | V2.3 |
|--------|------|------|
| Train-Test Gap | 0.0910 | 0.0568 |
| Status | Acceptable | **Better** |

---

## Files Modified Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `src/tune.py` | **NEW** | GridSearchCV tuning script |
| `model/model.pkl` | REPLACED | Tuned model for production |
| `model/metrics.json` | UPDATED | V2.3 performance metrics |
| `model/tuning_results.json` | **NEW** | Best parameters and detailed results |
| `model/model_v2.3_tuned.pkl` | **NEW** | Tuned model artifact |
| `logs/v2.3_grid_search_results.csv` | **NEW** | All 126 combinations |
| `logs/v2.3_implementation_log.md` | **NEW** | Implementation log |

---

## API Behavior Changes

**None** - The API code is unchanged. It automatically loads the updated `model/model.pkl`.

### Model Version Detection

The model service auto-detects version based on feature count:
- 33 features → "v1"
- 43 features → "v2.1" (same as V2.3, both use 43 features)

**Note:** Consider adding explicit version tracking in future.

---

## What Did NOT Change

- ✅ Feature count: Still 43 (17 home + 26 demographic)
- ✅ Feature names: Identical to V2.1
- ✅ API endpoints: All 4 endpoints unchanged
- ✅ Data pipeline: Same data loading and preprocessing
- ✅ RobustScaler: Still used for normalization

---

## Methodology Notes

### Why GridSearchCV?

| Method | Pros | Cons |
|--------|------|------|
| Manual tuning | Fast | Misses interactions, biased |
| **GridSearchCV** | Exhaustive, CV prevents overfitting | Slower |
| RandomizedSearchCV | Faster for large spaces | May miss optimum |
| Bayesian optimization | Efficient | Complex setup |

GridSearchCV was chosen because:
1. Parameter space is small (126 combinations)
2. 5-fold CV gives robust estimates
3. Easy to parallelize

### Why Manhattan Distance?

Manhattan (L1) vs Euclidean (L2):

```
High dimensions (43 features):
- Euclidean: All points seem equally far
- Manhattan: Distances remain discriminative
```

This is the "curse of dimensionality" - Manhattan suffers less from it.

---

## Reproducibility

To reproduce these results:

```bash
python src/tune.py --cv-folds 5 --test-size 0.2 --random-state 42
```

**Fixed seeds:**
- `train_test_split(random_state=42)`
- 5-fold CV (default sklearn behavior)

---

**Document Version:** 1.0  
**Last Updated:** 2025-12-08
