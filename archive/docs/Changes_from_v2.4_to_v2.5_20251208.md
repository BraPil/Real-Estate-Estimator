# Changes from V2.4 to V2.5

**Date:** 2025-12-08  
**Branch:** `feature/v2.5-robust-evaluation`

---

## Overview

V2.5 adds statistically rigorous model evaluation without changing the production model. The XGBoost model from V2.4.1 remains in production, but we now have better confidence in its performance metrics.

---

## Files Added

| File | Purpose |
|------|---------|
| `src/robust_evaluate.py` | Comprehensive evaluation with K-fold CV, bootstrap CI, log transform experiment, residual analysis |
| `logs/v2.5_robust_evaluation_20251208_094937.json` | Detailed JSON results from robust evaluation |
| `docs/V2.5_Robust_Evaluation_Summary.md` | Summary documentation of V2.5 findings |
| `docs/Changes_from_v2.4_to_v2.5_20251208.md` | This file |

---

## Files Modified

### `model/metrics.json`
**Changes:**
- Added `robust_evaluation` section with:
  - K-fold CV results (5-fold)
  - Bootstrap 95% confidence intervals
  - Log transform experiment results
  - Residual analysis metrics
- Updated version to `v2.5`
- Updated `total_improvement` to reflect CV metrics

**Key additions:**
```json
"robust_evaluation": {
  "kfold_cv": {
    "k_folds": 5,
    "mae_mean": 63529,
    "mae_std": 2150,
    "r2_mean": 0.8945,
    "r2_std": 0.0168
  },
  "bootstrap_ci_95": {
    "mae": { "ci_lower": 63590, "ci_upper": 70971 },
    "r2": { "ci_lower": 0.8421, "ci_upper": 0.9089 }
  }
}
```

### `docs/V2_Detailed_Roadmap.md`
**Changes:**
- V2.5 section updated from planned to ✅ COMPLETE
- Added results table with key metrics
- Documented log transform findings
- Listed files created

### `RESTART_20251208.md`
**Changes:**
- Status updated to "V2.5 COMPLETE"
- Performance history table updated with V2.5 row
- Model configuration updated with robust evaluation metrics
- Version status table updated

### `logs/human_in_the_loop_corrections.md`
**Changes:**
- Added Correction #8: numpy bool JSON serialization error

---

## No Changes To

| File | Reason |
|------|--------|
| `model/model.pkl` | V2.5 is evaluation only, no model changes |
| `src/train.py` | No training changes |
| `src/services/*` | No API changes |
| `requirements.txt` | scipy already available (used for stats) |

---

## Key Metrics Comparison

| Metric | V2.4.1 (Test Set) | V2.5 (5-Fold CV) |
|--------|-------------------|------------------|
| MAE | $67,041 | $63,529 ± $2,150 |
| R² | 0.8755 | 0.8945 ± 0.0168 |
| RMSE | $137,174 | $119,038 ± $14,312 |

---

## Key Findings

1. **Cross-validation confirms strong performance**
   - CV metrics slightly better than single test set
   - Consistent across all 5 folds

2. **Log transform NOT recommended**
   - 1% worse overall
   - 4.3% worse on high-value homes
   - Keep normal target

3. **Model is unbiased**
   - Mean residual only $735
   - No systematic over/under prediction

4. **Heteroscedasticity confirmed**
   - Variance ratio 28.20
   - Errors scale with price
   - Potential future work: price-tiered models

---

## Dependencies

No new dependencies added. Uses existing:
- `scipy.stats` (for normality test, skew, kurtosis)
- `numpy` (for bootstrap sampling)
- `sklearn` (for KFold, metrics)

---

## Breaking Changes

None. V2.5 is purely additive evaluation - no API or model changes.

---

## Upgrade Path

No upgrade needed. V2.5 adds evaluation capabilities without changing production behavior.

```powershell
# Run robust evaluation anytime
python src/robust_evaluate.py --k-folds 5 --bootstrap-samples 500
```

---

## Version Summary

| Version | MAE | R² | Key Change |
|---------|-----|----|-----------| 
| V1 | $102,045 | 0.7281 | Baseline |
| V2.1 | $89,769 | 0.7682 | +10 features |
| V2.3 | $84,494 | 0.7932 | Hyperparameter tuning |
| V2.4.1 | $67,041 | 0.8755 | XGBoost |
| **V2.5** | **$63,529** (CV) | **0.8945** (CV) | **Robust evaluation** |

**Total improvement from V1: -37.7% MAE, +22.8% R²**
