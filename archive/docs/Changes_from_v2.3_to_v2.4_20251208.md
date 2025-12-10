# Changes from V2.3 to V2.4

**Date:** 2025-12-08  
**Branch:** `feature/v2.4-model-alternatives`  
**Focus:** Model Alternatives Comparison

---

## Summary

V2.4 replaced the KNN model with XGBoost, achieving a **20.7% MAE improvement**.

| Metric | V2.3 (KNN) | V2.4.1 (XGBoost) | Change |
|--------|------------|------------------|--------|
| MAE | $84,494 | $67,041 | **-20.7%** |
| R² | 0.7932 | 0.8755 | **+10.4%** |
| RMSE | $176,796 | $137,174 | **-22.4%** |
| Model Size | 6.0 MB | 0.96 MB | **-84%** |

---

## Files Added

### Source Code
| File | Purpose |
|------|---------|
| `src/compare_models.py` | Compare 5 models (KNN, RF, XGBoost, LightGBM, Ridge) |
| `src/tune_xgboost.py` | Focused XGBoost hyperparameter tuning |
| `src/tune_top_models.py` | Multi-model tuning (slower alternative) |

### Model Artifacts
| File | Purpose |
|------|---------|
| `model/model_v2.4.1_xgboost_tuned.pkl` | Tuned XGBoost model backup |
| `model/metrics_v2.4.1.json` | V2.4.1 specific metrics |
| `model/comparison_results.json` | Model comparison summary |

### Documentation
| File | Purpose |
|------|---------|
| `docs/V2.4_Completion_Summary.md` | Version completion summary |
| `docs/V2.4_Model_Comparison_Guide.md` | How to use compare_models.py |
| `docs/V2.4_Deep_Dive_Educational.md` | Educational deep dive |
| `docs/feature_wrap_up_protocol.md` | Reusable wrap-up checklist |
| `docs/Changes_from_v2.3_to_v2.4_20251208.md` | This file |

### Logs
| File | Purpose |
|------|---------|
| `logs/v2.4_model_comparison_*.csv` | Full comparison results |
| `logs/v2.4_feature_importance_*.csv` | Feature importance rankings |
| `logs/v2.4.1_xgboost_feature_importance.csv` | Final XGBoost feature importance |

---

## Files Modified

### Production Model
| File | Change |
|------|--------|
| `model/model.pkl` | **Replaced KNN with XGBoost** |
| `model/metrics.json` | Updated to V2.4.1 with XGBoost metrics |
| `model/evaluation_report.json` | Updated with XGBoost evaluation |

### Source Code
| File | Change |
|------|--------|
| `src/services/model_service.py` | Added `model_type` attribute; reads version from metrics.json |

### Dependencies
| File | Change |
|------|--------|
| `requirements.txt` | Added `xgboost>=2.0.0` and `lightgbm>=4.0.0` |

### Documentation
| File | Change |
|------|--------|
| `docs/V2_Detailed_Roadmap.md` | V2.4 status → COMPLETE |
| `RESTART_20251208.md` | Updated model config to XGBoost |
| `logs/human_in_the_loop_corrections.md` | Added V2.4 corrections |

---

## Configuration Changes

### Model Configuration

**Before (V2.3 KNN):**
```python
KNeighborsRegressor(
    n_neighbors=7,
    weights='distance',
    metric='manhattan'
)
```

**After (V2.4.1 XGBoost):**
```python
XGBRegressor(
    n_estimators=239,
    max_depth=7,
    learning_rate=0.0863,
    subsample=0.7472,
    colsample_bytree=0.8388,
    min_child_weight=6,
    gamma=0.1589,
    reg_alpha=0.2791,
    reg_lambda=1.3826
)
```

### Pipeline Structure

**Before:**
```python
Pipeline([
    ('scaler', RobustScaler()),
    ('knn', KNeighborsRegressor(...))
])
```

**After:**
```python
Pipeline([
    ('scaler', RobustScaler()),
    ('model', XGBRegressor(...))
])
```

Note: Pipeline step name changed from `'knn'` to `'model'`.

---

## Dependencies Added

```txt
# requirements.txt additions
xgboost>=2.0.0
lightgbm>=4.0.0
```

Install with:
```powershell
pip install xgboost lightgbm
```

---

## API Changes

### No Breaking Changes
The API endpoints remain unchanged:
- `/api/v1/health`
- `/api/v1/predict`
- `/api/v1/predict-minimal`
- `/api/v1/predict-full`
- `/api/v1/predict-adaptive`

### Version Reporting Enhancement
The health endpoint and predictions now report the correct version from `metrics.json`:
```json
{
  "model_version": "v2.4.1"
}
```

Previously, version was guessed from feature count (always showed "v2.1" for 43 features).

---

## Model Service Changes

### `src/services/model_service.py`

**Added:**
- `model_type` attribute to track model class
- Reading version info from `metrics.json` instead of guessing

**Changed:**
```python
# Before: Version detection by feature count
if len(self.feature_names) >= 40:
    self.model_version = "v2.1"

# After: Read from metrics.json
metrics_path = model_path.parent / "metrics.json"
if metrics_path.exists():
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    self.model_version = metrics.get("version", "unknown")
    self.model_type = metrics.get("model_type", "unknown")
```

---

## Performance Comparison

### Error by Price Range

| Price Range | V2.3 MAE | V2.4.1 MAE | Improvement |
|-------------|----------|------------|-------------|
| Under $300k | $43,538 | $36,848 | -15.4% |
| $300k-$500k | $46,500 | $40,987 | -11.9% |
| $500k-$750k | $70,291 | $59,890 | -14.8% |
| $750k-$1M | $115,372 | $109,589 | -5.0% |
| Over $1M | $386,349 | $243,015 | **-37.1%** |

### Bias Reduction
| Metric | V2.3 | V2.4.1 | Change |
|--------|------|--------|--------|
| Mean Residual | $27,281 | $815 | **-97%** |

---

## Breaking Changes

**None.** The API interface is unchanged. Existing clients will work without modification.

---

## Migration Guide

### For Developers
1. Install new dependencies:
   ```powershell
   pip install xgboost lightgbm
   ```

2. The model file changed format (smaller, different structure), but the Pipeline interface is identical.

### For API Users
No changes required. The API behaves the same way.

---

## Rollback Instructions

If you need to revert to V2.3 KNN:
```powershell
# Restore KNN model
Copy-Item model\model_v2.3_tuned.pkl model\model.pkl -Force

# Restart API server
```

---

**Document Version:** 1.0  
**Last Updated:** 2025-12-08
