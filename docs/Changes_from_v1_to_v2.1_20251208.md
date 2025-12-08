# Changes from V1 to V2.1

**Project:** Real Estate Price Predictor  
**Date:** 2025-12-08  
**Branch:** `feature/v2-model-improvements`

---

## Overview

This document captures all changes made from V1 (baseline API) to V2.1 (expanded feature set).

**Key Achievement:** Improved model accuracy by 12% (MAE) while reducing overfitting by 20%.

---

## V1 Baseline (Reference)

**Commit:** `b4a413d` (main branch)  
**Model Metrics:**
- Test R²: 0.7281
- Test MAE: $102,045
- Test RMSE: $201,659
- Overfitting Gap: 0.1133
- Features: 33 (7 home + 26 demographic)

---

## Complete List of Changes

### 1. Feature Expansion: 7 → 17 Home Features

| Change | V1 | V2.1 |
|--------|-----|------|
| Total features | 33 | 43 |
| Home features | 7 | 17 |
| Demographic features | 26 | 26 (unchanged) |

**V1 Features (7):**
```
bedrooms, bathrooms, sqft_living, sqft_lot, floors, sqft_above, sqft_basement
```

**V2.1 Features (17) - Added 10:**
```
V1 features +
waterfront, view, condition, grade,      # Property characteristics
yr_built, yr_renovated,                   # Age features
lat, long,                                # Spatial features
sqft_living15, sqft_lot15                 # Neighborhood context
```

---

### 2. Training Script Updates (`src/train.py`)

| Aspect | V1 | V2.1 |
|--------|-----|------|
| SALES_COLUMN_SELECTION | 9 columns | 19 columns |
| Default experiment name | "real-estate-v1" | "real-estate-v2" |
| Metrics output | Basic | Enhanced with version tracking |

**Code Change:**
```python
# V1
SALES_COLUMN_SELECTION = [
    'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'sqft_above', 'sqft_basement', 'zipcode'
]

# V2.1
SALES_COLUMN_SELECTION = [
    'price',
    'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'sqft_above', 'sqft_basement',
    'waterfront', 'view', 'condition', 'grade',  # NEW
    'yr_built', 'yr_renovated',                   # NEW
    'lat', 'long',                                # NEW
    'sqft_living15', 'sqft_lot15',                # NEW
    'zipcode'
]
```

---

### 3. API Models Update (`src/models.py`)

| Method | V1 | V2.1 |
|--------|-----|------|
| `get_model_features()` | Returns 7 features | Returns 17 features |
| Module docstring | "7 home features" | "17 home features" |

**Code Change:**
```python
# V1: get_model_features() returned 7 features
# V2.1: get_model_features() returns all 17 home features
def get_model_features(self) -> dict:
    return {
        # V1 structural features (7)
        "bedrooms": self.bedrooms,
        "bathrooms": self.bathrooms,
        "sqft_living": self.sqft_living,
        "sqft_lot": self.sqft_lot,
        "floors": self.floors,
        "sqft_above": self.sqft_above,
        "sqft_basement": self.sqft_basement,
        # V2.1 NEW features (10)
        "waterfront": self.waterfront,
        "view": self.view,
        "condition": self.condition,
        "grade": self.grade,
        "yr_built": self.yr_built,
        "yr_renovated": self.yr_renovated,
        "lat": self.lat,
        "long": self.long,
        "sqft_living15": self.sqft_living15,
        "sqft_lot15": self.sqft_lot15,
    }
```

---

### 4. Feature Service Updates (`src/services/feature_service.py`)

**New Addition:** `V21_DEFAULT_FEATURES` constant for minimal endpoint

```python
V21_DEFAULT_FEATURES = {
    "waterfront": 0,          # Most homes not waterfront
    "view": 0,                # Most have no special view
    "condition": 3,           # Average (1-5 scale)
    "grade": 7,               # Average construction grade
    "yr_built": 1975,         # Median year built
    "yr_renovated": 0,        # Most never renovated
    "lat": 47.5601,           # King County center
    "long": -122.2139,        # King County center
    "sqft_living15": 1986,    # Median neighbor sqft
    "sqft_lot15": 7620,       # Median neighbor lot
}
```

**Updated Method:** `enrich_features_with_average()` now fills in V2.1 defaults for minimal endpoint.

---

### 5. Model Service Updates (`src/services/model_service.py`)

**New Feature:** Auto-detection of model version based on feature count

```python
# V1: 33 features → version "v1"
# V2.1: 43 features → version "v2.1"
if len(self.feature_names) >= 40:
    self.model_version = "v2.1"
else:
    self.model_version = "v1"
```

---

### 6. Model Artifacts Updated

| File | V1 | V2.1 |
|------|-----|------|
| `model/model.pkl` | 33-feature KNN | 43-feature KNN |
| `model/model_features.json` | 33 features | 43 features |
| `model/metrics.json` | R²=0.728, MAE=$102k | R²=0.768, MAE=$90k |

---

### 7. MLflow Registry

| Aspect | V1 | V2.1 |
|--------|-----|------|
| Model version | 1 | 2 |
| Experiment | real-estate-v1 | real-estate-v2 |

---

## Performance Comparison

| Metric | V1 | V2.1 | Change | % Improvement |
|--------|-----|------|--------|---------------|
| Test R² | 0.7281 | 0.7682 | +0.0401 | **+5.5%** |
| Test MAE | $102,045 | $89,769 | -$12,276 | **-12.0%** |
| Test RMSE | $201,659 | $186,207 | -$15,452 | **-7.7%** |
| Overfitting Gap | 0.1133 | 0.0910 | -0.0223 | **-19.7%** |
| Features | 33 | 43 | +10 | +30.3% |

---

## Files Modified Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `src/train.py` | MODIFIED | Expanded SALES_COLUMN_SELECTION, updated experiment name |
| `src/models.py` | MODIFIED | Updated get_model_features() to return 17 features |
| `src/services/feature_service.py` | MODIFIED | Added V21_DEFAULT_FEATURES |
| `src/services/model_service.py` | MODIFIED | Added version auto-detection |
| `model/model.pkl` | REGENERATED | Retrained with 43 features |
| `model/model_features.json` | REGENERATED | 43 feature names |
| `model/metrics.json` | REGENERATED | New metrics |
| `logs/v2.1_implementation_log.md` | NEW | Full implementation log |

---

## API Behavior Changes

### Full Prediction Endpoint (`/predict`)

| Aspect | V1 | V2.1 |
|--------|-----|------|
| Input | 18 columns (all accepted) | 18 columns (all accepted) |
| Features used | 7 home + 26 demo = 33 | 17 home + 26 demo = 43 |
| Unused inputs | 10 columns ignored | 0 columns ignored |

### Minimal Prediction Endpoint (`/predict-minimal`)

| Aspect | V1 | V2.1 |
|--------|-----|------|
| Input | 7 features required | 7 features required |
| Missing features | Filled with avg demographics | Filled with avg demographics + V21_DEFAULT_FEATURES |

---

**Document Version:** 1.0  
**Last Updated:** 2025-12-08
