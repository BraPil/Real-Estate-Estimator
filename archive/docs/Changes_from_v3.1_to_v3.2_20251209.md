# Changes from V3.1 to V3.2 (2025-12-09)

## Overview

V3.2 integrated fresh King County Assessment data (2020-2024) and achieved 100% feature mapping from the original model schema.

| Aspect | V3.1 | V3.2 |
|--------|------|------|
| Training Data | 2014-2015 (21,613 records) | 2020-2024 (155,855 records) |
| CV MAE | $63,529 | $236,161 |
| R2 | 0.88 | 0.68 |
| MAPE | ~14% | 28.7% |
| Median Price | $450,000 | $845,000 |
| Feature Mapping | N/A (original data) | 100% from assessment data |

**Note:** Higher absolute MAE is expected due to 88% price inflation. MAPE is the relevant comparison.

---

## Files Added

### Data Pipeline

| File | Lines | Purpose |
|------|-------|---------|
| `src/data/__init__.py` | 1 | Data module initialization |
| `src/data/transform_assessment_data.py` | 342 | ETL pipeline for KC Assessment data |

### Training & Evaluation

| File | Lines | Purpose |
|------|-------|---------|
| `src/train_fresh_data.py` | 288 | Training script with MLflow for 2020+ data |
| `src/evaluate_fresh.py` | 290 | Evaluation script for fresh data |

### Scripts

| File | Lines | Purpose |
|------|-------|---------|
| `scripts/extract_parcel_centroids.py` | 180 | Extract lat/long from GIS GeoJSON |
| `scripts/evaluate_zero_shot.py` | 200 | Zero-shot evaluation on 2020+ data |

### Model Artifacts

| File | Size | Purpose |
|------|------|---------|
| `model/model.pkl` | 1.8 MB | V3.2 trained model |
| `model/model_features.json` | 2 KB | 43 feature names |
| `model/evaluation_fresh_report.json` | 2 KB | Evaluation results |

### Data Files (Generated, not in repo)

| File | Size | Purpose |
|------|------|---------|
| `data/parcel_centroids.csv` | 32 MB | 637,540 parcel lat/long coordinates |
| `data/assessment_2020_plus_v3.csv` | 24 MB | Transformed training data |

### Documentation

| File | Purpose |
|------|---------|
| `docs/V3.2_Fresh_Data_Results.md` | Comprehensive results documentation |
| `RESTART_20251209.md` | Session restart protocol |

---

## Files Modified

### Configuration

| File | Change |
|------|--------|
| `.gitignore` | Removed `model/` from ignore (model now tracked) |

### Documentation

| File | Change |
|------|--------|
| `docs/V3_Detailed_Roadmap.md` | Marked V3.2 complete, added V3.3 priorities |

---

## Key Technical Changes

### 1. Data Transformation Pipeline

Created `src/data/transform_assessment_data.py` with:
- Loads 3 assessment files (Sales, Buildings, Parcels)
- Joins on Major/Minor parcel ID
- Maps 17 original features from assessment columns
- Integrates GIS centroid data for real lat/long
- Calculates neighbor averages by zipcode

### 2. Feature Mapping (100% Coverage)

| Original Feature | Assessment Source | Mapping |
|------------------|-------------------|---------|
| price | SalePrice | Direct |
| bedrooms | Bedrooms | Direct |
| bathrooms | BathFullCount + 0.75*Bath3qtr + 0.5*BathHalf | Computed |
| sqft_living | SqFtTotLiving | Direct |
| sqft_lot | SqFtLot | Direct |
| floors | Stories | Direct |
| waterfront | WfntLocation > 0 | Binary conversion |
| view | ViewUtilization | Ordinal mapping |
| condition | Condition | Direct |
| grade | BldgGrade | Direct |
| sqft_above | SqFt1stFloor + SqFt2ndFloor + Upper | Computed |
| sqft_basement | SqFtTotBasement | Direct |
| yr_built | YrBuilt | Direct |
| yr_renovated | YrRenovated | Direct |
| zipcode | ZipCode | Direct |
| lat | GIS Parcel Centroids | Extracted from GeoJSON |
| long | GIS Parcel Centroids | Extracted from GeoJSON |
| sqft_living15 | Zipcode average | Computed |
| sqft_lot15 | Zipcode average | Computed |

### 3. GIS Integration

- Parsed 844MB GeoJSON file with 637,540 parcels
- Extracted polygon centroids for lat/long
- 100% match rate with sales data

### 4. MLflow Integration

All training runs logged to MLflow with:
- Hyperparameters
- Cross-validation metrics
- Model artifacts
- Feature information
- Evaluation reports

---

## Dependencies

No new dependencies added. Uses existing:
- pandas, numpy, scikit-learn
- xgboost
- mlflow

---

## Breaking Changes

None. V3.2 model is backward-compatible with API endpoints.

---

## Performance Progression (V3.2 Development)

| Iteration | lat/long | Neighbors | CV MAE | R2 | MAPE |
|-----------|----------|-----------|--------|-----|------|
| V3.2-v1 | Synthetic | Self | $258,958 | 0.641 | 31.8% |
| V3.2-v2 | Synthetic | Zipcode avg | $256,302 | 0.645 | 31.3% |
| **V3.2-v3** | **Real GIS** | Zipcode avg | **$236,161** | **0.684** | **28.7%** |

Real lat/long from GIS data improved MAE by $20,141 (7.8%).

---

## Known Issues

1. **Under $500K tier has 107% MAPE** - Likely distressed sales/family transfers
2. **Residual skewness of 1.78** - Right-skewed errors suggest log transform may help
3. **V2.5 hyperparameters** - May not be optimal for 2020+ data distribution

These are addressed in V3.3 priorities.

---

## Testing

All 13 tests passing after V3.2 changes:
- Model loading tests (4)
- Prediction tests (4)
- Input validation tests (2)
- Performance tests (1)
- MLflow tests (2)

