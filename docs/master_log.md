# Master Log: Real-Estate-Estimator Project

**Project:** phData Machine Learning Engineer Coding Test
**Start Date:** December 6, 2025
**Current Phase:** V2.4 Model Alternatives (STARTING)
**Last Updated:** 2025-12-08

---

## Log Index

| Log File | Purpose | Status | Last Updated |
|----------|---------|--------|--------------|
| master_log.md | Central index | Active | 2025-12-08 |
| 2025-12-06_analysis_log.md | Initial analysis | Complete | 2025-12-06 |
| 2025-12-06_data_provenance_log.md | Data documentation | Complete | 2025-12-06 |
| 2025-12-07_implementation_log.md | V1 implementation | Complete | 2025-12-07 |
| v2.1_implementation_log.md | V2.1 feature expansion | Complete | 2025-12-08 |
| v2.3_implementation_log.md | V2.3 hyperparameter tuning | Complete | 2025-12-08 |
| v2.3_grid_search_results.csv | GridSearchCV results | Complete | 2025-12-08 |
| human_in_the_loop_corrections.md | Corrections | Active | 2025-12-08 |
| v2_roadmap.md | Version planning | Superseded | 2025-12-08 |

---

## Version History

### V1: Initial Implementation
**Status:** COMPLETE (2025-12-07)
**Tag:** v1.0.0

| Metric | Value |
|--------|-------|
| Test R^2 | 0.7281 |
| Test MAE | $102,045 |
| Features | 33 (7 home + 26 demographic) |

**Deliverables:**
- FastAPI application with /health, /predict, /predict-minimal
- Training and evaluation scripts
- Docker containerization
- Comprehensive documentation

---

### V2.1: Feature Expansion
**Status:** COMPLETE (2025-12-08)
**Branch:** feature/v2-model-improvements (merged to develop)

| Metric | V1 | V2.1 | Change |
|--------|-----|------|--------|
| Test R^2 | 0.7281 | 0.7682 | +5.5% |
| Test MAE | $102,045 | $89,769 | -12.0% |
| Features | 33 | 43 | +30% |

**Key Changes:**
- Added 10 home features: waterfront, view, condition, grade, yr_built, yr_renovated, lat, long, sqft_living15, sqft_lot15
- Updated train.py, models.py, feature_service.py
- Added V21_DEFAULT_FEATURES for minimal endpoint

---

### V2.1.1: Full Features Endpoint
**Status:** COMPLETE (2025-12-08)

**Deliverable:** `/predict-full` endpoint
- Accepts all 17 home features
- No zipcode required
- Uses average demographics
- Best single strategy for no-zipcode predictions

---

### V2.1.2: Adaptive Routing
**Status:** LOW PRIORITY (deferred)

**Finding:** Price-tier pattern confirmed (62%/88% win rates by tier)
**Outcome:** Routing accuracy (52%) too low to beat always-use-`/predict-full`
**Decision:** Document for future exploration, proceed with other improvements

---

### V2.3: Hyperparameter Tuning
**Status:** COMPLETE (2025-12-08)
**Branch:** feature/v2.3-hyperparameter-tuning (merged to develop)

| Metric | V2.1 | V2.3 | Change |
|--------|------|------|--------|
| Test R^2 | 0.7682 | 0.7932 | +3.3% |
| Test MAE | $89,769 | $84,494 | -5.9% |
| CV MAE | N/A | $79,976 | - |

**Best Parameters Found:**
- n_neighbors: 7 (was 5)
- weights: distance (was uniform)
- metric: manhattan (was minkowski/euclidean)
- p: 1

**Key Learnings:**
- Distance weighting improves accuracy
- Manhattan distance better for high-dimensional data
- GridSearchCV with 5-fold CV prevents overfitting

---

### V2.4: Model Alternatives
**Status:** NEXT (starting)
**Branch:** feature/v2.4-model-alternatives

**Objective:** Compare KNN vs tree-based models
**Models:** Random Forest, XGBoost, LightGBM, Ridge

---

## Cumulative Improvement

| Metric | V1 Baseline | V2.3 Current | Total Change |
|--------|-------------|--------------|--------------|
| Test MAE | $102,045 | $84,494 | **-17.2%** |
| Test R^2 | 0.7281 | 0.7932 | **+8.9%** |

---

## Documentation Index

| Document | Purpose |
|----------|---------|
| docs/V2_Detailed_Roadmap.md | Version planning |
| docs/Changes_from_original_to_v1_20251207.md | Original -> V1 changes |
| docs/Changes_from_v1_to_v2.1_20251208.md | V1 -> V2.1 changes |
| docs/Changes_from_v2.1_to_v2.3_20251208.md | V2.1 -> V2.3 changes |
| docs/V2.1_Lessons_Learned.md | V2.1 lessons |
| docs/V2.3_Lessons_Learned.md | V2.3 lessons |
| docs/V2.1.x_Completion_Summary.md | V2.1.x summary |
| docs/V2.3_Completion_Summary.md | V2.3 summary |
| RESTART_20251208.md | Session restart guide |

---

## Git Branch Status

| Branch | Status | Contains |
|--------|--------|----------|
| main | Needs update | V1 only |
| develop | Current | V1 + V2.1 + V2.3 |
| feature/v2.4-model-alternatives | Active | Starting V2.4 |

**Action Required:** PR develop -> main to update main branch

---

## Key Files

```
Production Model: model/model.pkl (V2.3 tuned)
Model Config: model/metrics.json
Feature List: model/model_features.json (43 features)
API Entry: src/main.py
Training: src/train.py
Tuning: src/tune.py
Evaluation: src/evaluate.py
```

---

**Last Updated:** 2025-12-08
**Maintained By:** AI Assistant following established protocols
