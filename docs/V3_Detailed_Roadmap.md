# V3 Detailed Roadmap: MLOps & Data Pipeline

**Project:** Real Estate Price Predictor  
**Date:** 2025-12-09  
**Status:** V3.2 Complete, V3.3 Planned

---

## Version Overview

| Version | Focus | Status | Key Deliverable |
|---------|-------|--------|-----------------|
| **V3.1** | MLOps & CI/CD | ✅ **COMPLETE** | GitHub Actions, MLflow, automated pipelines |
| **V3.2** | Fresh Data Integration | ✅ **COMPLETE** | King County 2020+ data, 100% feature mapping |
| **V3.3** | Model Optimization | 🚀 **NEXT** | Quick wins for performance improvement |
| V3.4 | Model Registry | 📋 PLANNED | Production/Staging model promotion |
| V3.5 | Deploy Pipeline | 📋 PLANNED | Docker build + push workflow |

---

## V3.1: MLOps & CI/CD Infrastructure ✅ **COMPLETE**

**Goal:** Production-ready ML infrastructure with automated pipelines.

### Status: COMPLETE (2025-12-08)

**What Was Delivered:**

| Component | Status | Details |
|-----------|--------|---------|
| MLflow Experiment Tracking | ✅ | `mlflow_config.py`, training logs metrics/params/artifacts |
| MLflow-Integrated Training | ✅ | `src/train_with_mlflow.py` with full tracking |
| GitHub Actions CI | ✅ | `ci.yml` - lint, test, validate on every PR |
| GitHub Actions Training | ✅ | `train.yml` - manual trigger with params |
| Test Suite | ✅ | 13 tests in `tests/test_model.py` |
| Code Quality Tools | ✅ | ruff + black configured in `pyproject.toml` |
| First CI/CD Cycle | ✅ | PR → CI Passes → Merge completed |

### Files Created

```
.github/workflows/ci.yml      - CI pipeline (lint, test, validate)
.github/workflows/train.yml   - Training pipeline (manual trigger)
mlflow_config.py              - MLflow configuration
pyproject.toml                - Tool configurations
src/train_with_mlflow.py      - MLflow-integrated training
tests/test_model.py           - Test suite (13 tests)
```

### Key Learnings
- MLflow needs `Path.as_uri()` for Windows artifact paths
- Test fixtures must use exact feature names from model
- Document rationale for lint rule ignores

**See:** `docs/V3.1_Completion_Summary.md` for full details.

---

## V3.2: Fresh Data Integration ✅ **COMPLETE**

**Goal:** Update model with current King County assessment data (2020+).

### Status: COMPLETE (2025-12-09)

**What Was Delivered:**

| Component | Status | Details |
|-----------|--------|---------|
| Data Pipeline | ✅ | `src/data/transform_assessment_data.py` - Full ETL |
| GIS Integration | ✅ | Real lat/long from King County GIS parcel centroids |
| Training Script | ✅ | `src/train_fresh_data.py` with MLflow tracking |
| Evaluation Script | ✅ | `src/evaluate_fresh.py` for 2020+ data |
| 100% Feature Mapping | ✅ | All 17 original features properly mapped |
| 155,855 Records | ✅ | 2020-2024 single-family residential sales |

### Performance Results

| Metric | V2.5 (2014-15 data) | V3.2 (2020+ data) |
|--------|---------------------|-------------------|
| CV MAE | $63,529 | $236,161 |
| Test R2 | 0.88 | 0.68 |
| MAPE | ~14% | 28.7% |
| Median Price | $450,000 | $845,000 |

**Note:** Higher absolute MAE expected due to 88% price inflation. MAPE is the relevant comparison metric.

### Files Created

```
src/data/__init__.py
src/data/transform_assessment_data.py    # ETL pipeline
src/train_fresh_data.py                  # Training with MLflow
src/evaluate_fresh.py                    # Evaluation for 2020+ data
scripts/extract_parcel_centroids.py      # GIS centroid extraction
scripts/evaluate_zero_shot.py            # Zero-shot evaluation
data/parcel_centroids.csv                # 637,540 parcel lat/long
model/model_features.json                # 43 feature names
model/evaluation_fresh_report.json       # Latest eval results
```

### Key Accomplishments

1. **100% Feature Mapping** - All 17 original features mapped from assessment data
2. **Real Coordinates** - Extracted lat/long from 844MB GIS GeoJSON
3. **Zero-Shot Validation** - Confirmed retraining was necessary (60% MAPE without)
4. **MLflow Integration** - All experiments tracked and logged

**See:** `docs/V3.2_Fresh_Data_Results.md` for full details.

---

## V3.3: Model Optimization 🚀 **NEXT**

**Goal:** Quick wins to improve V3.2 model performance.

### Background

V3.2 evaluation revealed optimization opportunities:
- Under $500K tier has 107% MAPE (distressed sales?)
- Residual skewness of 1.78 (right-skewed errors)
- Hyperparameters from V2.5 may not be optimal for 2020+ data

### Implementation Plan (Priority Order)

#### Priority 1: Filter Distressed Sales (5 min)
- [ ] Filter out sales below $400K (likely distressed/family transfers)
- [ ] Expected impact: -5% overall MAPE
- [ ] Rationale: 3,740 samples (12%) dragging down performance

#### Priority 2: Log-Transform Target (10 min)
- [ ] Train on log(price), predict exp(log_price)
- [ ] Expected impact: -2-3% MAPE, better residual distribution
- [ ] Rationale: Price distributions are right-skewed

#### Priority 3: Add Temporal Features (10 min)
- [ ] Add `sale_year`, `sale_month`, `sale_quarter` features
- [ ] Expected impact: -1-2% MAPE
- [ ] Rationale: Housing prices vary by season/year

#### Priority 4: Hyperparameter Tuning (15 min)
- [ ] Run Optuna tuning on fresh data
- [ ] Expected impact: -2-3% MAPE
- [ ] Rationale: V2.5 params may not be optimal for 2020+ distribution

#### Priority 5: Cap Outliers at $3M (5 min)
- [ ] Remove extreme high-end properties from training
- [ ] Expected impact: Better predictions in $1.5M+ tier
- [ ] Rationale: Luxury properties have unique features

### Success Criteria

- [ ] MAPE reduced from 28.7% to below 25%
- [ ] Under $500K tier MAPE reduced from 107% to below 50%
- [ ] R2 improved from 0.68 to above 0.72
- [ ] All improvements logged in MLflow

### Estimated Effort

| Priority | Task | Time | Expected MAPE Reduction |
|----------|------|------|------------------------|
| 1 | Filter distressed sales | 5 min | -5% |
| 2 | Log-transform target | 10 min | -2-3% |
| 3 | Add temporal features | 10 min | -1-2% |
| 4 | Hyperparameter tuning | 15 min | -2-3% |
| 5 | Cap outliers | 5 min | Better tier distribution |
| **Total** | | **45 min** | **-10-13% MAPE** |

---

## V3.4: Model Registry (Future)

**Goal:** Implement model versioning with Production/Staging stages.

### Features

- MLflow Model Registry integration
- Model promotion workflow (Staging → Production)
- A/B testing capability
- Rollback mechanism

### Not Started - Dependencies

- V3.3 optimizations provide multiple model versions to compare

---

## V3.5: Deploy Pipeline (Future)

**Goal:** Automated deployment via GitHub Actions.

### Features
- Docker image build on model approval
- Push to container registry
- Staged deployment (staging → production)
- Health check validation
- Rollback on failure

### Not Started - Dependencies
- V3.3 model registry provides deployment trigger

---

## Architecture (Current State)

```
┌─────────────────────────────────────────────────────────────────────┐
│                         GitHub Actions                               │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐                    │
│  │   ci.yml   │  │ train.yml  │  │ deploy.yml │                    │
│  │ lint+test  │  │ train+eval │  │   (TODO)   │                    │
│  │    ✅      │  │     ✅     │  │    📋      │                    │
│  └─────┬──────┘  └─────┬──────┘  └────────────┘                    │
└────────│───────────────│────────────────────────────────────────────┘
         │               │
         ▼               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                           MLflow                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │  Experiments │  │Model Registry│  │  Artifacts   │              │
│  │     ✅       │  │    (TODO)    │  │     ✅       │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Data Pipeline (V3.2)                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ Assessment   │  │  Transform   │  │   Training   │              │
│  │    Data      │→ │   Pipeline   │→ │     Data     │              │
│  │   (TODO)     │  │    (TODO)    │  │    (TODO)    │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Decision Log

### 2025-12-09

- **V3.2 Complete**: Fresh data integration with 100% feature mapping
- **V3.3 Planned**: Quick wins for model optimization (5 priorities identified)
- **Key Insight**: Under $500K tier has 107% MAPE - needs filtering/investigation

### 2025-12-08

- **V3.1 Complete**: MLOps infrastructure delivered with CI/CD, MLflow, tests
- **V3.2 Planned**: User has King County assessment data ready for integration
- **Focus**: Data pipeline over model registry (practical value first)

---

## Commands Reference

### MLflow
```powershell
# Train with tracking
python src/train_with_mlflow.py

# View UI
mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db
```

### CI/CD
```powershell
# Run tests locally
pytest tests/ -v

# Run linting locally
python -m ruff check .
python -m black . --check

# Trigger training (GitHub UI)
# Actions → Train Model → Run workflow
```

### Git Workflow
```powershell
# Create feature branch
git checkout -b feature/v3.2-fresh-data

# After work, push and create PR
git push -u origin feature/v3.2-fresh-data
# Create PR on GitHub → CI runs automatically
```

---

**Document Version:** 1.0  
**Last Updated:** 2025-12-08
