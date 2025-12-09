# V3 Detailed Roadmap: MLOps & Data Pipeline

**Project:** Real Estate Price Predictor  
**Date:** 2025-12-08  
**Status:** V3.1 Complete, V3.2 Planned

---

## Version Overview

| Version | Focus | Status | Key Deliverable |
|---------|-------|--------|-----------------|
| **V3.1** | MLOps & CI/CD | ✅ **COMPLETE** | GitHub Actions, MLflow, automated pipelines |
| **V3.2** | Fresh Data Integration | 🚀 **NEXT** | King County 2024 assessment data |
| V3.3 | Model Registry | 📋 PLANNED | Production/Staging model promotion |
| V3.4 | Deploy Pipeline | 📋 PLANNED | Docker build + push workflow |

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

## V3.2: Fresh Data Integration 🚀 **NEXT**

**Goal:** Update model with current King County assessment data (2024).

### Background

The current model is trained on 2014-2015 data:
- Seattle market has changed dramatically (~80-100% price increase)
- Model predictions are systematically low for current market
- Demographics may have shifted

### Data Source

**King County Assessor's Office** - You have downloaded:
```
Reference_Docs/King_County_Assessment_data_ALL/
├── EXTR_RPSale.csv           # Real property sales (606 MB)
├── EXTR_ResBldg.csv          # Residential buildings (146 MB)
├── EXTR_Parcel.csv           # Parcel information (234 MB)
├── EXTR_LookUp.csv           # Lookup codes
└── ... (30+ files)
```

### Implementation Plan

#### Phase 1: Data Exploration
- [ ] Analyze `EXTR_RPSale.csv` schema and compare to original `kc_house_data.csv`
- [ ] Identify which columns map to existing features
- [ ] Determine date range of available sales data
- [ ] Assess data quality (missing values, outliers)

#### Phase 2: Data Pipeline
- [ ] Create `src/data/load_assessment_data.py` - Load and parse assessment files
- [ ] Create `src/data/transform_assessment_data.py` - Transform to model schema
- [ ] Map assessment columns to existing 17 home features:
  | Original Feature | Assessment Source | Notes |
  |-----------------|-------------------|-------|
  | `bedrooms` | EXTR_ResBldg | BldgNbr? |
  | `bathrooms` | EXTR_ResBldg | |
  | `sqft_living` | EXTR_ResBldg | SqFtTotLiving? |
  | `sqft_lot` | EXTR_Parcel | SqFtLot? |
  | `price` | EXTR_RPSale | SalePrice |
  | ... | ... | |

#### Phase 3: Demographics Update
- [ ] Check if demographics have changed significantly since 2014
- [ ] If needed, find updated census/ACS data for King County zipcodes
- [ ] Create demographics update pipeline

#### Phase 4: Model Retraining
- [ ] Train model on new data using MLflow tracking
- [ ] Compare performance to V2.5 baseline
- [ ] Evaluate if schema changes affect prediction quality

#### Phase 5: Validation
- [ ] Test API with new model
- [ ] Validate predictions against recent sales
- [ ] Document any breaking changes

### Expected Challenges

1. **Schema Mapping** - Assessment data likely uses different column names
2. **Data Quality** - May have more missing values or outliers
3. **Price Range** - 2024 prices much higher than 2015 ($500k → $900k median)
4. **Demographics** - May need to source updated demographic data

### Success Criteria

- [ ] Successfully load and transform assessment data
- [ ] Train model on 2020+ sales data
- [ ] Model performs comparably to V2.5 on new data
- [ ] API serves predictions based on current market values
- [ ] Document schema mapping and transformation logic

### Estimated Effort

| Task | Hours |
|------|-------|
| Data exploration | 2-3 |
| Schema mapping | 3-4 |
| Pipeline development | 4-6 |
| Demographics update | 2-4 |
| Training & validation | 2-3 |
| Documentation | 1-2 |
| **Total** | **14-22** |

---

## V3.3: Model Registry (Future)

**Goal:** Implement model versioning with Production/Staging stages.

### Features
- MLflow Model Registry integration
- Model promotion workflow (Staging → Production)
- A/B testing capability
- Rollback mechanism

### Not Started - Dependencies
- V3.2 provides reason to have multiple model versions

---

## V3.4: Deploy Pipeline (Future)

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
