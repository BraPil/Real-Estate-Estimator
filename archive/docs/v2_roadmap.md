# V2 Roadmap: Future Enhancements

**Project:** Real-Estate-Estimator
**Purpose:** Track ideas and requirements for future versions beyond v1 MVP
**Status:** Planning (to be implemented after v1 complete)

---

## V2 Philosophy: Platform-Agnostic Design

**Date Added:** 2025-12-07
**Source:** Architecture review
**Priority:** GUIDING PRINCIPLE

**Decision:**
Keep V2 platform-agnostic. Do NOT add Databricks, Kubernetes, or other enterprise platforms unless there's a clear requirement.

**Rationale:**
- Dataset is small (~21k rows, ~5 MB) - fits in memory trivially
- scikit-learn trains in seconds locally - no distributed computing needed
- MLflow already integrated - provides experiment tracking without external dependencies
- Docker + any cloud VM = production-ready deployment
- Simplicity > Complexity when requirements are met

**What We Have (and it's enough):**
- Local training with `src/train.py`
- MLflow for experiment tracking (self-hosted or cloud)
- FastAPI + Docker for serving
- GitHub for version control
- Works on laptop, cloud VM, or any container platform

**When to Reconsider:**
- Dataset grows to millions of rows
- Real-time data pipelines required
- Multi-team collaboration needs
- Enterprise compliance mandates specific platforms

**Bottom Line:** The current stack is production-capable for this scale. Don't add complexity without clear benefit.

---

## V2 Enhancement: Spatial Feature Engineering

**Date Added:** 2025-12-07
**Source:** Feature analysis discussion
**Priority:** HIGH

### Quick Wins: Add Raw Lat/Long

**Current State:** `lat` and `long` are accepted by API but NOT used by model.

**Recommendation:** Add them to the model's feature set as a first step.

**Rationale:**
- For KNN, having both as separate features works because distance calculation naturally combines them
- Immediate accuracy improvement with minimal effort
- No new data required

### Engineered Location Features

| Feature | Formula/Approach | Rationale |
|---------|------------------|-----------|
| `distance_to_downtown` | `haversine(lat, long, 47.6062, -122.3321)` | Captures "urban premium" directly |
| `distance_to_water` | Distance to nearest waterfront point | Captures view/location value |
| `distance_to_employers` | Distance to major employers (Amazon, Microsoft) | Commute convenience premium |

**Implementation Notes:**
- Use Haversine formula for accurate geographic distance
- Downtown Seattle coordinates: (47.6062, -122.3321)
- Puget Sound/Lake Washington coordinates needed for water distance
- Consider creating a `src/features/geo_features.py` module

---

## V2 Enhancement: Neighborhood Context Features

**Date Added:** 2025-12-07
**Source:** Feature analysis discussion
**Priority:** MEDIUM-HIGH

### Keep Existing Neighborhood Features

**Current State:** `sqft_living15` and `sqft_lot15` are in the data but NOT used by model.

**Recommendation:** Add them to the model - they are NOT duplicative of `sqft_living`/`sqft_lot`.

**Why They're Different:**
- `sqft_living` = size of THIS house (property characteristic)
- `sqft_living15` = avg size of 15 nearest houses (neighborhood characteristic)

**Example:** A 2,000 sqft house is valued differently:
- In a 1,200 sqft avg neighborhood → "big for the area" → premium
- In a 3,500 sqft avg neighborhood → "small for the area" → discount

### Engineered Relative Features

| Feature | Formula | Interpretation |
|---------|---------|----------------|
| `relative_living_size` | `sqft_living / sqft_living15` | >1 = bigger than neighbors |
| `relative_lot_size` | `sqft_lot / sqft_lot15` | >1 = larger lot than neighbors |

**Rationale:**
- Captures whether a house is "typical" or "unusual" for its neighborhood
- Tree-based models can learn this implicitly, but explicit features help all model types

---

## V2 Enhancement: Unused Data Features

**Date Added:** 2025-12-07
**Source:** PROJECT_ANALYSIS.md
**Priority:** MEDIUM

Features currently in data but NOT used by model:

| Feature | Type | Why Valuable |
|---------|------|--------------|
| `waterfront` | Binary (0/1) | Major price driver |
| `view` | Ordinal (0-4) | Quality of view affects price |
| `condition` | Ordinal (1-5) | House maintenance state |
| `grade` | Ordinal (1-13) | Construction quality |
| `yr_built` | Numeric | House age matters |
| `yr_renovated` | Numeric | Recent renovations add value |

**Engineered from existing:**
- `house_age = 2025 - yr_built`
- `years_since_renovation = 2025 - yr_renovated` (if renovated)
- `was_renovated = yr_renovated > 0` (binary)

---

## Other V2 Candidates (Prioritized)

| Priority | Enhancement | Description | Source |
|----------|-------------|-------------|--------|
| HIGH | Fresh Data | Update from 2014-2015 to current housing data | Data provenance analysis |
| HIGH | Spatial Features | Add lat/long, distance_to_downtown, distance_to_water | Feature discussion |
| HIGH | Neighborhood Features | Add sqft_living15, sqft_lot15, relative_size ratios | Feature discussion |
| MEDIUM | Unused Features | Add waterfront, view, condition, grade | PROJECT_ANALYSIS.md |
| MEDIUM | KNN Hyperparameter Tuning | Tune k, weights, distance metric | Evaluation results |
| MEDIUM | Tree-Based Models | Try Random Forest, XGBoost, LightGBM | Evaluation results |
| MEDIUM | Target Transformation | Model log(price) instead of price | Evaluation results |
| MEDIUM | Cross-Validation | Use k-fold CV, report mean +/- std | Evaluation results |
| LOW | CI/CD Pipeline | GitHub Actions for automated training | MLFLOW_INTEGRATION_ARCHITECTURE.md |
| LOW | Monitoring | Track prediction drift over time | MLOps best practices |

---

## V2 Implementation Order (Recommended)

1. **Phase V2.1 - Feature Expansion (Quick Wins)**
   - Add lat, long to model features
   - Add sqft_living15, sqft_lot15
   - Add waterfront, view, condition, grade
   - Retrain and compare metrics

2. **Phase V2.2 - Feature Engineering**
   - Add distance_to_downtown
   - Add relative_living_size, relative_lot_size
   - Add house_age, was_renovated
   - Retrain and compare metrics

3. **Phase V2.3 - Model Improvements**
   - Hyperparameter tuning (k, weights, metric)
   - Try tree-based models
   - Implement cross-validation
   - Try log(price) transformation

4. **Phase V2.4 - Data Refresh**
   - Acquire current housing data
   - Validate schema compatibility
   - Retrain on fresh data

5. **Phase V2.5 - Production Hardening (Optional)**
   - CI/CD pipelines (GitHub Actions)
   - Basic monitoring (logs, health checks)
   - Cloud deployment if needed (any VM + Docker)
   
   Note: Databricks/Kubernetes NOT required unless scale demands it

---

**Last Updated:** 2025-12-07
**Next Review:** After v1 deployment to main branch
