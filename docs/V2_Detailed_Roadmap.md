# V2 Detailed Roadmap

**Project:** Real Estate Price Predictor  
**Date:** 2025-12-08  
**Status:** V2.5 Complete, V2.7 Explored, V3.1 Next

---

## Version Overview

| Version | Focus | Status | Key Deliverable |
|---------|-------|--------|-----------------|
| V2.1 | Feature Expansion | ✅ **COMPLETE** | +10 features, MAE -12% |
| V2.1.1 | Full Features Endpoint | ✅ **COMPLETE** | `/predict-full` - all 17 features, no zipcode |
| V2.1.2 | Adaptive Routing | ⏸️ **LOW PRIORITY** | Explored but deferred - `/predict-full` is sufficient |
| V2.2 | Feature Engineering | ⏸️ DEFERRED | Skipping for now - can revisit later |
| V2.3 | Hyperparameter Tuning | ✅ **COMPLETE** | **MAE -5.9%**, manhattan + distance-weighted |
| V2.4 | Model Alternatives | ✅ **COMPLETE** | XGBoost wins: MAE $67,041 (-20.7%) |
| V2.5 | Robust Evaluation | ✅ **COMPLETE** | K-fold CV MAE $63,529, 95% CI |
| V2.6 | Fresh Data (Future) | 📋 PLANNED | Updated housing data (if available) |
| V2.7 | Price-Tiered Models | ⏸️ **EXPLORED - NO ROI** | Best: +0.17% improvement, not worth complexity |
| **V3.1** | **MLOps & CI/CD** | 📋 **NEXT** | GitHub Actions, MLflow, automated retraining |

### Decision Log (2025-12-08)
- **V2.1.2 Adaptive Routing:** Discovered price-tier pattern (confirmed statistically) but routing accuracy too low (52%) to beat always-use-`/predict-full`. Documented as interesting finding for future exploration.
- **V2.2 Feature Engineering:** Deferred in favor of V2.3. Can revisit after V2.4 if needed.
- **V2.3 Hyperparameter Tuning:** ✅ COMPLETE - GridSearchCV found optimal params: `n_neighbors=7, weights=distance, metric=manhattan`. MAE improved from $89,769 to $84,494 (-5.9%).
- **V2.4 Model Alternatives:** Starting next - compare KNN to tree-based models.

---

## V2.2: Feature Engineering

**Goal:** Create calculated features that capture domain knowledge about home valuation.

### New Engineered Features

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `distance_to_downtown` | `haversine(lat, long, 47.6062, -122.3321)` | Urban premium - Seattle downtown is (47.6062, -122.3321) |
| `relative_living_size` | `sqft_living / sqft_living15` | Is this home big/small for the neighborhood? >1 = bigger than neighbors |
| `relative_lot_size` | `sqft_lot / sqft_lot15` | Is this lot big/small for the neighborhood? |
| `house_age` | `2015 - yr_built` | Using 2015 as reference (data vintage) |
| `years_since_renovation` | `2015 - yr_renovated` if renovated, else `house_age` | How long since last update? |
| `was_renovated` | `1 if yr_renovated > 0 else 0` | Binary flag for renovation status |
| `total_rooms` | `bedrooms + bathrooms` | Simple total room count |
| `bath_per_bed` | `bathrooms / bedrooms` | Bathroom ratio (luxury indicator) |

### Implementation Plan

1. Create `src/features/engineering.py` module
2. Add feature engineering step to `train.py`
3. Update `feature_service.py` to compute features at prediction time
4. Compare metrics with V2.1 baseline

### Expected Impact
- Moderate improvement expected from `distance_to_downtown`
- `relative_living_size` captures "big for the area" premium
- May add 1-3% to R² based on similar projects

---

## V2.3: Hyperparameter Tuning

**Goal:** Optimize KNN hyperparameters for best performance.

### Parameters to Tune

| Parameter | Current | Search Range | Description |
|-----------|---------|--------------|-------------|
| `n_neighbors` (k) | 5 | 3, 5, 7, 10, 15, 20 | Number of neighbors |
| `weights` | uniform | uniform, distance | Equal vs distance-weighted |
| `metric` | minkowski | euclidean, manhattan, minkowski | Distance calculation |
| `p` | 2 | 1, 2 | Power for minkowski (1=manhattan, 2=euclidean) |

### Tuning Approach

1. **Grid Search with Cross-Validation**
   - 5-fold CV to avoid overfitting to test set
   - Score metric: neg_mean_absolute_error (MAE)

2. **Implementation**
   ```python
   from sklearn.model_selection import GridSearchCV
   
   param_grid = {
       'kneighborsregressor__n_neighbors': [3, 5, 7, 10, 15, 20],
       'kneighborsregressor__weights': ['uniform', 'distance'],
       'kneighborsregressor__metric': ['euclidean', 'manhattan'],
   }
   
   grid_search = GridSearchCV(
       pipeline, param_grid, cv=5, 
       scoring='neg_mean_absolute_error',
       n_jobs=-1, verbose=2
   )
   ```

3. **Deliverables**
   - Best hyperparameters
   - CV score vs test score comparison
   - Tuning results logged to MLflow

### Expected Impact
- KNN performance is sensitive to k and weights
- `weights='distance'` often helps (closer neighbors matter more)
- May improve MAE by 2-5%

---

## V2.4: Model Alternatives

**Goal:** Compare KNN against other model types.

### Models to Evaluate

| Model | Pros | Cons |
|-------|------|------|
| **KNN** (current) | Simple, interpretable neighbors | Slow at prediction time for large data |
| **Random Forest** | Handles non-linearity, feature importance | Less interpretable |
| **XGBoost** | Often best accuracy, handles missing data | Complex tuning, less interpretable |
| **LightGBM** | Fast training, handles large data | Similar to XGBoost |
| **Ridge Regression** | Simple baseline, fast | Assumes linearity |

### Evaluation Framework

```python
models = {
    'KNN': KNeighborsRegressor(n_neighbors=5),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, random_state=42),
    'Ridge': Ridge(alpha=1.0),
}

for name, model in models.items():
    pipeline = make_pipeline(RobustScaler(), model)
    # Cross-validation
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_absolute_error')
    # Fit and evaluate
    pipeline.fit(X_train, y_train)
    test_score = mean_absolute_error(y_test, pipeline.predict(X_test))
```

### Deliverables
- Comparison table (R², MAE, RMSE for each model)
- Feature importance from tree-based models
- Recommendation for production model

### Expected Impact
- Tree-based models often beat KNN by 5-15% on tabular data
- XGBoost/LightGBM likely to perform best

---

## V2.5: Robust Evaluation ✅ **COMPLETE**

**Goal:** Implement statistically rigorous model evaluation.

### Status: COMPLETE (2025-12-08)

**Key Results:**
| Metric | Value | Notes |
|--------|-------|-------|
| **5-Fold CV MAE** | $63,529 ± $2,150 | More reliable than single split |
| **5-Fold CV R²** | 0.8945 ± 0.0168 | Strong, consistent performance |
| **95% CI (MAE)** | [$63,590, $70,971] | Bootstrap confidence interval |
| **95% CI (R²)** | [0.8421, 0.9089] | Bootstrap confidence interval |

**Log Transform Experiment:** ❌ NOT RECOMMENDED
- Normal MAE: $63,529
- Log Transform MAE: $64,135 (1% worse)
- High-value homes: 4.3% worse with log transform

**Residual Analysis:**
- Mean bias: Only $735 (essentially unbiased ✅)
- Heteroscedastic: Yes (variance ratio 28.20)
- 82.6% of predictions within $100k

**Files Created:**
- `src/robust_evaluate.py` - Comprehensive evaluation script
- `logs/v2.5_robust_evaluation_*.json` - Detailed results

### Components (All Implemented)

1. **K-Fold Cross-Validation** ✅
   - 5-fold CV with mean ± standard deviation
   - More reliable estimate than single train/test split

2. **Confidence Intervals** ✅
   - 500-sample bootstrap for 95% CI
   - Shows range of expected performance

3. **Target Transformation** ✅
   - Tested log(price) vs price
   - Result: Keep normal target (log is worse)

4. **Residual Analysis** ✅
   - Error distribution by price range
   - Heteroscedasticity detection
   - Systematic bias analysis

### Key Insights
1. **Log transform hurts, not helps** - Counter to initial hypothesis
2. **Model is essentially unbiased** - Mean residual only $735
3. **High-value homes are hardest** - MAE $241k for >$1M homes
4. **Performance is stable** - CV std dev only $2,150

---

## V2.6: Fresh Data (Future)

**Goal:** Update training data from 2014-2015 to current.

### Current Limitation
- Model trained on 2014-2015 King County data
- Seattle market has changed significantly (prices up ~80%)
- Model predictions are systematically biased low

### Data Sources to Investigate
- Zillow Research Data
- Redfin Data Center
- King County Assessor's Office
- Realtor.com Research

### Challenges
- Schema may differ from original
- Demographics data would need updating
- May require significant preprocessing

### When to Prioritize
- After V2.2-V2.5 optimizations
- If model is deployed for real use
- Currently lower priority (research/learning project)

---

## Implementation Schedule

| Version | Estimated Effort | Dependencies |
|---------|------------------|--------------|
| V2.2 | 2-3 hours | None (start anytime) |
| V2.3 | 2-3 hours | V2.2 recommended first |
| V2.4 | 3-4 hours | V2.2, V2.3 complete |
| V2.5 | 2-3 hours | Can be done in parallel |
| V2.6 | 8+ hours | After V2.2-V2.5 |

---

## Success Criteria

| Version | Target |
|---------|--------|
| V2.2 | R² > 0.78, MAE < $88k |
| V2.3 | MAE improvement from tuning |
| V2.4 | Best model identified, documented comparison |
| V2.5 | 95% CI for all metrics, CV scores reported |

---

## V2.7: Price-Tiered Models ⏸️ **EXPLORED - INSUFFICIENT ROI**

**Goal:** Improve accuracy by routing predictions to specialized models based on price tier.

### Status: EXPLORED - NOT ADOPTED (2025-12-08)

**Experiments Conducted:**

| Approach | MAE | vs Baseline | Decision |
|----------|-----|-------------|----------|
| V2.7a: 2-Tier Split (50th percentile) | $69,006 | -2.9% worse | ❌ Rejected |
| V2.7b: 3-Zone Hybrid (25/50/25) | $69,050 | -4.9% worse | ❌ Rejected |
| V2.7c: Confidence Routing (both tiers) | $66,352 | -0.8% worse | ❌ Rejected |
| V2.7d: Low-Only Specialist (40th @ 99%) | $65,687 | **+0.17% better** | ⚠️ Marginal |

**Key Learnings:**

1. **Misrouting is catastrophic** - When classifier is wrong, MAE explodes ($119k+ for misrouted)
2. **High-price specialist hurts** - Huge variance in luxury segment, specialist loses context
3. **Low-price specialist helps** - $26k MAE for routed samples (21% better than baseline)
4. **Best config barely wins** - 0.17% improvement doesn't justify complexity

**Decision: NOT ADOPTED FOR PRODUCTION**
- Added complexity (multiple models, routing logic) not worth 0.17% gain
- Single XGBoost model already handles price variation internally
- Focus engineering effort on MLOps infrastructure instead

**Files Created (for reference):**
- `src/tiered_model_system.py` - 2-tier system
- `src/hybrid_tiered_system.py` - 3-zone hybrid
- `src/confidence_routing_system.py` - Confidence-based routing
- `src/low_specialist_optimization.py` - Grid search optimization
- `src/optimize_tier_split.py` - Percentile optimization
- `src/explore_price_tier_predictors.py` - Feature analysis

---

## V3.1: MLOps & CI/CD Infrastructure 📋 **NEXT**

**Goal:** Production-ready ML infrastructure with automated pipelines.

### Planned Components

| Component | Tool | Purpose |
|-----------|------|---------|
| Experiment Tracking | MLflow | Track experiments, metrics, artifacts |
| CI/CD Pipeline | GitHub Actions | Automated testing, deployment |
| Model Registry | MLflow | Version control for models |
| Data Validation | Great Expectations / custom | Ensure data quality |
| Monitoring | Custom + logs | Track prediction quality over time |
| Retraining | Scheduled/triggered | Keep model current |

### GitHub Actions Workflows

1. **`ci.yml`** - On every PR:
   - Lint (ruff/black)
   - Unit tests
   - Integration tests
   - Model validation

2. **`train.yml`** - On data update or manual trigger:
   - Data validation
   - Model training
   - Evaluation against baseline
   - MLflow logging
   - Model registration (if improved)

3. **`deploy.yml`** - On model approval:
   - Build Docker image
   - Push to registry
   - Deploy to staging
   - Smoke tests
   - Deploy to production (manual gate)

### MLflow Integration

```
mlflow/
├── mlruns/           # Experiment tracking
├── models/           # Model registry
└── artifacts/        # Saved artifacts
```

### Data Pipeline

```
New Data → Validation → Preprocessing → Training → Evaluation → Registry
                ↓
          Great Expectations
          (schema, ranges, distributions)
```

### Monitoring & Alerting

- **Prediction drift**: Compare recent predictions to training distribution
- **Data drift**: Monitor input feature distributions
- **Performance decay**: Track MAE on holdout samples over time
- **Alert thresholds**: Trigger retrain when metrics degrade >5%

### Implementation Plan

| Phase | Deliverable | Effort |
|-------|-------------|--------|
| 3.1.1 | MLflow setup + experiment tracking | 2-3 hours |
| 3.1.2 | GitHub Actions CI (lint, test) | 2-3 hours |
| 3.1.3 | Training pipeline workflow | 3-4 hours |
| 3.1.4 | Model registry + versioning | 2-3 hours |
| 3.1.5 | Deployment pipeline | 3-4 hours |
| 3.1.6 | Monitoring dashboard | 3-4 hours |

### Success Criteria

- [ ] All experiments logged to MLflow with metrics + artifacts
- [ ] PR requires passing CI checks before merge
- [ ] Model retraining can be triggered via GitHub Actions
- [ ] New models require evaluation gate before deployment
- [ ] Production model version tracked in registry
- [ ] Monitoring alerts on performance degradation

---

**Document Version:** 2.0  
**Last Updated:** 2025-12-08
