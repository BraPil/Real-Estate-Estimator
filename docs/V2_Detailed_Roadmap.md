# V2 Detailed Roadmap

**Project:** Real Estate Price Predictor  
**Date:** 2025-12-08  
**Status:** V2.1.x Complete, V2.3 In Progress (skipping V2.2 for now)

---

## Version Overview

| Version | Focus | Status | Key Deliverable |
|---------|-------|--------|-----------------|
| V2.1 | Feature Expansion | ✅ **COMPLETE** | +10 features, MAE -12% |
| V2.1.1 | Full Features Endpoint | ✅ **COMPLETE** | `/predict-full` - all 17 features, no zipcode |
| V2.1.2 | Adaptive Routing | ⏸️ **LOW PRIORITY** | Explored but deferred - `/predict-full` is sufficient |
| V2.2 | Feature Engineering | ⏸️ DEFERRED | Skipping for now - V2.3 has higher ROI |
| V2.3 | Hyperparameter Tuning | 🔜 **IN PROGRESS** | Optimal k, weights, distance metric |
| V2.4 | Model Alternatives | 📋 PLANNED | Random Forest, XGBoost comparison |
| V2.5 | Robust Evaluation | 📋 PLANNED | K-fold CV, confidence intervals |
| V2.6 | Fresh Data (Future) | 📋 PLANNED | Updated housing data (if available) |

### Decision Log (2025-12-08)
- **V2.1.2 Adaptive Routing:** Discovered price-tier pattern (confirmed statistically) but routing accuracy too low (52%) to beat always-use-`/predict-full`. Documented as interesting finding for future exploration.
- **V2.2 Feature Engineering:** Deferred in favor of V2.3. Hyperparameter tuning expected to provide cleaner, more defensible improvements.
- **V2.3 Hyperparameter Tuning:** Starting now - proper cross-validation avoids test-set peeking issues from V2.1.2 exploration.

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

## V2.5: Robust Evaluation

**Goal:** Implement statistically rigorous model evaluation.

### Components

1. **K-Fold Cross-Validation**
   - Replace single train/test split with 5-fold or 10-fold CV
   - Report mean ± standard deviation
   - More reliable estimate of true performance

2. **Confidence Intervals**
   - Bootstrap confidence intervals for MAE/R²
   - Report 95% CI, not just point estimates

3. **Target Transformation**
   - Try `log(price)` instead of `price`
   - Residuals may be more normally distributed
   - Better performance on high-value homes

4. **Residual Analysis**
   - Plot residuals vs predicted
   - Check for heteroscedasticity
   - Identify systematic errors by price range

### Implementation

```python
from sklearn.model_selection import cross_val_score, KFold
import numpy as np

# K-Fold CV
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_absolute_error')

print(f"MAE: ${-cv_scores.mean():,.0f} ± ${cv_scores.std():,.0f}")

# Log transform
y_log = np.log(y)
model.fit(X, y_log)
y_pred_log = model.predict(X_test)
y_pred = np.exp(y_pred_log)  # Transform back
```

### Expected Impact
- More confident in reported metrics
- Log transform may improve high-end predictions
- Better understanding of model limitations

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

**Document Version:** 1.0  
**Last Updated:** 2025-12-08
