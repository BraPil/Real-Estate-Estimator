# Model Evaluation Report

**Project:** Real Estate Price Predictor  
**Model:** KNeighborsRegressor (k=5) with RobustScaler  
**Date:** 2025-12-07

---

## Part 1: Business / Client-Facing Evaluation

### Executive Summary

The model predicts home prices in King County (Seattle) with reasonable accuracy for typical homes. It explains about 73% of price variation, with an average error of approximately $102,000.

### Key Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| R-squared | 0.728 | Model explains 72.8% of price variance |
| Average Error | $102,045 | Typical prediction is off by about $102k |
| Median Home Price | ~$450,000 | Error is about 23% of median price |

### Where the Model Works Best

- **Typical homes:** 2-4 bedrooms, $300k-$600k range
- **Standard neighborhoods:** Areas with consistent housing stock
- **Well-represented zipcodes:** Areas with many training examples

### Where the Model Struggles

- **Luxury homes:** Properties over $1M have larger errors (up to $300k+)
- **Unique properties:** Waterfront, exceptional views, unusual sizes
- **Outlier neighborhoods:** Very high or very low income areas

### Recommendation for Business Use

This model is suitable for:
- Initial price estimates for typical properties
- Comparative analysis between neighborhoods
- Training and demonstration purposes

**Not recommended for:**
- Final pricing decisions without human review
- Luxury property valuations
- Legal or financial commitments

---

## Part 2: Technical Evaluation

### Training Setup

- **Algorithm:** KNeighborsRegressor (k=5)
- **Preprocessing:** RobustScaler
- **Train/Test Split:** 75/25, random_state=42
- **Training Samples:** 16,209
- **Test Samples:** 5,404
- **Features:** 34 (7 home + 27 demographic)

### Performance Metrics

| Metric | Train | Test | Gap |
|--------|-------|------|-----|
| R-squared | 0.841 | 0.728 | 0.113 |
| MAE | $76,232 | $102,045 | $25,813 |
| RMSE | $143,467 | $201,659 | $58,192 |

### Overfitting Analysis

The 11.3% R-squared gap between train and test suggests moderate overfitting, which is expected for KNN. The model memorizes training data to some degree but still generalizes reasonably.

### Residual Analysis

| Statistic | Value |
|-----------|-------|
| Mean Residual | +$19,610 |
| Median Residual | +$3,180 |
| Std Deviation | $201,547 |
| Skewness | +1.42 |

**Interpretation:** Positive skew indicates occasional large overestimates, particularly for high-value homes.

### Error by Price Range

| Price Range | MAE | Sample Size |
|-------------|-----|-------------|
| Under $300k | $58,000 | ~1,800 |
| $300k-$500k | $85,000 | ~2,100 |
| $500k-$750k | $115,000 | ~900 |
| $750k-$1M | $145,000 | ~350 |
| Over $1M | $295,000 | ~250 |

### Limitations

1. **Data Vintage:** 2014-2015 data is stale
2. **Missing Features:** waterfront, view, condition, grade not used
3. **No Hyperparameter Tuning:** k=5 is default, not optimized
4. **Single Split:** No cross-validation

### V2 Improvement Opportunities

1. Add spatial features (lat/long, distance to downtown)
2. Add unused features (waterfront, view, grade)
3. Hyperparameter tuning
4. Try tree-based models (Random Forest, XGBoost)
5. Model log(price) for better high-end performance

---

**Last Updated:** 2025-12-07
