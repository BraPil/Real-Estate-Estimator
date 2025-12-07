## Model Evaluation – Real-Estate-Estimator

**Project:** phData Machine Learning Engineer Coding Test  
**Model:** KNeighborsRegressor (k=5) with RobustScaler  
**Data:** King County (Seattle) housing sales 2014–2015 + zipcode demographics  
**Author (MLE Candidate):** [Your Name]

---

## Part 1 – Business / Client-Facing Evaluation

### 1.1 Executive Summary

- The current model is a **solid first-pass prototype**. On unseen data, it explains about **73% of the variance** in home sale prices.
- For typical properties in the **mid-market price range**, it provides **directionally useful estimates** that can support exploratory analysis and rough comparisons.
- However, the model has **large errors for higher-priced homes** and is trained on **2014–2015 data only**, with **no built-in retraining process**. As a result, it is **not yet ready to be the primary engine for high-stakes pricing or underwriting decisions**.
- I recommend treating this as a **strong baseline to build on**, and investing in a focused **v2** effort that refreshes the data, expands the feature set, and tightens the model and its operational controls.

### 1.2 What the Model Does

- Uses historical home sale data from King County (Seattle) and **merges it with zipcode-level demographic data**.
- Learns relationships between:
  - Home features: bedrooms, bathrooms, square footage, lot size, etc.
  - Area features: income, education, and other demographic indicators.
- Given a new home’s features (and zipcode), it **predicts an estimated sale price**.

### 1.3 How Well It Performs (Overall Metrics)

On a hold-out test set of **5,404 homes** (25% of the data):

| **Metric** | **Value** | **Interpretation** |
|-----------|----------:|--------------------|
| **R²** | 0.7281 | Model explains ~73% of price variance |
| **MAE** | $102,044.70 | Average absolute error per home |
| **RMSE** | $201,659.43 | Larger errors penalized more strongly |
| **MAPE** | 17.90% | Typical relative error ≈ 18% |

**Plain-language interpretation:**

- The model captures the **major pricing patterns** in the data.
- On average, each prediction is about **$100k away from the actual sale price**, which is often **15–25%** of the home’s value.
- For some use cases (e.g. portfolio-level analysis, rough scenario planning), this can be acceptable. For others (e.g. precise pricing or underwriting), it is **not tight enough**.

### 1.4 Where the Model Works Best (and Where It Doesn’t)

The table below shows average errors by price segment:

| **Price Range (Test Set)** | **Count (n)** | **MAE** | **RMSE** |
|----------------------------|--------------:|--------:|---------:|
| Under $300k                | 1,107         | $53,808 | $79,338  |
| $300k–$500k                | 1,963         | $58,305 | $85,968  |
| $500k–$750k                | 1,381         | $88,223 | $122,339 |
| $750k–$1M                  | 552           | $148,683 | $223,207 |
| Over $1M                   | 401           | $432,723 | $611,844 |

**Key observations:**

- In the **core of the market** (roughly $200k–$700k), the model’s average error is typically in the **$50k–$90k** range.
  - This is large, but still useful as a **directional tool** and for comparing neighborhoods or property types.
- In the **upper segments** (especially **> $1M**), the model’s errors become **very large in absolute terms**:
  - Average error in the $1M+ segment is **over $400k**.
  - This means that for a $1.5M home, the model might easily be off by **30–50% or more**.

From a business perspective:

- The model is **much more reliable in the mid-market** than for **luxury or atypical properties**.
- If we were to deploy this as-is for high-value properties, **the dollar value lost in those large mistakes could easily outweigh the value of being “roughly right” on average**.

### 1.5 Bias and Risk Considerations

The evaluation shows:

- **Mean residual:** +$19,604  
- **Median residual:** +$1,832  
- **Residual skewness:** 3.76 (strong right skew)

**What this means in business terms:**

- For a **typical home**, the model is roughly unbiased (median error is close to zero).
- However, the **average error is pulled upward** by a subset of homes where we **significantly overestimate the price** (predicted value much higher than actual).
- Statistically, this is a **long right tail**: most errors are modest, but a smaller number are **very large overestimates**.

From a risk perspective:

- The model is generally reasonable in the middle of the market, but in some cases – particularly at **higher price points** – it can be **overly optimistic about value**.
- If such a model were used for decisions like **lending, pricing, or guarantees**, these large overestimates could result in **material financial risk**.

### 1.6 Summary for Business Stakeholders

- The current model is a **good prototype** and a **strong starting point**:
  - It leverages both home and demographic data.
  - It demonstrates that we can explain a large portion of price variation.
  - It provides a working foundation for an API and an MLOps pipeline.
- At the same time, it is **not yet the finished product** for enterprise use:
  - The data is **old (2014–2015)** and there is **no retraining mechanism**.
  - It **underperforms badly on luxury properties** (errors > $400k).
  - It lacks some important features (e.g. condition, grade, renovation history, time trends).

**Recommendation:**

> Treat this model as a **strong v1 baseline**, suitable for exploration and learning. Invest in a focused **v2 iteration** that uses fresher data, more complete features, and a modest amount of tuning and model comparison to bring errors down to a level that is comfortable for business use.

---

## Part 2 – Technical Evaluation for Interviewers

### 2.1 Data, Splits, and Setup

- **Source data:**
  - `kc_house_data.csv` – 21,613 home sales in King County (2014–2015)
  - `zipcode_demographics.csv` – 83 zipcodes with 27 demographic features
- **Feature construction:**
  - Subset of home features (price, bedrooms, bathrooms, sqft, lot size, floors, above/basement sqft, zipcode).
  - Join on `zipcode` to demographics; `zipcode` column dropped afterward.
  - Final feature space: **33 features**.
- **Target:** `price`
- **Split:**
  - Single train/test split with `test_size=0.25`, `random_state=42`.
  - Train: 16,209 samples  
    Test: 5,404 samples
- **Model:**
  - `Pipeline([RobustScaler(), KNeighborsRegressor(n_neighbors=5)])`
- **Tracking:**
  - Metrics, params, and artifacts logged with **MLflow**.
  - Model registered as `real-estate-price-predictor` v1 in MLflow Model Registry.

### 2.2 Metrics (Train vs Test)

From `train.py`:

- **Train R²:** 0.8414  
- **Test R²:** 0.7281  
- **Overfitting gap:** 0.1133 (~11.3%)

- **Train MAE:** $76,232.25  
- **Test MAE:** $102,044.70

- **Train RMSE:** $143,466.79  
- **Test RMSE:** $201,659.43

**Interpretation:**

- The model is **fitting the training data well** and maintains **reasonable generalization** to the test set.
- The ~11% R² gap is **moderate overfitting**, consistent with KNN at k=5 on tabular data.
- The jump from train MAE/RMSE to test MAE/RMSE is expected but non-trivial, reinforcing the narrative that the model is a good baseline, not a polished solution.

### 2.3 Residual and Error Distribution

From `evaluate.py` (test set):

- **R²:** 0.7281  
- **MAE:** $102,044.70  
- **RMSE:** $201,659.43  
- **MAPE:** 17.90%

- **Mean residual:** +$19,604.28  
- **Median residual:** +$1,832.50  
- **Std residual:** $200,704.26  
- **Skewness:** 3.7647

- **Actual price range:** $78,000 – $5,570,000  
- **Predicted price range:** $132,640 – $4,021,360  
- **Correlation(y, ŷ):** 0.8576

**Technical interpretation:**

- **RMSE >> MAE** (≈ 2x) indicates a **fat-tailed error distribution**:
  - Many errors are in a moderate band (tens of thousands).
  - A non-trivial subset of errors are **very large**, pushing up RMSE.
- **Positive mean residual** with near-zero median and strong positive skew:
  - Most residuals are small and relatively symmetric.
  - A smaller number of cases yield **large positive residuals** (i.e. **overestimates**).
- **Prediction range compression**:
  - The model never predicts prices as low or as high as the true extremes; KNN is pulling extremes toward the mean of the training data.

In short: the model behaves reasonably in the bulk of the distribution but exhibits **heavy tails and optimistic bias** in certain regions, especially at high prices.

### 2.4 Segment-Level Analysis

Error by price range (test set):

| Price Range      | n    | MAE     | RMSE      |
|------------------|------|---------|-----------|
| under_300k       | 1107 | $53,808 | $79,338   |
| 300k_to_500k     | 1963 | $58,305 | $85,968   |
| 500k_to_750k     | 1381 | $88,223 | $122,339  |
| 750k_to_1m       | 552  | $148,683 | $223,207  |
| over_1m          | 401  | $432,723 | $611,844  |

**Technical implications:**

- The model’s **global metrics are dominated by mid-market behavior**.
- Error **scales up aggressively** in higher price bands, particularly **> $1M**:
  - The effective error distribution is far from homoscedastic.
  - For luxury properties, the model’s signal-to-noise ratio is poor.
- A more production-ready system would likely:
  - Use **segment-aware modeling** (e.g. separate models or at least separate evaluation constraints per price band).
  - Possibly adjust loss functions or evaluation criteria to reflect the business risk at higher price points.

### 2.5 Limitations and Risks (Technical)

Key limitations from a technical perspective:

- **Data vintage and drift:**
  - Only 2014–2015 data used; no explicit handling of market drift.
  - No scheduled retraining, no time-aware validation splits.

- **Model choice and tuning:**
  - KNN with fixed k=5, default weights and distance metric.
  - No hyperparameter tuning or model comparison (e.g. trees, boosting).

- **Feature utilization:**
  - Not all available features are used (e.g. condition, grade, renovation history, temporal features).
  - No advanced feature engineering (ratios, composite indices, log-transform of price, etc.).

- **Evaluation depth:**
  - Single random train/test split; no k-fold cross-validation to quantify variability.
  - Segment-level analysis present but not yet tied to hard business thresholds.

- **Operationalization:**
  - MLflow integration exists, but:
    - No automated CI/CD pipeline is currently wired.
    - No monitoring/drift detection rules have been defined.

### 2.6 Planned v2 Improvements (Technical Roadmap)

Planned and documented in `logs/v2_roadmap.md`:

- **Data & Retraining:**
  - Incorporate **fresher data** beyond 2014–2015.
  - Introduce a **retraining cadence** (e.g. scheduled jobs, MLflow-backed pipeline).

- **Feature Engineering:**
  - Add and engineer features such as:
    - Condition, grade, renovation/age of property.
    - Time-based features (sale month/year, market trend indicators).
    - Ratios (e.g. sqft_living / sqft_lot) and composite demographic indices.

- **Model Tuning & Comparison:**
  - KNN hyperparameter search (k, weights, distance metric).
  - Benchmark against tree-based models (Random Forest, Gradient Boosting, XGBoost/LightGBM).
  - Consider log(price) as target to stabilize variance.

- **Evaluation & Monitoring:**
  - Use **k-fold cross-validation** to estimate generalization more robustly.
  - Maintain **segment-level monitoring** (price bands, zipcodes) and set business-aligned thresholds.
  - Log all metrics and artifacts via MLflow for comparison across versions.

---

**Intended Use of This Document:**

- **Sections 1.1–1.6**: For business stakeholders / clients – accessible, non-technical summary of what the model can and cannot do today.
- **Sections 2.1–2.6**: For technical interviewers – demonstrates depth of evaluation, awareness of limitations, and a clear path to an improved, production-ready system.
