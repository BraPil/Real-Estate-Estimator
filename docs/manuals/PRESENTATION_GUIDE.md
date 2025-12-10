# Presentation Guide: Real Estate Estimator (V3.4)

**Purpose:** This guide provides a structured script and talking points for presenting the Real Estate Estimator project to technical stakeholders. It covers the narrative arc, architectural diagrams for whiteboarding, and detailed code walkthroughs.

---

## 1. The Narrative Arc (Whiteboard Script)

**Theme:** "From Broken Script to Enterprise MLOps"

### Opening: The Problem (Draw a broken line)
*   "We started with a legacy script (`create_model.py`) that was functionally broken."
*   "It pointed to non-existent data, had leakage issues, and used a simple KNN model that was 10 years out of date."
*   "Our goal was not just to fix it, but to build a scalable, production-ready system."

### Act 1: The Fix (V1 - MVP)
*   **Action:** Fixed the data merge bug (Sales + Demographics).
*   **Result:** A working API serving predictions.
*   **Metric:** MAE ~$102k (Baseline).

### Act 2: The Optimization (V2 - Data Science)
*   **Action:** We didn't just tune parameters; we engineered features.
*   **Key Insight:** "Location is more than a zipcode." We added 26 demographic features (income, education).
*   **Model Selection:** Benchmarked KNN vs. Random Forest vs. XGBoost.
*   **Winner:** XGBoost (Gradient Boosting) reduced error by 35%.

### Act 3: The Reality Check (V3 - MLOps)
*   **The Twist:** "A model trained on 2015 data cannot predict 2024 prices." (Inflation, Market Shift).
*   **The Solution:** We built a **Fresh Data Pipeline**.
    *   Ingested 2020-2024 King County Assessment data.
    *   **Critical Fix:** Implemented `GroupKFold` validation to prevent "Repeat Sale Leakage" (where the model memorizes a house it saw in training).
*   **Result:** A robust system that understands *current* market value.

---

## 2. Whiteboard Diagrams

### Diagram A: High-Level Architecture (The "What")

```text
[ User / Client ]
       |
       v
[ Load Balancer / Nginx ]
       |
       v
+---------------------------------------------------------------+
|  FastAPI Application (src/main.py)                            |
|                                                               |
|  1. Request  -> [ Pydantic Validation ]                       |
|  2. Enrich   -> [ FeatureService ] (Adds Demographics/GIS)    |
|  3. Predict  -> [ ModelService ] (XGBoost Pipeline)           |
|  4. Response <- { "price": 850000, "confidence": "high" }     |
+---------------------------------------------------------------+
       ^
       | (Loads Artifacts)
[ MLflow Model Registry ]
```

**Talking Points:**
*   **Separation of Concerns:** The API doesn't know about XGBoost; it just asks `ModelService` for a prediction.
*   **Enrichment:** The user sends 17 features; `FeatureService` expands this to 43 features using cached demographic data.

### Diagram B: The Training Pipeline (The "How")

```text
[ Raw County Data ] (CSVs)
       |
       v
[ transform_assessment_data.py ] <--- (Cleaning & Mapping)
       |
       v
[ Cleaned DataFrame ]
       |
       v
+-------------------------------------------------------+
|  train_with_mlflow.py                                 |
|                                                       |
|  1. Split (GroupKFold) ----------------------------+  |
|  2. Preprocessing (Imputer -> Scaler)              |  |
|  3. Training (XGBoost)                             |  |
|  4. Logging (MLflow: Params, Metrics, Artifacts) <-+  |
+-------------------------------------------------------+
       |
       v
[ MLflow Tracking Server ]
```

**Talking Points:**
*   **Reproducibility:** Every run is logged with git commit hash, parameters, and metrics.
*   **Leakage Prevention:** We split by `ParcelID`, not random rows, ensuring we test on *unseen houses*, not just unseen transactions.

---

## 3. File-by-File Talking Points

### `src/main.py` (The Entry Point)
*   **What it is:** The FastAPI application definition.
*   **Key Tech:** `lifespan` context manager.
*   **Why it matters:** It handles the "heavy lifting" of loading the model and demographic data *once* at startup, not on every request. This ensures low latency (<50ms).
*   **Look for:** The `startup` event where `get_model_service().load_model()` is called.

### `src/services/feature_service.py` (The Business Logic)
*   **What it is:** The brain that enriches raw inputs.
*   **Key Tech:** Pandas caching, Dictionary lookups.
*   **Why it matters:**
    *   **V2.1 Defaults:** Handles missing data intelligently (e.g., assumes "Average Grade" if not provided).
    *   **Temporal Features:** Calculates `sale_year`, `sale_quarter` dynamically based on the current date.
*   **Look for:** `V21_DEFAULT_FEATURES` dictionary.

### `src/train_with_mlflow.py` (The Engine)
*   **What it is:** The script that builds the model.
*   **Key Tech:** `sklearn.pipeline.Pipeline`, `xgboost.XGBRegressor`, `mlflow`.
*   **Why it matters:** It encapsulates the entire recipe. You can run this on a laptop or a massive GPU cluster.
*   **Look for:** The `Pipeline` definition steps: `('imputer', SimpleImputer), ('scaler', StandardScaler), ('model', XGBRegressor)`.

### `src/evaluate_fresh.py` (The Auditor)
*   **What it is:** The script that keeps us honest.
*   **Key Tech:** `GroupKFold`, `matplotlib` (for residual plots).
*   **Why it matters:** Standard cross-validation overestimates performance on real estate data because of repeat sales. This script forces the model to predict on houses it has *never* seen before.
*   **Look for:** `cv = GroupKFold(n_splits=5)` and the `groups=data['id']` parameter.

### `src/tune_v33.py` (The Optimizer)
*   **What it is:** The hyperparameter search script.
*   **Key Tech:** `optuna`.
*   **Why it matters:** Manual tuning is guessing. Optuna uses Bayesian optimization to "surf" the parameter space and find the global minimum error efficiently.
*   **Look for:** The `objective` function that defines what "good" looks like (minimizing MAE).

### `src/data/transform_assessment_data.py` (The Bridge)
*   **What it is:** The ETL (Extract, Transform, Load) script.
*   **Key Tech:** Pandas data manipulation.
*   **Why it matters:** Real-world data is messy. This script filters out "distressed sales" (e.g., foreclosures sold for $1) and maps legacy column names to our modern schema.
*   **Look for:** `FEATURE_MAPPING` dictionary and the `load_assessment_data` function.

