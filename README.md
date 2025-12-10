# Real Estate Price Estimator

**A Journey from Broken Script to Production MLOps**

> This project showcases the complete lifecycle of ML engineering: inheriting broken code, building a working MVP, optimizing for performance, and deploying an honest, production-ready system with fresh 2024 data.

---

## Quick Start (5 minutes)

```bash
# Clone
git clone https://github.com/BraPil/Real-Estate-Estimator.git
cd Real-Estate-Estimator

# Start all 3 model versions (V1 on 8000, V2.5 on 8001, V3.3 on 8002)
docker compose -f demo/docker-compose.demo.yml up -d

# Install Python dependencies
pip install -r requirements.txt

# Look up any King County address!
python scripts/compare_by_address.py "1523 15th Ave S" Seattle 98144
```

### Expected Output

```
================================================================================
                   REAL ESTATE ESTIMATOR - ADDRESS LOOKUP                       
================================================================================

Timestamp: 2025-12-10 18:47:42

--------------------------------------------------------------------------------
SERVICE HEALTH CHECK
--------------------------------------------------------------------------------
  V1     (Port 8000): HEALTHY
  V2.5   (Port 8001): HEALTHY
  V3.3   (Port 8002): HEALTHY

--------------------------------------------------------------------------------
PROPERTY LOOKUP
--------------------------------------------------------------------------------
    Searching: 1523 15th Ave S, Seattle, WA 98144
    Found property!

    PIN:             8850000080
    Matched Address:  1523 15TH AVE S, Seattle, 98144
    Geocode Score:    97%

--------------------------------------------------------------------------------
PROPERTY DETAILS
--------------------------------------------------------------------------------
    Location:      Zipcode 98144 (Beacon Hill/Mt Baker - Light rail, diverse, views)
    Size:          1,150 sqft living space
    Lot:           3,000 sqft
    Layout:        2 bedrooms, 1.0 bathrooms, 1.5 floors
    Year Built:    1903
    Condition:     3 (1=Poor, 3=Average, 5=Excellent)
    Grade:         7 (1-13 scale, 7=Average, 10+=Luxury)

--------------------------------------------------------------------------------
PREDICTIONS
--------------------------------------------------------------------------------
  V1 - KNN (k=5)
  --------------------------------------------------
  Model Version:    1.0.0
  Training Data:    2014-2015
  Features Used:    33
  Predicted Price:  $414,600

  V2.5 - XGBoost + RandomizedSearchCV
  --------------------------------------------------
  Model Version:    v2.5
  Training Data:    2014-2015
  Features Used:    43
  Predicted Price:  $337,363

  V3.3 - XGBoost + Optuna (30 trials)
  --------------------------------------------------
  Model Version:    v3.3
  Training Data:    2020-2024
  Features Used:    47
  Predicted Price:  $628,040

--------------------------------------------------------------------------------
COMPARISON SUMMARY
--------------------------------------------------------------------------------
    Version    Algorithm                           Data              Prediction
    ---------------------------------------------------------------------------
    1.0.0      KNN (k=5)                           2014-2015    $       414,600
    v2.5       XGBoost + RandomizedSearchCV        2014-2015    $       337,363
    v3.3       XGBoost + Optuna (30 trials)        2020-2024    $       628,040

================================================================================
                              ANALYSIS COMPLETE                                 
================================================================================
    Recommended Estimate (V3.3): $628,040
```

**What This Demonstrates:**
- **V3.3** uses fresh 2020-2024 data and is the recommended estimate
- **V1 and V2.5** use 2014-2015 data for comparison
- All 3 versions run simultaneously - zero downtime for updates

---

## The Story: Three Versions, One Journey

### V1: The MVP (Emergency Stabilization)

**Challenge:** Inherited a broken `create_model.py` with a critical bug on line 14 - it was trying to join sales data with demographic data on the wrong column.

**Solution:** Fixed the bug, built a simple KNN model, and wrapped it in a FastAPI application.

| Metric | Value |
|--------|-------|
| Model | KNN (k=5) |
| R-squared | 0.728 |
| MAE | $102,045 |
| Features | 33 (7 home + 26 demographic) |
| Data | 2014-2015 (21,613 samples) |

**Result:** A functional, deployable API serving predictions in under 50ms.

---

### V2.5: The Optimizer (Performance Maximization)

**Challenge:** The KNN model was leaving money on the table. Can we do better?

**Solution:** Systematic model comparison (KNN, Random Forest, LightGBM, XGBoost) with proper cross-validation and hyperparameter tuning via RandomizedSearchCV.

| Metric | Value |
|--------|-------|
| Model | XGBoost (RandomizedSearchCV tuned) |
| CV R-squared | 0.8945 |
| CV MAE | $63,529 |
| Features | 43 (17 home + 26 demographic) |
| Data | 2014-2015 (21,613 samples) |

**Key Improvements:**
- 38% reduction in MAE vs V1
- Expanded to 17 home features (added `yr_renovated`, `grade`, `condition`, etc.)
- Bootstrap confidence intervals for uncertainty quantification
- Residual analysis to understand error patterns

---

### V3.3: The Production System (Enterprise MLOps)

**Challenge:** The 2014-2015 training data was nearly 10 years old. Models trained on it were predicting 2015 prices, not 2024 prices.

**Solution:** Complete data pipeline modernization with fresh 2020-2024 King County Assessment data, plus rigorous evaluation to catch data leakage.

| Metric | Value |
|--------|-------|
| Model | XGBoost (Optuna Bayesian optimization) |
| CV R-squared | 0.868 |
| CV MAE | $115,247 |
| Features | 47 (17 home + 26 demographic + 4 temporal) |
| Data | 2020-2024 (143,476 samples) |

**Key Innovations:**
- **Fresh Data Pipeline:** Ingested 143k+ real transaction records from King County Assessor
- **Honest Evaluation:** GroupKFold cross-validation prevents leakage from repeat sales
- **Temporal Features:** `sale_year`, `sale_month`, `sale_quarter`, `sale_dow`
- **Real Coordinates:** GIS parcel centroids (100% geocode match rate)
- **MLflow Tracking:** Full experiment reproducibility
- **CI/CD Pipeline:** GitHub Actions for automated testing

> **Why is V3.3 MAE higher than V2.5?** Because 2024 home prices are much higher than 2015 prices! A $115k MAE on $850k median homes (13.5%) is comparable to V2.5's $64k MAE on $450k median homes (14.2%). The model is actually slightly more accurate in relative terms.

---

## Complete Version Comparison Matrix

| Dimension | V1.0 (MVP) | V2.5 (Optimized) | V3.3 (Production) |
|-----------|------------|------------------|-------------------|
| **Algorithm** | KNN (k=5) | XGBoost | XGBoost (Optuna) |
| **Tuning Method** | None | RandomizedSearchCV | Optuna Bayesian (30 trials) |
| **R-squared** | 0.728 | 0.8945 | 0.868 |
| **MAE** | $102,045 | $63,529 (CV) | $115,247 |
| **Data Vintage** | 2014-2015 | 2014-2015 | 2020-2024 |
| **Training Samples** | 21,613 | 21,613 | 143,476 |
| **Home Features** | 7 | 17 | 17 |
| **Demographic Features** | 26 | 26 | 26 |
| **Temporal Features** | 0 | 0 | 4 |
| **Total Features** | 33 | 43 | 47 |
| **CV Strategy** | Train/Test Split | KFold (5) | GroupKFold (5) |
| **Leakage Prevention** | None | None | GroupKFold by parcel_id |
| **Experiment Tracking** | None | Basic | MLflow (full) |
| **API Prefix** | `/predict`, `/health` | `/api/v1/...` | `/api/v1/...` |
| **Docker** | Yes | Yes | Yes |

### Key Source Files by Version

| Purpose | V1.0 | V2.5 | V3.3 |
|---------|------|------|------|
| Training | `train.py` | `train.py`, `tune_xgboost.py` | `train_with_mlflow.py`, `tune_v33.py` |
| Evaluation | `evaluate.py` | `robust_evaluate.py` | `evaluate_fresh.py` |
| API | `main.py` | `main.py`, `api/prediction.py` | `main.py`, `api/prediction.py` |
| Config | `config.py` | `config.py` | `config.py` |

---

## Technical Architecture

```
Client Request
      |
      v
+------------------+
|    FastAPI       |
|  (src/main.py)   |
+--------+---------+
         |
    +----+----+
    v         v
+-------+  +----------+
|Feature|  |  Model   |
|Service|  | Service  |
+---+---+  +----+-----+
    |           |
    v           v
+-------+  +--------+
|Zipcode|  |sklearn |
|  CSV  |  |Pipeline|
+-------+  +--------+
```

### Core Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.12 |
| API Framework | FastAPI |
| ML Framework | scikit-learn, XGBoost |
| Validation | Pydantic |
| Experiment Tracking | MLflow |
| Hyperparameter Optimization | Optuna |
| Containerization | Docker |
| CI/CD | GitHub Actions |

---

## Repository Structure

```
Real-Estate-Estimator/
|-- scripts/
|   |-- compare_by_address.py  # <-- Main demo: address lookup + 3-model comparison
|   |-- download_kc_data.py    # Optional: refresh King County data
|-- demo/
|   |-- Dockerfile.v1          # V1 MVP container
|   |-- Dockerfile.v2.5        # V2.5 Optimized container
|   |-- Dockerfile.v3.3        # V3.3 Production container
|   |-- docker-compose.demo.yml
|   |-- compare_versions.py    # Sample property comparison
|-- src/
|   |-- main.py                # FastAPI application
|   |-- config.py              # Configuration management
|   |-- train_with_mlflow.py   # MLflow-integrated training
|   |-- evaluate_fresh.py      # Honest evaluation with GroupKFold
|   |-- tune_v33.py            # Optuna hyperparameter optimization
|   |-- api/prediction.py      # Prediction endpoints
|   |-- services/
|       |-- feature_service.py     # Feature engineering
|       |-- model_service.py       # Model loading/prediction
|       |-- address_service_v2.py  # King County address lookup
|-- data/
|   |-- king_county/           # Compressed KC Assessor data (44MB)
|   |-- kc_house_data.csv      # Original 2014-2015 training data
|   |-- assessment_2020_plus.csv  # Fresh 2020-2024 data
|-- model/                     # Serialized model artifacts
|-- docs/                      # Development documentation
|-- tests/                     # Test suite
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture V1-V2](docs/ARCHITECTURE_REFERENCE-v1-v2.md) | System design evolution |
| [Architecture V2-V3](docs/ARCHITECTURE_REFERENCE-v2-v3.md) | Production architecture |
| [Model Evolution V1-V2](docs/MODEL_VERSION_EVOLUTION-v1-v2.md) | Algorithm progression |
| [Model Evolution V2-V3](docs/MODEL_VERSION_EVOLUTION-v2-v3.md) | Fresh data integration |
| [Master Log](docs/master_log.md) | Development timeline |

---

## License

MIT

---

**Built with care by BraPil**

*Demonstrating ML engineering best practices: honest evaluation, proper versioning, and production-ready deployment.*
