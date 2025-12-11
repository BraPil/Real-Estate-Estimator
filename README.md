# Real Estate Price Estimator

**A Journey from Broken Script to Production MLOps**

> This project showcases a complete lifecycle of ML engineering: Inheriting a solution in need of a little guidance, building a working MVP, optimizing for performance, and deploying a monitored, evaluated, production-ready system with CI/CD, fresh data and a useful UX.

---

## Quick Start (5 minutes)

### Prerequisites

Before starting, ensure you have installed:

| Tool | Version | Purpose | Install Link |
|------|---------|---------|--------------|
| **Git** | 2.0+ | Clone repository | [git-scm.com](https://git-scm.com/downloads) |
| **Docker Desktop** | 20.0+ | Run containerized APIs | [docker.com/get-started](https://www.docker.com/get-started/) |
| **Python** | 3.10+ | Run demo scripts | [python.org](https://www.python.org/downloads/) |
| **pip** | 21.0+ | Install dependencies | Included with Python |

> **Note:** Docker Desktop includes Docker Compose. On Linux, you may need to install `docker-compose-plugin` separately.

### Run the Demo

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

**Key Discovery #1 - Unused Features:** The original code only used 7 of 21 available columns, ignoring critical predictors like `grade` (construction quality), `waterfront`, `view`, `condition`, `yr_built`, and `lat/long`. Adding these features alone improved predictions significantly.

**Key Discovery #2 - Algorithm Limitation:** The original KNN algorithm was fundamentally limited. Our head-to-head comparison revealed XGBoost reduced prediction error by 38% - a finding that justified the complete algorithm change from V1 to V2.

| Original V1 Features (7) | Added in V2.5 (10) |
|--------------------------|---------------------|
| bedrooms, bathrooms | **grade** (most important!) |
| sqft_living, sqft_lot | **waterfront**, **view** |
| floors, sqft_above | **condition**, **yr_built** |
| sqft_basement | **yr_renovated**, **lat**, **long** |
| | sqft_living15, sqft_lot15 |

**Tuning Methodology:**

| Technique | Details |
|-----------|---------|
| Cross-Validation | 5-fold CV on all model comparisons |
| Hyperparameter Search | RandomizedSearchCV (50 iterations) |
| Evaluation Metric | Negative MAE (to find minimum error) |
| Parameters Tuned | 8 XGBoost params (n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_weight, gamma, reg_alpha) |

| Metric | Value |
|--------|-------|
| Model | XGBoost (RandomizedSearchCV tuned) |
| CV R-squared | 0.8945 |
| CV MAE | $63,529 |
| Features | 43 (17 home + 26 demographic) |
| Data | 2014-2015 (21,613 samples) |

**Key Improvements:**

- 38% reduction in MAE vs V1 (KNN to XGBoost)
- Expanded from 7 to 17 home features (added `grade`, `waterfront`, `view`, `condition`, `yr_built`, `yr_renovated`, `lat`, `long`, etc.)
- Bootstrap confidence intervals for uncertainty quantification
- Residual analysis to understand error patterns

**Lesson Learned: Knowing When to Stop**

Between V2.5 and V3, we explored **tiered model routing** (V2.7) - training separate specialists for low/mid/high-price homes with a classifier routing predictions. After 6 experimental approaches:
- Best result: +0.17% MAE improvement
- Cost: 2x models to maintain, routing logic, increased failure modes
- Decision: **Abandoned** - insufficient ROI for added complexity

This demonstrates engineering judgment: not all improvements are worth implementing.

---

### V3.3: The Production System (Enterprise MLOps)

**Challenge:** The 2014-2015 training data was nearly 10 years old. Models trained on it were predicting 2015 prices, not 2024 prices.

**Solution:** Complete data pipeline modernization with fresh 2020-2024 King County Assessment data, plus rigorous evaluation to catch data leakage.

**Tuning Methodology (Upgraded from V2.5):**
| Technique | Details |
|-----------|---------|
| Cross-Validation | 5-fold GroupKFold (grouped by parcel to prevent leakage) |
| Hyperparameter Search | Optuna Bayesian optimization (30 trials) |
| Sampler | TPE (Tree-structured Parzen Estimator) |
| Parameters Tuned | 9 XGBoost params including regularization (reg_alpha, reg_lambda) |
| Best Learning Rate | 0.113 (discovered via log-scale search) |
| Best Max Depth | 10 (deeper than V2.5's 7) |

| Metric | Value |
|--------|-------|
| Model | XGBoost (Optuna Bayesian optimization) |
| CV R-squared | 0.877 |
| CV MAE | $115,344 |
| Test MAE | $112,955 |
| Test MAPE | 11.4% |
| Features | 47 (17 home + 26 demographic + 4 temporal) |
| Data | 2020-2024 (143,476 samples) |

**Key Innovations:**

- **Fresh Data Pipeline:** Ingested 143k+ real transaction records from King County Assessor
- **Honest Evaluation:** GroupKFold cross-validation prevents leakage from repeat sales
- **Temporal Features:** `sale_year`, `sale_month`, `sale_quarter`, `sale_dow`
- **Real Coordinates:** GIS parcel centroids from 844MB King County GeoJSON (8.8% MAE improvement over synthetic lat/long)
- **MLflow Tracking:** Full experiment reproducibility
- **CI/CD Pipeline:** GitHub Actions for automated testing

**Key Discovery #3 - Data Leakage Detection:**

Initial V3.3 results showed R² = 0.966 - suspiciously high. User skepticism prompted investigation:
- Found 12.4% repeat sales (same property sold multiple times in dataset)
- Random train/test split leaked information (property in both sets)
- Implemented GroupKFold CV splitting by parcel_id
- Honest R² = 0.868 (still excellent, now trustworthy)

**Same House, Different Data:**

The same property predicts dramatically different prices based on data vintage:
- 2014-2015 data (V1/V2.5): ~$670,000
- 2020-2024 data (V3.3): ~$1,290,000

This 92% increase reflects actual King County real estate appreciation, validating why fresh data was essential.

> **Why is V3.3 MAE higher than V2.5?** Because 2024 home prices are much higher than 2015 prices! A $115k MAE on $850k median homes (13.5%) is comparable to V2.5's $64k MAE on $450k median homes (14.2%). The model is actually slightly more accurate in relative terms.

---

### V4.1: Address-Based Lookup (User Experience)

**Challenge:** Users don't know their exact square footage. Manual feature entry is tedious and error-prone.

**Solution:** Type any King County address, get an instant prediction with property details automatically populated.

| Feature | Implementation |
|---------|----------------|
| Geocoding | King County Official Geocoder API |
| Building Data | EXTR_ResBldg.csv (44MB compressed to 7MB) |
| Parcel Data | EXTR_Parcel.csv (lot sizes) |
| Coordinates | ArcGIS Residential Parcels API |

**Key Innovations:**

- **Zero Manual Entry:** Address-to-prediction in one command
- **Official Data Source:** King County Assessor records (authoritative)
- **Compressed Storage:** 84% smaller repository with auto-decompression
- **Fallback Strategy:** CSV text search when API is unavailable
- **Multi-Model Demo:** `compare_by_address.py` queries all 3 versions simultaneously

**New Component:** `src/services/address_service_v2.py` (606 lines)

```
User enters: "1523 15th Ave S, Seattle 98144"
      |
      v
KC Geocoder API -> PIN: 8850000080
      |
      v
EXTR_ResBldg.csv -> 2 bed, 1 bath, 1150 sqft, grade 7
      |
      v
EXTR_Parcel.csv -> 3000 sqft lot
      |
      v
Prediction: $628,040 (V3.3)
```

---

### Human-in-the-Loop: Responsible AI Collaboration

This project documents substantive instances where human oversight improved outcomes:

| Correction | Impact |
|------------|--------|
| **ROI vs Complexity (V2.7)** | User recognized 0.17% gain didn't justify 2x model complexity - abandoned tiered approach |
| **Data Leakage Detection (V3.3)** | User skepticism of 96.6% R² caught repeat-sale leakage - led to GroupKFold |
| **Real GIS Coordinates (V3.2)** | User provided 844MB GeoJSON with actual parcel coordinates - 8.8% MAE improvement |
| **Feature Name Schema (V3.1)** | User caught test fixtures using invented names instead of actual feature schema |

Full documentation: [Human-in-the-Loop Corrections](docs/human_in_the_loop_corrections.md)

---

## Complete Version Comparison Matrix

| Dimension | V1.0 (MVP) | V2.5 (Optimized) | V3.3 (Production) | V4.1 (UX) |
|-----------|------------|------------------|-------------------|-----------|
| **Algorithm** | KNN (k=5) | XGBoost | XGBoost (Optuna) | XGBoost (Optuna) |
| **Tuning Method** | None | RandomizedSearchCV | Optuna Bayesian (30 trials) | Same as V3.3 |
| **R-squared** | 0.728 | 0.8945 | 0.868 | 0.868 |
| **MAE** | $102,045 | $63,529 (CV) | $115,247 | $115,247 |
| **Data Vintage** | 2014-2015 | 2014-2015 | 2020-2024 | 2020-2024 |
| **Training Samples** | 21,613 | 21,613 | 143,476 | 143,476 |
| **Home Features** | 7 | 17 | 17 | 17 |
| **Demographic Features** | 26 | 26 | 26 | 26 |
| **Temporal Features** | 0 | 0 | 4 | 4 |
| **Total Features** | 33 | 43 | 47 | 47 |
| **CV Strategy** | Train/Test Split | KFold (5) | GroupKFold (5) | GroupKFold (5) |
| **Leakage Prevention** | None | None | GroupKFold by parcel_id | GroupKFold by parcel_id |
| **Experiment Tracking** | None | Basic | MLflow (full) | MLflow (full) |
| **API Prefix** | `/predict`, `/health` | `/api/v1/...` | `/api/v1/...` | `/api/v1/...` |
| **Address Lookup** | No | No | No | **Yes** |
| **Docker** | Yes | Yes | Yes | Yes |

### Key Source Files by Version

| Purpose | V1.0 | V2.5 | V3.3 | V4.1 |
|---------|------|------|------|------|
| Training | `train.py` | `train.py`, `tune_xgboost.py` | `train_with_mlflow.py`, `tune_v33.py` | Same as V3.3 |
| Evaluation | `evaluate.py` | `robust_evaluate.py` | `evaluate_fresh.py` | Same as V3.3 |
| API | `main.py` | `main.py`, `api/prediction.py` | `main.py`, `api/prediction.py` | Same + `address_service_v2.py` |
| Config | `config.py` | `config.py` | `config.py` | Same |
| Demo | - | - | `compare_versions.py` | `compare_by_address.py` |

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
|       |-- address_service_v2.py  # V4.1: King County address lookup (606 lines)
|-- data/
|   |-- king_county/               # V4.1: Compressed KC Assessor data
|   |   |-- EXTR_ResBldg.csv.gz    # Building data (7MB, auto-decompresses to 44MB)
|   |   |-- EXTR_Parcel.csv.gz     # Parcel/lot data
|   |-- kc_house_data.csv          # Original 2014-2015 training data
|   |-- assessment_2020_plus.csv   # Fresh 2020-2024 data
|-- model/                     # Serialized model artifacts
|-- docs/                      # Development documentation
|   |-- manuals/               # Technical education and presentation slides
|-- tests/                     # Test suite
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture V1-V2](docs/ARCHITECTURE_REFERENCE-v1-v2.md) | System design evolution |
| [Architecture V2-V3](docs/ARCHITECTURE_REFERENCE-v2-v3.md) | Production architecture |
| [Architecture V3-V4](docs/ARCHITECTURE_REFERENCE-v3_3-v4_1.md) | Address lookup service |
| [Model Evolution V1-V2](docs/MODEL_VERSION_EVOLUTION-v1-v2.md) | Algorithm progression |
| [Model Evolution V2-V3](docs/MODEL_VERSION_EVOLUTION-v2-v3.md) | Fresh data integration |
| [Session Log V3-V4](docs/SESSION_LOG_20251210-v3_3-v4_1.md) | Demo and address lookup development |
| [Technical Education](docs/manuals/TECHNICAL_EDUCATION.md) | Deep-dive into all components |
| [Presentation Slides](docs/manuals/PRESENTATION_SLIDES.md) | 10-slide presentation (5 client + 5 technical) |
| [Master Log](docs/master_log.md) | Development timeline |

---

## License

MIT

---

**Built by [Brandt Pileggi](https://www.linkedin.com/in/brandtpileggi/)**

*Demonstrating ML engineering best practices: honest evaluation, proper versioning, and production-ready deployment.*
