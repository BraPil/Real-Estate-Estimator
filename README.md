# Real Estate Price Estimator

**A Journey from Broken Script to Production MLOps**

> This project showcases the complete lifecycle of ML engineering: inheriting broken code, building a working MVP, optimizing for performance, and deploying an honest, production-ready system with fresh 2024 data.

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

## For Interviewers: Quick Demo

**See the evolution in action!** Run all three versions side-by-side and compare predictions.

### Prerequisites

- Docker and Docker Compose
- Git

### Step 1: Clone and Navigate

```bash
git clone https://github.com/BraPil/Real-Estate-Estimator.git
cd Real-Estate-Estimator/demo
```

### Step 2: Start All Three Versions

```bash
docker compose -f docker-compose.demo.yml up -d
```

This starts:
- **V1 MVP** on port 8000
- **V2.5 Optimized** on port 8001
- **V3.3 Production** on port 8002

### Step 3: Compare Predictions

Run the comparison script for a typical Seattle home (3BR/2.5BA, 2000 sqft, zipcode 98103):

```bash
./compare_versions.sh
```

Or use curl directly:

### Expected Output

```
========================================
  Real Estate API Version Comparison
========================================

Checking service availability...

[V1 MVP] Health Check (Port 8000)
Endpoint: GET /health  (NO /api/v1 prefix)
{ "status": "healthy" }

[V2.5] Health Check (Port 8001)
Endpoint: GET /api/v1/health
{ "status": "healthy", "model_version": "v2.5" }

[V3.3] Health Check (Port 8002)
Endpoint: GET /api/v1/health
{ "status": "healthy", "model_version": "v3.3" }

========================================
  Prediction Comparison (Same Input)
========================================

[V1 MVP] Prediction (7 features used)
Endpoint: POST /predict  (NO /api/v1 prefix)
{ "predicted_price": 670600.25 }

[V2.5] Full Prediction
Endpoint: POST /api/v1/predict
{ "predicted_price": 720261.50, "model_version": "v2.5" }

[V3.3] Full Prediction
Endpoint: POST /api/v1/predict
{ "predicted_price": 1291832.75, "model_version": "v3.3" }

========================================
  Key Differences Summary
========================================

V1 MVP:
  - Endpoints: /health, /predict (no /api/v1 prefix)
  - Features: 7 home + 26 demographic = 33 total
  - Data: 2014-2015 vintage

V2.5:
  - Endpoints: /api/v1/predict, /predict-minimal, /predict-adaptive
  - Features: 17 home + 26 demographic = 43 total
  - Data: 2014-2015 vintage
  - Added: Tier-based adaptive routing experiment

V3.3:
  - Endpoints: Same as V2.5
  - Features: 17 home + 26 demographic + 4 temporal = 47 total
  - Data: 2020-2024 vintage (fresh data)
  - Added: MLflow integration, production hardening
```

**What This Demonstrates:**
1. **V1 to V2.5:** Pure model improvement (same data, better algorithm) = +7% more accurate
2. **V2.5 to V3.3:** Massive difference reflects real market appreciation over 9 years

### Step 4: Try Your Own Properties

```bash
# V1 MVP
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"bedrooms": 4, "bathrooms": 3, "sqft_living": 2500, "zipcode": "98115"}'

# V2.5 Optimized
curl -X POST http://localhost:8001/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"bedrooms": 4, "bathrooms": 3, "sqft_living": 2500, "zipcode": "98115"}'

# V3.3 Production
curl -X POST http://localhost:8002/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"bedrooms": 4, "bathrooms": 3, "sqft_living": 2500, "zipcode": "98115"}'
```

### Step 5: Cleanup

```bash
docker compose -f docker-compose.demo.yml down
```

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
|-- demo/                  # Multi-version Docker demo
|   |-- Dockerfile.v1      # V1 MVP container
|   |-- Dockerfile.v2.5    # V2.5 Optimized container
|   |-- Dockerfile.v3.3    # V3.3 Production container
|   |-- docker-compose.demo.yml # Orchestration
|   |-- compare_versions.sh # Comparison script
|-- src/
|   |-- main.py            # FastAPI application
|   |-- config.py          # Configuration management
|   |-- train_with_mlflow.py  # MLflow-integrated training
|   |-- evaluate_fresh.py  # Honest evaluation with GroupKFold
|   |-- tune_v33.py        # Optuna hyperparameter optimization
|   |-- api/
|   |   |-- prediction.py  # Prediction endpoints
|   |-- services/
|       |-- feature_service.py  # Feature engineering
|       |-- model_service.py    # Model loading/prediction
|-- model/                 # Serialized model artifacts
|-- data/                  # Training data
|-- docs/                  # Documentation
|-- tests/                 # Test suite
```

---

## Development Setup

### Local Installation

```bash
# Clone
git clone https://github.com/BraPil/Real-Estate-Estimator.git
cd Real-Estate-Estimator

# Virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Dependencies
pip install -r requirements.txt
```

### Run the API

```bash
PYTHONPATH=. uvicorn src.main:app --reload
```

### Train a New Model

```bash
# Train on fresh 2020-2024 data
python src/train_with_mlflow.py --data-source fresh

# Or train on original 2014-2015 data
python src/train_with_mlflow.py --data-source original
```

### Run Tests

```bash
pytest tests/ -v
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [API Reference](docs/API.md) | Endpoint specifications |
| [Architecture](docs/ARCHITECTURE.md) | System design details |
| [Evaluation](docs/EVALUATION.md) | Model performance analysis |
| [V3.1 Summary](docs/V3.1_Completion_Summary.md) | CI/CD and MLOps implementation |

---

## License

MIT

---

**Built with care by BraPil**

*Demonstrating ML engineering best practices: honest evaluation, proper versioning, and production-ready deployment.*
