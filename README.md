# Real Estate Price Estimator: From MVP to Enterprise MLOps

**Current Version:** v3.3.1  
**Status:** Production-Ready  
**Model:** XGBoost (Optuna Tuned)  
**Data Vintage:** 2020-2024 (King County Assessment Data)

---

## 📖 Project Story

This project represents the evolution of a simple machine learning script into a robust, enterprise-grade MLOps solution. It demonstrates not just model building, but the complete lifecycle of software engineering for AI: from fixing broken legacy code to deploying a scalable API with continuous integration and honest evaluation protocols.

### Phase 1: The Foundation (MVP)
**Goal:** Stabilize the client's assets and establish a baseline.

We started with a broken script (`create_model.py`) that pointed to the wrong data paths.
- **Fixed:** Critical bug in data merging (sales data vs demographics).
- **Built:** A FastAPI application to serve predictions.
- **Tracked:** Integrated MLflow for experiment tracking from day one.
- **Result:** A functional, deployable API with a baseline KNN model (MAE: ~$102k).

### Phase 2: Optimization & Rigor
**Goal:** Maximize performance and deeply understand model behavior.

We moved beyond the baseline, exploring feature engineering and model alternatives.
- **Expanded:** Integrated 26 demographic features (income, education, etc.).
- **Competed:** Benchmarked KNN against Random Forest, LightGBM, and XGBoost.
- **Selected:** XGBoost emerged as the winner (MAE reduced by 34%).
- **Validated:** Implemented rigorous evaluation (Bootstrap Confidence Intervals, Residual Analysis) to prove the model's stability.
- **Result:** A highly optimized model (R² 0.875) with understood error bounds.

### Phase 3: Enterprise MLOps & Reality
**Goal:** Modernize the data pipeline and ensure honest evaluation on fresh data.

We transitioned from static 2015 data to a dynamic pipeline using 2020-2024 assessment records.
- **Modernized:** Ingested 143k+ fresh records from King County Assessor data.
- **Geocoded:** Replaced synthetic coordinates with real GIS parcel centroids (100% match rate).
- **Corrected:** Identified and fixed critical data leakage (repeat sales) using GroupKFold cross-validation.
- **Automated:** Built a full CI/CD pipeline with GitHub Actions for automated testing and training.
- **Result:** An honest, robust system (R² 0.868) reflecting current market realities, not just historical patterns.

---

## 🚀 Key Achievements

| Metric | V1 (MVP) | V2.4 (Optimized) | V3.3 (Enterprise) |
|--------|----------|------------------|-------------------|
| **Model** | KNN | XGBoost | XGBoost (Tuned) |
| **Data Era** | 2014-2015 | 2014-2015 | **2020-2024** |
| **Evaluation** | Simple Split | Bootstrap CI | **GroupKFold (Honest)** |
| **R² Score** | 0.728 | 0.876 | **0.868** |
| **MAE** | $102,045 | $67,041 | **$115,247*** |
| **Pipeline** | Manual | Scripted | **CI/CD Automated** |

*\*Note: V3 MAE is higher because 2024 home prices are significantly higher than 2015 prices. The R² score confirms the model's predictive power remains excellent.*

---

## 🛠️ Technical Architecture

### Core Stack
- **Language:** Python 3.12
- **Framework:** FastAPI
- **ML Engine:** XGBoost + scikit-learn Pipeline
- **Tracking:** MLflow (SQLite backend)
- **Optimization:** Optuna (Bayesian Hyperparameter Tuning)

### Pipeline Components
1.  **Data Ingestion:** Transforms raw assessment CSVs, filters distressed sales (bottom 5%), and caps outliers (>$3M).
2.  **Feature Engineering:**
    - Temporal features (`sale_year`, `sale_quarter`)
    - Demographic enrichment (Census data)
    - GIS coordinate mapping
3.  **Training:**
    - `src/train_fresh_data.py`: Main training logic.
    - `src/tune_v33.py`: Optuna optimization loop.
4.  **Evaluation:**
    - `src/evaluate_fresh.py`: Generates metrics.json and plots.
    - **GroupKFold:** Ensures no data leakage from repeat sales.
5.  **Serving:**
    - REST API with health checks and metadata endpoints.
    - Input validation via Pydantic models.

---

## 📂 Repository Structure

```
├── .github/workflows/    # CI/CD Pipelines (Test & Train)
├── docs/                 # Comprehensive Documentation
│   ├── V3.3_Completion_Summary.md
│   ├── V2.5_Robust_Evaluation_Summary.md
│   └── human_in_the_loop_corrections.md
├── src/
│   ├── api/              # FastAPI Endpoints
│   ├── data/             # Data Transformation Scripts
│   ├── services/         # Business Logic
│   ├── train_fresh_data.py
│   └── tune_v33.py
├── tests/                # Pytest Suite
└── model/                # Serialized Model & Metrics
```

---

## 🚦 Getting Started

### Prerequisites
- Python 3.12+
- Virtual Environment

### Installation
```bash
# Clone repository
git clone https://github.com/BraPil/Real-Estate-Estimator.git
cd Real-Estate-Estimator

# Setup environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Running the API
```bash
# Start the server
PYTHONPATH=. uvicorn src.main:app --reload

# Test health endpoint
curl http://localhost:8000/api/v1/health
```

### Training the Model
```bash
# Train on fresh data
python src/train_with_mlflow.py --data-source fresh

# Run evaluation
python src/evaluate_fresh.py
```

---

## 🧠 Human-in-the-Loop

This project emphasizes the importance of human oversight in AI development. See `docs/human_in_the_loop_corrections.md` for a log of critical interventions, including:
- **Correction #15:** Replacing synthetic coordinates with real GIS data.
- **Correction #16:** Catching overfitting (inflated R² of 0.966) and enforcing honest evaluation.
- **Correction #17:** Aligning CI/CD pipelines with evolving data sources.

---

**Maintained by:** BraPil & AI Assistant  
**License:** MIT
