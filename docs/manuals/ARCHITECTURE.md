# Architecture Documentation

**Project:** Real Estate Price Predictor API  
**Version:** 1.0.0  
**Date:** 2025-12-07

---

## System Overview

The Real Estate Price Predictor is a RESTful API service that predicts home prices in King County (Seattle), WA using a machine learning model trained on historical housing data.

### High-Level Architecture

```
                                    +------------------+
                                    |   Load Balancer  |
                                    +--------+---------+
                                             |
              +------------------------------+------------------------------+
              |                              |                              |
    +---------v---------+        +-----------v---------+        +-----------v---------+
    |    API Instance   |        |    API Instance     |        |    API Instance     |
    |    (Container)    |        |    (Container)      |        |    (Container)      |
    +-------------------+        +---------------------+        +---------------------+
              |                              |                              |
              +------------------------------+------------------------------+
                                             |
                             +---------------v---------------+
                             |      Shared Volume / S3       |
                             |   (Model + Demographics)      |
                             +-------------------------------+
```

### Key Design Principles

1. **Stateless API:** No session state, enabling horizontal scaling
2. **In-Memory Caching:** Demographics data loaded at startup for fast lookup
3. **Graceful Degradation:** API runs without MLflow if not configured
4. **Container-First:** Designed for Docker and orchestration platforms

---

## Component Architecture

### Project Structure

```
Real-Estate-Estimator/
├── src/
│   ├── __init__.py
│   ├── main.py              # FastAPI application entry point
│   ├── config.py            # Configuration management (Pydantic Settings)
│   ├── models.py            # Pydantic request/response schemas
│   ├── api/
│   │   ├── __init__.py
│   │   └── prediction.py    # API endpoints (/health, /predict, /predict-minimal)
│   ├── services/
│   │   ├── __init__.py
│   │   ├── model_service.py # Model loading and prediction
│   │   └── feature_service.py # Demographics lookup and enrichment
│   ├── train.py             # Model training script
│   └── evaluate.py          # Model evaluation script
├── data/
│   ├── kc_house_data.csv    # Training data (21,613 samples)
│   ├── zipcode_demographics.csv # Demographics by zipcode (83 zipcodes)
│   └── future_unseen_examples.csv # Test examples (300 samples)
├── model/
│   ├── model.pkl            # Trained sklearn Pipeline
│   ├── model_features.json  # Feature names in training order
│   └── metrics.json         # Training/test metrics
├── scripts/
│   └── test_api.py          # API test script
├── Dockerfile               # Container definition
├── docker-compose.yml       # Multi-container orchestration
└── requirements.txt         # Python dependencies
```

---

## Data Flow

### Prediction Request Flow

```
1. Client sends POST /api/v1/predict with JSON body
   |
2. FastAPI validates request against PredictionRequest schema
   |
3. prediction.py extracts model features from request
   |
4. feature_service looks up demographics by zipcode (O(1) in-memory)
   |
5. Features merged: 7 home features + 26 demographic features = 33 total
   |
6. model_service orders features to match training order
   |
7. sklearn Pipeline (RobustScaler -> KNN) makes prediction
   |
8. Response built with prediction + metadata + warnings
   |
9. Client receives PredictionResponse JSON
```

---

## Scalability Design

### Horizontal Scaling Strategy

The API is designed for horizontal scaling without stopping the service:

1. **Stateless Design:** No session state stored in the API
2. **In-Memory Caching:** Demographics loaded once per replica
3. **Model Loading:** Model loaded once at startup per replica

### Scaling Options

| Platform | Approach |
|----------|----------|
| Docker Compose | Increase replicas with load balancer |
| Kubernetes | HorizontalPodAutoscaler based on CPU |
| AWS ECS | Service Auto Scaling |
| Cloud Run | Automatic scaling |

---

## Model Versioning

### Zero-Downtime Updates

1. **Blue/Green:** Deploy new version alongside old, switch traffic
2. **Rolling Update:** Kubernetes updates pods one at a time
3. **MLflow Registry:** Load model by stage (Production, Staging)
4. **Hot Reload:** Call `model_service.reload()` without restart

---

## Technology Stack

| Layer | Technology | Rationale |
|-------|------------|-----------|
| Web Framework | FastAPI | Modern, async, auto-docs, Pydantic |
| Server | Uvicorn | ASGI, high performance |
| Validation | Pydantic | Type safety, serialization |
| ML Model | scikit-learn | Simple, reliable |
| MLOps | MLflow | Tracking, registry |
| Container | Docker | Reproducibility, portability |

---

**Last Updated:** 2025-12-07
