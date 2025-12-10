# Multi-Version Demo

This folder runs **V1 MVP**, **V2.5**, and **V3.3** APIs side-by-side for interview demonstration.

## Quick Start

```bash
cd demo
docker-compose -f docker-compose.demo.yml up --build
```

Wait for services to initialize (~2-3 minutes for model training).

## Endpoints

| Version | Port | Health Check | Predict | Features |
|---------|------|--------------|---------|----------|
| **V1 MVP** | 8000 | `GET /health` | `POST /predict` | 33 (7+26) |
| **V2.5** | 8001 | `GET /api/v1/health` | `POST /api/v1/predict` | 43 (17+26) |
| **V3.3** | 8002 | `GET /api/v1/health` | `POST /api/v1/predict` | 43 (17+26) |

## Run Demo Script

```bash
chmod +x compare_versions.sh
./compare_versions.sh
```

## Key Differences

### V1 MVP (Bare Minimum)
- Meets minimum requirements only
- Simple `/health` and `/predict` endpoints (NO `/api/v1/` prefix)
- Uses 7 home features + 26 demographics = 33 total
- 2014-2015 data vintage
- No experimental endpoints

### V2.5 (Experimentation Phase)
- Added `/api/v1/` prefix for versioning
- Expanded to 17 home features + 26 demographics = 43 total
- Experimental endpoints:
  - `/api/v1/predict-minimal` - 7 features with default demographics
  - `/api/v1/predict-adaptive` - Tier-based routing (discovered strategy)
- Same 2014-2015 data

### V3.3 (Production Release)
- Same endpoints as V2.5
- Updated to 2020-2024 data vintage
- MLflow integration for model tracking
- Production-ready error handling

## Swagger Documentation

- V1 MVP: http://localhost:8000/docs
- V2.5:   http://localhost:8001/docs
- V3.3:   http://localhost:8002/docs

## Cleanup

```bash
docker-compose -f docker-compose.demo.yml down
docker rmi real-estate-v1 real-estate-v2.5 real-estate-v3.3
```

