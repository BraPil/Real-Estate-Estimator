# API Documentation

**Base URL:** `http://localhost:8000`  
**API Version:** v1  
**Prefix:** `/api/v1`

---

## Quick Start

```bash
# Start the API
uvicorn src.main:app --host 0.0.0.0 --port 8000

# Health check
curl http://localhost:8000/api/v1/health

# Make a prediction
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "bedrooms": 3, "bathrooms": 2.5, "sqft_living": 2000,
    "sqft_lot": 5000, "floors": 2.0, "waterfront": 0,
    "view": 0, "condition": 4, "grade": 8, "sqft_above": 1500,
    "sqft_basement": 500, "yr_built": 1990, "yr_renovated": 0,
    "zipcode": "98103", "lat": 47.5354, "long": -122.273,
    "sqft_living15": 1560, "sqft_lot15": 5765
  }'
```

---

## Endpoints

### GET /api/v1/health

Check if the API is healthy and dependencies are loaded.

**Response 200:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "demographics_loaded": true,
  "model_version": "v1",
  "data_vintage": "2014-2015",
  "timestamp": "2025-12-07T12:00:00.000000"
}
```

---

### POST /api/v1/predict

Predict home price with full feature set and zipcode-based demographics.

**Request Body (all 18 fields required):**

| Field | Type | Description |
|-------|------|-------------|
| bedrooms | int | Number of bedrooms |
| bathrooms | float | Number of bathrooms |
| sqft_living | int | Living space square footage |
| sqft_lot | int | Lot square footage |
| floors | float | Number of floors |
| waterfront | int | Waterfront (0/1) |
| view | int | View quality (0-4) |
| condition | int | Condition (1-5) |
| grade | int | Construction grade (1-13) |
| sqft_above | int | Above-ground sqft |
| sqft_basement | int | Basement sqft |
| yr_built | int | Year built |
| yr_renovated | int | Year renovated (0 if never) |
| zipcode | str | 5-digit King County zipcode |
| lat | float | Latitude |
| long | float | Longitude |
| sqft_living15 | int | Avg sqft of 15 nearest neighbors |
| sqft_lot15 | int | Avg lot sqft of 15 nearest neighbors |

**Response 200:**
```json
{
  "predicted_price": 485000.00,
  "prediction_id": "pred-20251207-123456-abc12345",
  "model_version": "v1",
  "confidence_note": "Prediction based on King County 2014-2015 data...",
  "data_vintage_warning": "This model was trained on 2014-2015 data...",
  "timestamp": "2025-12-07T12:00:00.000000"
}
```

---

### POST /api/v1/predict-minimal (BONUS)

Predict using only the 7 features the model requires. Uses average demographics.

**Request Body (7 fields required):**

| Field | Type | Description |
|-------|------|-------------|
| bedrooms | int | Number of bedrooms |
| bathrooms | float | Number of bathrooms |
| sqft_living | int | Living space square footage |
| sqft_lot | int | Lot square footage |
| floors | float | Number of floors |
| sqft_above | int | Above-ground sqft |
| sqft_basement | int | Basement sqft |

**Response 200:** Same format as /predict

---

## Interactive Documentation

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

---

## Error Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Bad Request (invalid zipcode, validation error) |
| 422 | Unprocessable Entity (schema validation failed) |
| 500 | Internal Server Error |
