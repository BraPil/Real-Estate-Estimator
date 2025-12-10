# Complete Code Walkthrough

**Purpose:** Interview preparation - understand every piece of code

---

## Project Structure

```
src/
├── config.py          # Settings with env var support
├── models.py          # Pydantic request/response schemas
├── main.py            # FastAPI application entry point
├── train.py           # Model training with MLflow
├── evaluate.py        # Model evaluation
├── api/
│   └── prediction.py  # API endpoints
└── services/
    ├── model_service.py   # Model loading
    └── feature_service.py # Demographics lookup
```

---

## Key Files Explained

### src/config.py - Configuration

**Purpose:** Centralized settings using Pydantic.

```python
class Settings(BaseSettings):
    model_path: str = "model/model.pkl"
    demographics_path: str = "data/zipcode_demographics.csv"
    
    class Config:
        env_file = ".env"
```

**Key Points:**
- Reads from environment variables automatically
- Supports .env file
- `@lru_cache` ensures settings loaded once

---

### src/models.py - Pydantic Schemas

**Purpose:** Request validation and response serialization.

```python
class PredictionRequest(BaseModel):
    bedrooms: int = Field(..., ge=0, le=33)
    # ... 18 total fields
    
    def get_model_features(self) -> dict:
        """Extract only the 7 features model uses."""
```

**Key Points:**
- Accepts ALL 18 columns from test data
- `get_model_features()` extracts only what model needs
- Automatic validation and OpenAPI schema generation

---

### src/services/model_service.py - Model Loading

**Purpose:** Load model, make predictions.

```python
class ModelService:
    def __init__(self, settings):
        self._load_model()  # Load on init
    
    def predict_single(self, features_dict) -> float:
        df = pd.DataFrame([features_dict])
        return self.model.predict(df[self.feature_names])[0]
```

**Key Points:**
- Singleton pattern (one instance shared)
- Supports local pickle or MLflow registry
- Feature ordering matches training

---

### src/services/feature_service.py - Demographics

**Purpose:** Lookup demographics by zipcode.

```python
class FeatureService:
    def __init__(self, settings):
        self.demographics_df = pd.read_csv(...)
        self.demographics_df.set_index("zipcode", inplace=True)
    
    def enrich_features(self, home_features, zipcode):
        demographics = self.demographics_df.loc[zipcode].to_dict()
        return {**home_features, **demographics}
```

**Key Points:**
- Loaded once at startup (O(1) lookup)
- Merges 7 home features + 27 demographics = 34 total

---

### src/api/prediction.py - Endpoints

**Purpose:** HTTP request handling.

```python
@router.post("/predict")
async def predict(
    request: PredictionRequest,
    model_service: ModelService = Depends(get_model_service),
    feature_service: FeatureService = Depends(get_feature_service)
):
    home_features = request.get_model_features()
    enriched = feature_service.enrich_features(home_features, request.zipcode)
    price = model_service.predict_single(enriched)
    return PredictionResponse(predicted_price=price, ...)
```

**Key Points:**
- Dependency injection via `Depends()`
- Services are singletons
- Automatic validation from Pydantic

---

### src/main.py - Application Entry

**Purpose:** Configure and start FastAPI.

```python
@asynccontextmanager
async def lifespan(app):
    # Startup: load model and demographics
    get_model_service()
    get_feature_service()
    yield
    # Shutdown: cleanup

app = FastAPI(lifespan=lifespan)
app.include_router(prediction_router, prefix="/api/v1")
```

**Key Points:**
- Lifespan context manager for startup/shutdown
- CORS middleware configured
- Router mounted at /api/v1

---

## Interview Questions

### Q: How does a prediction request work?

1. FastAPI validates JSON against PredictionRequest
2. Extract 7 home features with `get_model_features()`
3. Lookup demographics by zipcode (O(1))
4. Merge: 7 home + 26 demographics = 33 features
5. Order features to match training
6. sklearn Pipeline predicts
7. Return response with metadata

### Q: How did you fix the bug?

Original `create_model.py` had `DEMOGRAPHICS_PATH = 'data/kc_house_data.csv'` which was wrong. I fixed it to point to the actual demographics file.

### Q: How would you scale this?

Stateless API + Docker = horizontal scaling. Each replica loads model/demographics independently. Add load balancer in front.

---

**Last Updated:** 2025-12-07
