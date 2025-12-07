"""
FastAPI application entry point for the Real Estate Price Predictor API.

This module creates and configures the FastAPI application, including:
- CORS middleware
- Logging configuration
- Router mounting
- Startup/shutdown event handlers
- API metadata for documentation

Usage:
    Development:
        uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
    
    Production:
        uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 4

API Documentation:
    - Swagger UI: http://localhost:8000/docs
    - ReDoc: http://localhost:8000/redoc
    - OpenAPI JSON: http://localhost:8000/openapi.json
"""

import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.prediction import router as prediction_router
from src.config import get_settings
from src.services.feature_service import get_feature_service, reset_feature_service
from src.services.model_service import get_model_service, reset_model_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifespan manager for startup and shutdown events.
    
    On startup:
        - Load settings
        - Initialize model service (loads model)
        - Initialize feature service (loads demographics)
    
    On shutdown:
        - Clean up resources
    """
    # Startup
    logger.info("Starting Real Estate Price Predictor API...")
    settings = get_settings()
    
    try:
        # Initialize services (this loads model and demographics into memory)
        logger.info("Loading model...")
        model_service = get_model_service()
        logger.info("Model loaded: version=%s, features=%d", 
                   model_service.model_version,
                   len(model_service.feature_names))
        
        logger.info("Loading demographics data...")
        feature_service = get_feature_service()
        logger.info("Demographics loaded: zipcodes=%d, features=%d",
                   len(feature_service.valid_zipcodes),
                   len(feature_service.demographic_columns))
        
        logger.info("API startup complete. Ready to serve predictions.")
        
    except FileNotFoundError as e:
        logger.error("Failed to load required files: %s", str(e))
        logger.error("Ensure model is trained (python src/train.py) and data files are in place.")
        raise
    except Exception as e:
        logger.error("Unexpected error during startup: %s", str(e), exc_info=True)
        raise
    
    yield  # Application runs here
    
    # Shutdown
    logger.info("Shutting down Real Estate Price Predictor API...")
    reset_model_service()
    reset_feature_service()
    logger.info("Cleanup complete.")


# Get settings
settings = get_settings()

# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    description="""
## Real Estate Price Predictor API

Predict home prices in King County (Seattle), WA using machine learning.

### Overview

This API serves predictions from a KNeighborsRegressor model trained on 
King County housing data from 2014-2015, enriched with zipcode-level 
demographic information.

### Endpoints

- **GET /health** - Check API health and service status
- **POST /predict** - Full prediction with zipcode demographics
- **POST /predict-minimal** - Prediction using only required features (BONUS)

### Data Vintage Warning

The model was trained on 2014-2015 data. Predictions should be adjusted 
for current market conditions. This is a demonstration model for the 
phData Machine Learning Engineer interview project.

### Technical Details

- **Model:** KNeighborsRegressor (k=5) with RobustScaler preprocessing
- **Features:** 7 home features + 26 demographic features (33 total)
- **Performance:** R2 = 0.728, MAE = $102k, RMSE = $202k on test set
- **Training Data:** 21,613 King County home sales

### Source Code

[GitHub Repository](https://github.com/BraPil/Real-Estate-Estimator)
    """,
    version=settings.app_version,
    openapi_tags=[
        {
            "name": "Health",
            "description": "API health and status endpoints"
        },
        {
            "name": "Prediction",
            "description": "Home price prediction endpoints"
        }
    ],
    lifespan=lifespan
)

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(prediction_router, prefix="/api/v1")


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint - redirect to docs."""
    return {
        "message": "Real Estate Price Predictor API",
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/api/v1/health"
    }


# For running directly with Python (not recommended for production)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
