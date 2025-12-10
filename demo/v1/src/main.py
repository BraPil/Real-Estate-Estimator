"""
V1 MVP - FastAPI Application Entry Point

This is the BARE MINIMUM API that meets the requirements:
- Single /predict endpoint (POST)
- Single /health endpoint (GET)
- NO versioned path prefix (/api/v1/) - that's a V2+ feature
- NO experimental endpoints - those came in V2
"""

import logging
import sys
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI

from src.api.prediction import router as prediction_router
from src.config import get_settings
from src.services.feature_service import get_feature_service, reset_feature_service
from src.services.model_service import get_model_service, reset_model_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application startup and shutdown."""
    logger.info("Starting V1 MVP API...")
    
    try:
        # Load model
        model_service = get_model_service()
        logger.info("Model loaded: %s features", len(model_service.feature_names))
        
        # Load demographics
        feature_service = get_feature_service()
        logger.info("Demographics loaded: %d zipcodes", len(feature_service.valid_zipcodes))
        
        logger.info("V1 MVP API ready.")
    except Exception as e:
        logger.error("Startup failed: %s", str(e))
        raise
    
    yield
    
    # Cleanup
    logger.info("Shutting down V1 MVP API...")
    reset_model_service()
    reset_feature_service()


settings = get_settings()

# V1 MVP: Simple app, no versioned prefix
app = FastAPI(
    title="Real Estate Price Predictor",
    description="""
    **V1 MVP** - Minimum Viable Product
    
    This is the baseline API that meets the minimum requirements:
    - Accepts 18 input columns from future_unseen_examples.csv
    - Uses 7 home features + 26 demographic features
    - Returns prediction with metadata
    - Demographics enriched on backend
    
    **Endpoints:**
    - `GET /health` - Health check
    - `POST /predict` - Price prediction
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# V1: Mount directly at root - no /api/v1 prefix
app.include_router(prediction_router)
