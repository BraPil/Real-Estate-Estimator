"""
V1 MVP Prediction Endpoints

V1 has only TWO endpoints (meeting minimum requirements):
- GET  /health   - Health check
- POST /predict  - Prediction with demographics enrichment

NO /api/v1/ prefix - this is the bare MVP.
NO /predict-minimal, /predict-full, /predict-adaptive - those came in V2.
"""

import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status

from src.config import Settings, get_settings
from src.models import (
    ErrorResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.services.feature_service import FeatureService, get_feature_service
from src.services.model_service import ModelService, get_model_service

logger = logging.getLogger(__name__)

router = APIRouter()


def generate_prediction_id() -> str:
    """Generate a unique prediction ID."""
    now = datetime.utcnow()
    date_str = now.strftime("%Y%m%d-%H%M%S")
    short_uuid = uuid.uuid4().hex[:8]
    return f"pred-{date_str}-{short_uuid}"


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="V1 MVP health check.",
    tags=["Health"],
)
async def health_check(
    settings: Settings = Depends(get_settings),
    model_service: ModelService = Depends(get_model_service),
    feature_service: FeatureService = Depends(get_feature_service),
) -> HealthResponse:
    """V1 Health check - minimal response."""
    model_status = model_service.get_status()
    feature_status = feature_service.get_status()
    
    is_healthy = model_status["is_loaded"] and feature_status["is_loaded"]
    
    return HealthResponse(
        status="healthy" if is_healthy else "unhealthy",
        model_loaded=model_status["is_loaded"],
        demographics_loaded=feature_status["is_loaded"],
        model_version=model_status["model_version"],
        data_vintage=f"{settings.data_vintage_start}-{settings.data_vintage_end}",
        timestamp=datetime.utcnow(),
    )


@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Server error"},
    },
    summary="Predict Home Price",
    description="""
    V1 MVP - Predict home price using 7 features + demographics.
    
    Accepts all 18 columns from future_unseen_examples.csv but only uses:
    bedrooms, bathrooms, sqft_living, sqft_lot, floors, sqft_above, sqft_basement
    
    Demographics are enriched on the backend based on zipcode.
    """,
    tags=["Prediction"],
)
async def predict(
    request: PredictionRequest,
    settings: Settings = Depends(get_settings),
    model_service: ModelService = Depends(get_model_service),
    feature_service: FeatureService = Depends(get_feature_service),
) -> PredictionResponse:
    """V1 Prediction - uses only 7 home features + demographics."""
    prediction_id = generate_prediction_id()
    logger.info("V1 Prediction request. ID: %s, Zipcode: %s", prediction_id, request.zipcode)
    
    try:
        # Validate zipcode
        if not feature_service.is_valid_zipcode(request.zipcode):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "InvalidZipcode",
                    "message": f"Zipcode {request.zipcode} is not valid.",
                },
            )
        
        # Get only the 7 features V1 uses
        home_features = request.get_model_features()
        
        # Enrich with demographics
        enriched_features = feature_service.enrich_features(home_features, request.zipcode)
        
        # Make prediction
        predicted_price = model_service.predict_single(enriched_features)
        
        logger.info("V1 Prediction complete. ID: %s, Price: $%.2f", prediction_id, predicted_price)
        
        return PredictionResponse(
            predicted_price=round(predicted_price, 2),
            prediction_id=prediction_id,
            model_version=model_service.model_version,
            confidence_note=f"V1 MVP: Uses 7 home features + {len(feature_service.demographic_columns)} demographic features.",
            data_vintage_warning=f"Model trained on {settings.data_vintage_start}-{settings.data_vintage_end} data.",
            timestamp=datetime.utcnow(),
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("V1 Prediction failed: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "PredictionFailed", "message": str(e)},
        )
