"""
Prediction API endpoints for the Real Estate Price Predictor.

Endpoints:
    GET  /health          - Health check
    POST /predict         - Full prediction with demographics lookup
    POST /predict-minimal - Prediction using only required home features (BONUS)

All endpoints include data vintage warnings and metadata in responses.
"""

import logging
import uuid
from datetime import datetime
from typing import Union

from fastapi import APIRouter, Depends, HTTPException, status

from src.config import Settings, get_settings
from src.models import (
    ErrorResponse,
    HealthResponse,
    PredictionRequest,
    PredictionRequestMinimal,
    PredictionResponse,
)
from src.services.feature_service import FeatureService, get_feature_service
from src.services.model_service import ModelService, get_model_service

logger = logging.getLogger(__name__)

router = APIRouter()


def generate_prediction_id() -> str:
    """Generate a unique prediction ID.
    
    Returns:
        String in format: pred-YYYYMMDD-HHMMSS-uuid8
    """
    now = datetime.utcnow()
    date_str = now.strftime("%Y%m%d-%H%M%S")
    short_uuid = uuid.uuid4().hex[:8]
    return f"pred-{date_str}-{short_uuid}"


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check if the API is healthy and all dependencies are loaded.",
    tags=["Health"]
)
async def health_check(
    settings: Settings = Depends(get_settings),
    model_service: ModelService = Depends(get_model_service),
    feature_service: FeatureService = Depends(get_feature_service)
) -> HealthResponse:
    """Health check endpoint.
    
    Returns the status of the API, model, and demographics data.
    """
    model_status = model_service.get_status()
    feature_status = feature_service.get_status()
    
    is_healthy = model_status["is_loaded"] and feature_status["is_loaded"]
    
    return HealthResponse(
        status="healthy" if is_healthy else "unhealthy",
        model_loaded=model_status["is_loaded"],
        demographics_loaded=feature_status["is_loaded"],
        model_version=model_status["model_version"],
        data_vintage=f"{settings.data_vintage_start}-{settings.data_vintage_end}",
        timestamp=datetime.utcnow()
    )


@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Server error"}
    },
    summary="Predict Home Price",
    description="""
    Predict the price of a home in King County, WA.
    
    This endpoint accepts all columns from the test data (future_unseen_examples.csv)
    and enriches them with demographic data based on the provided zipcode.
    
    **Note:** The model was trained on 2014-2015 data. Predictions should be
    adjusted for current market conditions.
    """,
    tags=["Prediction"]
)
async def predict(
    request: PredictionRequest,
    settings: Settings = Depends(get_settings),
    model_service: ModelService = Depends(get_model_service),
    feature_service: FeatureService = Depends(get_feature_service)
) -> PredictionResponse:
    """Predict home price with full feature set and demographics.
    
    Args:
        request: PredictionRequest with all 18 columns from test data
        
    Returns:
        PredictionResponse with predicted price and metadata
        
    Raises:
        HTTPException: If zipcode is invalid or prediction fails
    """
    prediction_id = generate_prediction_id()
    logger.info(
        "Prediction request received. ID: %s, Zipcode: %s",
        prediction_id,
        request.zipcode
    )
    
    try:
        # Validate zipcode
        if not feature_service.is_valid_zipcode(request.zipcode):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "InvalidZipcode",
                    "message": f"Zipcode {request.zipcode} is not a valid King County zipcode.",
                    "valid_examples": list(feature_service.valid_zipcodes)[:5]
                }
            )
        
        # Extract only the features the model uses
        home_features = request.get_model_features()
        
        # Enrich with demographics
        enriched_features = feature_service.enrich_features(
            home_features,
            request.zipcode
        )
        
        # Make prediction
        predicted_price = model_service.predict_single(enriched_features)
        
        logger.info(
            "Prediction completed. ID: %s, Price: $%.2f",
            prediction_id,
            predicted_price
        )
        
        return PredictionResponse(
            predicted_price=round(predicted_price, 2),
            prediction_id=prediction_id,
            model_version=model_service.model_version,
            confidence_note=f"Prediction based on King County {settings.data_vintage_start}-{settings.data_vintage_end} data. Actual market values may differ.",
            data_vintage_warning=f"This model was trained on {settings.data_vintage_start}-{settings.data_vintage_end} data. For current valuations, consider market adjustments.",
            timestamp=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.error("Validation error in prediction: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "ValidationError", "message": str(e)}
        )
    except Exception as e:
        logger.error("Unexpected error in prediction: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "PredictionError", "message": "An unexpected error occurred during prediction."}
        )


@router.post(
    "/predict-minimal",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Server error"}
    },
    summary="Predict Home Price (Minimal Features) - BONUS",
    description="""
    **BONUS ENDPOINT**: Predict home price using only the required model features.
    
    This endpoint accepts only the 7 home features that the model actually uses,
    without requiring a zipcode. Demographics are filled with King County averages.
    
    **Use cases:**
    - Quick estimates when zipcode is unknown
    - Properties outside King County (less accurate)
    - Simpler integrations
    
    **Note:** Predictions using average demographics are less accurate than
    zipcode-specific predictions.
    """,
    tags=["Prediction"]
)
async def predict_minimal(
    request: PredictionRequestMinimal,
    settings: Settings = Depends(get_settings),
    model_service: ModelService = Depends(get_model_service),
    feature_service: FeatureService = Depends(get_feature_service)
) -> PredictionResponse:
    """Predict home price using only required features (BONUS endpoint).
    
    Uses average King County demographics instead of zipcode-specific data.
    
    Args:
        request: PredictionRequestMinimal with 7 home features
        
    Returns:
        PredictionResponse with predicted price and metadata
    """
    prediction_id = generate_prediction_id()
    logger.info(
        "Minimal prediction request received. ID: %s",
        prediction_id
    )
    
    try:
        # Get home features
        home_features = request.get_model_features()
        
        # Enrich with AVERAGE demographics (no zipcode)
        enriched_features = feature_service.enrich_features_with_average(home_features)
        
        # Make prediction
        predicted_price = model_service.predict_single(enriched_features)
        
        logger.info(
            "Minimal prediction completed. ID: %s, Price: $%.2f",
            prediction_id,
            predicted_price
        )
        
        return PredictionResponse(
            predicted_price=round(predicted_price, 2),
            prediction_id=prediction_id,
            model_version=model_service.model_version,
            confidence_note="Prediction uses King County average demographics. For more accurate predictions, use /predict with a specific zipcode.",
            data_vintage_warning=f"This model was trained on {settings.data_vintage_start}-{settings.data_vintage_end} data. For current valuations, consider market adjustments.",
            timestamp=datetime.utcnow()
        )
        
    except ValueError as e:
        logger.error("Validation error in minimal prediction: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "ValidationError", "message": str(e)}
        )
    except Exception as e:
        logger.error("Unexpected error in minimal prediction: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "PredictionError", "message": "An unexpected error occurred during prediction."}
        )
