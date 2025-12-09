"""
Prediction API endpoints for the Real Estate Price Predictor.

Endpoints:
    GET  /health              - Health check
    POST /predict             - Full prediction with demographics lookup (GOLD STANDARD)
    POST /predict-minimal     - Prediction using only 7 home features (BONUS)
    POST /predict-full        - Prediction using all 17 home features (V2.1.1 Experiment)
    POST /predict-adaptive    - Adaptive tier-based routing (V2.1.2 Experiment)

All endpoints include data vintage warnings and metadata in responses.

V2.1.2 Discovery:
    Empirical analysis showed that /predict-minimal is more accurate for homes
    under $400K, while /predict-full is better for homes over $400K. The
    /predict-adaptive endpoint implements this insight.
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
    PredictionRequestFullFeatures,
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
    tags=["Health"],
)
async def health_check(
    settings: Settings = Depends(get_settings),
    model_service: ModelService = Depends(get_model_service),
    feature_service: FeatureService = Depends(get_feature_service),
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
    Predict the price of a home in King County, WA.
    
    This endpoint accepts all columns from the test data (future_unseen_examples.csv)
    and enriches them with demographic data based on the provided zipcode.
    
    **Note:** The model was trained on 2014-2015 data. Predictions should be
    adjusted for current market conditions.
    """,
    tags=["Prediction"],
)
async def predict(
    request: PredictionRequest,
    settings: Settings = Depends(get_settings),
    model_service: ModelService = Depends(get_model_service),
    feature_service: FeatureService = Depends(get_feature_service),
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
    logger.info("Prediction request received. ID: %s, Zipcode: %s", prediction_id, request.zipcode)

    try:
        # Validate zipcode
        if not feature_service.is_valid_zipcode(request.zipcode):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "InvalidZipcode",
                    "message": f"Zipcode {request.zipcode} is not a valid King County zipcode.",
                    "valid_examples": list(feature_service.valid_zipcodes)[:5],
                },
            )

        # Extract only the features the model uses
        home_features = request.get_model_features()

        # Enrich with demographics
        enriched_features = feature_service.enrich_features(home_features, request.zipcode)

        # Make prediction
        predicted_price = model_service.predict_single(enriched_features)

        logger.info("Prediction completed. ID: %s, Price: $%.2f", prediction_id, predicted_price)

        return PredictionResponse(
            predicted_price=round(predicted_price, 2),
            prediction_id=prediction_id,
            model_version=model_service.model_version,
            confidence_note=f"Prediction based on King County {settings.data_vintage_start}-{settings.data_vintage_end} data. Actual market values may differ.",
            data_vintage_warning=f"This model was trained on {settings.data_vintage_start}-{settings.data_vintage_end} data. For current valuations, consider market adjustments.",
            timestamp=datetime.utcnow(),
        )

    except HTTPException:
        raise
    except ValueError as e:
        logger.error("Validation error in prediction: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "ValidationError", "message": str(e)},
        )
    except Exception as e:
        logger.error("Unexpected error in prediction: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "PredictionError",
                "message": "An unexpected error occurred during prediction.",
            },
        )


@router.post(
    "/predict-minimal",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Server error"},
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
    tags=["Prediction"],
)
async def predict_minimal(
    request: PredictionRequestMinimal,
    settings: Settings = Depends(get_settings),
    model_service: ModelService = Depends(get_model_service),
    feature_service: FeatureService = Depends(get_feature_service),
) -> PredictionResponse:
    """Predict home price using only required features (BONUS endpoint).

    Uses average King County demographics instead of zipcode-specific data.

    Args:
        request: PredictionRequestMinimal with 7 home features

    Returns:
        PredictionResponse with predicted price and metadata
    """
    prediction_id = generate_prediction_id()
    logger.info("Minimal prediction request received. ID: %s", prediction_id)

    try:
        # Get home features
        home_features = request.get_model_features()

        # Enrich with AVERAGE demographics (no zipcode)
        enriched_features = feature_service.enrich_features_with_average(home_features)

        # Make prediction
        predicted_price = model_service.predict_single(enriched_features)

        logger.info(
            "Minimal prediction completed. ID: %s, Price: $%.2f", prediction_id, predicted_price
        )

        return PredictionResponse(
            predicted_price=round(predicted_price, 2),
            prediction_id=prediction_id,
            model_version=model_service.model_version,
            confidence_note="Prediction uses King County average demographics. For more accurate predictions, use /predict with a specific zipcode.",
            data_vintage_warning=f"This model was trained on {settings.data_vintage_start}-{settings.data_vintage_end} data. For current valuations, consider market adjustments.",
            timestamp=datetime.utcnow(),
        )

    except ValueError as e:
        logger.error("Validation error in minimal prediction: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "ValidationError", "message": str(e)},
        )
    except Exception as e:
        logger.error("Unexpected error in minimal prediction: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "PredictionError",
                "message": "An unexpected error occurred during prediction.",
            },
        )


@router.post(
    "/predict-full",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Server error"},
    },
    summary="Predict Home Price (All 17 Features) - V2.1.1 EXPERIMENT",
    description="""
    **EXPERIMENTAL ENDPOINT V2.1.1**: Predict home price using all 17 home features WITHOUT zipcode.
    
    This endpoint accepts all 17 home features that the V2.1 model uses,
    but does NOT require a zipcode. Uses average King County demographics.
    
    **Purpose:** Test if providing actual values for all 17 features improves
    accuracy compared to /predict-minimal (which uses defaults for 10 features).
    
    **Hypothesis:** Actual feature values should give more accurate predictions
    than using default values.
    
    **Use cases:**
    - Properties outside King County where zipcode demographics don't apply
    - Quick estimates when you have property details but not exact location
    - Comparing accuracy vs /predict-minimal
    """,
    tags=["Prediction"],
)
async def predict_full_features(
    request: PredictionRequestFullFeatures,
    settings: Settings = Depends(get_settings),
    model_service: ModelService = Depends(get_model_service),
    feature_service: FeatureService = Depends(get_feature_service),
) -> PredictionResponse:
    """Predict home price using all 17 home features (V2.1.1 experimental endpoint).

    Uses average King County demographics (no zipcode required).
    All 17 home features are provided by the user (no defaults).

    Args:
        request: PredictionRequestFullFeatures with all 17 home features

    Returns:
        PredictionResponse with predicted price and metadata
    """
    prediction_id = generate_prediction_id()
    logger.info("Full-features prediction request received. ID: %s", prediction_id)

    try:
        # Get all 17 home features (no defaults needed)
        home_features = request.get_model_features()

        # Enrich with AVERAGE demographics only (no V21 defaults needed)
        demographics = feature_service.get_average_demographics()
        enriched_features = {**home_features, **demographics}

        # Make prediction
        predicted_price = model_service.predict_single(enriched_features)

        logger.info(
            "Full-features prediction completed. ID: %s, Price: $%.2f",
            prediction_id,
            predicted_price,
        )

        return PredictionResponse(
            predicted_price=round(predicted_price, 2),
            prediction_id=prediction_id,
            model_version=model_service.model_version,
            confidence_note="Prediction uses all 17 home features with King County average demographics. More accurate than /predict-minimal.",
            data_vintage_warning=f"This model was trained on {settings.data_vintage_start}-{settings.data_vintage_end} data. For current valuations, consider market adjustments.",
            timestamp=datetime.utcnow(),
        )

    except ValueError as e:
        logger.error("Validation error in full-features prediction: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "ValidationError", "message": str(e)},
        )
    except Exception as e:
        logger.error("Unexpected error in full-features prediction: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "PredictionError",
                "message": "An unexpected error occurred during prediction.",
            },
        )


# Price tier threshold for adaptive routing (discovered via empirical analysis)
ADAPTIVE_PRICE_THRESHOLD = 400_000


@router.post(
    "/predict-adaptive",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Server error"},
    },
    summary="Predict Home Price (Adaptive Tier-Based) - V2.1.2 EXPERIMENT",
    description=f"""
    **EXPERIMENTAL ENDPOINT V2.1.2**: Adaptive prediction using price-tier routing.
    
    **Discovery:** Analysis showed that:
    - `/predict-minimal` (7 features + defaults) is more accurate for homes < ${ADAPTIVE_PRICE_THRESHOLD:,}
    - `/predict-full` (17 features) is more accurate for homes >= ${ADAPTIVE_PRICE_THRESHOLD:,}
    
    **Strategy:** This endpoint:
    1. First estimates price using all 17 features
    2. Routes to appropriate strategy based on estimated price tier
    3. Returns the prediction with metadata about which strategy was used
    
    **Use case:** Best accuracy when zipcode is unknown but all property details available.
    """,
    tags=["Prediction"],
)
async def predict_adaptive(
    request: PredictionRequestFullFeatures,
    settings: Settings = Depends(get_settings),
    model_service: ModelService = Depends(get_model_service),
    feature_service: FeatureService = Depends(get_feature_service),
) -> PredictionResponse:
    """Predict home price using adaptive tier-based routing (V2.1.2 experiment).

    Uses initial estimate to determine price tier, then applies the
    most accurate strategy for that tier.

    Args:
        request: PredictionRequestFullFeatures with all 17 home features

    Returns:
        PredictionResponse with predicted price and strategy metadata
    """
    prediction_id = generate_prediction_id()
    logger.info("Adaptive prediction request received. ID: %s", prediction_id)

    try:
        # Get all 17 home features
        home_features = request.get_model_features()

        # Get average demographics
        demographics = feature_service.get_average_demographics()

        # STEP 1: Make initial estimate with all 17 features (full strategy)
        full_features = {**home_features, **demographics}
        initial_estimate = model_service.predict_single(full_features)

        # STEP 2: Determine price tier
        price_tier = "HIGH" if initial_estimate >= ADAPTIVE_PRICE_THRESHOLD else "LOW"

        # STEP 3: Apply tier-appropriate strategy
        if price_tier == "HIGH":
            # HIGH tier: use full features (already computed)
            predicted_price = initial_estimate
            strategy_used = "full_features"
            strategy_reason = f"Estimated price ${initial_estimate:,.0f} >= ${ADAPTIVE_PRICE_THRESHOLD:,} threshold"
        else:
            # LOW tier: use minimal features + V21 defaults
            minimal_features = {
                "bedrooms": home_features["bedrooms"],
                "bathrooms": home_features["bathrooms"],
                "sqft_living": home_features["sqft_living"],
                "sqft_lot": home_features["sqft_lot"],
                "floors": home_features["floors"],
                "sqft_above": home_features["sqft_above"],
                "sqft_basement": home_features["sqft_basement"],
            }
            enriched = feature_service.enrich_features_with_average(minimal_features)
            predicted_price = model_service.predict_single(enriched)
            strategy_used = "minimal_with_defaults"
            strategy_reason = f"Estimated price ${initial_estimate:,.0f} < ${ADAPTIVE_PRICE_THRESHOLD:,} threshold"

        logger.info(
            "Adaptive prediction completed. ID: %s, Tier: %s, Strategy: %s, Price: $%.2f",
            prediction_id,
            price_tier,
            strategy_used,
            predicted_price,
        )

        return PredictionResponse(
            predicted_price=round(predicted_price, 2),
            prediction_id=prediction_id,
            model_version=f"{model_service.model_version}-adaptive",
            confidence_note=f"Adaptive routing: {price_tier} tier → {strategy_used}. {strategy_reason}.",
            data_vintage_warning=f"This model was trained on {settings.data_vintage_start}-{settings.data_vintage_end} data. For current valuations, consider market adjustments.",
            timestamp=datetime.utcnow(),
        )

    except ValueError as e:
        logger.error("Validation error in adaptive prediction: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "ValidationError", "message": str(e)},
        )
    except Exception as e:
        logger.error("Unexpected error in adaptive prediction: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "PredictionError",
                "message": "An unexpected error occurred during prediction.",
            },
        )
