"""
Pydantic models for request/response validation in the Real Estate Price Predictor API.

This module defines the data structures for:
- Prediction requests (full and minimal)
- Prediction responses
- Health check responses
- Error responses

All models include validation, examples, and documentation for OpenAPI schema generation.

Note on Request Schema:
    The PredictionRequest accepts ALL 18 columns from future_unseen_examples.csv.

    Columns accepted: bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront,
    view, condition, grade, sqft_above, sqft_basement, yr_built, yr_renovated,
    zipcode, lat, long, sqft_living15, sqft_lot15

    V2.1 Model uses ALL 17 home features (all except zipcode which is used for
    demographics lookup) plus 26 demographic features = 43 total features.

Version History:
    V1: 7 home features + 26 demographics = 33 features
    V2.1: 17 home features + 26 demographics = 43 features (current)
"""

from datetime import datetime

from pydantic import BaseModel, Field, field_validator

# ==============================================================================
# REQUEST MODELS
# ==============================================================================


class PredictionRequest(BaseModel):
    """Full prediction request matching future_unseen_examples.csv schema.

    This request accepts ALL 18 columns from the test data file.
    The service extracts only the features needed by the model and
    enriches with demographic data based on zipcode.

    Required by model: bedrooms, bathrooms, sqft_living, sqft_lot, floors,
                       sqft_above, sqft_basement, zipcode (for demographics)
    Accepted but unused: waterfront, view, condition, grade, yr_built,
                         yr_renovated, lat, long, sqft_living15, sqft_lot15
    """

    # Features USED by the model
    bedrooms: int = Field(..., ge=0, le=33, description="Number of bedrooms", examples=[3])
    bathrooms: float = Field(
        ...,
        ge=0,
        le=10,
        description="Number of bathrooms (can be fractional, e.g., 2.5)",
        examples=[2.5],
    )
    sqft_living: int = Field(
        ..., ge=100, le=15000, description="Square footage of living space", examples=[2000]
    )
    sqft_lot: int = Field(
        ..., ge=500, le=2000000, description="Square footage of the lot", examples=[5000]
    )
    floors: float = Field(
        ...,
        ge=1,
        le=4,
        description="Number of floors (can be fractional, e.g., 1.5 for split-level)",
        examples=[2.0],
    )
    sqft_above: int = Field(
        ..., ge=0, le=15000, description="Square footage above ground level", examples=[1500]
    )
    sqft_basement: int = Field(
        ...,
        ge=0,
        le=5000,
        description="Square footage of basement (0 if no basement)",
        examples=[500],
    )

    # Zipcode for demographics lookup (USED by model via demographics join)
    zipcode: str = Field(
        ...,
        min_length=5,
        max_length=5,
        pattern=r"^\d{5}$",
        description="5-digit zipcode for King County, WA (used for demographics lookup)",
        examples=["98103"],
    )

    # Features ACCEPTED but NOT USED by current model (for API compatibility)
    waterfront: int = Field(
        ...,
        ge=0,
        le=1,
        description="Waterfront property (0=No, 1=Yes) - not used by current model",
        examples=[0],
    )
    view: int = Field(
        ...,
        ge=0,
        le=4,
        description="Quality of view (0-4 scale) - not used by current model",
        examples=[0],
    )
    condition: int = Field(
        ...,
        ge=1,
        le=5,
        description="Condition of the house (1-5 scale) - not used by current model",
        examples=[4],
    )
    grade: int = Field(
        ...,
        ge=1,
        le=13,
        description="Construction grade (1-13 scale) - not used by current model",
        examples=[8],
    )
    yr_built: int = Field(
        ...,
        ge=1800,
        le=2030,
        description="Year the house was built - not used by current model",
        examples=[1990],
    )
    yr_renovated: int = Field(
        ...,
        ge=0,
        le=2030,
        description="Year of last renovation (0 if never) - not used by current model",
        examples=[0],
    )
    lat: float = Field(
        ...,
        ge=47.0,
        le=48.0,
        description="Latitude coordinate - not used by current model",
        examples=[47.5354],
    )
    long: float = Field(
        ...,
        ge=-123.0,
        le=-121.0,
        description="Longitude coordinate - not used by current model",
        examples=[-122.273],
    )
    sqft_living15: int = Field(
        ...,
        ge=0,
        le=15000,
        description="Average sqft of 15 nearest neighbors - not used by current model",
        examples=[1560],
    )
    sqft_lot15: int = Field(
        ...,
        ge=0,
        le=2000000,
        description="Average lot sqft of 15 nearest neighbors - not used by current model",
        examples=[5765],
    )

    @field_validator("zipcode")
    @classmethod
    def validate_zipcode(cls, v: str) -> str:
        """Ensure zipcode is a valid 5-digit string."""
        if not v.isdigit():
            raise ValueError("Zipcode must contain only digits")
        return v

    def get_model_features(self) -> dict:
        """Extract the home features used by the model.

        V2.1: Returns all 17 home features (expanded from 7 in V1).
        These are combined with 26 demographic features for prediction.

        Returns:
            Dictionary with all 17 home features the model needs.
        """
        return {
            # V1 structural features (7)
            "bedrooms": self.bedrooms,
            "bathrooms": self.bathrooms,
            "sqft_living": self.sqft_living,
            "sqft_lot": self.sqft_lot,
            "floors": self.floors,
            "sqft_above": self.sqft_above,
            "sqft_basement": self.sqft_basement,
            # V2.1 property characteristics (4)
            "waterfront": self.waterfront,
            "view": self.view,
            "condition": self.condition,
            "grade": self.grade,
            # V2.1 age features (2)
            "yr_built": self.yr_built,
            "yr_renovated": self.yr_renovated,
            # V2.1 spatial features (2)
            "lat": self.lat,
            "long": self.long,
            # V2.1 neighborhood context (2)
            "sqft_living15": self.sqft_living15,
            "sqft_lot15": self.sqft_lot15,
        }

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "bedrooms": 4,
                    "bathrooms": 1.0,
                    "sqft_living": 1680,
                    "sqft_lot": 5043,
                    "floors": 1.5,
                    "waterfront": 0,
                    "view": 0,
                    "condition": 4,
                    "grade": 6,
                    "sqft_above": 1680,
                    "sqft_basement": 0,
                    "yr_built": 1911,
                    "yr_renovated": 0,
                    "zipcode": "98118",
                    "lat": 47.5354,
                    "long": -122.273,
                    "sqft_living15": 1560,
                    "sqft_lot15": 5765,
                }
            ]
        }
    }


class PredictionRequestMinimal(BaseModel):
    """Minimal prediction request with only the features used by the model.

    BONUS ENDPOINT: This request type uses only the 7 home features
    that the model actually uses, without zipcode-based demographic
    enrichment. Predictions use average demographics and may be less
    accurate but do not require a valid King County zipcode.

    This is useful for quick estimates when zipcode is unknown or
    for properties outside King County.
    """

    bedrooms: int = Field(..., ge=0, le=33, description="Number of bedrooms", examples=[3])
    bathrooms: float = Field(
        ...,
        ge=0,
        le=10,
        description="Number of bathrooms (can be fractional, e.g., 2.5)",
        examples=[2.5],
    )
    sqft_living: int = Field(
        ..., ge=100, le=15000, description="Square footage of living space", examples=[2000]
    )
    sqft_lot: int = Field(
        ..., ge=500, le=2000000, description="Square footage of the lot", examples=[5000]
    )
    floors: float = Field(
        ...,
        ge=1,
        le=4,
        description="Number of floors (can be fractional, e.g., 1.5 for split-level)",
        examples=[2.0],
    )
    sqft_above: int = Field(
        ..., ge=0, le=15000, description="Square footage above ground level", examples=[1500]
    )
    sqft_basement: int = Field(
        ...,
        ge=0,
        le=5000,
        description="Square footage of basement (0 if no basement)",
        examples=[500],
    )

    def get_model_features(self) -> dict:
        """Extract the features as a dictionary.

        Returns:
            Dictionary with the 7 home features.
        """
        return {
            "bedrooms": self.bedrooms,
            "bathrooms": self.bathrooms,
            "sqft_living": self.sqft_living,
            "sqft_lot": self.sqft_lot,
            "floors": self.floors,
            "sqft_above": self.sqft_above,
            "sqft_basement": self.sqft_basement,
        }

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "bedrooms": 3,
                    "bathrooms": 2.5,
                    "sqft_living": 2000,
                    "sqft_lot": 5000,
                    "floors": 2.0,
                    "sqft_above": 1500,
                    "sqft_basement": 500,
                }
            ]
        }
    }


class PredictionRequestFullFeatures(BaseModel):
    """Full home features request WITHOUT zipcode (V2.1.1 Experiment).

    EXPERIMENTAL ENDPOINT: This request requires ALL 17 home features
    but does NOT require zipcode. Uses average demographics.

    This is to test if providing actual values for all 17 features
    (vs. 7 features + defaults) improves accuracy when zipcode is unknown.

    Hypothesis: Actual values should be more accurate than defaults.
    """

    # V1 structural features (7)
    bedrooms: int = Field(..., ge=0, le=33, description="Number of bedrooms", examples=[3])
    bathrooms: float = Field(..., ge=0, le=10, description="Number of bathrooms", examples=[2.5])
    sqft_living: int = Field(
        ..., ge=100, le=15000, description="Square footage of living space", examples=[2000]
    )
    sqft_lot: int = Field(
        ..., ge=500, le=2000000, description="Square footage of the lot", examples=[5000]
    )
    floors: float = Field(..., ge=1, le=4, description="Number of floors", examples=[2.0])
    sqft_above: int = Field(
        ..., ge=0, le=15000, description="Square footage above ground", examples=[1500]
    )
    sqft_basement: int = Field(
        ..., ge=0, le=5000, description="Basement square footage", examples=[500]
    )

    # V2.1 property characteristics (4)
    waterfront: int = Field(
        ..., ge=0, le=1, description="Waterfront property (0=No, 1=Yes)", examples=[0]
    )
    view: int = Field(..., ge=0, le=4, description="Quality of view (0-4 scale)", examples=[0])
    condition: int = Field(
        ..., ge=1, le=5, description="Condition of the house (1-5 scale)", examples=[3]
    )
    grade: int = Field(
        ..., ge=1, le=13, description="Construction grade (1-13 scale)", examples=[7]
    )

    # V2.1 age features (2)
    yr_built: int = Field(..., ge=1800, le=2030, description="Year built", examples=[1990])
    yr_renovated: int = Field(
        ..., ge=0, le=2030, description="Year renovated (0 if never)", examples=[0]
    )

    # V2.1 spatial features (2)
    lat: float = Field(..., ge=47.0, le=48.0, description="Latitude coordinate", examples=[47.5354])
    long: float = Field(
        ..., ge=-123.0, le=-121.0, description="Longitude coordinate", examples=[-122.273]
    )

    # V2.1 neighborhood context (2)
    sqft_living15: int = Field(
        ..., ge=0, le=15000, description="Avg sqft of 15 nearest neighbors", examples=[1560]
    )
    sqft_lot15: int = Field(
        ..., ge=0, le=2000000, description="Avg lot sqft of 15 nearest neighbors", examples=[5765]
    )

    def get_model_features(self) -> dict:
        """Extract all 17 home features as a dictionary."""
        return {
            "bedrooms": self.bedrooms,
            "bathrooms": self.bathrooms,
            "sqft_living": self.sqft_living,
            "sqft_lot": self.sqft_lot,
            "floors": self.floors,
            "sqft_above": self.sqft_above,
            "sqft_basement": self.sqft_basement,
            "waterfront": self.waterfront,
            "view": self.view,
            "condition": self.condition,
            "grade": self.grade,
            "yr_built": self.yr_built,
            "yr_renovated": self.yr_renovated,
            "lat": self.lat,
            "long": self.long,
            "sqft_living15": self.sqft_living15,
            "sqft_lot15": self.sqft_lot15,
        }

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "bedrooms": 3,
                    "bathrooms": 2.5,
                    "sqft_living": 2000,
                    "sqft_lot": 5000,
                    "floors": 2.0,
                    "sqft_above": 1500,
                    "sqft_basement": 500,
                    "waterfront": 0,
                    "view": 0,
                    "condition": 3,
                    "grade": 7,
                    "yr_built": 1990,
                    "yr_renovated": 0,
                    "lat": 47.5354,
                    "long": -122.273,
                    "sqft_living15": 1560,
                    "sqft_lot15": 5765,
                }
            ]
        }
    }


class BatchPredictionRequest(BaseModel):
    """Batch prediction request for multiple homes."""

    homes: list[PredictionRequest] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of homes to predict prices for (max 100)",
    )


# ==============================================================================
# RESPONSE MODELS
# ==============================================================================


class AddressPredictionRequest(BaseModel):
    """Prediction request using just an address (V4 feature).

    This request type allows users to get a price prediction by providing
    only an address. The system will:
    1. Geocode the address to get lat/long
    2. Find the nearest property in King County records
    3. Auto-populate all property features
    4. Make a prediction using the standard model

    This is the most user-friendly way to get a prediction.
    """

    address: str = Field(
        ...,
        min_length=10,
        max_length=200,
        description="Full street address including city, state, and zip",
        examples=["123 Main St, Seattle, WA 98103"],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"address": "400 Broad St, Seattle, WA 98109"},
                {"address": "123 Main St, Bellevue, WA 98004"},
            ]
        }
    }


class AddressPredictionResponse(BaseModel):
    """Response for address-based prediction (V4 feature).

    Includes the standard prediction plus the auto-populated property details.
    """

    predicted_price: float = Field(
        ..., ge=0, description="Predicted home price in USD", examples=[850000.0]
    )
    prediction_id: str = Field(
        ...,
        description="Unique identifier for this prediction",
        examples=["pred-20251210-123456-abc123"],
    )
    model_version: str = Field(
        ..., description="Version of the model used for prediction", examples=["v3.3"]
    )
    
    # Property details that were auto-populated
    property_details: dict = Field(
        ...,
        description="Property details retrieved from King County records",
        examples=[{
            "bedrooms": 3,
            "bathrooms": 2.5,
            "sqft_living": 2000,
            "sqft_lot": 5000,
            "yr_built": 2010,
            "grade": 8,
        }],
    )
    
    # Address lookup metadata
    geocoded_address: str = Field(
        ...,
        description="Full address as resolved by geocoder",
        examples=["400 Broad St, Seattle, King County, Washington, 98109, United States"],
    )
    match_confidence: str = Field(
        ...,
        description="Confidence in property match (high/medium/low)",
        examples=["high"],
    )
    distance_meters: float = Field(
        ...,
        description="Distance from geocoded location to matched property",
        examples=[25.5],
    )
    
    confidence_note: str = Field(
        default="Prediction based on King County records. Property details were auto-populated from assessor data.",
        description="Confidence note about the prediction",
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Timestamp of the prediction"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "predicted_price": 850000.0,
                    "prediction_id": "pred-20251210-123456-abc123",
                    "model_version": "v3.3",
                    "property_details": {
                        "bedrooms": 3,
                        "bathrooms": 2.5,
                        "sqft_living": 2000,
                        "grade": 8,
                    },
                    "geocoded_address": "400 Broad St, Seattle, WA 98109",
                    "match_confidence": "high",
                    "distance_meters": 25.5,
                    "confidence_note": "Prediction based on King County records.",
                    "timestamp": "2025-12-10T12:00:00Z",
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Response containing the predicted price and metadata."""

    predicted_price: float = Field(
        ..., ge=0, description="Predicted home price in USD", examples=[450000.0]
    )
    prediction_id: str = Field(
        ...,
        description="Unique identifier for this prediction",
        examples=["pred-20251207-123456-abc123"],
    )
    model_version: str = Field(
        ..., description="Version of the model used for prediction", examples=["v1"]
    )
    confidence_note: str = Field(
        default="Prediction based on King County 2014-2015 data. Actual market values may differ.",
        description="Confidence note about the prediction",
    )
    data_vintage_warning: str = Field(
        default="This model was trained on 2014-2015 data. For current valuations, consider market adjustments.",
        description="Warning about data vintage",
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Timestamp of the prediction"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "predicted_price": 450000.0,
                    "prediction_id": "pred-20251207-123456-abc123",
                    "model_version": "v1",
                    "confidence_note": "Prediction based on King County 2014-2015 data. Actual market values may differ.",
                    "data_vintage_warning": "This model was trained on 2014-2015 data. For current valuations, consider market adjustments.",
                    "timestamp": "2025-12-07T12:00:00Z",
                }
            ]
        }
    }


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions."""

    predictions: list[PredictionResponse] = Field(
        ..., description="List of predictions for each home"
    )
    total_count: int = Field(..., description="Total number of predictions made")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Timestamp of the batch prediction"
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Health status of the API", examples=["healthy"])
    model_loaded: bool = Field(
        ..., description="Whether the model is loaded and ready", examples=[True]
    )
    demographics_loaded: bool = Field(
        ..., description="Whether demographics data is loaded", examples=[True]
    )
    model_version: str = Field(..., description="Version of the loaded model", examples=["v1"])
    data_vintage: str = Field(
        ..., description="Data vintage (training data date range)", examples=["2014-2015"]
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Timestamp of the health check"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "healthy",
                    "model_loaded": True,
                    "demographics_loaded": True,
                    "model_version": "v1",
                    "data_vintage": "2014-2015",
                    "timestamp": "2025-12-07T12:00:00Z",
                }
            ]
        }
    }


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str = Field(..., description="Error type", examples=["ValidationError"])
    message: str = Field(
        ...,
        description="Human-readable error message",
        examples=["Invalid zipcode: 00000 not found in King County"],
    )
    details: dict | None = Field(default=None, description="Additional error details")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Timestamp of the error"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "error": "ValidationError",
                    "message": "Invalid zipcode: 00000 not found in King County",
                    "details": {"field": "zipcode", "value": "00000"},
                    "timestamp": "2025-12-07T12:00:00Z",
                }
            ]
        }
    }
