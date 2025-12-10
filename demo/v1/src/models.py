"""
V1 MVP Models - Request/Response schemas for baseline API.

V1 accepts all 18 columns from future_unseen_examples.csv but only uses 7.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """V1 Prediction request - accepts all 18 columns, uses 7."""
    
    # All 18 columns from future_unseen_examples.csv
    bedrooms: int = Field(..., ge=0, le=33)
    bathrooms: float = Field(..., ge=0, le=10)
    sqft_living: int = Field(..., ge=200, le=15000)
    sqft_lot: int = Field(..., ge=500, le=2000000)
    floors: float = Field(..., ge=1, le=4)
    waterfront: int = Field(default=0, ge=0, le=1)
    view: int = Field(default=0, ge=0, le=4)
    condition: int = Field(default=3, ge=1, le=5)
    grade: int = Field(default=7, ge=1, le=13)
    sqft_above: int = Field(..., ge=200, le=10000)
    sqft_basement: int = Field(..., ge=0, le=5000)
    yr_built: int = Field(default=1980, ge=1900, le=2025)
    yr_renovated: int = Field(default=0, ge=0, le=2025)
    zipcode: str = Field(..., min_length=5, max_length=5)
    lat: float = Field(default=47.5, ge=47.0, le=48.0)
    long: float = Field(default=-122.0, ge=-123.0, le=-121.0)
    sqft_living15: int = Field(default=1500, ge=200, le=10000)
    sqft_lot15: int = Field(default=5000, ge=500, le=1000000)
    
    def get_model_features(self) -> dict:
        """Return only the 7 features V1 model uses."""
        return {
            "bedrooms": self.bedrooms,
            "bathrooms": self.bathrooms,
            "sqft_living": self.sqft_living,
            "sqft_lot": self.sqft_lot,
            "floors": self.floors,
            "sqft_above": self.sqft_above,
            "sqft_basement": self.sqft_basement,
        }


class PredictionResponse(BaseModel):
    """V1 Prediction response - minimal metadata."""
    
    predicted_price: float
    prediction_id: str
    model_version: str = "1.0.0"
    confidence_note: str
    data_vintage_warning: str
    timestamp: datetime


class HealthResponse(BaseModel):
    """V1 Health check response."""
    
    status: str
    model_loaded: bool
    demographics_loaded: bool
    model_version: str
    data_vintage: str
    timestamp: datetime


class ErrorResponse(BaseModel):
    """V1 Error response."""
    
    error: str
    message: str
    detail: Optional[str] = None
