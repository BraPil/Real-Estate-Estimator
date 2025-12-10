"""
API package for the Real Estate Price Predictor.

This package contains FastAPI routers for:
- Health check endpoint
- Prediction endpoints (/predict, /predict-minimal)
"""

from src.api.prediction import router as prediction_router

__all__ = ["prediction_router"]
