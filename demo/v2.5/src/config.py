"""
Configuration management for the Real Estate Price Predictor API.

This module provides centralized configuration using Pydantic settings,
supporting environment variables and sensible defaults.

Environment Variables:
    MODEL_PATH: Path to the pickled model file
    FEATURES_PATH: Path to the model features JSON file
    DEMOGRAPHICS_PATH: Path to the demographics CSV file
    MLFLOW_TRACKING_URI: MLflow tracking server URI (optional)
    MLFLOW_MODEL_NAME: MLflow model registry name
    MLFLOW_MODEL_STAGE: MLflow model stage to load (Production, Staging, None)
    API_VERSION: API version string
    LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR)
"""

import os
from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application metadata
    app_name: str = "Real Estate Price Predictor API"
    app_version: str = "1.0.0"
    api_version: str = "v1"
    debug: bool = Field(default=False, description="Enable debug mode")
    
    # Model paths (local file-based loading)
    model_path: str = Field(
        default="model/model.pkl",
        description="Path to the pickled sklearn model"
    )
    features_path: str = Field(
        default="model/model_features.json",
        description="Path to the model features JSON file"
    )
    demographics_path: str = Field(
        default="data/zipcode_demographics.csv",
        description="Path to the demographics CSV file"
    )
    
    # MLflow integration (optional)
    mlflow_tracking_uri: Optional[str] = Field(
        default=None,
        description="MLflow tracking server URI"
    )
    mlflow_model_name: str = Field(
        default="real-estate-price-predictor",
        description="MLflow model registry name"
    )
    mlflow_model_stage: Optional[str] = Field(
        default="Production",
        description="MLflow model stage to load (Production, Staging, or None for latest)"
    )
    use_mlflow_model: bool = Field(
        default=False,
        description="Whether to load model from MLflow registry instead of local file"
    )
    
    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    
    # Data vintage warning
    data_vintage_start: int = 2014
    data_vintage_end: int = 2015
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance.
    
    Returns:
        Settings: Application settings singleton
    """
    return Settings()


# Convenience function to get settings in dependency injection
def get_settings_dependency() -> Settings:
    """Dependency injection wrapper for settings.
    
    This is used as a FastAPI dependency to provide settings
    to endpoints.
    
    Returns:
        Settings: Application settings
    """
    return get_settings()
