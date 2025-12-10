"""
V1 MVP Configuration - Minimal settings for baseline API.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """V1 MVP Configuration."""
    
    app_name: str = "Real Estate Price Predictor"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Data vintage (V1 uses original 2014-2015 data)
    data_vintage_start: str = "2014"
    data_vintage_end: str = "2015"
    
    # Model paths
    model_path: str = "model/model.pkl"
    features_path: str = "model/model_features.json"
    demographics_path: str = "data/zipcode_demographics.csv"
    
    class Config:
        env_prefix = "API_"


@lru_cache
def get_settings() -> Settings:
    return Settings()
