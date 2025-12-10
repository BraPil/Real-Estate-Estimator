"""
V1 MVP Feature Service - Demographics lookup and enrichment.
"""

import logging
from functools import lru_cache
from typing import Dict, Set

import pandas as pd

logger = logging.getLogger(__name__)


class FeatureService:
    """V1 Demographics lookup service."""
    
    def __init__(self, demographics_path: str = "data/zipcode_demographics.csv"):
        self.demographics: pd.DataFrame = None
        self.valid_zipcodes: Set[str] = set()
        self.demographic_columns: list = []
        self._load_demographics(demographics_path)
    
    def _load_demographics(self, path: str) -> None:
        """Load demographics data."""
        logger.info("Loading demographics from: %s", path)
        
        self.demographics = pd.read_csv(path)
        self.demographics["zipcode"] = self.demographics["zipcode"].astype(str)
        self.valid_zipcodes = set(self.demographics["zipcode"].values)
        self.demographic_columns = [c for c in self.demographics.columns if c != "zipcode"]
        
        logger.info("Demographics loaded. Zipcodes: %d", len(self.valid_zipcodes))
    
    def is_valid_zipcode(self, zipcode: str) -> bool:
        """Check if zipcode is valid."""
        return str(zipcode) in self.valid_zipcodes
    
    def get_demographics(self, zipcode: str) -> Dict[str, float]:
        """Get demographic features for a zipcode."""
        row = self.demographics[self.demographics["zipcode"] == str(zipcode)]
        if row.empty:
            raise ValueError(f"Unknown zipcode: {zipcode}")
        
        return row[self.demographic_columns].iloc[0].to_dict()
    
    def enrich_features(self, home_features: Dict[str, float], zipcode: str) -> Dict[str, float]:
        """Combine home features with demographics."""
        demographics = self.get_demographics(zipcode)
        return {**home_features, **demographics}
    
    def get_status(self) -> Dict:
        """Return service status."""
        return {
            "is_loaded": self.demographics is not None,
            "n_zipcodes": len(self.valid_zipcodes),
            "n_features": len(self.demographic_columns)
        }


_feature_service = None


@lru_cache
def get_feature_service() -> FeatureService:
    global _feature_service
    if _feature_service is None:
        _feature_service = FeatureService()
    return _feature_service


def reset_feature_service() -> None:
    global _feature_service
    _feature_service = None
    get_feature_service.cache_clear()
