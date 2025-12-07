"""
Feature service for demographics lookup and feature enrichment.

This service handles:
- Loading and caching demographics data
- Enriching home features with zipcode demographics
- Providing average demographics for minimal predictions
- Validating zipcodes against King County

The demographics data is loaded once at startup and cached in memory
for fast lookup during predictions.
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.config import Settings, get_settings

logger = logging.getLogger(__name__)


class FeatureService:
    """Service for feature enrichment and demographics lookup.
    
    This service loads demographics data and provides methods to
    enrich home features with zipcode-based demographic information.
    
    Attributes:
        demographics_df: DataFrame containing demographics by zipcode
        valid_zipcodes: Set of valid King County zipcodes
        average_demographics: Dictionary of average values for each demographic feature
        is_loaded: Whether the demographics data is loaded
    """
    
    def __init__(self, settings: Settings):
        """Initialize the feature service.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        self.demographics_df: Optional[pd.DataFrame] = None
        self.valid_zipcodes: set = set()
        self.average_demographics: Dict[str, float] = {}
        self.demographic_columns: List[str] = []
        self.is_loaded: bool = False
        self._load_demographics()
    
    def _load_demographics(self) -> None:
        """Load demographics data from CSV file.
        
        Raises:
            FileNotFoundError: If demographics file not found
        """
        demographics_path = Path(self.settings.demographics_path)
        
        if not demographics_path.exists():
            raise FileNotFoundError(
                f"Demographics file not found: {demographics_path}. "
                "Ensure data files are in place."
            )
        
        logger.info("Loading demographics from: %s", demographics_path)
        
        # Load demographics with zipcode as string
        self.demographics_df = pd.read_csv(
            demographics_path,
            dtype={"zipcode": str}
        )
        
        # Index by zipcode for fast lookup
        self.demographics_df.set_index("zipcode", inplace=True)
        
        # Store valid zipcodes
        self.valid_zipcodes = set(self.demographics_df.index)
        
        # Store demographic column names (all columns except zipcode index)
        self.demographic_columns = list(self.demographics_df.columns)
        
        # Compute average demographics for minimal predictions
        self.average_demographics = self.demographics_df.mean().to_dict()
        
        self.is_loaded = True
        logger.info(
            "Demographics loaded successfully. Zipcodes: %d, Features: %d",
            len(self.valid_zipcodes),
            len(self.demographic_columns)
        )
    
    def is_valid_zipcode(self, zipcode: str) -> bool:
        """Check if a zipcode is valid for King County.
        
        Args:
            zipcode: 5-digit zipcode string
            
        Returns:
            True if zipcode is in the demographics data
        """
        return zipcode in self.valid_zipcodes
    
    def get_demographics(self, zipcode: str) -> Dict[str, float]:
        """Get demographics for a specific zipcode.
        
        Args:
            zipcode: 5-digit zipcode string
            
        Returns:
            Dictionary of demographic feature values
            
        Raises:
            ValueError: If zipcode is not found
        """
        if not self.is_valid_zipcode(zipcode):
            raise ValueError(
                f"Invalid zipcode: {zipcode}. "
                f"Must be a valid King County zipcode. "
                f"Valid examples: {list(self.valid_zipcodes)[:5]}..."
            )
        
        row = self.demographics_df.loc[zipcode]
        return row.to_dict()
    
    def get_average_demographics(self) -> Dict[str, float]:
        """Get average demographics across all zipcodes.
        
        Used for minimal predictions where no zipcode is provided.
        
        Returns:
            Dictionary of average demographic feature values
        """
        return self.average_demographics.copy()
    
    def enrich_features(
        self,
        home_features: Dict[str, float],
        zipcode: str
    ) -> Dict[str, float]:
        """Enrich home features with demographics from zipcode.
        
        Args:
            home_features: Dictionary of home-level features
            zipcode: 5-digit zipcode for demographics lookup
            
        Returns:
            Dictionary with home features plus demographic features
            
        Raises:
            ValueError: If zipcode is invalid
        """
        demographics = self.get_demographics(zipcode)
        
        # Merge home features with demographics
        enriched = {**home_features, **demographics}
        
        return enriched
    
    def enrich_features_with_average(
        self,
        home_features: Dict[str, float]
    ) -> Dict[str, float]:
        """Enrich home features with average demographics.
        
        Used when no zipcode is provided (minimal prediction).
        
        Args:
            home_features: Dictionary of home-level features
            
        Returns:
            Dictionary with home features plus average demographic features
        """
        demographics = self.get_average_demographics()
        
        # Merge home features with average demographics
        enriched = {**home_features, **demographics}
        
        return enriched
    
    def get_status(self) -> dict:
        """Get the current status of the feature service.
        
        Returns:
            Dictionary with feature service status information
        """
        return {
            "is_loaded": self.is_loaded,
            "zipcode_count": len(self.valid_zipcodes),
            "demographic_feature_count": len(self.demographic_columns),
            "sample_zipcodes": list(self.valid_zipcodes)[:5],
        }


# Singleton instance
_feature_service: Optional[FeatureService] = None


def get_feature_service() -> FeatureService:
    """Get the singleton FeatureService instance.
    
    Returns:
        FeatureService: The feature service singleton
        
    Raises:
        RuntimeError: If demographics loading fails
    """
    global _feature_service
    if _feature_service is None:
        settings = get_settings()
        _feature_service = FeatureService(settings)
    return _feature_service


def reset_feature_service() -> None:
    """Reset the feature service singleton.
    
    Useful for testing or forcing a data reload.
    """
    global _feature_service
    _feature_service = None
