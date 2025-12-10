"""
Address lookup service for geocoding and property data retrieval.

This service provides:
- Geocoding: Convert address to lat/long using Nominatim (OpenStreetMap)
- Property Lookup: Find nearest property in King County database
- Auto-population: Get all property features from address alone

V4 Feature: Allows users to get predictions from just an address,
without needing to know property details like bedrooms, sqft, etc.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

from src.config import Settings, get_settings

logger = logging.getLogger(__name__)


class AddressService:
    """Service for address geocoding and property lookup."""

    def __init__(self, settings: Settings):
        """Initialize the address service.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self._property_db: Optional[pd.DataFrame] = None
        self._geocoder_url = "https://nominatim.openstreetmap.org/search"
        self._user_agent = "RealEstateEstimator/4.0 (King County Price Predictor)"

    def load_property_database(self) -> None:
        """Load the property lookup database."""
        # Use same directory as model path
        model_dir = Path(self.settings.model_path).parent.parent
        lookup_path = model_dir / "data" / "property_lookup.csv"
        
        # Also try relative to current directory
        if not lookup_path.exists():
            lookup_path = Path("data") / "property_lookup.csv"
        
        if not lookup_path.exists():
            logger.warning("Property lookup database not found at %s", lookup_path)
            return
        
        self._property_db = pd.read_csv(lookup_path)
        logger.info(
            "Loaded property database: %d properties", len(self._property_db)
        )

    @property
    def is_loaded(self) -> bool:
        """Check if property database is loaded."""
        return self._property_db is not None and len(self._property_db) > 0

    def geocode_address(self, address: str) -> Optional[dict]:
        """Convert an address to lat/long coordinates.

        Uses the free Nominatim (OpenStreetMap) geocoding service.

        Args:
            address: Full address string (e.g., "123 Main St, Seattle, WA 98103")

        Returns:
            Dict with 'lat', 'long', 'display_name' or None if not found
        """
        # Ensure address includes King County context
        if "king county" not in address.lower() and "wa" not in address.lower():
            address = f"{address}, King County, WA"

        params = {
            "q": address,
            "format": "json",
            "limit": 1,
            "countrycodes": "us",
        }
        headers = {"User-Agent": self._user_agent}

        try:
            response = requests.get(
                self._geocoder_url, 
                params=params, 
                headers=headers, 
                timeout=10
            )
            response.raise_for_status()
            
            results = response.json()
            if not results:
                logger.warning("No geocoding results for: %s", address)
                return None

            result = results[0]
            geocoded = {
                "lat": float(result["lat"]),
                "long": float(result["lon"]),
                "display_name": result.get("display_name", address),
            }
            logger.info(
                "Geocoded '%s' to (%.6f, %.6f)", 
                address, geocoded["lat"], geocoded["long"]
            )
            return geocoded

        except requests.RequestException as e:
            logger.error("Geocoding request failed: %s", e)
            return None
        except (KeyError, ValueError, IndexError) as e:
            logger.error("Failed to parse geocoding response: %s", e)
            return None

    def find_nearest_property(
        self, lat: float, long: float, max_distance_km: float = 0.5
    ) -> Optional[dict]:
        """Find the nearest property in the database to given coordinates.

        Uses haversine distance for accurate geographic matching.

        Args:
            lat: Latitude coordinate
            long: Longitude coordinate  
            max_distance_km: Maximum distance to search (default 0.5km)

        Returns:
            Dict with all property features, or None if no match found
        """
        if not self.is_loaded:
            logger.error("Property database not loaded")
            return None

        # Calculate distances using haversine formula
        lat_rad = np.radians(lat)
        long_rad = np.radians(long)
        
        db_lat_rad = np.radians(self._property_db["lat_centroid"].values)
        db_long_rad = np.radians(self._property_db["long_centroid"].values)

        # Haversine formula
        dlat = db_lat_rad - lat_rad
        dlong = db_long_rad - long_rad
        
        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(lat_rad) * np.cos(db_lat_rad) * np.sin(dlong / 2) ** 2
        )
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Earth radius in km
        earth_radius_km = 6371
        distances = earth_radius_km * c

        # Find nearest
        min_idx = np.argmin(distances)
        min_distance = distances[min_idx]

        if min_distance > max_distance_km:
            logger.warning(
                "Nearest property is %.2f km away (max: %.2f km)",
                min_distance, max_distance_km
            )
            return None

        # Get property data
        row = self._property_db.iloc[min_idx]
        
        property_data = {
            "bedrooms": int(row["bedrooms"]),
            "bathrooms": float(row["bathrooms"]),
            "sqft_living": int(row["sqft_living"]),
            "sqft_lot": int(row["sqft_lot"]),
            "floors": float(row["floors"]),
            "waterfront": int(row["waterfront"]),
            "view": int(row["view"]),
            "condition": int(row["condition"]),
            "grade": int(row["grade"]),
            "sqft_above": int(row["sqft_above"]),
            "sqft_basement": int(row["sqft_basement"]),
            "yr_built": int(row["yr_built"]),
            "yr_renovated": int(row["yr_renovated"]),
            "zipcode": str(int(row["zipcode"])),  # Convert float->int->str to avoid "98122.0"
            "lat": float(row["lat"]),
            "long": float(row["long"]),
            "sqft_living15": float(row["sqft_living15"]),
            "sqft_lot15": float(row["sqft_lot15"]),
            # Metadata
            "_parcel_major": int(row["Major"]),
            "_parcel_minor": int(row["Minor"]),
            "_distance_km": float(min_distance),
            "_match_confidence": "high" if min_distance < 0.1 else ("medium" if min_distance < 0.3 else "low"),
        }

        logger.info(
            "Found property at %.4f km: %d bed, %d sqft, grade %d",
            min_distance,
            property_data["bedrooms"],
            property_data["sqft_living"],
            property_data["grade"],
        )

        return property_data

    def lookup_address(self, address: str) -> Optional[dict]:
        """Full address lookup: geocode and find property data.

        This is the main entry point for V4 address-based predictions.

        Args:
            address: Full address string

        Returns:
            Dict with all property features needed for prediction,
            plus geocoding and match metadata, or None if not found
        """
        # Step 1: Geocode the address
        geocoded = self.geocode_address(address)
        if not geocoded:
            return None

        # Step 2: Find nearest property
        property_data = self.find_nearest_property(
            geocoded["lat"], geocoded["long"]
        )
        if not property_data:
            return None

        # Add geocoding metadata
        property_data["_geocoded_address"] = geocoded["display_name"]
        property_data["_geocoded_lat"] = geocoded["lat"]
        property_data["_geocoded_long"] = geocoded["long"]

        return property_data


# Singleton instance
_address_service: Optional[AddressService] = None


def get_address_service() -> AddressService:
    """Get the singleton AddressService instance.

    Returns:
        The initialized AddressService
    """
    global _address_service
    if _address_service is None:
        settings = get_settings()
        _address_service = AddressService(settings)
        _address_service.load_property_database()
    return _address_service


def reset_address_service() -> None:
    """Reset the singleton instance (for testing)."""
    global _address_service
    _address_service = None
