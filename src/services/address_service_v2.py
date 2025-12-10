"""
Address lookup service V2 using King County official data.

This service provides accurate property data by:
1. Geocoding: King County Address Locator (official) to get PIN
2. Property Data: King County Assessor CSV export (EXTR_ResBldg.csv)
3. Coordinates: ArcGIS Residential Parcels for lat/long

V4.1 Feature: Accurate address-to-property lookup using official sources.
No scraping required - uses authoritative CSV data from King County Assessor.
"""

import gzip
import logging
import re
import shutil
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from src.config import Settings

logger = logging.getLogger(__name__)


def _ensure_decompressed(gz_path: Path, csv_path: Path) -> bool:
    """Decompress a gzipped file if the CSV doesn't exist.
    
    Args:
        gz_path: Path to the .gz file
        csv_path: Path where the CSV should be
        
    Returns:
        True if CSV exists (or was decompressed), False otherwise
    """
    if csv_path.exists():
        return True
    
    if not gz_path.exists():
        return False
    
    logger.info("Decompressing %s -> %s (one-time operation)...", gz_path, csv_path)
    try:
        with gzip.open(gz_path, 'rb') as f_in:
            with open(csv_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        logger.info("Decompression complete: %s", csv_path)
        return True
    except Exception as e:
        logger.error("Failed to decompress %s: %s", gz_path, e)
        return False


class AddressServiceV2:
    """Service for address lookup using King County official data."""

    # King County API endpoints
    KC_GEOCODER_URL = (
        "https://gismaps.kingcounty.gov/arcgis/rest/services/Address/"
        "Composite_locator/GeocodeServer/findAddressCandidates"
    )
    KC_ARCGIS_PARCELS_URL = (
        "https://services.arcgis.com/Ej0PsM5Aw677QF1W/arcgis/rest/services/"
        "Residential_Parcels_with_Building_Age/FeatureServer/2/query"
    )

    # Path to King County Assessor data (check multiple locations)
    # Primary: data/king_county/ (new clean location with .gz files)
    # Fallback: references/King_County_Assessment_data_ALL/ (legacy location)
    DATA_PATHS = [
        Path("data/king_county"),
        Path("references/King_County_Assessment_data_ALL"),
    ]

    def __init__(self, settings: Settings):
        """Initialize the address service.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "RealEstateEstimator/4.1 (King County Price Predictor)"
        })
        self._resbldg_df: Optional[pd.DataFrame] = None
        self._parcel_df: Optional[pd.DataFrame] = None
        self._data_dir: Optional[Path] = None

    def _find_data_dir(self) -> Optional[Path]:
        """Find the directory containing King County data files."""
        if self._data_dir:
            return self._data_dir
            
        for path in self.DATA_PATHS:
            # Check for CSV or GZ files
            if (path / "EXTR_ResBldg.csv").exists() or (path / "EXTR_ResBldg.csv.gz").exists():
                self._data_dir = path
                logger.info("Found King County data in: %s", path)
                return path
        
        logger.error("King County data not found in any of: %s", self.DATA_PATHS)
        return None

    def _get_csv_path(self, filename: str) -> Optional[Path]:
        """Get path to a CSV file, decompressing from .gz if needed."""
        data_dir = self._find_data_dir()
        if not data_dir:
            return None
        
        csv_path = data_dir / filename
        gz_path = data_dir / f"{filename}.gz"
        
        # Try to ensure the CSV exists (decompress if needed)
        if _ensure_decompressed(gz_path, csv_path):
            return csv_path
        
        # Check if CSV exists directly
        if csv_path.exists():
            return csv_path
            
        return None

    def _load_resbldg_data(self) -> bool:
        """Load the residential buildings CSV data.

        Returns:
            True if loaded successfully, False otherwise
        """
        if self._resbldg_df is not None:
            return True

        csv_path = self._get_csv_path("EXTR_ResBldg.csv")
        if not csv_path:
            logger.error("ResBldg CSV not found. Run: python scripts/download_kc_data.py")
            return False

        try:
            logger.info("Loading King County ResBldg data from %s...", csv_path)
            self._resbldg_df = pd.read_csv(csv_path)
            # Create PIN column from Major + Minor (padded to 10 digits)
            self._resbldg_df["PIN"] = (
                self._resbldg_df["Major"].astype(str).str.zfill(6) +
                self._resbldg_df["Minor"].astype(str).str.zfill(4)
            )
            logger.info(
                "Loaded %d residential buildings from King County Assessor data",
                len(self._resbldg_df)
            )
            return True
        except Exception as e:
            logger.error("Failed to load ResBldg CSV: %s", e)
            return False

    def _load_parcel_data(self) -> bool:
        """Load the parcel CSV data for lot size info.

        Returns:
            True if loaded successfully, False otherwise
        """
        if self._parcel_df is not None:
            return True

        csv_path = self._get_csv_path("EXTR_Parcel.csv")
        if not csv_path:
            logger.warning("Parcel CSV not found. Lot sizes will use defaults.")
            return False

        try:
            logger.info("Loading King County Parcel data from %s...", csv_path)
            # Only load columns we need to save memory
            # Use latin-1 encoding as the file contains special characters
            self._parcel_df = pd.read_csv(
                csv_path,
                usecols=["Major", "Minor", "SqFtLot"],
                encoding="latin-1"
            )
            # Create PIN column from Major + Minor (padded to 10 digits)
            self._parcel_df["PIN"] = (
                self._parcel_df["Major"].astype(str).str.zfill(6) +
                self._parcel_df["Minor"].astype(str).str.zfill(4)
            )
            logger.info(
                "Loaded %d parcels from King County Parcel data",
                len(self._parcel_df)
            )
            return True
        except Exception as e:
            logger.error("Failed to load Parcel CSV: %s", e)
            return False

    def get_lot_size(self, pin: str) -> int:
        """Get lot size from parcel data.

        Args:
            pin: Parcel Identification Number (10 digits)

        Returns:
            Lot size in sqft, or 5000 as default
        """
        if not self._load_parcel_data():
            return 5000  # Default fallback

        matches = self._parcel_df[self._parcel_df["PIN"] == pin]
        if matches.empty:
            logger.warning("No parcel data found for PIN: %s, using default lot size", pin)
            return 5000

        sqft_lot = int(matches.iloc[0].get("SqFtLot", 0) or 0)
        if sqft_lot < 500:
            logger.warning(
                "Lot size %d too small for PIN %s, using default",
                sqft_lot, pin
            )
            return 5000
        
        return sqft_lot

    @property
    def is_loaded(self) -> bool:
        """Check if the CSV data can be loaded.
        
        This triggers loading if not already done.
        """
        return self._load_resbldg_data()

    def geocode_address(self, address: str) -> Optional[dict]:
        """Geocode address using King County's official geocoder.

        Returns PIN (parcel number) if found as a point address.

        Args:
            address: Full address string

        Returns:
            Dict with 'pin', 'matched_address', 'score' or None
        """
        try:
            params = {
                "SingleLine": address,
                "outFields": "*",
                "f": "json",
            }
            response = self._session.get(
                self.KC_GEOCODER_URL, params=params, timeout=15
            )
            response.raise_for_status()
            data = response.json()

            candidates = data.get("candidates", [])
            if not candidates:
                logger.warning("No geocoding results for: %s", address)
                return None

            # Find best candidate with a PIN (point address match)
            for candidate in candidates:
                attrs = candidate.get("attributes", {})
                pin = attrs.get("User_fld", "")
                if pin:
                    result = {
                        "pin": pin,
                        "matched_address": candidate.get("address", address),
                        "score": candidate.get("score", 0),
                        "addr_type": attrs.get("Addr_type", ""),
                    }
                    logger.info(
                        "Geocoded '%s' to PIN %s (score: %.1f)",
                        address, pin, result["score"]
                    )
                    return result

            # No PIN found - try CSV fallback
            logger.info(
                "Geocoder returned no PIN for '%s', trying CSV search...",
                address
            )
            return self._search_csv_by_address(address)

        except requests.RequestException as e:
            logger.error("Geocoding request failed: %s", e)
            return self._search_csv_by_address(address)
        except (KeyError, ValueError) as e:
            logger.error("Failed to parse geocoding response: %s", e)
            return self._search_csv_by_address(address)

    def _search_csv_by_address(self, address: str) -> Optional[dict]:
        """Search for property by address text in CSV data.

        This is a fallback when the geocoder doesn't return a PIN.

        Args:
            address: Full address string

        Returns:
            Dict with 'pin', 'matched_address', 'score' or None
        """
        if not self._load_resbldg_data():
            return None

        # Parse address components
        # Example: "1523 15th Ave, Seattle, WA 98122"
        import re

        # Extract house number
        num_match = re.match(r"(\d+)", address.strip())
        if not num_match:
            logger.warning("Could not extract house number from: %s", address)
            return None
        house_num = num_match.group(1)

        # Extract street name (uppercase, remove common prefixes/suffixes)
        addr_upper = address.upper()
        # Remove city, state, zip
        addr_clean = re.sub(r",.*", "", addr_upper).strip()
        # Remove house number
        addr_clean = re.sub(r"^\d+\s*", "", addr_clean).strip()
        # Get street name parts
        street_parts = addr_clean.split()
        if not street_parts:
            return None
        
        # Build search pattern - house number at START, followed by spaces and street
        # The CSV format is like: "1523   15TH AVE   98122"
        # Use \b for word boundary to avoid "1523" matching "11523"
        search_pattern = r"^\s*" + house_num + r"\s+.*" + street_parts[0]

        matches = self._resbldg_df[
            self._resbldg_df["Address"].str.contains(search_pattern, na=False, case=False, regex=True)
        ]

        if matches.empty:
            # Try just house number at start + any part of street
            for part in street_parts[:3]:
                if len(part) > 2:
                    pattern2 = r"^\s*" + house_num + r"\s+.*" + part
                    matches = self._resbldg_df[
                        self._resbldg_df["Address"].str.contains(pattern2, na=False, case=False, regex=True)
                    ]
                    if not matches.empty:
                        break

        if matches.empty:
            logger.warning("No CSV match found for address: %s", address)
            return None

        # If multiple matches, prefer ones with a valid zipcode
        if len(matches) > 1:
            # Extract zipcode from input address
            zip_match = re.search(r"\b(\d{5})\b", address)
            if zip_match:
                input_zip = zip_match.group(1)
                zip_matches = matches[matches["ZipCode"].astype(str) == input_zip]
                if not zip_matches.empty:
                    matches = zip_matches

        # Take the first match
        row = matches.iloc[0]
        pin = str(row["Major"]).zfill(6) + str(row["Minor"]).zfill(4)

        # Get zipcode - prefer from CSV, fallback to input address
        csv_zip = str(row.get("ZipCode", ""))
        if csv_zip == "nan" or not csv_zip or csv_zip == "":
            zip_match = re.search(r"\b(\d{5})\b", address)
            csv_zip = zip_match.group(1) if zip_match else ""

        result = {
            "pin": pin,
            "matched_address": str(row["Address"]).strip(),
            "score": 80.0,  # Lower score since this is a fuzzy match
            "addr_type": "CSVMatch",
            "_csv_zipcode": csv_zip,
        }
        logger.info(
            "CSV search matched '%s' to PIN %s (address: %s)",
            address, pin, result["matched_address"]
        )
        return result

    def get_arcgis_parcel_data(self, pin: str) -> Optional[dict]:
        """Get parcel data from ArcGIS Residential Parcels layer.

        Args:
            pin: Parcel Identification Number (10 digits)

        Returns:
            Dict with parcel attributes or None
        """
        try:
            params = {
                "where": f"PIN='{pin}'",
                "outFields": "*",
                "f": "json",
            }
            response = self._session.get(
                self.KC_ARCGIS_PARCELS_URL, params=params, timeout=15
            )
            response.raise_for_status()
            data = response.json()

            features = data.get("features", [])
            if not features:
                logger.warning("No ArcGIS parcel data for PIN: %s", pin)
                return None

            attrs = features[0].get("attributes", {})
            result = {
                "address": attrs.get("ADDR_FULL"),
                "zipcode": attrs.get("ZIP5"),
                "sqft_living": attrs.get("SqFtTotLiv"),
                "stories": attrs.get("Stories"),
                "grade": attrs.get("BldgGrade"),
                "yr_built": attrs.get("YrBuilt"),
                "yr_renovated": attrs.get("YrRenovate", 0),
                "condition": attrs.get("Condition"),
                "lat": attrs.get("LAT"),
                "long": attrs.get("LON"),
            }
            logger.info(
                "ArcGIS parcel data for PIN %s: %s sqft, grade %s, coords (%s, %s)",
                pin, result["sqft_living"], result["grade"], result["lat"], result["long"]
            )
            return result

        except requests.RequestException as e:
            logger.error("ArcGIS parcel query failed: %s", e)
            return None

    def get_csv_property_data(self, pin: str) -> Optional[dict]:
        """Get detailed property data from King County Assessor CSV.

        This is the authoritative source for bedrooms, bathrooms, and
        detailed square footage breakdowns.

        Args:
            pin: Parcel Identification Number (10 digits)

        Returns:
            Dict with detailed property attributes or None
        """
        if not self._load_resbldg_data():
            return None

        # Find matching property by PIN
        matches = self._resbldg_df[self._resbldg_df["PIN"] == pin]

        if matches.empty:
            logger.warning("No CSV data found for PIN: %s", pin)
            return None

        # Use the first building (most properties have one main building)
        row = matches.iloc[0]

        # Calculate total bathrooms (full + 0.75*3/4 + 0.5*half)
        full_baths = int(row.get("BathFullCount", 0) or 0)
        three_quarter_baths = int(row.get("Bath3qtrCount", 0) or 0)
        half_baths = int(row.get("BathHalfCount", 0) or 0)
        total_baths = full_baths + 0.75 * three_quarter_baths + 0.5 * half_baths

        # Map condition code to numeric (1-5 scale)
        condition = int(row.get("Condition", 3) or 3)

        # ViewUtilization is Y/N, convert to 0-4 scale (0=none, higher=better)
        view_util = str(row.get("ViewUtilization", "N") or "N").strip().upper()
        view = 1 if view_util == "Y" else 0

        result = {
            "address": str(row.get("Address", "")),
            "bedrooms": int(row.get("Bedrooms", 3) or 3),
            "bathrooms": total_baths if total_baths > 0 else 1.0,
            "full_baths": full_baths,
            "three_quarter_baths": three_quarter_baths,
            "half_baths": half_baths,
            "sqft_living": int(row.get("SqFtTotLiving", 0) or 0),
            "sqft_above": int(row.get("SqFt1stFloor", 0) or 0) + int(row.get("SqFt2ndFloor", 0) or 0),
            "sqft_basement": int(row.get("SqFtTotBasement", 0) or 0),
            "sqft_finished_basement": int(row.get("SqFtFinBasement", 0) or 0),
            "sqft_lot": self.get_lot_size(pin),  # Get lot size from Parcel CSV
            "stories": float(row.get("Stories", 1) or 1),
            "floors": float(row.get("Stories", 1) or 1),
            "grade": int(row.get("BldgGrade", 7) or 7),
            "condition": condition,
            "yr_built": int(row.get("YrBuilt", 1990) or 1990),
            "yr_renovated": int(row.get("YrRenovated", 0) or 0),
            "zipcode": str(row.get("ZipCode", "")),
            "view": view,
            "fireplace": int(row.get("FpSingleStory", 0) or 0) + int(row.get("FpMultiStory", 0) or 0),
        }

        logger.info(
            "CSV data for PIN %s: %d bed, %.1f bath, %d sqft, grade %d",
            pin, result["bedrooms"], result["bathrooms"],
            result["sqft_living"], result["grade"]
        )
        return result

    def lookup_address(self, address: str) -> Optional[dict]:
        """Complete address lookup: geocode -> get property data.

        Combines data from:
        1. King County Geocoder (PIN lookup)
        2. King County Assessor CSV (bedrooms, bathrooms, sqft)
        3. ArcGIS Residential Parcels (lat/long)

        Args:
            address: Full address string

        Returns:
            Complete property data dict ready for prediction, or None
        """
        # Step 1: Geocode to get PIN
        geocode_result = self.geocode_address(address)
        if not geocode_result:
            return None

        pin = geocode_result["pin"]

        # Step 2: Get CSV data (authoritative source for building details)
        csv_data = self.get_csv_property_data(pin)
        if not csv_data:
            logger.warning("No CSV data for PIN %s", pin)
            return None

        # Step 3: Get ArcGIS data for lat/long (CSV doesn't have coords)
        arcgis_data = self.get_arcgis_parcel_data(pin)

        # Determine zipcode - multiple fallback sources
        zipcode = csv_data["zipcode"]
        if not zipcode or zipcode == "nan" or str(zipcode) == "nan":
            # Try from geocode result (CSV fallback stores it here)
            zipcode = geocode_result.get("_csv_zipcode", "")
        if not zipcode or zipcode == "nan" or str(zipcode) == "nan":
            # Try from ArcGIS
            zipcode = arcgis_data["zipcode"] if arcgis_data else ""
        if not zipcode or zipcode == "nan" or str(zipcode) == "nan":
            # Extract from input address
            zipcode = self._extract_zipcode(address)

        # Combine data, preferring CSV for property details
        property_data = {
            # From CSV (authoritative building data)
            "bedrooms": csv_data["bedrooms"],
            "bathrooms": csv_data["bathrooms"],
            "sqft_living": csv_data["sqft_living"],
            "sqft_lot": csv_data["sqft_lot"],  # From EXTR_Parcel.csv via get_lot_size()
            "floors": csv_data["floors"],
            "waterfront": 0,  # Would need EXTR_Parcel.csv waterfront column
            "view": csv_data["view"],
            "condition": csv_data["condition"],
            "grade": csv_data["grade"],
            "sqft_above": csv_data["sqft_above"],
            "sqft_basement": csv_data["sqft_basement"],
            "yr_built": csv_data["yr_built"],
            "yr_renovated": csv_data["yr_renovated"],
            # From ArcGIS (for coordinates)
            "lat": arcgis_data["lat"] if arcgis_data else None,
            "long": arcgis_data["long"] if arcgis_data else None,
            "zipcode": zipcode,
            # Neighborhood averages (placeholder - use same as property)
            "sqft_living15": csv_data["sqft_living"],  # Use same as estimate
            "sqft_lot15": csv_data["sqft_lot"],  # Use same as property
            # Metadata
            "_pin": pin,
            "_matched_address": geocode_result["matched_address"],
            "_geocode_score": geocode_result["score"],
            "_data_source": "King County Assessor (EXTR_ResBldg.csv + EXTR_Parcel.csv)",
        }

        # Validate we have coordinates
        if property_data["lat"] is None or property_data["long"] is None:
            logger.warning("No coordinates found for PIN %s", pin)
            # Try to continue anyway - some models may not need coords
            property_data["lat"] = 0.0
            property_data["long"] = 0.0

        logger.info(
            "Complete property lookup for '%s': %d bed, %.1f bath, %d sqft @ (%s, %s)",
            address,
            property_data["bedrooms"],
            property_data["bathrooms"],
            property_data["sqft_living"],
            property_data["lat"],
            property_data["long"],
        )

        return property_data

    def _extract_zipcode(self, address: str) -> str:
        """Extract zipcode from address string."""
        match = re.search(r"\b(\d{5})\b", address)
        return match.group(1) if match else "98101"


# Singleton instance
_address_service_v2: Optional[AddressServiceV2] = None


def get_address_service_v2() -> AddressServiceV2:
    """Get or create the singleton AddressServiceV2 instance."""
    global _address_service_v2
    if _address_service_v2 is None:
        from src.config import get_settings
        _address_service_v2 = AddressServiceV2(get_settings())
    return _address_service_v2
