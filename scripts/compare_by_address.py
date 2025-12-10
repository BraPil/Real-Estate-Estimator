#!/usr/bin/env python3
"""
Compare Model Predictions by Address

This script takes a street address, city, and zip code, looks up the property
data from King County Assessor records, and compares predictions from all
three dockerized model versions (V1, V2.5, V3.3).

Usage:
    python scripts/compare_by_address.py "1523 15th Ave S" "Seattle" "98144"
    python scripts/compare_by_address.py --interactive

Requirements:
    - Docker containers running on ports 8000 (V1), 8001 (V2.5), 8002 (V3.3)
    - King County Assessor CSV data in references/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import requests

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.address_service_v2 import AddressServiceV2
from src.config import get_settings


# ANSI colors for terminal output
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    BLUE = '\033[0;34m'
    YELLOW = '\033[1;33m'
    CYAN = '\033[0;36m'
    MAGENTA = '\033[0;35m'
    BOLD = '\033[1m'
    NC = '\033[0m'  # No Color


def print_header(text: str, color: str = Colors.BLUE):
    """Print a formatted header."""
    print(f"\n{color}{'=' * 60}{Colors.NC}")
    print(f"{color}{text:^60}{Colors.NC}")
    print(f"{color}{'=' * 60}{Colors.NC}\n")


def print_section(text: str, color: str = Colors.CYAN):
    """Print a section header."""
    print(f"\n{color}--- {text} ---{Colors.NC}")


def lookup_property(street: str, city: str, zipcode: str) -> Optional[dict]:
    """Look up property data from King County Assessor records."""
    address = f"{street}, {city}, WA {zipcode}"
    
    print(f"{Colors.YELLOW}Looking up property: {address}{Colors.NC}")
    
    try:
        svc = AddressServiceV2(get_settings())
        result = svc.lookup_address(address)
        
        if result:
            print(f"{Colors.GREEN}Found property: PIN {result['_pin']}{Colors.NC}")
            return result
        else:
            print(f"{Colors.RED}Property not found in King County records{Colors.NC}")
            return None
    except Exception as e:
        print(f"{Colors.RED}Error looking up property: {e}{Colors.NC}")
        return None


def call_v1_api(property_data: dict) -> Optional[dict]:
    """Call V1 MVP API on port 8000."""
    # V1 uses /predict (no /api/v1 prefix)
    # V1 expects: bedrooms, bathrooms, sqft_living, sqft_lot, floors, 
    #             waterfront, view, condition, grade, sqft_above, sqft_basement,
    #             yr_built, yr_renovated, zipcode, lat, long, sqft_living15, sqft_lot15
    
    # Get sqft values from CSV data
    sqft_living = property_data.get("sqft_living", 0)
    sqft_basement = property_data.get("sqft_basement", 0) or 0  # Default to 0 if not found
    sqft_above = property_data.get("sqft_above", 0) or 0
    
    # Fallback: if sqft_above not in data, calculate as sqft_living (treat all as above ground)
    if sqft_above == 0:
        sqft_above = sqft_living
    
    payload = {
        "bedrooms": property_data.get("bedrooms", 3),
        "bathrooms": property_data.get("bathrooms", 1.0),
        "sqft_living": sqft_living,
        "sqft_lot": property_data.get("sqft_lot", 5000),
        "floors": property_data.get("floors", 1.0),
        "waterfront": property_data.get("waterfront", 0),
        "view": property_data.get("view", 0),
        "condition": property_data.get("condition", 3),
        "grade": property_data.get("grade", 7),
        "sqft_above": sqft_above,
        "sqft_basement": sqft_basement,
        "yr_built": property_data.get("yr_built", 1990),
        "yr_renovated": property_data.get("yr_renovated", 0) or 0,
        "zipcode": str(property_data.get("zipcode", "98101")),
        "lat": property_data.get("lat", 47.6),
        "long": property_data.get("long", -122.3),
        "sqft_living15": property_data.get("sqft_living15", sqft_living),
        "sqft_lot15": property_data.get("sqft_lot15", 5000),
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/predict",
            json=payload,
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        else:
            print(f"{Colors.RED}V1 API error: {response.status_code} - {response.text[:200]}{Colors.NC}")
            return None
    except requests.RequestException as e:
        print(f"{Colors.RED}V1 API connection error: {e}{Colors.NC}")
        return None


def call_v25_api(property_data: dict) -> Optional[dict]:
    """Call V2.5 API on port 8001."""
    # V2.5 uses /api/v1/predict
    
    # Get sqft values from CSV data
    sqft_living = property_data.get("sqft_living", 0)
    sqft_basement = property_data.get("sqft_basement", 0) or 0  # Default to 0 if not found
    sqft_above = property_data.get("sqft_above", 0) or 0
    
    # Fallback: if sqft_above not in data, calculate as sqft_living (treat all as above ground)
    if sqft_above == 0:
        sqft_above = sqft_living
    
    payload = {
        "bedrooms": property_data.get("bedrooms", 3),
        "bathrooms": property_data.get("bathrooms", 1.0),
        "sqft_living": sqft_living,
        "sqft_lot": property_data.get("sqft_lot", 5000),
        "floors": property_data.get("floors", 1.0),
        "waterfront": property_data.get("waterfront", 0),
        "view": property_data.get("view", 0),
        "condition": property_data.get("condition", 3),
        "grade": property_data.get("grade", 7),
        "sqft_above": sqft_above,
        "sqft_basement": sqft_basement,
        "yr_built": property_data.get("yr_built", 1990),
        "yr_renovated": property_data.get("yr_renovated", 0) or 0,
        "zipcode": str(property_data.get("zipcode", "98101")),
        "lat": property_data.get("lat", 47.6),
        "long": property_data.get("long", -122.3),
        "sqft_living15": property_data.get("sqft_living15", sqft_living),
        "sqft_lot15": property_data.get("sqft_lot15", 5000),
    }
    
    try:
        response = requests.post(
            "http://localhost:8001/api/v1/predict",
            json=payload,
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        else:
            print(f"{Colors.RED}V2.5 API error: {response.status_code} - {response.text[:200]}{Colors.NC}")
            return None
    except requests.RequestException as e:
        print(f"{Colors.RED}V2.5 API connection error: {e}{Colors.NC}")
        return None


def call_v33_api(property_data: dict) -> Optional[dict]:
    """Call V3.3 API on port 8002."""
    # V3.3 uses /api/v1/predict
    
    # Get sqft values from CSV data
    sqft_living = property_data.get("sqft_living", 0)
    sqft_basement = property_data.get("sqft_basement", 0) or 0  # Default to 0 if not found
    sqft_above = property_data.get("sqft_above", 0) or 0
    
    # Fallback: if sqft_above not in data, calculate as sqft_living (treat all as above ground)
    if sqft_above == 0:
        sqft_above = sqft_living
    
    payload = {
        "bedrooms": property_data.get("bedrooms", 3),
        "bathrooms": property_data.get("bathrooms", 1.0),
        "sqft_living": sqft_living,
        "sqft_lot": property_data.get("sqft_lot", 5000),
        "floors": property_data.get("floors", 1.0),
        "waterfront": property_data.get("waterfront", 0),
        "view": property_data.get("view", 0),
        "condition": property_data.get("condition", 3),
        "grade": property_data.get("grade", 7),
        "sqft_above": sqft_above,
        "sqft_basement": sqft_basement,
        "yr_built": property_data.get("yr_built", 1990),
        "yr_renovated": property_data.get("yr_renovated", 0) or 0,
        "zipcode": str(property_data.get("zipcode", "98101")),
        "lat": property_data.get("lat", 47.6),
        "long": property_data.get("long", -122.3),
        "sqft_living15": property_data.get("sqft_living15", sqft_living),
        "sqft_lot15": property_data.get("sqft_lot15", 5000),
    }
    
    try:
        response = requests.post(
            "http://localhost:8002/api/v1/predict",
            json=payload,
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        else:
            print(f"{Colors.RED}V3.3 API error: {response.status_code} - {response.text[:200]}{Colors.NC}")
            return None
    except requests.RequestException as e:
        print(f"{Colors.RED}V3.3 API connection error: {e}{Colors.NC}")
        return None


def get_neighborhood_info(zipcode: str) -> Optional[dict]:
    """Get neighborhood demographic info from V3.3 API."""
    try:
        # Get demographics from V3.3 health endpoint or feature service
        response = requests.get("http://localhost:8002/api/v1/health", timeout=10)
        if response.status_code == 200:
            health = response.json()
            return {
                "demographics_loaded": health.get("demographics_loaded", False),
                "data_vintage": health.get("data_vintage", "Unknown"),
            }
    except:
        pass
    return None


def print_property_details(property_data: dict, address: str):
    """Print property details in a nice format."""
    print_section("Property Details", Colors.CYAN)
    
    print(f"  {Colors.BOLD}Address:{Colors.NC} {address}")
    print(f"  {Colors.BOLD}PIN:{Colors.NC} {property_data.get('_pin', 'N/A')}")
    print(f"  {Colors.BOLD}Matched Address:{Colors.NC} {property_data.get('_matched_address', 'N/A')}")
    print(f"  {Colors.BOLD}Geocode Score:{Colors.NC} {property_data.get('_geocode_score', 0):.0f}%")
    print()
    
    print(f"  {Colors.YELLOW}Bedrooms:{Colors.NC}    {property_data.get('bedrooms', 'N/A')}")
    print(f"  {Colors.YELLOW}Bathrooms:{Colors.NC}   {property_data.get('bathrooms', 'N/A')}")
    print(f"  {Colors.YELLOW}Sqft Living:{Colors.NC} {property_data.get('sqft_living', 'N/A'):,}")
    print(f"  {Colors.YELLOW}Sqft Lot:{Colors.NC}    {property_data.get('sqft_lot', 0):,}")
    print(f"  {Colors.YELLOW}Floors:{Colors.NC}      {property_data.get('floors', 'N/A')}")
    print(f"  {Colors.YELLOW}Grade:{Colors.NC}       {property_data.get('grade', 'N/A')}")
    print(f"  {Colors.YELLOW}Condition:{Colors.NC}   {property_data.get('condition', 'N/A')}")
    print(f"  {Colors.YELLOW}Year Built:{Colors.NC}  {property_data.get('yr_built', 'N/A')}")
    print(f"  {Colors.YELLOW}Zipcode:{Colors.NC}     {property_data.get('zipcode', 'N/A')}")
    print(f"  {Colors.YELLOW}Lat/Long:{Colors.NC}    {property_data.get('lat', 'N/A')}, {property_data.get('long', 'N/A')}")


def print_prediction_comparison(v1_result: dict, v25_result: dict, v33_result: dict):
    """Print a comparison of predictions from all three models."""
    print_section("Model Predictions Comparison", Colors.GREEN)
    
    v1_price = v1_result.get("predicted_price", 0) if v1_result else 0
    v25_price = v25_result.get("predicted_price", 0) if v25_result else 0
    v33_price = v33_result.get("predicted_price", 0) if v33_result else 0
    
    # Calculate statistics
    prices = [p for p in [v1_price, v25_price, v33_price] if p > 0]
    avg_price = sum(prices) / len(prices) if prices else 0
    min_price = min(prices) if prices else 0
    max_price = max(prices) if prices else 0
    spread = max_price - min_price if prices else 0
    spread_pct = (spread / avg_price * 100) if avg_price > 0 else 0
    
    # Print table header
    print(f"  {'Model':<12} {'Predicted Price':>18} {'Diff from Avg':>15} {'Data Vintage':>15}")
    print(f"  {'-' * 12} {'-' * 18} {'-' * 15} {'-' * 15}")
    
    # V1 MVP
    if v1_result:
        diff = v1_price - avg_price
        diff_pct = (diff / avg_price * 100) if avg_price > 0 else 0
        diff_str = f"{'+' if diff >= 0 else ''}{diff_pct:.1f}%"
        vintage = v1_result.get("data_vintage", "2014-2015")
        print(f"  {Colors.CYAN}V1 MVP{Colors.NC}      {Colors.BOLD}${v1_price:>15,.2f}{Colors.NC} {diff_str:>15} {vintage:>15}")
    else:
        print(f"  {Colors.CYAN}V1 MVP{Colors.NC}      {Colors.RED}{'FAILED':>18}{Colors.NC}")
    
    # V2.5
    if v25_result:
        diff = v25_price - avg_price
        diff_pct = (diff / avg_price * 100) if avg_price > 0 else 0
        diff_str = f"{'+' if diff >= 0 else ''}{diff_pct:.1f}%"
        vintage = v25_result.get("data_vintage", "2014-2015")
        print(f"  {Colors.GREEN}V2.5{Colors.NC}        {Colors.BOLD}${v25_price:>15,.2f}{Colors.NC} {diff_str:>15} {vintage:>15}")
    else:
        print(f"  {Colors.GREEN}V2.5{Colors.NC}        {Colors.RED}{'FAILED':>18}{Colors.NC}")
    
    # V3.3
    if v33_result:
        diff = v33_price - avg_price
        diff_pct = (diff / avg_price * 100) if avg_price > 0 else 0
        diff_str = f"{'+' if diff >= 0 else ''}{diff_pct:.1f}%"
        vintage = v33_result.get("data_vintage", "2020-2024")
        print(f"  {Colors.MAGENTA}V3.3{Colors.NC}        {Colors.BOLD}${v33_price:>15,.2f}{Colors.NC} {diff_str:>15} {vintage:>15}")
    else:
        print(f"  {Colors.MAGENTA}V3.3{Colors.NC}        {Colors.RED}{'FAILED':>18}{Colors.NC}")
    
    print()
    print(f"  {Colors.BOLD}Average:{Colors.NC}     ${avg_price:>15,.2f}")
    print(f"  {Colors.BOLD}Range:{Colors.NC}       ${min_price:,.0f} - ${max_price:,.0f} (spread: ${spread:,.0f}, {spread_pct:.1f}%)")


def print_model_notes(v1_result: dict, v25_result: dict, v33_result: dict):
    """Print notes about each model's methodology."""
    print_section("Model Notes", Colors.YELLOW)
    
    print(f"  {Colors.CYAN}V1 MVP (2014-2015 data):{Colors.NC}")
    print(f"    - Uses 7 home features + 26 demographic features")
    print(f"    - Original training data from Kaggle King County dataset")
    if v1_result:
        note = v1_result.get("confidence_note", "")
        if note:
            print(f"    - Note: {note[:80]}...")
    
    print()
    print(f"  {Colors.GREEN}V2.5 (2014-2015 data):{Colors.NC}")
    print(f"    - Uses 17 home features + 26 demographic features")
    print(f"    - Added experimental tier-based adaptive routing")
    if v25_result:
        note = v25_result.get("confidence_note", "")
        if note:
            print(f"    - Note: {note[:80]}...")
    
    print()
    print(f"  {Colors.MAGENTA}V3.3 (2020-2024 data):{Colors.NC}")
    print(f"    - Uses 17 home features + 26 demographic features")
    print(f"    - Fresh training data, MLflow integration")
    print(f"    - {Colors.BOLD}Recommended for current market estimates{Colors.NC}")
    if v33_result:
        note = v33_result.get("confidence_note", "")
        if note:
            print(f"    - Note: {note[:80]}...")


def print_neighborhood_notes(property_data: dict):
    """Print neighborhood context notes."""
    print_section("Neighborhood Context", Colors.BLUE)
    
    zipcode = property_data.get("zipcode", "Unknown")
    print(f"  {Colors.BOLD}Zipcode:{Colors.NC} {zipcode}")
    
    # Zipcode-specific notes for known areas
    zipcode_notes = {
        "98101": "Downtown Seattle - Urban core, mixed commercial/residential",
        "98102": "Capitol Hill/Eastlake - Dense urban, walkable, transit-rich",
        "98103": "Fremont/Wallingford - Urban residential, family-friendly",
        "98104": "Pioneer Square/ID - Historic district, urban core",
        "98105": "University District - Near UW, student housing prevalent",
        "98106": "White Center/Highland Park - Diverse, affordable options",
        "98107": "Ballard - Trendy, breweries, Nordic heritage",
        "98108": "South Seattle - Industrial transition, emerging",
        "98109": "South Lake Union - Tech hub, Amazon HQ, rapid growth",
        "98112": "Madison Park/Montlake - Affluent, waterfront access",
        "98115": "Wedgwood/View Ridge - Quiet residential, families",
        "98116": "West Seattle/Alki - Beach community, views",
        "98117": "Ballard/Crown Hill - Residential, good schools",
        "98118": "Columbia City/Rainier Valley - Diverse, light rail",
        "98119": "Queen Anne - Views, urban village, steep hills",
        "98122": "Central District - Historic Black neighborhood, gentrifying",
        "98125": "Lake City/Northgate - Suburban feel, affordable",
        "98126": "West Seattle/Delridge - Mix of housing types",
        "98133": "Shoreline/Bitter Lake - North Seattle, suburban",
        "98136": "West Seattle/Fauntleroy - Ferry access, views",
        "98144": "Beacon Hill/Mt Baker - Light rail, diverse, views",
        "98199": "Magnolia - Quiet, Discovery Park, family-oriented",
        "98004": "Downtown Bellevue - Urban, high-rise, tech jobs",
        "98005": "Bellevue (SE) - Suburban, good schools",
        "98006": "Bellevue (SW) - Affluent, waterfront, Mercer Island access",
        "98007": "Bellevue (Crossroads) - Diverse, suburban",
        "98008": "Bellevue (Lake Hills) - Family-oriented, parks",
        "98033": "Kirkland - Waterfront, walkable downtown",
        "98034": "Kirkland (Totem Lake) - Suburban, retail",
        "98052": "Redmond - Microsoft HQ, tech hub, trails",
        "98053": "Redmond (Education Hill) - Schools, newer homes",
    }
    
    if zipcode in zipcode_notes:
        print(f"  {Colors.YELLOW}Area:{Colors.NC} {zipcode_notes[zipcode]}")
    else:
        print(f"  {Colors.YELLOW}Area:{Colors.NC} King County residential area")
    
    # Add coordinate-based notes
    lat = property_data.get("lat", 0)
    long = property_data.get("long", 0)
    
    if lat and long:
        # General location
        if lat > 47.7:
            print(f"  {Colors.YELLOW}Location:{Colors.NC} North King County")
        elif lat < 47.5:
            print(f"  {Colors.YELLOW}Location:{Colors.NC} South King County")
        else:
            print(f"  {Colors.YELLOW}Location:{Colors.NC} Central King County (Seattle area)")
        
        # Waterfront proximity (rough)
        if long > -122.25:
            print(f"  {Colors.YELLOW}Note:{Colors.NC} East side (Bellevue/Kirkland area)")
        elif long < -122.4:
            print(f"  {Colors.YELLOW}Note:{Colors.NC} West side (near Puget Sound)")


def check_services():
    """Check if all Docker services are running."""
    print_section("Checking Services", Colors.YELLOW)
    
    services = [
        ("V1 MVP", "http://localhost:8000/health"),
        ("V2.5", "http://localhost:8001/api/v1/health"),
        ("V3.3", "http://localhost:8002/api/v1/health"),
    ]
    
    all_healthy = True
    for name, url in services:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"  {Colors.GREEN}[OK]{Colors.NC} {name} is running")
            else:
                print(f"  {Colors.RED}[FAIL]{Colors.NC} {name} returned {response.status_code}")
                all_healthy = False
        except requests.RequestException:
            print(f"  {Colors.RED}[FAIL]{Colors.NC} {name} not reachable")
            all_healthy = False
    
    return all_healthy


def main():
    parser = argparse.ArgumentParser(
        description="Compare model predictions for a King County address",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/compare_by_address.py "1523 15th Ave S" Seattle 98144
    python scripts/compare_by_address.py "119 NW 41st St" Seattle 98107
    python scripts/compare_by_address.py --interactive
        """
    )
    parser.add_argument("street", nargs="?", help="Street address (e.g., '1523 15th Ave S')")
    parser.add_argument("city", nargs="?", default="Seattle", help="City (default: Seattle)")
    parser.add_argument("zipcode", nargs="?", help="ZIP code (e.g., 98144)")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    
    args = parser.parse_args()
    
    print_header("Real Estate Price Comparison by Address", Colors.BLUE)
    
    # Check services first
    if not check_services():
        print(f"\n{Colors.RED}Error: Not all services are running.{Colors.NC}")
        print(f"Please start Docker containers with: docker-compose -f docker-compose.demo.yml up -d")
        sys.exit(1)
    
    # Get address input
    if args.interactive or not args.street:
        print_section("Enter Address", Colors.CYAN)
        street = input("  Street address: ").strip()
        city = input("  City [Seattle]: ").strip() or "Seattle"
        zipcode = input("  ZIP code: ").strip()
    else:
        street = args.street
        city = args.city
        zipcode = args.zipcode
    
    if not street or not zipcode:
        print(f"{Colors.RED}Error: Street address and ZIP code are required{Colors.NC}")
        sys.exit(1)
    
    full_address = f"{street}, {city}, WA {zipcode}"
    
    # Look up property
    print_header(f"Analyzing: {full_address}", Colors.CYAN)
    
    property_data = lookup_property(street, city, zipcode)
    
    if not property_data:
        print(f"\n{Colors.RED}Could not find property. Please check the address.{Colors.NC}")
        print(f"Tips:")
        print(f"  - Ensure the address is in King County, WA")
        print(f"  - Use the format: '1523 15th Ave S' (include directional suffix)")
        print(f"  - The property must be residential (not commercial)")
        sys.exit(1)
    
    # Print property details
    print_property_details(property_data, full_address)
    
    # Call all three models
    print_section("Calling Models...", Colors.YELLOW)
    
    print(f"  Calling V1 MVP (port 8000)...")
    v1_result = call_v1_api(property_data)
    
    print(f"  Calling V2.5 (port 8001)...")
    v25_result = call_v25_api(property_data)
    
    print(f"  Calling V3.3 (port 8002)...")
    v33_result = call_v33_api(property_data)
    
    # Print comparison
    print_prediction_comparison(v1_result, v25_result, v33_result)
    
    # Print model notes
    print_model_notes(v1_result, v25_result, v33_result)
    
    # Print neighborhood context
    print_neighborhood_notes(property_data)
    
    print_header("Analysis Complete", Colors.GREEN)
    
    # Summary
    prices = []
    if v1_result:
        prices.append(("V1", v1_result.get("predicted_price", 0)))
    if v25_result:
        prices.append(("V2.5", v25_result.get("predicted_price", 0)))
    if v33_result:
        prices.append(("V3.3", v33_result.get("predicted_price", 0)))
    
    if prices:
        avg = sum(p[1] for p in prices) / len(prices)
        print(f"  {Colors.BOLD}Recommended estimate (V3.3):{Colors.NC} ${v33_result.get('predicted_price', 0):,.2f}" if v33_result else "")
        print(f"  {Colors.BOLD}Average across models:{Colors.NC} ${avg:,.2f}")
        print()


if __name__ == "__main__":
    main()
