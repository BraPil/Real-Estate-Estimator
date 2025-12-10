#!/usr/bin/env python3
"""
Real Estate Estimator - Address-Based Version Comparison
=========================================================
Look up a King County property by address and compare predictions
from all three model versions (V1, V2.5, V3.3).

Usage:
    python scripts/compare_by_address.py "1523 15th Ave S" Seattle 98144
    python scripts/compare_by_address.py --interactive

Requirements:
    - Docker containers running on ports 8000 (V1), 8001 (V2.5), 8002 (V3.3)
    - King County Assessor CSV data (auto-decompresses on first use)
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.address_service_v2 import AddressServiceV2
from src.config import get_settings


# ANSI Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    MAGENTA = '\033[35m'
    BOLD = '\033[1m'
    END = '\033[0m'


# Version configurations
VERSIONS = [
    {
        "name": "V1",
        "label": "1.0.0",
        "port": 8000,
        "predict_url": "http://localhost:8000/predict",
        "health_url": "http://localhost:8000/health",
        "algorithm": "KNN (k=5)",
        "data_vintage": "2014-2015",
        "features": 33,
        "description": "MVP - Simple, fast, interpretable",
        "color": Colors.CYAN
    },
    {
        "name": "V2.5",
        "label": "v2.5",
        "port": 8001,
        "predict_url": "http://localhost:8001/api/v1/predict",
        "health_url": "http://localhost:8001/api/v1/health",
        "algorithm": "XGBoost + RandomizedSearchCV",
        "data_vintage": "2014-2015",
        "features": 43,
        "description": "Optimized - Better accuracy, same data",
        "color": Colors.GREEN
    },
    {
        "name": "V3.3",
        "label": "v3.3",
        "port": 8002,
        "predict_url": "http://localhost:8002/api/v1/predict",
        "health_url": "http://localhost:8002/api/v1/health",
        "algorithm": "XGBoost + Optuna (100 trials)",
        "data_vintage": "2020-2024",
        "features": 47,
        "description": "Production - Current market data",
        "color": Colors.MAGENTA
    }
]

# Zipcode neighborhood descriptions
ZIPCODE_INFO = {
    "98101": "Downtown Seattle - Urban core, mixed commercial/residential",
    "98102": "Capitol Hill/Eastlake - Dense urban, walkable, transit-rich",
    "98103": "Fremont/Wallingford - Urban residential, family-friendly",
    "98104": "Pioneer Square/ID - Historic district, urban core",
    "98105": "University District - Near UW, student housing",
    "98106": "White Center/Highland Park - Diverse, affordable",
    "98107": "Ballard - Trendy, breweries, Nordic heritage",
    "98108": "South Seattle - Industrial transition, emerging",
    "98109": "South Lake Union - Tech hub, Amazon HQ, rapid growth",
    "98112": "Madison Park/Montlake - Affluent, waterfront access",
    "98115": "Wedgwood/View Ridge - Quiet residential, families",
    "98116": "West Seattle/Alki - Beach community, views",
    "98117": "Ballard/Crown Hill - Residential, good schools",
    "98118": "Columbia City/Rainier Valley - Diverse, light rail",
    "98119": "Queen Anne - Views, urban village, steep hills",
    "98122": "Central District - Historic neighborhood, gentrifying",
    "98125": "Lake City/Northgate - Suburban feel, affordable",
    "98126": "West Seattle/Delridge - Mix of housing types",
    "98133": "Shoreline/Bitter Lake - North Seattle, suburban",
    "98136": "West Seattle/Fauntleroy - Ferry access, views",
    "98144": "Beacon Hill/Mt Baker - Light rail, diverse, views",
    "98199": "Magnolia - Quiet, Discovery Park, family-oriented",
    "98004": "Downtown Bellevue - Urban, high-rise, tech jobs",
    "98005": "Bellevue (SE) - Suburban, good schools",
    "98006": "Bellevue (SW) - Affluent, waterfront",
    "98007": "Bellevue (Crossroads) - Diverse, suburban",
    "98008": "Bellevue (Lake Hills) - Family-oriented, parks",
    "98033": "Kirkland - Waterfront, walkable downtown",
    "98034": "Kirkland (Totem Lake) - Suburban, retail",
    "98052": "Redmond - Microsoft HQ, tech hub, trails",
    "98053": "Redmond (Education Hill) - Schools, newer homes",
}


def print_header(text: str):
    """Print a main header with equals signs."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(80)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 80}{Colors.END}")


def print_section(text: str):
    """Print a section header with dashes."""
    print(f"\n{Colors.CYAN}{'-' * 80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text}{Colors.END}")
    print(f"{Colors.CYAN}{'-' * 80}{Colors.END}")


def check_health(version: dict) -> tuple[bool, Optional[dict]]:
    """Check if a version's API is healthy."""
    try:
        r = requests.get(version["health_url"], timeout=5)
        if r.status_code == 200:
            return True, r.json()
    except Exception:
        pass
    return False, None


def get_prediction(version: dict, property_data: dict) -> tuple[bool, Optional[dict]]:
    """Get a prediction from a version's API."""
    sqft_living = property_data.get("sqft_living", 0)
    sqft_basement = property_data.get("sqft_basement", 0) or 0
    sqft_above = property_data.get("sqft_above", 0) or 0
    
    # Fallback: if sqft_above not in data, use sqft_living
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
        r = requests.post(version["predict_url"], json=payload, timeout=10)
        if r.status_code == 200:
            return True, r.json()
        return False, {"error": f"HTTP {r.status_code}"}
    except Exception as e:
        return False, {"error": str(e)}


def lookup_property(street: str, city: str, zipcode: str) -> Optional[dict]:
    """Look up property data from King County Assessor records."""
    address = f"{street}, {city}, WA {zipcode}"
    
    try:
        svc = AddressServiceV2(get_settings())
        result = svc.lookup_address(address)
        return result
    except Exception as e:
        print(f"{Colors.RED}Error looking up property: {e}{Colors.END}")
        return None


def get_location_description(lat: float, long: float) -> str:
    """Get location description based on coordinates."""
    if lat > 47.7:
        return "North King County"
    elif lat < 47.5:
        return "South King County"
    else:
        if long > -122.25:
            return "Eastside (Bellevue/Kirkland area)"
        elif long < -122.4:
            return "West Seattle / Puget Sound area"
        else:
            return "Central Seattle area"


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
    
    # Main header
    print_header("REAL ESTATE ESTIMATOR - ADDRESS LOOKUP")
    print(f"\n{Colors.YELLOW}Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.END}")
    
    # Health check
    print_section("SERVICE HEALTH CHECK")
    all_healthy = True
    for v in VERSIONS:
        healthy, _ = check_health(v)
        status = f"{Colors.GREEN}HEALTHY{Colors.END}" if healthy else f"{Colors.RED}NOT RUNNING{Colors.END}"
        print(f"  {v['name']:6} (Port {v['port']}): {status}")
        if not healthy:
            all_healthy = False
    
    if not all_healthy:
        print(f"\n{Colors.RED}ERROR: Not all services are running.{Colors.END}")
        print("Start them with: docker compose -f docker-compose.demo.yml up -d --build")
        sys.exit(1)
    
    # Get address input
    if args.interactive or not args.street:
        print_section("ENTER ADDRESS")
        street = input("  Street address: ").strip()
        city = input("  City [Seattle]: ").strip() or "Seattle"
        zipcode = input("  ZIP code: ").strip()
    else:
        street = args.street
        city = args.city
        zipcode = args.zipcode
    
    if not street or not zipcode:
        print(f"{Colors.RED}Error: Street address and ZIP code are required{Colors.END}")
        sys.exit(1)
    
    full_address = f"{street}, {city}, WA {zipcode}"
    
    # Property lookup
    print_section("PROPERTY LOOKUP")
    print(f"\n    {Colors.BOLD}Searching:{Colors.END} {full_address}")
    
    property_data = lookup_property(street, city, zipcode)
    
    if not property_data:
        print(f"\n{Colors.RED}Could not find property. Please check the address.{Colors.END}")
        print(f"\n    Tips:")
        print(f"    - Ensure the address is in King County, WA")
        print(f"    - Use the format: '1523 15th Ave S' (include directional suffix)")
        print(f"    - The property must be residential (not commercial)")
        sys.exit(1)
    
    # Display property details
    print(f"\n    {Colors.GREEN}Found property!{Colors.END}")
    print(f"\n    {Colors.BOLD}PIN:{Colors.END}             {property_data.get('_pin', 'N/A')}")
    print(f"    {Colors.BOLD}Matched Address:{Colors.END}  {property_data.get('_matched_address', 'N/A')}")
    print(f"    {Colors.BOLD}Geocode Score:{Colors.END}    {property_data.get('_geocode_score', 0):.0f}%")
    
    print_section("PROPERTY DETAILS")
    sqft_living = property_data.get('sqft_living', 0)
    sqft_lot = property_data.get('sqft_lot', 0)
    sqft_basement = property_data.get('sqft_basement', 0) or 0
    
    print(f"""
    {Colors.BOLD}Location:{Colors.END}      Zipcode {property_data.get('zipcode', 'N/A')} ({ZIPCODE_INFO.get(str(property_data.get('zipcode', '')), 'King County')})
    {Colors.BOLD}Size:{Colors.END}          {sqft_living:,} sqft living space
    {Colors.BOLD}Lot:{Colors.END}           {sqft_lot:,} sqft
    {Colors.BOLD}Layout:{Colors.END}        {property_data.get('bedrooms', 'N/A')} bedrooms, {property_data.get('bathrooms', 'N/A')} bathrooms, {property_data.get('floors', 'N/A')} floors
    {Colors.BOLD}Year Built:{Colors.END}    {property_data.get('yr_built', 'N/A')}
    {Colors.BOLD}Condition:{Colors.END}     {property_data.get('condition', 'N/A')} (1=Poor, 3=Average, 5=Excellent)
    {Colors.BOLD}Grade:{Colors.END}         {property_data.get('grade', 'N/A')} (1-13 scale, 7=Average, 10+=Luxury)
    {Colors.BOLD}Basement:{Colors.END}      {sqft_basement:,} sqft
    {Colors.BOLD}Waterfront:{Colors.END}    {'Yes' if property_data.get('waterfront', 0) else 'No'}
    {Colors.BOLD}View Rating:{Colors.END}   {property_data.get('view', 0)}/4
    {Colors.BOLD}Coordinates:{Colors.END}   {property_data.get('lat', 'N/A')}, {property_data.get('long', 'N/A')}
    """)
    
    # Get predictions from all versions
    print_section("PREDICTIONS")
    results = []
    for v in VERSIONS:
        success, data = get_prediction(v, property_data)
        if success:
            price = data.get("predicted_price", 0)
            model_ver = data.get("model_version", v["label"])
            results.append({
                "version": v,
                "price": price,
                "model_version": model_ver,
                "response": data
            })
            color = v["color"]
            print(f"\n  {Colors.BOLD}{color}{v['name']} - {v['algorithm']}{Colors.END}")
            print(f"  {'-' * 50}")
            print(f"  Model Version:    {model_ver}")
            print(f"  Training Data:    {v['data_vintage']}")
            print(f"  Features Used:    {v['features']}")
            print(f"  {Colors.GREEN}{Colors.BOLD}Predicted Price:  ${price:,.0f}{Colors.END}")
        else:
            print(f"\n  {Colors.RED}{v['name']}: FAILED - {data.get('error', 'Unknown error')}{Colors.END}")
            results.append({"version": v, "price": 0, "error": True})
    
    # Comparison table
    print_section("COMPARISON SUMMARY")
    print(f"""
    {Colors.BOLD}{'Version':<10} {'Algorithm':<35} {'Data':<12} {'Prediction':>15}{Colors.END}
    {'-' * 75}""")
    for r in results:
        v = r["version"]
        price = r.get("price", 0)
        print(f"    {v['label']:<10} {v['algorithm']:<35} {v['data_vintage']:<12} ${price:>14,.0f}")
    
    # Price evolution analysis
    if len(results) == 3 and all(r.get("price", 0) > 0 for r in results):
        v1_price = results[0]["price"]
        v2_price = results[1]["price"]
        v3_price = results[2]["price"]
        
        print_section("PRICE EVOLUTION ANALYSIS")
        print(f"""
    {Colors.BOLD}V1 -> V2.5:{Colors.END}  ${v2_price - v1_price:>+12,.0f}  ({(v2_price - v1_price) / v1_price * 100:+.1f}%)
                 {Colors.CYAN}Better algorithm (XGBoost vs KNN) on same 2014-2015 data{Colors.END}

    {Colors.BOLD}V2.5 -> V3.3:{Colors.END} ${v3_price - v2_price:>+12,.0f}  ({(v3_price - v2_price) / v2_price * 100:+.1f}%)
                 {Colors.CYAN}Fresh 2020-2024 data reflects Seattle's housing boom{Colors.END}

    {Colors.BOLD}V1 -> V3.3:{Colors.END}  ${v3_price - v1_price:>+12,.0f}  ({(v3_price - v1_price) / v1_price * 100:+.1f}%)
                 {Colors.CYAN}Combined effect of algorithm + market appreciation{Colors.END}
        """)
        
        # Price per sqft analysis
        print_section("PRICE PER SQUARE FOOT")
        print(f"""
    {Colors.BOLD}V1:  {Colors.END}  ${v1_price / sqft_living:,.0f}/sqft  (2014-2015 Seattle market)
    {Colors.BOLD}V2.5:{Colors.END}  ${v2_price / sqft_living:,.0f}/sqft  (2014-2015 Seattle market, better model)
    {Colors.BOLD}V3.3:{Colors.END}  ${v3_price / sqft_living:,.0f}/sqft  (2020-2024 Seattle market)
    
    {Colors.YELLOW}Note: Seattle median home price rose ~80% from 2015 to 2024.{Colors.END}
    {Colors.YELLOW}V3.3's prediction reflects current market conditions.{Colors.END}
        """)
    
    # Neighborhood context
    print_section("NEIGHBORHOOD CONTEXT")
    zipcode_str = str(property_data.get('zipcode', ''))
    lat = property_data.get('lat', 0)
    long = property_data.get('long', 0)
    
    print(f"""
    {Colors.BOLD}Zipcode:{Colors.END}   {zipcode_str}
    {Colors.BOLD}Area:{Colors.END}      {ZIPCODE_INFO.get(zipcode_str, 'King County residential area')}
    {Colors.BOLD}Location:{Colors.END}  {get_location_description(lat, long)}
    """)
    
    # Version details
    print_section("VERSION DETAILS")
    print(f"""
    {Colors.BOLD}{Colors.CYAN}V1 (MVP){Colors.END}
      - Algorithm: K-Nearest Neighbors (k=5)
      - Features: 7 home + 26 zipcode demographics = 33 total
      - Endpoint: POST /predict (no /api/v1 prefix)
      - Use case: Quick baseline, interpretable results
    
    {Colors.BOLD}{Colors.GREEN}V2.5 (Optimized){Colors.END}
      - Algorithm: XGBoost with RandomizedSearchCV tuning
      - Features: 17 home + 26 demographics = 43 total
      - Endpoint: POST /api/v1/predict
      - Bonus: /api/v1/predict-minimal (11 required fields only)
      - Use case: Better accuracy with same vintage data
    
    {Colors.BOLD}{Colors.MAGENTA}V3.3 (Production) - RECOMMENDED{Colors.END}
      - Algorithm: XGBoost with Optuna hyperparameter tuning (100 trials)
      - Features: 17 home + 26 demographics + 4 temporal = 47 total
      - Endpoint: POST /api/v1/predict
      - Extras: MLflow tracking, model versioning, production hardening
      - Use case: Current market valuations
    """)
    
    # Final summary
    print_header("ANALYSIS COMPLETE")
    
    if len(results) == 3 and all(r.get("price", 0) > 0 for r in results):
        v3_price = results[2]["price"]
        print(f"""
    {Colors.GREEN}{Colors.BOLD}Recommended Estimate (V3.3): ${v3_price:,.0f}{Colors.END}
    
    {Colors.BOLD}Key Takeaways:{Colors.END}
    1. V3.3 uses 2020-2024 data - most relevant for current market
    2. V1 and V2.5 are included for comparison (same 2014-2015 data)
    3. Algorithm improvements (V1->V2.5) add ~5-10% accuracy
    4. Market appreciation (V2.5->V3.3) reflects real price growth
    
    {Colors.CYAN}Compare with Redfin/Zillow to validate the estimate!{Colors.END}
    """)


if __name__ == "__main__":
    main()
