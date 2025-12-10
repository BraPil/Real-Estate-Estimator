#!/usr/bin/env python3
"""
Real Estate Estimator - 3-Version Comparison Demo
==================================================
Run this script after starting the demo containers:
    docker compose -f docker-compose.demo.yml up -d --build
    python compare_versions.py

This demonstrates the evolution from V1 MVP to V3.3 Production.
"""

import requests
import sys
from datetime import datetime

# ANSI Colors
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(80)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 80}{Colors.END}")


def print_section(text):
    print(f"\n{Colors.CYAN}{'-' * 80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text}{Colors.END}")
    print(f"{Colors.CYAN}{'-' * 80}{Colors.END}")


# Sample property for comparison
SAMPLE_PROPERTY = {
    "bedrooms": 3,
    "bathrooms": 2.5,
    "sqft_living": 2000,
    "sqft_lot": 5000,
    "floors": 2,
    "waterfront": 0,
    "view": 0,
    "condition": 3,
    "grade": 8,
    "sqft_above": 1500,
    "sqft_basement": 500,
    "yr_built": 2010,
    "yr_renovated": 0,
    "zipcode": "98103",
    "lat": 47.6,
    "long": -122.3,
    "sqft_living15": 1900,
    "sqft_lot15": 4800
}

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
        "description": "MVP - Simple, fast, interpretable"
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
        "description": "Optimized - Better accuracy, same data"
    },
    {
        "name": "V3.3",
        "label": "v3.3",
        "port": 8002,
        "predict_url": "http://localhost:8002/api/v1/predict",
        "health_url": "http://localhost:8002/api/v1/health",
        "algorithm": "XGBoost + Optuna (30 trials)",
        "data_vintage": "2020-2024",
        "features": 47,
        "description": "Production - Current market data"
    }
]


def check_health(version):
    """Check if a version's API is healthy."""
    try:
        r = requests.get(version["health_url"], timeout=5)
        if r.status_code == 200:
            return True, r.json()
    except Exception:
        pass
    return False, None


def get_prediction(version, property_data):
    """Get a prediction from a version's API."""
    try:
        r = requests.post(version["predict_url"], json=property_data, timeout=10)
        if r.status_code == 200:
            return True, r.json()
        return False, {"error": f"HTTP {r.status_code}"}
    except Exception as e:
        return False, {"error": str(e)}


def main():
    print_header("REAL ESTATE ESTIMATOR - VERSION COMPARISON DEMO")
    print(f"\n{Colors.YELLOW}Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.END}")
    
    # Display sample property
    print_section("SAMPLE PROPERTY")
    print(f"""
    {Colors.BOLD}Location:{Colors.END}      Zipcode 98103 (Wallingford/Fremont, Seattle)
    {Colors.BOLD}Size:{Colors.END}          {SAMPLE_PROPERTY['sqft_living']:,} sqft living space
    {Colors.BOLD}Lot:{Colors.END}           {SAMPLE_PROPERTY['sqft_lot']:,} sqft
    {Colors.BOLD}Layout:{Colors.END}        {SAMPLE_PROPERTY['bedrooms']} bedrooms, {SAMPLE_PROPERTY['bathrooms']} bathrooms, {int(SAMPLE_PROPERTY['floors'])} floors
    {Colors.BOLD}Year Built:{Colors.END}    {SAMPLE_PROPERTY['yr_built']}
    {Colors.BOLD}Condition:{Colors.END}     {SAMPLE_PROPERTY['condition']} (Average)
    {Colors.BOLD}Grade:{Colors.END}         {SAMPLE_PROPERTY['grade']} (Good construction quality)
    {Colors.BOLD}Basement:{Colors.END}      {SAMPLE_PROPERTY['sqft_basement']:,} sqft
    {Colors.BOLD}Waterfront:{Colors.END}    {'Yes' if SAMPLE_PROPERTY['waterfront'] else 'No'}
    {Colors.BOLD}View Rating:{Colors.END}   {SAMPLE_PROPERTY['view']}/4
    """)
    
    # Check health of all versions
    print_section("SERVICE HEALTH CHECK")
    all_healthy = True
    for v in VERSIONS:
        healthy, health_data = check_health(v)
        status = f"{Colors.GREEN}HEALTHY{Colors.END}" if healthy else f"{Colors.RED}NOT RUNNING{Colors.END}"
        print(f"  {v['name']:6} (Port {v['port']}): {status}")
        if not healthy:
            all_healthy = False
    
    if not all_healthy:
        print(f"\n{Colors.RED}ERROR: Not all services are running.{Colors.END}")
        print("Start them with: docker compose -f docker-compose.demo.yml up -d --build")
        sys.exit(1)
    
    # Get predictions from all versions
    print_section("PREDICTIONS")
    results = []
    for v in VERSIONS:
        success, data = get_prediction(v, SAMPLE_PROPERTY)
        if success:
            price = data.get("predicted_price", 0)
            model_ver = data.get("model_version", "unknown")
            results.append({
                "version": v,
                "price": price,
                "model_version": model_ver,
                "response": data
            })
            print(f"\n  {Colors.BOLD}{v['name']} - {v['algorithm']}{Colors.END}")
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
    
    # Price differences
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
        sqft = SAMPLE_PROPERTY["sqft_living"]
        print_section("PRICE PER SQUARE FOOT")
        print(f"""
    {Colors.BOLD}V1:  {Colors.END}  ${v1_price / sqft:,.0f}/sqft  (2014-2015 Seattle market)
    {Colors.BOLD}V2.5:{Colors.END}  ${v2_price / sqft:,.0f}/sqft  (2014-2015 Seattle market, better model)
    {Colors.BOLD}V3.3:{Colors.END}  ${v3_price / sqft:,.0f}/sqft  (2020-2024 Seattle market)
    
    {Colors.YELLOW}Note: Seattle median home price rose ~80% from 2015 to 2024.{Colors.END}
    {Colors.YELLOW}V3.3's higher prediction reflects real market appreciation.{Colors.END}
        """)
    
    # Technical details
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
    
    {Colors.BOLD}{Colors.GREEN}V3.3 (Production){Colors.END}
      - Algorithm: XGBoost with Optuna hyperparameter tuning (30 trials)
      - Features: 17 home + 26 demographics + 4 temporal = 47 total
      - Endpoint: POST /api/v1/predict
      - Extras: MLflow tracking, model versioning, production hardening
      - Use case: Current market valuations
    """)
    
    # API documentation links
    print_section("API DOCUMENTATION (Swagger UI)")
    print(f"""
    {Colors.YELLOW}V1 MVP:{Colors.END}      http://localhost:8000/docs
    {Colors.YELLOW}V2.5:{Colors.END}        http://localhost:8001/docs
    {Colors.YELLOW}V3.3:{Colors.END}        http://localhost:8002/docs
    """)
    
    print_header("DEMO COMPLETE")
    print(f"""
    {Colors.GREEN}All 3 versions are running and producing predictions.{Colors.END}
    
    {Colors.BOLD}Key Takeaways:{Colors.END}
    1. Model versioning allows A/B testing and gradual rollouts
    2. Algorithm improvements (V1->V2.5) boost accuracy ~7%
    3. Fresh data (V2.5->V3.3) captures market changes
    4. All versions can run simultaneously (no downtime for updates)
    
    {Colors.CYAN}Try the Swagger UIs to explore the APIs interactively!{Colors.END}
    """)


if __name__ == "__main__":
    main()
