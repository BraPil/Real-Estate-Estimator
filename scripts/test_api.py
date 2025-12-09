"""
Test script for the Real Estate Price Predictor API.

This script demonstrates the API functionality by:
1. Checking the health endpoint
2. Sending prediction requests from future_unseen_examples.csv
3. Testing the minimal prediction endpoint (BONUS)
4. Displaying results

Usage:
    # First, start the API server:
    uvicorn src.main:app --host 0.0.0.0 --port 8000

    # Then run this script:
    python scripts/test_api.py

    # Or with a custom API URL:
    python scripts/test_api.py --url http://localhost:8000

Requirements:
    - API server must be running
    - httpx or requests library installed
    - data/future_unseen_examples.csv must exist
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Try httpx first, fall back to requests
try:
    import httpx

    HTTP_CLIENT = "httpx"
except ImportError:
    try:
        import requests

        HTTP_CLIENT = "requests"
    except ImportError:
        print("[ERROR] Neither httpx nor requests is installed.")
        print("Install with: pip install httpx")
        sys.exit(1)


# Default configuration
DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_DATA_PATH = "data/future_unseen_examples.csv"
NUM_EXAMPLES = 5  # Number of examples to test


def make_request(url: str, method: str = "GET", json_data: dict = None) -> dict:
    """Make an HTTP request using available client.

    Args:
        url: Full URL to request
        method: HTTP method (GET or POST)
        json_data: JSON data for POST requests

    Returns:
        Response as dictionary
    """
    if HTTP_CLIENT == "httpx":
        if method == "GET":
            response = httpx.get(url, timeout=30)
        else:
            response = httpx.post(url, json=json_data, timeout=30)
        return {"status_code": response.status_code, "json": response.json()}
    else:
        if method == "GET":
            response = requests.get(url, timeout=30)
        else:
            response = requests.post(url, json=json_data, timeout=30)
        return {"status_code": response.status_code, "json": response.json()}


def print_separator(title: str = "") -> None:
    """Print a formatted separator."""
    width = 70
    if title:
        print(f"\n{'='*width}")
        print(f" {title}")
        print(f"{'='*width}")
    else:
        print(f"\n{'-'*width}")


def test_health(api_url: str) -> bool:
    """Test the health endpoint.

    Args:
        api_url: Base API URL

    Returns:
        True if healthy, False otherwise
    """
    print_separator("HEALTH CHECK")

    try:
        url = f"{api_url}/api/v1/health"
        print(f"GET {url}")

        response = make_request(url, "GET")

        print(f"\nStatus Code: {response['status_code']}")
        print("Response:")
        print(json.dumps(response["json"], indent=2))

        if response["status_code"] == 200 and response["json"].get("status") == "healthy":
            print("\n[SUCCESS] API is healthy and ready.")
            return True
        else:
            print("\n[WARNING] API returned unhealthy status.")
            return False

    except Exception as e:
        print(f"\n[ERROR] Health check failed: {e}")
        print("Make sure the API server is running:")
        print("  uvicorn src.main:app --host 0.0.0.0 --port 8000")
        return False


def test_predict(api_url: str, examples: pd.DataFrame, num_examples: int = 5) -> None:
    """Test the /predict endpoint with examples.

    Args:
        api_url: Base API URL
        examples: DataFrame with test examples
        num_examples: Number of examples to test
    """
    print_separator("PREDICTION TESTS (/predict)")

    url = f"{api_url}/api/v1/predict"
    print(f"POST {url}")
    print(f"Testing with {num_examples} examples from future_unseen_examples.csv\n")

    # Get column names for the request
    # These are all 18 columns from future_unseen_examples.csv
    columns = list(examples.columns)

    success_count = 0
    error_count = 0
    predictions = []

    for i in range(min(num_examples, len(examples))):
        row = examples.iloc[i]

        # Build request payload
        payload = {col: row[col] for col in columns}
        # Convert zipcode to string
        payload["zipcode"] = str(int(payload["zipcode"]))

        print(f"\n--- Example {i+1} ---")
        print(f"Zipcode: {payload['zipcode']}")
        print(f"Bedrooms: {payload['bedrooms']}, Bathrooms: {payload['bathrooms']}")
        print(f"Sqft Living: {payload['sqft_living']}, Sqft Lot: {payload['sqft_lot']}")

        try:
            response = make_request(url, "POST", payload)

            if response["status_code"] == 200:
                pred = response["json"]
                print(f"[SUCCESS] Predicted Price: ${pred['predicted_price']:,.2f}")
                print(f"          Prediction ID: {pred['prediction_id']}")
                print(f"          Model Version: {pred['model_version']}")
                success_count += 1
                predictions.append(
                    {
                        "example": i + 1,
                        "zipcode": payload["zipcode"],
                        "predicted_price": pred["predicted_price"],
                    }
                )
            else:
                print(f"[ERROR] Status: {response['status_code']}")
                print(f"        Response: {response['json']}")
                error_count += 1

        except Exception as e:
            print(f"[ERROR] Request failed: {e}")
            error_count += 1

    print_separator("PREDICTION SUMMARY")
    print(f"Successful predictions: {success_count}/{num_examples}")
    print(f"Failed predictions: {error_count}/{num_examples}")

    if predictions:
        print("\nPredictions:")
        for p in predictions:
            print(f"  Example {p['example']} (ZIP {p['zipcode']}): ${p['predicted_price']:,.2f}")


def test_predict_minimal(api_url: str, examples: pd.DataFrame, num_examples: int = 3) -> None:
    """Test the /predict-minimal endpoint (BONUS).

    Args:
        api_url: Base API URL
        examples: DataFrame with test examples
        num_examples: Number of examples to test
    """
    print_separator("MINIMAL PREDICTION TESTS (/predict-minimal) - BONUS")

    url = f"{api_url}/api/v1/predict-minimal"
    print(f"POST {url}")
    print(f"Testing with {num_examples} examples (only required features)\n")

    # Only the 7 features the model actually uses
    required_features = [
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "floors",
        "sqft_above",
        "sqft_basement",
    ]

    success_count = 0

    for i in range(min(num_examples, len(examples))):
        row = examples.iloc[i]

        # Build minimal payload
        payload = {col: row[col] for col in required_features}

        print(f"\n--- Example {i+1} ---")
        print(f"Bedrooms: {payload['bedrooms']}, Bathrooms: {payload['bathrooms']}")
        print(f"Sqft Living: {payload['sqft_living']}")

        try:
            response = make_request(url, "POST", payload)

            if response["status_code"] == 200:
                pred = response["json"]
                print(f"[SUCCESS] Predicted Price: ${pred['predicted_price']:,.2f}")
                print("          Note: Uses average demographics")
                success_count += 1
            else:
                print(f"[ERROR] Status: {response['status_code']}")
                print(f"        Response: {response['json']}")

        except Exception as e:
            print(f"[ERROR] Request failed: {e}")

    print(f"\nSuccessful minimal predictions: {success_count}/{num_examples}")


def main():
    """Main entry point for the test script."""
    parser = argparse.ArgumentParser(description="Test the Real Estate Price Predictor API")
    parser.add_argument(
        "--url", default=DEFAULT_API_URL, help=f"API base URL (default: {DEFAULT_API_URL})"
    )
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA_PATH,
        help=f"Path to test data CSV (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=NUM_EXAMPLES,
        help=f"Number of examples to test (default: {NUM_EXAMPLES})",
    )
    args = parser.parse_args()

    print_separator("REAL ESTATE PRICE PREDICTOR - API TEST SCRIPT")
    print(f"API URL: {args.url}")
    print(f"Test Data: {args.data}")
    print(f"HTTP Client: {HTTP_CLIENT}")

    # Check if data file exists
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"\n[ERROR] Test data file not found: {data_path}")
        sys.exit(1)

    # Load test data
    print(f"\nLoading test data from {args.data}...")
    examples = pd.read_csv(args.data)
    print(f"Loaded {len(examples)} examples with {len(examples.columns)} columns")

    # Test health endpoint
    if not test_health(args.url):
        print("\n[ERROR] API is not healthy. Exiting.")
        sys.exit(1)

    # Test predict endpoint
    test_predict(args.url, examples, args.examples)

    # Test predict-minimal endpoint (BONUS)
    test_predict_minimal(args.url, examples, min(3, args.examples))

    print_separator("TEST COMPLETE")
    print("All API tests completed. Review results above.")


if __name__ == "__main__":
    main()
