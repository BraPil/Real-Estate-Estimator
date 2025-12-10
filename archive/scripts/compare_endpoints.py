"""
Compare prediction endpoints to test V2.1.1 hypothesis.

This script compares predictions from:
1. /predict         - Full features + zipcode demographics
2. /predict-minimal - 7 features + defaults + avg demographics
3. /predict-full    - 17 features + avg demographics (V2.1.1 experiment)

Hypothesis: /predict-full should be more accurate than /predict-minimal
            because it uses actual values instead of defaults.

Usage:
    # Start API first:
    uvicorn src.main:app --host 0.0.0.0 --port 8000

    # Then run comparison:
    python scripts/compare_endpoints.py
"""

import sys
from pathlib import Path

import pandas as pd

try:
    import httpx
except ImportError:
    print("[ERROR] httpx not installed. Run: pip install httpx")
    sys.exit(1)


API_URL = "http://localhost:8000"
DATA_PATH = "data/future_unseen_examples.csv"
NUM_EXAMPLES = 20  # Test with more examples for statistical significance


def make_prediction(endpoint: str, payload: dict) -> float:
    """Make a prediction request and return the price."""
    url = f"{API_URL}/api/v1/{endpoint}"
    response = httpx.post(url, json=payload, timeout=30)
    if response.status_code == 200:
        return response.json()["predicted_price"]
    else:
        print(f"[ERROR] {endpoint}: {response.status_code} - {response.json()}")
        return None


def main():
    print("=" * 70)
    print(" ENDPOINT COMPARISON: /predict vs /predict-minimal vs /predict-full")
    print("=" * 70)

    # Check API health
    try:
        health = httpx.get(f"{API_URL}/api/v1/health", timeout=10)
        health_data = health.json()
        print(f"\nAPI Health: {health_data['status']}")
        print(f"Model Version: {health_data['model_version']}")
    except Exception as e:
        print(f"\n[ERROR] API not responding: {e}")
        print("Start the API first: uvicorn src.main:app --host 0.0.0.0 --port 8000")
        sys.exit(1)

    # Load test data
    data_path = Path(DATA_PATH)
    if not data_path.exists():
        print(f"\n[ERROR] Test data not found: {data_path}")
        sys.exit(1)

    df = pd.read_csv(data_path)
    print(f"\nLoaded {len(df)} test examples")
    print(f"Testing with {NUM_EXAMPLES} examples\n")

    # Column mappings
    full_features_cols = [
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "floors",
        "sqft_above",
        "sqft_basement",
        "waterfront",
        "view",
        "condition",
        "grade",
        "yr_built",
        "yr_renovated",
        "lat",
        "long",
        "sqft_living15",
        "sqft_lot15",
    ]
    minimal_features_cols = [
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "floors",
        "sqft_above",
        "sqft_basement",
    ]

    results = []

    for i in range(min(NUM_EXAMPLES, len(df))):
        row = df.iloc[i]

        # Build payloads for each endpoint
        # 1. /predict - all 18 columns + zipcode
        payload_predict = {col: row[col] for col in df.columns}
        payload_predict["zipcode"] = str(int(payload_predict["zipcode"]))

        # 2. /predict-minimal - only 7 features
        payload_minimal = {col: row[col] for col in minimal_features_cols}

        # 3. /predict-full - all 17 features (no zipcode)
        payload_full = {col: row[col] for col in full_features_cols}

        # Make predictions
        price_predict = make_prediction("predict", payload_predict)
        price_minimal = make_prediction("predict-minimal", payload_minimal)
        price_full = make_prediction("predict-full", payload_full)

        if all(p is not None for p in [price_predict, price_minimal, price_full]):
            results.append(
                {
                    "example": i + 1,
                    "zipcode": payload_predict["zipcode"],
                    "price_predict": price_predict,
                    "price_minimal": price_minimal,
                    "price_full": price_full,
                    "diff_minimal_vs_full": abs(price_predict - price_minimal)
                    - abs(price_predict - price_full),
                }
            )

            print(f"Example {i+1} (ZIP {payload_predict['zipcode']}):")
            print(f"  /predict (gold):    ${price_predict:>12,.2f}")
            print(
                f"  /predict-minimal:   ${price_minimal:>12,.2f}  (diff: ${abs(price_predict - price_minimal):>10,.2f})"
            )
            print(
                f"  /predict-full:      ${price_full:>12,.2f}  (diff: ${abs(price_predict - price_full):>10,.2f})"
            )

            # Which is closer?
            if abs(price_predict - price_full) < abs(price_predict - price_minimal):
                print(
                    f"  → /predict-full is CLOSER by ${abs(price_predict - price_minimal) - abs(price_predict - price_full):,.2f}"
                )
            else:
                print(
                    f"  → /predict-minimal is CLOSER by ${abs(price_predict - price_full) - abs(price_predict - price_minimal):,.2f}"
                )
            print()

    # Summary statistics
    if results:
        print("=" * 70)
        print(" SUMMARY")
        print("=" * 70)

        df_results = pd.DataFrame(results)

        # Calculate average differences from "gold standard" (/predict)
        avg_diff_minimal = (df_results["price_predict"] - df_results["price_minimal"]).abs().mean()
        avg_diff_full = (df_results["price_predict"] - df_results["price_full"]).abs().mean()

        print("\nAverage absolute difference from /predict (gold standard):")
        print(f"  /predict-minimal: ${avg_diff_minimal:,.2f}")
        print(f"  /predict-full:    ${avg_diff_full:,.2f}")

        improvement = avg_diff_minimal - avg_diff_full
        improvement_pct = (improvement / avg_diff_minimal) * 100 if avg_diff_minimal > 0 else 0

        print("\nImprovement using /predict-full over /predict-minimal:")
        print(f"  ${improvement:,.2f} ({improvement_pct:.1f}%)")

        # Count wins
        full_wins = sum(1 for r in results if r["diff_minimal_vs_full"] > 0)
        minimal_wins = sum(1 for r in results if r["diff_minimal_vs_full"] < 0)
        ties = len(results) - full_wins - minimal_wins

        print("\nWin/Loss record (which is closer to /predict):")
        print(f"  /predict-full wins:    {full_wins}/{len(results)}")
        print(f"  /predict-minimal wins: {minimal_wins}/{len(results)}")
        print(f"  Ties:                  {ties}/{len(results)}")

        # Verdict
        print("\n" + "=" * 70)
        if improvement > 0:
            print(" VERDICT: Hypothesis CONFIRMED! /predict-full is more accurate.")
            print(
                f"          Provides ${improvement:,.2f} ({improvement_pct:.1f}%) better predictions on average."
            )
        else:
            print(
                " VERDICT: Hypothesis NOT confirmed. /predict-minimal is equally or more accurate."
            )
            print("          Defaults may be well-calibrated for typical homes.")
        print("=" * 70)


if __name__ == "__main__":
    main()
