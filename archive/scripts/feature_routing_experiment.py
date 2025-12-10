"""
Feature-Based Routing Experiment (V2.1.3 Exploration)

Question: Can we route based on input features instead of predicted price?

Hypothesis: Premium features (waterfront, high grade, good view) correlate
with homes where /predict-full outperforms /predict-minimal.

LEAKAGE WARNING: This is an exploratory analysis. Results are contaminated
because we're using the same data to discover patterns AND evaluate them.
For production use, this would need proper train/validation/test splits.
"""

import sys

import numpy as np
import pandas as pd

try:
    import httpx
except ImportError:
    print("[ERROR] httpx not installed. Run: pip install httpx")
    sys.exit(1)


API_URL = "http://localhost:8000"
DATA_PATH = "data/future_unseen_examples.csv"
NUM_EXAMPLES = 50


def make_prediction(endpoint: str, payload: dict) -> float:
    """Make a prediction request."""
    url = f"{API_URL}/api/v1/{endpoint}"
    response = httpx.post(url, json=payload, timeout=30)
    if response.status_code == 200:
        return response.json()["predicted_price"]
    return None


def compute_premium_score(row: pd.Series) -> float:
    """Compute a 'premium' score from raw features.

    Higher score = more premium/expensive home characteristics.

    Components:
    - waterfront: +5 points (huge premium indicator)
    - view: +0 to +4 points (view quality)
    - grade: (grade - 7) points (above/below average construction)
    - condition: (condition - 3) points (above/below average condition)
    - sqft_living: log ratio vs median (1986 sqft)
    """
    score = 0.0

    # Waterfront is huge (50-100% price premium)
    score += row.get("waterfront", 0) * 5

    # View quality (0-4)
    score += row.get("view", 0)

    # Construction grade (1-13, 7 is average)
    score += row.get("grade", 7) - 7

    # Condition (1-5, 3 is average)
    score += row.get("condition", 3) - 3

    # Size relative to median (log scale)
    sqft = row.get("sqft_living", 1986)
    score += np.log(sqft / 1986) * 2  # ~0 for average, +2 for 2x, -2 for 0.5x

    return score


def main():
    print("=" * 80)
    print(" FEATURE-BASED ROUTING EXPERIMENT (V2.1.3 Exploration)")
    print("=" * 80)
    print("\n⚠️  LEAKAGE WARNING: This uses test data to both discover and evaluate.")
    print("    Results are for exploration only, not production-ready.\n")

    # Check API
    try:
        health = httpx.get(f"{API_URL}/api/v1/health", timeout=10)
        print(f"API Health: {health.json()['status']}\n")
    except Exception as e:
        print(f"[ERROR] API not responding: {e}")
        sys.exit(1)

    # Load data
    df = pd.read_csv(DATA_PATH)
    print(f"Testing with {min(NUM_EXAMPLES, len(df))} examples\n")

    # Columns
    full_cols = [
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
    minimal_cols = [
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

        # Build payloads
        payload_predict = {col: row[col] for col in df.columns}
        payload_predict["zipcode"] = str(int(payload_predict["zipcode"]))
        payload_minimal = {col: row[col] for col in minimal_cols}
        payload_full = {col: row[col] for col in full_cols}

        # Make predictions
        price_gold = make_prediction("predict", payload_predict)
        price_minimal = make_prediction("predict-minimal", payload_minimal)
        price_full = make_prediction("predict-full", payload_full)

        if all(p is not None for p in [price_gold, price_minimal, price_full]):
            error_minimal = abs(price_gold - price_minimal)
            error_full = abs(price_gold - price_full)

            # Compute premium score
            premium_score = compute_premium_score(row)

            # Which strategy won?
            better = "full" if error_full < error_minimal else "minimal"

            results.append(
                {
                    "gold_price": price_gold,
                    "premium_score": premium_score,
                    "error_minimal": error_minimal,
                    "error_full": error_full,
                    "better_strategy": better,
                    "waterfront": row.get("waterfront", 0),
                    "view": row.get("view", 0),
                    "grade": row.get("grade", 7),
                    "condition": row.get("condition", 3),
                    "sqft_living": row.get("sqft_living", 1986),
                }
            )

    df_results = pd.DataFrame(results)

    # Analysis 1: Correlation between premium score and which strategy wins
    print("=" * 80)
    print(" ANALYSIS 1: Premium Score vs Winner")
    print("=" * 80)

    full_wins = df_results[df_results["better_strategy"] == "full"]
    minimal_wins = df_results[df_results["better_strategy"] == "minimal"]

    print(f"\n/predict-full wins (n={len(full_wins)}):")
    print(f"  Mean premium score: {full_wins['premium_score'].mean():.2f}")
    print(f"  Mean gold price: ${full_wins['gold_price'].mean():,.0f}")

    print(f"\n/predict-minimal wins (n={len(minimal_wins)}):")
    print(f"  Mean premium score: {minimal_wins['premium_score'].mean():.2f}")
    print(f"  Mean gold price: ${minimal_wins['gold_price'].mean():,.0f}")

    # Analysis 2: Find optimal premium score threshold
    print("\n" + "=" * 80)
    print(" ANALYSIS 2: Finding Optimal Premium Score Threshold")
    print("=" * 80)

    thresholds = np.arange(-4, 6, 0.5)
    best_threshold = 0
    best_accuracy = 0
    best_error = float("inf")

    print(f"\n{'Threshold':>10} {'Accuracy':>10} {'Avg Error':>15}")
    print("-" * 40)

    for thresh in thresholds:
        # Route: premium_score >= thresh → full, else → minimal
        df_results["predicted_strategy"] = df_results["premium_score"].apply(
            lambda x: "full" if x >= thresh else "minimal"
        )

        # Accuracy
        correct = (df_results["predicted_strategy"] == df_results["better_strategy"]).sum()
        accuracy = correct / len(df_results)

        # Average error with this routing
        errors = []
        for _, row in df_results.iterrows():
            if row["predicted_strategy"] == "full":
                errors.append(row["error_full"])
            else:
                errors.append(row["error_minimal"])
        avg_error = np.mean(errors)

        print(f"{thresh:>10.1f} {accuracy:>10.1%} ${avg_error:>14,.0f}")

        if avg_error < best_error:
            best_error = avg_error
            best_threshold = thresh
            best_accuracy = accuracy

    print("-" * 40)
    print(f"\nBest threshold: {best_threshold:.1f}")
    print(f"Best accuracy: {best_accuracy:.1%}")
    print(f"Best avg error: ${best_error:,.0f}")

    # Compare to baselines
    print("\n" + "=" * 80)
    print(" COMPARISON TO BASELINES")
    print("=" * 80)

    baseline_minimal = df_results["error_minimal"].mean()
    baseline_full = df_results["error_full"].mean()

    print(f"\n/predict-minimal (always):       ${baseline_minimal:>12,.0f}")
    print(f"/predict-full (always):          ${baseline_full:>12,.0f}")
    print(f"Price-based adaptive (V2.1.2):   ${166126:>12,.0f}")  # From previous run
    print(f"Feature-based routing (optimal): ${best_error:>12,.0f}")

    improvement_vs_full = baseline_full - best_error
    improvement_pct = (improvement_vs_full / baseline_full) * 100

    print("\nFeature-based improvement over /predict-full:")
    if improvement_vs_full > 0:
        print(f"  ${improvement_vs_full:,.0f} ({improvement_pct:.1f}%) BETTER")
    else:
        print(f"  ${-improvement_vs_full:,.0f} ({-improvement_pct:.1f}%) worse")

    # Analysis 3: Individual feature correlations
    print("\n" + "=" * 80)
    print(" ANALYSIS 3: Individual Feature Correlation with Winner")
    print("=" * 80)

    for feature in ["waterfront", "view", "grade", "condition", "sqft_living", "gold_price"]:
        corr_full = full_wins[feature].mean()
        corr_minimal = minimal_wins[feature].mean()
        print(f"\n{feature}:")
        print(f"  When /predict-full wins: {corr_full:,.1f}")
        print(f"  When /predict-minimal wins: {corr_minimal:,.1f}")
        print(f"  Difference: {corr_full - corr_minimal:+,.1f}")

    # Verdict
    print("\n" + "=" * 80)
    print(" VERDICT")
    print("=" * 80)

    if best_error < baseline_full:
        print("\n✓ Feature-based routing CAN beat /predict-full")
        print(f"  But only by ${improvement_vs_full:,.0f} ({improvement_pct:.1f}%)")
        print(f"\n⚠️  WARNING: This threshold ({best_threshold:.1f}) is overfit to test data!")
        print("   Would need proper CV to validate.")
    else:
        print("\n✗ Feature-based routing doesn't beat /predict-full")
        print("   Stick with /predict-full for simplicity.")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
