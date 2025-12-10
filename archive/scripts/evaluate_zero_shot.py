#!/usr/bin/env python3
"""
Zero-Shot Evaluation of V2.5 Model on Fresh Data
V3.2: Fresh Data Integration

This script evaluates how the V2.5 model (trained on 2014-2015 data)
performs on fresh 2020+ King County sales data WITHOUT retraining.

Key questions answered:
1. How large is the prediction error on new data?
2. Is the model systematically biased (under/over predicting)?
3. Which price ranges have the largest errors?
4. How does performance compare to the original test set?
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Paths
MODEL_PATH = Path("model/model.pkl")
ORIGINAL_DATA = Path("data/kc_house_data.csv")
DEMOGRAPHICS_DATA = Path("data/zipcode_demographics.csv")
FRESH_DATA = Path("data/assessment_2020_plus.csv")


def load_model():
    """Load the trained V2.5 model."""
    print("Loading V2.5 model...")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print(f"  Model type: {type(model).__name__}")
    return model


def prepare_features(df: pd.DataFrame, demographics: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features for model prediction.

    Joins with demographics and creates all required features.
    """
    # Ensure zipcode is string for joining
    df["zipcode"] = df["zipcode"].astype(str).str.strip()
    demographics["zipcode"] = demographics["zipcode"].astype(str).str.strip()

    # Join with demographics
    df = df.merge(demographics, on="zipcode", how="left")

    # Select features used by model (based on original training)
    feature_cols = [
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "floors",
        "waterfront",
        "view",
        "condition",
        "grade",
        "sqft_above",
        "sqft_basement",
        "yr_built",
        "yr_renovated",
        "lat",
        "long",
        "sqft_living15",
        "sqft_lot15",
    ]

    # Add demographic features if available
    demo_cols = [col for col in demographics.columns if col != "zipcode"]
    feature_cols.extend(demo_cols)

    # Fill missing values - use defaults for features we don't have
    for col in feature_cols:
        if col not in df.columns:
            # These features are not available in assessment data
            if col == "lat":
                df[col] = 47.5  # Approximate King County center latitude
            elif col == "long":
                df[col] = -122.3  # Approximate King County center longitude
            elif col == "sqft_living15":
                df[col] = df["sqft_living"]  # Use own living sqft
            elif col == "sqft_lot15":
                df[col] = df["sqft_lot"]  # Use own lot sqft
            else:
                df[col] = 0
        else:
            if df[col].dtype == "object":
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(0)

    return df, feature_cols


def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # Percentage errors
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # Bias (mean error - positive means underpredicting)
    bias = np.mean(y_true - y_pred)

    # Median absolute error (more robust)
    median_ae = np.median(np.abs(y_true - y_pred))

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
        "bias": bias,
        "median_ae": median_ae,
        "n_samples": len(y_true),
    }


def analyze_by_price_tier(y_true, y_pred):
    """Analyze performance by price tier."""
    tiers = [
        ("< $400K", 0, 400000),
        ("$400K - $600K", 400000, 600000),
        ("$600K - $800K", 600000, 800000),
        ("$800K - $1M", 800000, 1000000),
        ("$1M - $1.5M", 1000000, 1500000),
        ("> $1.5M", 1500000, float("inf")),
    ]

    results = []
    for tier_name, low, high in tiers:
        mask = (y_true >= low) & (y_true < high)
        if mask.sum() > 0:
            tier_metrics = calculate_metrics(y_true[mask], y_pred[mask])
            tier_metrics["tier"] = tier_name
            tier_metrics["pct_of_data"] = mask.sum() / len(y_true) * 100
            results.append(tier_metrics)

    return pd.DataFrame(results)


def main():
    """Run zero-shot evaluation."""
    print("=" * 80)
    print("ZERO-SHOT EVALUATION: V2.5 Model on 2020+ Data")
    print("=" * 80)

    # Load model
    model = load_model()

    # Load demographics
    print("\nLoading demographics data...")
    demographics = pd.read_csv(DEMOGRAPHICS_DATA, dtype={"zipcode": str})
    print(f"  Zipcodes: {len(demographics)}")

    # Load fresh data
    print("\nLoading fresh assessment data...")
    fresh_df = pd.read_csv(FRESH_DATA)
    print(f"  Records: {len(fresh_df):,}")
    print(f"  Date range: 2020+")
    print(f"  Median price: ${fresh_df['price'].median():,.0f}")

    # Prepare features
    print("\nPreparing features...")
    fresh_df, feature_cols = prepare_features(fresh_df, demographics)

    # Filter to records with valid features
    X_fresh = fresh_df[feature_cols].copy()
    y_true = fresh_df["price"].values

    # Check for missing values
    missing_mask = X_fresh.isna().any(axis=1)
    if missing_mask.sum() > 0:
        print(f"  Dropping {missing_mask.sum()} records with missing features")
        X_fresh = X_fresh[~missing_mask]
        y_true = y_true[~missing_mask]

    print(f"  Final evaluation set: {len(X_fresh):,} records")
    print(f"  Features: {len(feature_cols)}")

    # Make predictions
    print("\nMaking predictions...")
    y_pred = model.predict(X_fresh)
    print(f"  Predictions complete!")

    # Calculate overall metrics
    print("\n" + "=" * 80)
    print("OVERALL RESULTS")
    print("=" * 80)

    metrics = calculate_metrics(y_true, y_pred)

    print(f"\nZero-Shot Performance on 2020+ Data:")
    print(f"  MAE:        ${metrics['mae']:,.0f}")
    print(f"  RMSE:       ${metrics['rmse']:,.0f}")
    print(f"  R2:         {metrics['r2']:.4f}")
    print(f"  MAPE:       {metrics['mape']:.1f}%")
    print(f"  Median AE:  ${metrics['median_ae']:,.0f}")
    print(f"  Bias:       ${metrics['bias']:,.0f} (positive = underpredicting)")

    # Compare to original performance
    print("\n" + "-" * 40)
    print("Comparison to Original V2.5 Performance:")
    print("-" * 40)
    print(f"  Original CV MAE:  $63,529")
    print(f"  Fresh Data MAE:   ${metrics['mae']:,.0f}")
    print(f"  Difference:       ${metrics['mae'] - 63529:+,.0f} ({(metrics['mae'] - 63529) / 63529 * 100:+.1f}%)")

    # Analyze by price tier
    print("\n" + "=" * 80)
    print("PERFORMANCE BY PRICE TIER")
    print("=" * 80)

    tier_df = analyze_by_price_tier(y_true, y_pred)
    print("\n" + tier_df.to_string(index=False))

    # Key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    if metrics["bias"] > 0:
        print(f"\n1. SYSTEMATIC UNDERPREDICTION")
        print(f"   Model underpredicts by ${metrics['bias']:,.0f} on average")
        print(f"   This is expected: 2015 prices << 2020+ prices")
    else:
        print(f"\n1. SYSTEMATIC OVERPREDICTION")
        print(f"   Model overpredicts by ${abs(metrics['bias']):,.0f} on average")

    # Calculate price inflation
    original_median = 450000  # Approximate 2015 median
    fresh_median = fresh_df["price"].median()
    inflation = (fresh_median - original_median) / original_median * 100

    print(f"\n2. MARKET PRICE CHANGE")
    print(f"   2015 median price: ~$450,000")
    print(f"   2020+ median price: ${fresh_median:,.0f}")
    print(f"   Price inflation: {inflation:+.0f}%")

    print(f"\n3. MODEL STILL CAPTURES PATTERNS")
    print(f"   R2 of {metrics['r2']:.3f} means model explains {metrics['r2']*100:.1f}% of variance")
    print(f"   The relative ordering of prices is largely preserved")

    print(f"\n4. RECOMMENDATION")
    if metrics["mape"] > 30:
        print(f"   RETRAIN REQUIRED: MAPE of {metrics['mape']:.1f}% is too high")
        print(f"   Model needs retraining on 2020+ data")
    elif metrics["mape"] > 20:
        print(f"   RETRAIN RECOMMENDED: MAPE of {metrics['mape']:.1f}% is elevated")
        print(f"   Consider retraining for production use")
    else:
        print(f"   MODEL USABLE: MAPE of {metrics['mape']:.1f}% is acceptable")
        print(f"   May still benefit from retraining")

    # Save results
    results = {
        "evaluation_type": "zero_shot",
        "model_version": "V2.5",
        "data_source": "King County Assessment 2020+",
        "n_samples": len(y_true),
        "metrics": metrics,
        "price_inflation_pct": inflation,
    }

    import json

    with open("data/zero_shot_evaluation.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: data/zero_shot_evaluation.json")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)

    return metrics, tier_df


if __name__ == "__main__":
    main()
