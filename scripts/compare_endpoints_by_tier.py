"""
Compare prediction endpoints BY PRICE TIER.

User insight: /predict-minimal is more accurate for homes < $400K,
              /predict-full is more accurate for homes > $400K.

This script validates that hypothesis and tests an adaptive approach.

Usage:
    python scripts/compare_endpoints_by_tier.py
"""

import json
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
NUM_EXAMPLES = 50  # More examples for better tier analysis
PRICE_THRESHOLD = 400_000  # User's observed threshold


def make_prediction(endpoint: str, payload: dict) -> float:
    """Make a prediction request and return the price."""
    url = f"{API_URL}/api/v1/{endpoint}"
    response = httpx.post(url, json=payload, timeout=30)
    if response.status_code == 200:
        return response.json()["predicted_price"]
    else:
        return None


def main():
    print("=" * 80)
    print(" PRICE-TIER ANALYSIS: /predict-minimal vs /predict-full")
    print("=" * 80)
    print(f"\nHypothesis: /predict-minimal better for < ${PRICE_THRESHOLD:,}")
    print(f"            /predict-full better for > ${PRICE_THRESHOLD:,}")
    
    # Check API health
    try:
        health = httpx.get(f"{API_URL}/api/v1/health", timeout=10)
        health_data = health.json()
        print(f"\nAPI Health: {health_data['status']}, Model: {health_data['model_version']}")
    except Exception as e:
        print(f"\n[ERROR] API not responding: {e}")
        sys.exit(1)
    
    # Load test data
    df = pd.read_csv(DATA_PATH)
    print(f"Testing with {min(NUM_EXAMPLES, len(df))} examples\n")
    
    # Column mappings
    full_features_cols = [
        'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
        'sqft_above', 'sqft_basement', 'waterfront', 'view', 'condition',
        'grade', 'yr_built', 'yr_renovated', 'lat', 'long',
        'sqft_living15', 'sqft_lot15'
    ]
    minimal_features_cols = [
        'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
        'sqft_above', 'sqft_basement'
    ]
    
    results = []
    
    for i in range(min(NUM_EXAMPLES, len(df))):
        row = df.iloc[i]
        
        # Build payloads
        payload_predict = {col: row[col] for col in df.columns}
        payload_predict['zipcode'] = str(int(payload_predict['zipcode']))
        payload_minimal = {col: row[col] for col in minimal_features_cols}
        payload_full = {col: row[col] for col in full_features_cols}
        
        # Make predictions
        price_gold = make_prediction("predict", payload_predict)
        price_minimal = make_prediction("predict-minimal", payload_minimal)
        price_full = make_prediction("predict-full", payload_full)
        price_adaptive_api = make_prediction("predict-adaptive", payload_full)  # NEW: actual API adaptive
        
        if all(p is not None for p in [price_gold, price_minimal, price_full, price_adaptive_api]):
            error_minimal = abs(price_gold - price_minimal)
            error_full = abs(price_gold - price_full)
            error_adaptive_api = abs(price_gold - price_adaptive_api)  # NEW
            
            # Determine tier
            tier = "HIGH" if price_gold >= PRICE_THRESHOLD else "LOW"
            
            # Which would adaptive choose? (simulate locally)
            adaptive_tier = "HIGH" if price_full >= PRICE_THRESHOLD else "LOW"
            if adaptive_tier == "HIGH":
                price_adaptive = price_full
                adaptive_choice = "full"
            else:
                price_adaptive = price_minimal
                adaptive_choice = "minimal"
            error_adaptive = abs(price_gold - price_adaptive)
            
            # Which is actually better?
            better = "minimal" if error_minimal < error_full else "full"
            
            results.append({
                'example': i + 1,
                'gold_price': price_gold,
                'tier': tier,
                'price_minimal': price_minimal,
                'price_full': price_full,
                'price_adaptive': price_adaptive,
                'price_adaptive_api': price_adaptive_api,
                'error_minimal': error_minimal,
                'error_full': error_full,
                'error_adaptive': error_adaptive,
                'error_adaptive_api': error_adaptive_api,
                'better': better,
                'adaptive_choice': adaptive_choice,
                'adaptive_correct': (adaptive_choice == better),
            })
    
    df_results = pd.DataFrame(results)
    
    # Analysis by tier
    print("\n" + "=" * 80)
    print(" TIER ANALYSIS")
    print("=" * 80)
    
    for tier in ["LOW", "HIGH"]:
        tier_df = df_results[df_results['tier'] == tier]
        if len(tier_df) == 0:
            continue
        
        label = f"< ${PRICE_THRESHOLD:,}" if tier == "LOW" else f">= ${PRICE_THRESHOLD:,}"
        print(f"\n{tier} TIER ({label}): {len(tier_df)} examples")
        print("-" * 50)
        
        # Win counts
        minimal_wins = (tier_df['better'] == 'minimal').sum()
        full_wins = (tier_df['better'] == 'full').sum()
        
        print(f"  /predict-minimal wins: {minimal_wins}/{len(tier_df)} ({minimal_wins/len(tier_df)*100:.0f}%)")
        print(f"  /predict-full wins:    {full_wins}/{len(tier_df)} ({full_wins/len(tier_df)*100:.0f}%)")
        
        # Average errors
        avg_err_minimal = tier_df['error_minimal'].mean()
        avg_err_full = tier_df['error_full'].mean()
        avg_err_adaptive = tier_df['error_adaptive'].mean()
        
        print(f"\n  Average error:")
        print(f"    /predict-minimal: ${avg_err_minimal:>12,.2f}")
        print(f"    /predict-full:    ${avg_err_full:>12,.2f}")
        print(f"    /adaptive:        ${avg_err_adaptive:>12,.2f}")
        
        # Best strategy for this tier
        if avg_err_minimal < avg_err_full:
            print(f"\n  ✓ BEST for {tier} tier: /predict-minimal")
        else:
            print(f"\n  ✓ BEST for {tier} tier: /predict-full")
    
    # Overall adaptive performance
    print("\n" + "=" * 80)
    print(" ADAPTIVE STRATEGY EVALUATION")
    print("=" * 80)
    
    print(f"\nStrategy: Use /predict-full to estimate price tier, then:")
    print(f"  - If estimate < ${PRICE_THRESHOLD:,}: use /predict-minimal")
    print(f"  - If estimate >= ${PRICE_THRESHOLD:,}: use /predict-full")
    
    avg_minimal = df_results['error_minimal'].mean()
    avg_full = df_results['error_full'].mean()
    avg_adaptive = df_results['error_adaptive'].mean()
    avg_adaptive_api = df_results['error_adaptive_api'].mean()
    
    print(f"\nOverall average error:")
    print(f"  /predict-minimal (always): ${avg_minimal:>12,.2f}")
    print(f"  /predict-full (always):    ${avg_full:>12,.2f}")
    print(f"  /adaptive (simulated):     ${avg_adaptive:>12,.2f}")
    print(f"  /predict-adaptive (API):   ${avg_adaptive_api:>12,.2f}  ← V2.1.2 endpoint")
    
    # Compare to best single strategy
    best_single = min(avg_minimal, avg_full)
    best_name = "/predict-minimal" if avg_minimal < avg_full else "/predict-full"
    
    improvement = best_single - avg_adaptive
    improvement_pct = (improvement / best_single) * 100 if best_single > 0 else 0
    
    print(f"\nAdaptive improvement over best single ({best_name}):")
    if improvement > 0:
        print(f"  ${improvement:,.2f} ({improvement_pct:.1f}%) BETTER")
    else:
        print(f"  ${-improvement:,.2f} ({-improvement_pct:.1f}%) worse")
    
    # How often does adaptive pick correctly?
    adaptive_accuracy = df_results['adaptive_correct'].mean() * 100
    print(f"\nAdaptive strategy picks correct endpoint: {adaptive_accuracy:.0f}% of the time")
    
    print("\n" + "=" * 80)
    print(" VERDICT")
    print("=" * 80)
    
    # Validate hypothesis
    low_tier = df_results[df_results['tier'] == 'LOW']
    high_tier = df_results[df_results['tier'] == 'HIGH']
    
    if len(low_tier) > 0 and len(high_tier) > 0:
        low_minimal_wins_pct = (low_tier['better'] == 'minimal').mean() * 100
        high_full_wins_pct = (high_tier['better'] == 'full').mean() * 100
        
        print(f"\nHypothesis validation:")
        print(f"  LOW tier: /predict-minimal wins {low_minimal_wins_pct:.0f}% {'✓ CONFIRMED' if low_minimal_wins_pct > 50 else '✗ NOT confirmed'}")
        print(f"  HIGH tier: /predict-full wins {high_full_wins_pct:.0f}% {'✓ CONFIRMED' if high_full_wins_pct > 50 else '✗ NOT confirmed'}")
        
        if low_minimal_wins_pct > 50 and high_full_wins_pct > 50:
            print(f"\n✓ USER HYPOTHESIS CONFIRMED!")
            print(f"  Price-tier adaptive strategy is justified.")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
