"""
V2.7 Exploration: Optimal Tier Split Discovery

Goal: Find the optimal percentile to split prices into 2 tiers that:
1. Maximizes tier classification accuracy
2. Is resilient to price appreciation (percentile-based, not dollar-based)
3. Maintains balanced per-tier accuracy

Approach:
- Test splits from 25th to 75th percentile
- Measure Gradient Boosting classifier accuracy at each split
- Find the sweet spot where tiers are most separable
- Analyze per-tier accuracy to avoid imbalanced performance

Usage:
    python src/optimize_tier_split.py
    python src/optimize_tier_split.py --min-percentile 30 --max-percentile 70 --step 2
"""

import argparse
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, confusion_matrix

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path("data")
OUTPUT_DIR = Path("output/tier_optimization")

RANDOM_STATE = 42


def load_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Load and prepare data."""
    logger.info("Loading data...")
    
    sales_columns = [
        'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
        'sqft_above', 'sqft_basement', 'waterfront', 'view', 'condition', 'grade',
        'yr_built', 'yr_renovated', 'lat', 'long', 'sqft_living15', 'sqft_lot15', 'zipcode'
    ]
    
    sales = pd.read_csv(DATA_DIR / "kc_house_data.csv", usecols=sales_columns)
    demographics = pd.read_csv(DATA_DIR / "zipcode_demographics.csv")
    merged = sales.merge(demographics, on='zipcode', how='inner')
    
    y = merged['price']
    X = merged.drop(columns=['price', 'zipcode'])
    
    logger.info(f"Loaded {len(X)} samples, {len(X.columns)} features")
    return X, y


def create_tiers_at_percentile(y: pd.Series, percentile: float) -> pd.Series:
    """Create 2-tier labels at a given percentile."""
    threshold = y.quantile(percentile / 100)
    tiers = (y >= threshold).astype(int)
    return tiers, threshold


def evaluate_split(
    X: pd.DataFrame,
    y: pd.Series,
    percentile: float,
    n_cv_folds: int = 5
) -> Dict:
    """
    Evaluate tier classification accuracy at a given percentile split.
    
    Returns detailed metrics including per-tier accuracy.
    """
    # Create tiers
    tiers, threshold = create_tiers_at_percentile(y, percentile)
    
    # Class balance
    tier_0_pct = (tiers == 0).mean() * 100
    tier_1_pct = (tiers == 1).mean() * 100
    
    # Scale features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Use Gradient Boosting (our best classifier)
    clf = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=RANDOM_STATE
    )
    
    # Cross-validation with stratification
    cv = StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(clf, X_scaled, tiers, cv=cv, scoring='accuracy')
    
    # Train/test split for per-tier analysis
    X_train, X_test, y_train, y_test, tiers_train, tiers_test = train_test_split(
        X_scaled, y, tiers, test_size=0.2, random_state=RANDOM_STATE, stratify=tiers
    )
    
    clf.fit(X_train, tiers_train)
    predictions = clf.predict(X_test)
    
    # Per-tier accuracy
    tier_0_mask = tiers_test == 0
    tier_1_mask = tiers_test == 1
    
    tier_0_acc = accuracy_score(tiers_test[tier_0_mask], predictions[tier_0_mask]) if tier_0_mask.sum() > 0 else 0
    tier_1_acc = accuracy_score(tiers_test[tier_1_mask], predictions[tier_1_mask]) if tier_1_mask.sum() > 0 else 0
    
    # Minimum of both tiers (we want both to be good)
    min_tier_acc = min(tier_0_acc, tier_1_acc)
    
    # Balanced accuracy (average of per-tier accuracies)
    balanced_acc = (tier_0_acc + tier_1_acc) / 2
    
    # Confusion matrix
    cm = confusion_matrix(tiers_test, predictions)
    
    return {
        'percentile': percentile,
        'threshold_dollars': threshold,
        'tier_0_pct': tier_0_pct,
        'tier_1_pct': tier_1_pct,
        'cv_accuracy_mean': cv_scores.mean(),
        'cv_accuracy_std': cv_scores.std(),
        'tier_0_accuracy': tier_0_acc,
        'tier_1_accuracy': tier_1_acc,
        'min_tier_accuracy': min_tier_acc,
        'balanced_accuracy': balanced_acc,
        'accuracy_gap': abs(tier_0_acc - tier_1_acc),
        'confusion_matrix': cm.tolist()
    }


def find_optimal_split(
    X: pd.DataFrame,
    y: pd.Series,
    min_percentile: int = 25,
    max_percentile: int = 75,
    step: int = 5
) -> Tuple[Dict, List[Dict]]:
    """
    Search for optimal percentile split.
    
    Returns:
        best_result: The optimal split configuration
        all_results: Results for all tested percentiles
    """
    logger.info(f"\nSearching for optimal split from {min_percentile}th to {max_percentile}th percentile...")
    
    all_results = []
    
    percentiles = list(range(min_percentile, max_percentile + 1, step))
    
    for pct in percentiles:
        result = evaluate_split(X, y, pct)
        all_results.append(result)
        
        logger.info(f"  P{pct:02d}: CV={result['cv_accuracy_mean']:.1%} "
                   f"T0={result['tier_0_accuracy']:.1%} T1={result['tier_1_accuracy']:.1%} "
                   f"Min={result['min_tier_accuracy']:.1%} Gap={result['accuracy_gap']:.1%}")
    
    # Find optimal by different criteria
    best_by_cv = max(all_results, key=lambda x: x['cv_accuracy_mean'])
    best_by_min = max(all_results, key=lambda x: x['min_tier_accuracy'])
    best_by_balanced = max(all_results, key=lambda x: x['balanced_accuracy'])
    
    return {
        'best_by_cv': best_by_cv,
        'best_by_min_tier': best_by_min,
        'best_by_balanced': best_by_balanced
    }, all_results


def fine_tune_around_optimum(
    X: pd.DataFrame,
    y: pd.Series,
    center_percentile: int,
    range_width: int = 5,
    step: float = 1
) -> List[Dict]:
    """Fine-tune around a promising percentile with smaller steps."""
    logger.info(f"\nFine-tuning around {center_percentile}th percentile...")
    
    results = []
    percentiles = np.arange(
        center_percentile - range_width,
        center_percentile + range_width + step,
        step
    )
    
    for pct in percentiles:
        if 20 <= pct <= 80:  # Stay within reasonable bounds
            result = evaluate_split(X, y, pct)
            results.append(result)
            
            logger.info(f"  P{pct:05.1f}: CV={result['cv_accuracy_mean']:.2%} "
                       f"T0={result['tier_0_accuracy']:.2%} T1={result['tier_1_accuracy']:.2%} "
                       f"Min={result['min_tier_accuracy']:.2%}")
    
    return results


def analyze_feature_separability(
    X: pd.DataFrame,
    y: pd.Series,
    percentile: float
) -> Dict:
    """
    Analyze how well features separate tiers at a given percentile.
    """
    tiers, threshold = create_tiers_at_percentile(y, percentile)
    
    separability = {}
    
    for col in X.columns:
        tier_0_mean = X.loc[tiers == 0, col].mean()
        tier_1_mean = X.loc[tiers == 1, col].mean()
        tier_0_std = X.loc[tiers == 0, col].std()
        tier_1_std = X.loc[tiers == 1, col].std()
        
        # Cohen's d effect size
        pooled_std = np.sqrt((tier_0_std**2 + tier_1_std**2) / 2)
        if pooled_std > 0:
            cohens_d = abs(tier_1_mean - tier_0_mean) / pooled_std
        else:
            cohens_d = 0
        
        separability[col] = {
            'tier_0_mean': tier_0_mean,
            'tier_1_mean': tier_1_mean,
            'cohens_d': cohens_d
        }
    
    # Sort by effect size
    sorted_features = sorted(separability.items(), key=lambda x: x[1]['cohens_d'], reverse=True)
    
    return {
        'percentile': percentile,
        'threshold': threshold,
        'top_separating_features': sorted_features[:10]
    }


def main():
    parser = argparse.ArgumentParser(description='V2.7: Optimal Tier Split Discovery')
    parser.add_argument('--min-percentile', type=int, default=25, help='Minimum percentile to test')
    parser.add_argument('--max-percentile', type=int, default=75, help='Maximum percentile to test')
    parser.add_argument('--step', type=int, default=5, help='Percentile step size for coarse search')
    parser.add_argument('--fine-tune', action='store_true', help='Perform fine-tuning around optimal')
    args = parser.parse_args()
    
    print("\n")
    print("=" * 70)
    print(" V2.7 EXPLORATION: OPTIMAL TIER SPLIT DISCOVERY")
    print("=" * 70)
    print(f" Search Range: {args.min_percentile}th to {args.max_percentile}th percentile")
    print(f" Step Size: {args.step}")
    print("=" * 70)
    
    # Load data
    X, y = load_data()
    
    # Price statistics
    print(f"\nPrice Distribution:")
    print(f"  Min:    ${y.min():,.0f}")
    print(f"  25th:   ${y.quantile(0.25):,.0f}")
    print(f"  Median: ${y.quantile(0.50):,.0f}")
    print(f"  75th:   ${y.quantile(0.75):,.0f}")
    print(f"  Max:    ${y.quantile(1.0):,.0f}")
    
    # Coarse search
    print("\n" + "-" * 70)
    print(" COARSE SEARCH")
    print("-" * 70)
    
    optimal_results, all_results = find_optimal_split(
        X, y,
        min_percentile=args.min_percentile,
        max_percentile=args.max_percentile,
        step=args.step
    )
    
    # Display results table
    print("\n" + "-" * 70)
    print(" RESULTS SUMMARY")
    print("-" * 70)
    print(f"{'Pctl':>6} {'Threshold':>12} {'CV Acc':>8} {'T0 Acc':>8} {'T1 Acc':>8} {'Min':>8} {'Gap':>8}")
    print("-" * 70)
    
    for r in all_results:
        print(f"{r['percentile']:>5}% ${r['threshold_dollars']:>10,.0f} "
              f"{r['cv_accuracy_mean']:>7.1%} {r['tier_0_accuracy']:>7.1%} "
              f"{r['tier_1_accuracy']:>7.1%} {r['min_tier_accuracy']:>7.1%} "
              f"{r['accuracy_gap']:>7.1%}")
    
    print("-" * 70)
    
    # Best results
    print("\n" + "=" * 70)
    print(" OPTIMAL SPLITS")
    print("=" * 70)
    
    print(f"\nBest by CV Accuracy:")
    b = optimal_results['best_by_cv']
    print(f"  Percentile: {b['percentile']}th (${b['threshold_dollars']:,.0f})")
    print(f"  CV Accuracy: {b['cv_accuracy_mean']:.2%} +/- {b['cv_accuracy_std']:.2%}")
    print(f"  Per-Tier: T0={b['tier_0_accuracy']:.2%}, T1={b['tier_1_accuracy']:.2%}")
    
    print(f"\nBest by Minimum Tier Accuracy (ensures both tiers are good):")
    b = optimal_results['best_by_min_tier']
    print(f"  Percentile: {b['percentile']}th (${b['threshold_dollars']:,.0f})")
    print(f"  Min Tier Accuracy: {b['min_tier_accuracy']:.2%}")
    print(f"  Per-Tier: T0={b['tier_0_accuracy']:.2%}, T1={b['tier_1_accuracy']:.2%}")
    
    print(f"\nBest by Balanced Accuracy:")
    b = optimal_results['best_by_balanced']
    print(f"  Percentile: {b['percentile']}th (${b['threshold_dollars']:,.0f})")
    print(f"  Balanced Accuracy: {b['balanced_accuracy']:.2%}")
    print(f"  Per-Tier: T0={b['tier_0_accuracy']:.2%}, T1={b['tier_1_accuracy']:.2%}")
    
    # Fine-tune if requested or automatically around best
    fine_tune_results = []
    if args.fine_tune or True:  # Always fine-tune for best results
        best_pct = optimal_results['best_by_min_tier']['percentile']
        
        print("\n" + "-" * 70)
        print(" FINE-TUNING (1% steps)")
        print("-" * 70)
        
        fine_tune_results = fine_tune_around_optimum(X, y, best_pct, range_width=5, step=1)
        
        # Find best from fine-tuning
        best_fine = max(fine_tune_results, key=lambda x: x['min_tier_accuracy'])
        
        print(f"\nBest from fine-tuning:")
        print(f"  Percentile: {best_fine['percentile']:.1f}th (${best_fine['threshold_dollars']:,.0f})")
        print(f"  CV Accuracy: {best_fine['cv_accuracy_mean']:.2%}")
        print(f"  Per-Tier: T0={best_fine['tier_0_accuracy']:.2%}, T1={best_fine['tier_1_accuracy']:.2%}")
        print(f"  Min Tier: {best_fine['min_tier_accuracy']:.2%}")
    
    # Feature separability analysis
    print("\n" + "-" * 70)
    print(" FEATURE SEPARABILITY AT OPTIMAL SPLIT")
    print("-" * 70)
    
    best_pct = optimal_results['best_by_min_tier']['percentile']
    separability = analyze_feature_separability(X, y, best_pct)
    
    print(f"\nTop features separating tiers at {best_pct}th percentile:")
    print(f"{'Feature':30} {'Cohens d':>10} {'Interpretation':>20}")
    print("-" * 60)
    
    for feature, stats in separability['top_separating_features'][:10]:
        d = stats['cohens_d']
        if d > 0.8:
            interp = "Large effect"
        elif d > 0.5:
            interp = "Medium effect"
        elif d > 0.2:
            interp = "Small effect"
        else:
            interp = "Negligible"
        print(f"{feature:30} {d:>10.3f} {interp:>20}")
    
    # Final recommendation
    print("\n" + "=" * 70)
    print(" RECOMMENDATION")
    print("=" * 70)
    
    # Choose the best split
    if fine_tune_results:
        final_best = max(fine_tune_results, key=lambda x: x['min_tier_accuracy'])
    else:
        final_best = optimal_results['best_by_min_tier']
    
    print(f"\n  OPTIMAL PERCENTILE: {final_best['percentile']}th")
    print(f"  Current Threshold: ${final_best['threshold_dollars']:,.0f}")
    print(f"  Classification Accuracy: {final_best['cv_accuracy_mean']:.1%}")
    print(f"  Tier 0 (Lower): {final_best['tier_0_accuracy']:.1%} ({final_best['tier_0_pct']:.1f}% of data)")
    print(f"  Tier 1 (Higher): {final_best['tier_1_accuracy']:.1%} ({final_best['tier_1_pct']:.1f}% of data)")
    
    print("\n  Advantages of percentile-based split:")
    print("  - Resilient to price appreciation over time")
    print("  - Automatically adjusts with retraining")
    print("  - No hardcoded dollar thresholds")
    
    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    results_to_save = {
        'timestamp': timestamp,
        'optimal_percentile': final_best['percentile'],
        'coarse_search': all_results,
        'fine_tune_results': fine_tune_results,
        'optimal_results': optimal_results,
        'final_recommendation': final_best
    }
    
    # Convert numpy types
    def convert(obj):
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj
    
    with open(OUTPUT_DIR / f'tier_split_optimization_{timestamp}.json', 'w') as f:
        json.dump(convert(results_to_save), f, indent=2)
    
    print(f"\n  Results saved to: {OUTPUT_DIR / f'tier_split_optimization_{timestamp}.json'}")
    
    print("\n" + "=" * 70)
    
    return final_best


if __name__ == "__main__":
    main()
