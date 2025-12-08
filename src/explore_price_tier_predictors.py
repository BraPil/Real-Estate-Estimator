"""
V2.7 Exploration: Price Tier Prediction Analysis

Goal: Discover which features and feature combinations can predict price tiers
BEFORE we make the actual price prediction, enabling tiered model routing.

The Challenge:
- We want separate models for different price ranges (e.g., <$500k, $500k-$1M, >$1M)
- But we need to know the price tier BEFORE predicting to route correctly
- This script explores: Can we classify price tiers from features alone?

Analyses:
1. Individual feature correlations with price
2. Feature importance for tier classification
3. Non-linear relationships (polynomial features)
4. Feature interactions
5. Tier classifier accuracy experiments

Usage:
    python src/explore_price_tier_predictors.py
    python src/explore_price_tier_predictors.py --tiers 3
    python src/explore_price_tier_predictors.py --tiers 5 --output-dir output/tier_analysis
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
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")

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


def create_price_tiers(y: pd.Series, n_tiers: int = 3) -> Tuple[pd.Series, Dict]:
    """
    Create price tier labels.
    
    Args:
        y: Price series
        n_tiers: Number of tiers (3, 4, or 5)
        
    Returns:
        Tier labels and tier info dict
    """
    if n_tiers == 3:
        # Budget / Mid-range / Premium
        bins = [0, 400000, 750000, float('inf')]
        labels = ['budget', 'mid_range', 'premium']
    elif n_tiers == 4:
        bins = [0, 350000, 550000, 900000, float('inf')]
        labels = ['budget', 'mid_low', 'mid_high', 'premium']
    elif n_tiers == 5:
        bins = [0, 300000, 500000, 750000, 1000000, float('inf')]
        labels = ['budget', 'mid_low', 'mid_range', 'mid_high', 'luxury']
    else:
        # Use quantiles for custom tier counts
        bins = [0] + list(y.quantile(np.linspace(0, 1, n_tiers + 1)[1:-1])) + [float('inf')]
        labels = [f'tier_{i+1}' for i in range(n_tiers)]
    
    tiers = pd.cut(y, bins=bins, labels=labels)
    
    tier_info = {
        'n_tiers': n_tiers,
        'bins': bins,
        'labels': labels,
        'distribution': tiers.value_counts().to_dict()
    }
    
    return tiers, tier_info


# =============================================================================
# ANALYSIS 1: Individual Feature Correlations
# =============================================================================

def analyze_feature_correlations(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Calculate correlation of each feature with price.
    """
    logger.info("\n" + "="*60)
    logger.info("ANALYSIS 1: Feature Correlations with Price")
    logger.info("="*60)
    
    correlations = []
    for col in X.columns:
        # Pearson correlation (linear)
        pearson_r, pearson_p = stats.pearsonr(X[col], y)
        
        # Spearman correlation (monotonic, handles non-linear)
        spearman_r, spearman_p = stats.spearmanr(X[col], y)
        
        correlations.append({
            'feature': col,
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'abs_pearson': abs(pearson_r),
            'abs_spearman': abs(spearman_r)
        })
    
    df = pd.DataFrame(correlations).sort_values('abs_spearman', ascending=False)
    
    logger.info("\nTop 15 Features by Spearman Correlation:")
    for _, row in df.head(15).iterrows():
        logger.info(f"  {row['feature']:25} Pearson={row['pearson_r']:+.4f}  Spearman={row['spearman_r']:+.4f}")
    
    return df


# =============================================================================
# ANALYSIS 2: Feature Importance for Tier Classification
# =============================================================================

def analyze_tier_classification_importance(
    X: pd.DataFrame, 
    tiers: pd.Series
) -> pd.DataFrame:
    """
    Use Random Forest to find most important features for tier classification.
    """
    logger.info("\n" + "="*60)
    logger.info("ANALYSIS 2: Feature Importance for Tier Classification")
    logger.info("="*60)
    
    # Scale features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit Random Forest classifier
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf.fit(X_scaled, tiers)
    
    # Get feature importances
    importances = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("\nTop 15 Features for Tier Classification:")
    for _, row in importances.head(15).iterrows():
        logger.info(f"  {row['feature']:25} importance={row['importance']:.4f}")
    
    # Cross-validation score
    cv_scores = cross_val_score(rf, X_scaled, tiers, cv=5, scoring='accuracy')
    logger.info(f"\nRandom Forest Tier Classification Accuracy: {cv_scores.mean():.1%} +/- {cv_scores.std():.1%}")
    
    return importances


# =============================================================================
# ANALYSIS 3: Non-linear Relationships (Polynomial Features)
# =============================================================================

def analyze_polynomial_features(
    X: pd.DataFrame, 
    y: pd.Series,
    top_features: List[str]
) -> Dict:
    """
    Explore polynomial (squared, cubed) relationships for top features.
    """
    logger.info("\n" + "="*60)
    logger.info("ANALYSIS 3: Non-linear (Polynomial) Relationships")
    logger.info("="*60)
    
    results = {}
    
    for feature in top_features[:10]:  # Top 10 features
        # Original correlation
        base_r, _ = stats.spearmanr(X[feature], y)
        
        # Squared
        squared = X[feature] ** 2
        squared_r, _ = stats.spearmanr(squared, y)
        
        # Log (if all positive)
        if (X[feature] > 0).all():
            log_vals = np.log(X[feature])
            log_r, _ = stats.spearmanr(log_vals, y)
        else:
            log_r = None
        
        # Square root (if all non-negative)
        if (X[feature] >= 0).all():
            sqrt_vals = np.sqrt(X[feature])
            sqrt_r, _ = stats.spearmanr(sqrt_vals, y)
        else:
            sqrt_r = None
        
        results[feature] = {
            'base': base_r,
            'squared': squared_r,
            'log': log_r,
            'sqrt': sqrt_r
        }
        
        best = max(
            [(abs(base_r), 'base'), (abs(squared_r), 'squared')] +
            ([(abs(log_r), 'log')] if log_r else []) +
            ([(abs(sqrt_r), 'sqrt')] if sqrt_r else []),
            key=lambda x: x[0]
        )
        
        improvement = (best[0] - abs(base_r)) / abs(base_r) * 100 if base_r != 0 else 0
        
        logger.info(f"  {feature:25} base={base_r:+.4f}  best={best[1]}({best[0]:.4f}) "
                   f"{'[+' + f'{improvement:.1f}%]' if improvement > 1 else ''}")
    
    return results


# =============================================================================
# ANALYSIS 4: Feature Interactions
# =============================================================================

def analyze_feature_interactions(
    X: pd.DataFrame, 
    y: pd.Series,
    top_features: List[str]
) -> pd.DataFrame:
    """
    Explore two-feature interactions that might predict price better.
    """
    logger.info("\n" + "="*60)
    logger.info("ANALYSIS 4: Feature Interactions")
    logger.info("="*60)
    
    interactions = []
    top_10 = top_features[:10]
    
    for i, feat1 in enumerate(top_10):
        for feat2 in top_10[i+1:]:
            # Multiplication interaction
            mult = X[feat1] * X[feat2]
            mult_r, _ = stats.spearmanr(mult, y)
            
            # Ratio interaction (if denominator safe)
            if (X[feat2] != 0).all():
                ratio = X[feat1] / X[feat2]
                ratio_r, _ = stats.spearmanr(ratio, y)
            else:
                ratio_r = 0
            
            # Individual correlations
            r1, _ = stats.spearmanr(X[feat1], y)
            r2, _ = stats.spearmanr(X[feat2], y)
            
            # Does interaction beat both individual features?
            best_individual = max(abs(r1), abs(r2))
            mult_improvement = (abs(mult_r) - best_individual) / best_individual * 100
            ratio_improvement = (abs(ratio_r) - best_individual) / best_individual * 100
            
            interactions.append({
                'feature1': feat1,
                'feature2': feat2,
                'individual_best': best_individual,
                'multiply_r': mult_r,
                'ratio_r': ratio_r,
                'multiply_improvement': mult_improvement,
                'ratio_improvement': ratio_improvement
            })
    
    df = pd.DataFrame(interactions)
    
    # Find interactions that improve over individual features
    good_interactions = df[
        (df['multiply_improvement'] > 5) | (df['ratio_improvement'] > 5)
    ].sort_values('multiply_improvement', ascending=False)
    
    if len(good_interactions) > 0:
        logger.info("\nInteractions that improve over individual features (>5%):")
        for _, row in good_interactions.head(10).iterrows():
            logger.info(f"  {row['feature1']} x {row['feature2']}: "
                       f"mult={row['multiply_r']:.4f} (+{row['multiply_improvement']:.1f}%)")
    else:
        logger.info("\nNo significant interaction improvements found (>5%)")
    
    return df


# =============================================================================
# ANALYSIS 5: Tier Classifier Experiments
# =============================================================================

def tier_classifier_experiments(
    X: pd.DataFrame,
    tiers: pd.Series,
    tier_info: Dict
) -> Dict:
    """
    Test various classifiers for tier prediction accuracy.
    """
    logger.info("\n" + "="*60)
    logger.info("ANALYSIS 5: Tier Classifier Experiments")
    logger.info("="*60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, tiers, test_size=0.2, random_state=RANDOM_STATE, stratify=tiers
    )
    
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, 
                                                 random_state=RANDOM_STATE, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5,
                                                         random_state=RANDOM_STATE)
    }
    
    results = {}
    best_clf = None
    best_acc = 0
    
    logger.info(f"\nTier Distribution: {tier_info['distribution']}")
    logger.info(f"\nClassifier Comparison ({tier_info['n_tiers']} tiers):")
    
    for name, clf in classifiers.items():
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        
        acc = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=5, scoring='accuracy')
        
        results[name] = {
            'test_accuracy': acc,
            'cv_accuracy_mean': cv_scores.mean(),
            'cv_accuracy_std': cv_scores.std()
        }
        
        logger.info(f"  {name:25} Test: {acc:.1%}  CV: {cv_scores.mean():.1%} +/- {cv_scores.std():.1%}")
        
        if acc > best_acc:
            best_acc = acc
            best_clf = (name, clf, y_pred)
    
    # Detailed report for best classifier
    if best_clf:
        logger.info(f"\nBest Classifier: {best_clf[0]} ({best_acc:.1%})")
        logger.info("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, best_clf[2], labels=tier_info['labels'])
        
        # Print confusion matrix
        header = "Predicted:".ljust(15) + "  ".join([f"{l[:8]:>8}" for l in tier_info['labels']])
        logger.info(f"  {header}")
        for i, label in enumerate(tier_info['labels']):
            row = f"  Actual {label[:8]:8}:" + "  ".join([f"{cm[i,j]:8}" for j in range(len(tier_info['labels']))])
            logger.info(row)
        
        # Per-tier accuracy
        logger.info("\nPer-Tier Accuracy:")
        for i, label in enumerate(tier_info['labels']):
            tier_correct = cm[i, i]
            tier_total = cm[i, :].sum()
            tier_acc = tier_correct / tier_total if tier_total > 0 else 0
            logger.info(f"  {label:15}: {tier_acc:.1%} ({tier_correct}/{tier_total})")
    
    return results


# =============================================================================
# ANALYSIS 6: Simple Rules Discovery
# =============================================================================

def discover_simple_rules(
    X: pd.DataFrame,
    y: pd.Series,
    tiers: pd.Series,
    tier_info: Dict
) -> Dict:
    """
    Discover simple rules that can classify tiers with high accuracy.
    """
    logger.info("\n" + "="*60)
    logger.info("ANALYSIS 6: Simple Rules Discovery")
    logger.info("="*60)
    
    rules = []
    
    # Analyze thresholds for top features
    top_features = ['sqft_living', 'grade', 'sqft_above', 'bathrooms', 'sqft_living15', 'lat']
    
    for feature in top_features:
        if feature not in X.columns:
            continue
            
        # Find optimal thresholds using percentiles
        for percentile in [25, 50, 75]:
            threshold = X[feature].quantile(percentile / 100)
            
            # Binary classification: above vs below threshold
            pred_high = X[feature] >= threshold
            actual_high = y >= y.median()
            
            # Calculate accuracy of this simple rule
            rule_accuracy = (pred_high == actual_high).mean()
            
            rules.append({
                'feature': feature,
                'threshold': threshold,
                'percentile': percentile,
                'accuracy': rule_accuracy
            })
    
    # Sort by accuracy
    rules_df = pd.DataFrame(rules).sort_values('accuracy', ascending=False)
    
    logger.info("\nSimple Threshold Rules (Binary High/Low Classification):")
    for _, row in rules_df.head(10).iterrows():
        logger.info(f"  {row['feature']:20} >= {row['threshold']:>12,.0f} (p{row['percentile']:02d}) "
                   f"-> Accuracy: {row['accuracy']:.1%}")
    
    # Try combined rules
    logger.info("\nCombined Rules (AND conditions):")
    
    # Find best two-feature rules
    best_combined = []
    features_to_try = ['sqft_living', 'grade', 'bathrooms', 'lat']
    
    for i, f1 in enumerate(features_to_try):
        if f1 not in X.columns:
            continue
        for f2 in features_to_try[i+1:]:
            if f2 not in X.columns:
                continue
            
            t1 = X[f1].median()
            t2 = X[f2].median()
            
            pred_high = (X[f1] >= t1) & (X[f2] >= t2)
            actual_high = y >= y.median()
            
            acc = (pred_high == actual_high).mean()
            best_combined.append({
                'rule': f'{f1} >= {t1:.0f} AND {f2} >= {t2:.2f}',
                'accuracy': acc
            })
    
    combined_df = pd.DataFrame(best_combined).sort_values('accuracy', ascending=False)
    for _, row in combined_df.head(5).iterrows():
        logger.info(f"  {row['rule']} -> {row['accuracy']:.1%}")
    
    return {'threshold_rules': rules_df.to_dict('records'),
            'combined_rules': combined_df.to_dict('records')}


# =============================================================================
# ANALYSIS 7: Engineered Features for Tier Prediction
# =============================================================================

def engineer_tier_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features specifically for tier prediction.
    """
    logger.info("\n" + "="*60)
    logger.info("ANALYSIS 7: Engineered Features for Tier Prediction")
    logger.info("="*60)
    
    X_eng = X.copy()
    
    # Size ratios
    if 'sqft_living' in X.columns and 'sqft_living15' in X.columns:
        X_eng['relative_size'] = X['sqft_living'] / X['sqft_living15']
        
    if 'sqft_lot' in X.columns and 'sqft_lot15' in X.columns:
        X_eng['relative_lot'] = X['sqft_lot'] / (X['sqft_lot15'] + 1)
    
    # Quality indicators
    if 'grade' in X.columns and 'condition' in X.columns:
        X_eng['quality_score'] = X['grade'] * X['condition']
    
    # Space efficiency
    if 'sqft_living' in X.columns and 'bedrooms' in X.columns:
        X_eng['sqft_per_bedroom'] = X['sqft_living'] / (X['bedrooms'] + 1)
    
    if 'bathrooms' in X.columns and 'bedrooms' in X.columns:
        X_eng['bath_bed_ratio'] = X['bathrooms'] / (X['bedrooms'] + 1)
    
    # Location value proxy
    if 'lat' in X.columns:
        # Higher lat = closer to Seattle center (roughly)
        X_eng['lat_premium'] = (X['lat'] - X['lat'].min()) / (X['lat'].max() - X['lat'].min())
    
    # Age features
    if 'yr_built' in X.columns:
        X_eng['age'] = 2015 - X['yr_built']
        X_eng['is_new'] = (X['yr_built'] >= 2010).astype(int)
    
    # Luxury indicators
    if 'waterfront' in X.columns and 'view' in X.columns:
        X_eng['luxury_score'] = X['waterfront'] * 10 + X['view']
    
    new_features = [c for c in X_eng.columns if c not in X.columns]
    logger.info(f"\nCreated {len(new_features)} engineered features:")
    for f in new_features:
        logger.info(f"  - {f}")
    
    return X_eng


def test_engineered_features(
    X_original: pd.DataFrame,
    X_engineered: pd.DataFrame,
    tiers: pd.Series
) -> Dict:
    """
    Compare tier classification with and without engineered features.
    """
    scaler = RobustScaler()
    
    # Original features
    X_orig_scaled = scaler.fit_transform(X_original)
    rf_orig = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1)
    cv_orig = cross_val_score(rf_orig, X_orig_scaled, tiers, cv=5, scoring='accuracy')
    
    # Engineered features
    X_eng_scaled = scaler.fit_transform(X_engineered)
    rf_eng = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1)
    cv_eng = cross_val_score(rf_eng, X_eng_scaled, tiers, cv=5, scoring='accuracy')
    
    improvement = (cv_eng.mean() - cv_orig.mean()) / cv_orig.mean() * 100
    
    logger.info(f"\nTier Classification Comparison:")
    logger.info(f"  Original features:   {cv_orig.mean():.1%} +/- {cv_orig.std():.1%}")
    logger.info(f"  With engineered:     {cv_eng.mean():.1%} +/- {cv_eng.std():.1%}")
    logger.info(f"  Improvement:         {improvement:+.1f}%")
    
    return {
        'original_accuracy': cv_orig.mean(),
        'engineered_accuracy': cv_eng.mean(),
        'improvement_pct': improvement
    }


# =============================================================================
# MAIN
# =============================================================================

def save_results(results: Dict, output_dir: Path, filename: str):
    """Save results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / filename
    
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj
    
    with open(filepath, 'w') as f:
        json.dump(convert(results), f, indent=2)
    
    logger.info(f"\nResults saved to {filepath}")


def main():
    parser = argparse.ArgumentParser(description='V2.7 Exploration: Price Tier Predictors')
    parser.add_argument('--tiers', type=int, default=3, help='Number of price tiers (3, 4, or 5)')
    parser.add_argument('--output-dir', type=str, default='output/tier_analysis', 
                        help='Output directory for results')
    args = parser.parse_args()
    
    print("\n")
    print("=" * 70)
    print(" V2.7 EXPLORATION: PRICE TIER PREDICTION ANALYSIS")
    print("=" * 70)
    print(f" Number of Tiers: {args.tiers}")
    print("=" * 70)
    
    # Load data
    X, y = load_data()
    
    # Create price tiers
    tiers, tier_info = create_price_tiers(y, n_tiers=args.tiers)
    logger.info(f"\nPrice Tier Distribution:")
    for label, count in tier_info['distribution'].items():
        pct = count / len(y) * 100
        logger.info(f"  {label:15}: {count:5} ({pct:.1f}%)")
    
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'n_tiers': args.tiers,
        'tier_info': tier_info,
        'analyses': {}
    }
    
    # Analysis 1: Feature correlations
    correlations = analyze_feature_correlations(X, y)
    all_results['analyses']['correlations'] = correlations.head(20).to_dict('records')
    
    # Get top features for subsequent analyses
    top_features = correlations.head(15)['feature'].tolist()
    
    # Analysis 2: Feature importance for classification
    importances = analyze_tier_classification_importance(X, tiers)
    all_results['analyses']['classification_importance'] = importances.head(15).to_dict('records')
    
    # Analysis 3: Polynomial features
    poly_results = analyze_polynomial_features(X, y, top_features)
    all_results['analyses']['polynomial'] = poly_results
    
    # Analysis 4: Feature interactions
    interactions = analyze_feature_interactions(X, y, top_features)
    all_results['analyses']['interactions'] = interactions.head(20).to_dict('records')
    
    # Analysis 5: Classifier experiments
    classifier_results = tier_classifier_experiments(X, tiers, tier_info)
    all_results['analyses']['classifiers'] = classifier_results
    
    # Analysis 6: Simple rules
    rules_results = discover_simple_rules(X, y, tiers, tier_info)
    all_results['analyses']['simple_rules'] = rules_results
    
    # Analysis 7: Engineered features
    X_engineered = engineer_tier_features(X)
    eng_results = test_engineered_features(X, X_engineered, tiers)
    all_results['analyses']['engineered_features'] = eng_results
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_results(all_results, Path(args.output_dir), f'tier_analysis_{args.tiers}tiers_{timestamp}.json')
    
    # Final summary
    print("\n")
    print("=" * 70)
    print(" SUMMARY & RECOMMENDATIONS")
    print("=" * 70)
    
    best_classifier_acc = max(v['test_accuracy'] for v in classifier_results.values())
    
    print(f"\n1. TIER CLASSIFICATION FEASIBILITY:")
    print(f"   Best classifier accuracy: {best_classifier_acc:.1%}")
    if best_classifier_acc >= 0.75:
        print("   --> GOOD: Tier prediction is feasible for routing")
    elif best_classifier_acc >= 0.60:
        print("   --> MODERATE: Tier prediction possible but with errors")
    else:
        print("   --> CHALLENGING: Tier prediction has significant overlap")
    
    print(f"\n2. TOP PREDICTIVE FEATURES:")
    for i, row in correlations.head(5).iterrows():
        print(f"   - {row['feature']}: r={row['spearman_r']:.3f}")
    
    print(f"\n3. ENGINEERED FEATURES IMPACT:")
    print(f"   Improvement: {eng_results['improvement_pct']:+.1f}%")
    
    print(f"\n4. NEXT STEPS FOR V2.7:")
    print("   a. Build tier classifier using top features")
    print("   b. Train separate models per tier")
    print("   c. Route predictions through classifier first")
    print("   d. Compare against single-model baseline")
    
    print("\n" + "=" * 70)
    
    return all_results


if __name__ == "__main__":
    main()
