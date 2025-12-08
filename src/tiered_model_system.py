"""
V2.7: Two-Tier Model System

A tiered prediction system that:
1. Classifies homes into 2 price tiers (91.4% accuracy at 50th percentile)
2. Routes to specialized XGBoost models per tier
3. Compares against single-model baseline (V2.5)

Architecture:
    Input Features → Tier Classifier (GB) → Tier 0 Model OR Tier 1 Model → Prediction

Tier Split Configuration (Scientifically Validated):
    - Split: 50th percentile (median) - validated via optimize_tier_split.py
    - Tier 0 (Lower): 91.7% classification accuracy, ~50% of data
    - Tier 1 (Higher): 92.5% classification accuracy, ~50% of data
    - Gap: Only 0.9% between tiers (most balanced split)
    - Percentile-based: Resilient to price appreciation over time

Why 50th percentile?
    - Tested 25th-75th percentile in 5% steps + 1% fine-tuning
    - Higher percentiles (75th) have higher CV accuracy (93.7%) but imbalanced tiers
    - 50th percentile optimizes for BOTH tiers being accurate (min tier = 91.7%)
    - See: output/tier_optimization/tier_split_optimization_*.json

Usage:
    # Train the tiered system
    python src/tiered_model_system.py --mode train
    
    # Evaluate and compare
    python src/tiered_model_system.py --mode evaluate
    
    # Full pipeline (train + evaluate)
    python src/tiered_model_system.py --mode full
"""

import argparse
import json
import logging
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, classification_report
)
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path("data")
MODEL_DIR = Path("model")
OUTPUT_DIR = Path("output/tiered_models")

RANDOM_STATE = 42

# Validated tier split configuration
# Tested via optimize_tier_split.py on 2025-12-08
# 50th percentile provides best balanced accuracy (91.7% / 92.5% per tier)
TIER_SPLIT_PERCENTILE = 50


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data() -> Tuple[pd.DataFrame, pd.Series, List[str]]:
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
    return X, y, list(X.columns)


def create_tiers(y: pd.Series, percentile: float = TIER_SPLIT_PERCENTILE) -> Tuple[pd.Series, float]:
    """
    Create 2-tier labels based on validated percentile split.
    
    The 50th percentile (median) was scientifically validated as optimal:
    - Tested 25th-75th percentile via optimize_tier_split.py
    - 50th provides best balanced accuracy (91.7% T0, 92.5% T1)
    - Only 0.9% gap between tier accuracies (most balanced)
    - Percentile-based approach is resilient to price appreciation
    
    Args:
        y: Price series
        percentile: Percentile for split (default: 50, validated optimal)
    
    Returns:
        tiers: Series with tier labels (0 = lower, 1 = higher)
        threshold: The price threshold at the given percentile
    """
    threshold = y.quantile(percentile / 100)
    tiers = (y >= threshold).astype(int)
    
    logger.info(f"Tier split: {percentile}th percentile (validated optimal)")
    logger.info(f"Tier threshold: ${threshold:,.0f}")
    logger.info(f"Tier 0 (lower):  {(tiers == 0).sum()} samples ({(tiers == 0).mean():.1%})")
    logger.info(f"Tier 1 (higher): {(tiers == 1).sum()} samples ({(tiers == 1).mean():.1%})")
    
    return tiers, threshold


# =============================================================================
# V2.5 BASELINE (Single Model)
# =============================================================================

def get_v25_xgboost_params() -> dict:
    """Get the V2.5 tuned XGBoost parameters."""
    return {
        'n_estimators': 239,
        'max_depth': 7,
        'learning_rate': 0.0863,
        'subsample': 0.7472,
        'colsample_bytree': 0.8388,
        'min_child_weight': 6,
        'gamma': 0.1589,
        'reg_alpha': 0.2791,
        'reg_lambda': 1.3826,
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'verbosity': 0
    }


def train_baseline_model(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> Pipeline:
    """Train the V2.5 baseline single model."""
    logger.info("\nTraining V2.5 Baseline (Single Model)...")
    
    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('model', XGBRegressor(**get_v25_xgboost_params()))
    ])
    
    pipeline.fit(X_train, y_train)
    return pipeline


# =============================================================================
# TIERED MODEL SYSTEM
# =============================================================================

class TieredModelSystem:
    """
    Two-tier prediction system with specialized models per tier.
    
    Configuration (Validated 2025-12-08):
        - Split: 50th percentile (median)
        - Tier 0: Lower prices, 91.7% classification accuracy
        - Tier 1: Higher prices, 92.5% classification accuracy
        - Overall: 91.4% CV accuracy with only 0.9% gap between tiers
    """
    
    def __init__(self):
        self.tier_classifier = None
        self.tier_scaler = None
        self.tier_0_model = None  # Lower price tier (<= threshold)
        self.tier_1_model = None  # Higher price tier (> threshold)
        self.threshold = None
        self.percentile = TIER_SPLIT_PERCENTILE  # Validated optimal: 50
        self.feature_names = None
        
    def train_tier_classifier(
        self,
        X: pd.DataFrame,
        tiers: pd.Series
    ) -> float:
        """Train the tier classifier (Gradient Boosting)."""
        logger.info("\nTraining Tier Classifier...")
        
        self.tier_scaler = RobustScaler()
        X_scaled = self.tier_scaler.fit_transform(X)
        
        self.tier_classifier = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=RANDOM_STATE
        )
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.tier_classifier, X_scaled, tiers, 
            cv=5, scoring='accuracy'
        )
        
        # Fit on full training data
        self.tier_classifier.fit(X_scaled, tiers)
        
        logger.info(f"Tier Classifier CV Accuracy: {cv_scores.mean():.1%} +/- {cv_scores.std():.1%}")
        
        return cv_scores.mean()
    
    def train_tier_models(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        tiers: pd.Series
    ):
        """Train specialized XGBoost models for each tier."""
        
        # Get tier-specific data
        tier_0_mask = tiers == 0
        tier_1_mask = tiers == 1
        
        X_tier_0, y_tier_0 = X[tier_0_mask], y[tier_0_mask]
        X_tier_1, y_tier_1 = X[tier_1_mask], y[tier_1_mask]
        
        logger.info(f"\nTraining Tier 0 Model (Lower Prices)...")
        logger.info(f"  Samples: {len(X_tier_0)}, Price range: ${y_tier_0.min():,.0f} - ${y_tier_0.max():,.0f}")
        
        # Tier 0: Lower prices - may need different hyperparameters
        self.tier_0_model = Pipeline([
            ('scaler', RobustScaler()),
            ('model', XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=0
            ))
        ])
        self.tier_0_model.fit(X_tier_0, y_tier_0)
        
        logger.info(f"\nTraining Tier 1 Model (Higher Prices)...")
        logger.info(f"  Samples: {len(X_tier_1)}, Price range: ${y_tier_1.min():,.0f} - ${y_tier_1.max():,.0f}")
        
        # Tier 1: Higher prices - more complex, may need more trees
        self.tier_1_model = Pipeline([
            ('scaler', RobustScaler()),
            ('model', XGBRegressor(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.08,
                subsample=0.75,
                colsample_bytree=0.85,
                min_child_weight=4,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=0
            ))
        ])
        self.tier_1_model.fit(X_tier_1, y_tier_1)
        
        self.feature_names = list(X.columns)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict using the tiered system:
        1. Classify tier
        2. Route to appropriate model
        """
        # Classify tiers
        X_scaled = self.tier_scaler.transform(X)
        predicted_tiers = self.tier_classifier.predict(X_scaled)
        
        # Initialize predictions
        predictions = np.zeros(len(X))
        
        # Route to tier-specific models
        tier_0_mask = predicted_tiers == 0
        tier_1_mask = predicted_tiers == 1
        
        if tier_0_mask.sum() > 0:
            predictions[tier_0_mask] = self.tier_0_model.predict(X[tier_0_mask])
        
        if tier_1_mask.sum() > 0:
            predictions[tier_1_mask] = self.tier_1_model.predict(X[tier_1_mask])
        
        return predictions, predicted_tiers
    
    def save(self, output_dir: Path):
        """Save the tiered model system."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        artifacts = {
            'tier_classifier': self.tier_classifier,
            'tier_scaler': self.tier_scaler,
            'tier_0_model': self.tier_0_model,
            'tier_1_model': self.tier_1_model,
            'threshold': self.threshold,
            'percentile': self.percentile,
            'feature_names': self.feature_names,
            'config': {
                'split_method': 'percentile',
                'split_percentile': self.percentile,
                'validated_date': '2025-12-08',
                'validation_script': 'optimize_tier_split.py'
            }
        }
        
        with open(output_dir / 'tiered_model_system.pkl', 'wb') as f:
            pickle.dump(artifacts, f)
        
        logger.info(f"Tiered model system saved to {output_dir / 'tiered_model_system.pkl'}")
    
    def load(self, filepath: Path):
        """Load a saved tiered model system."""
        with open(filepath, 'rb') as f:
            artifacts = pickle.load(f)
        
        self.tier_classifier = artifacts['tier_classifier']
        self.tier_scaler = artifacts['tier_scaler']
        self.tier_0_model = artifacts['tier_0_model']
        self.tier_1_model = artifacts['tier_1_model']
        self.threshold = artifacts['threshold']
        self.percentile = artifacts.get('percentile', 50)
        self.feature_names = artifacts['feature_names']
        
        logger.info(f"Tiered model system loaded from {filepath}")
        logger.info(f"Split: {self.percentile}th percentile (${self.threshold:,.0f})")


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_model(
    y_true: pd.Series,
    y_pred: np.ndarray,
    name: str
) -> Dict:
    """Calculate evaluation metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'name': name,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape
    }


def evaluate_by_tier(
    y_true: pd.Series,
    y_pred: np.ndarray,
    actual_tiers: pd.Series,
    predicted_tiers: np.ndarray
) -> Dict:
    """Evaluate performance by actual tier."""
    results = {}
    
    for tier in [0, 1]:
        mask = actual_tiers == tier
        if mask.sum() > 0:
            tier_name = 'lower' if tier == 0 else 'higher'
            results[f'tier_{tier_name}'] = {
                'count': int(mask.sum()),
                'mae': float(mean_absolute_error(y_true[mask], y_pred[mask])),
                'r2': float(r2_score(y_true[mask], y_pred[mask])),
                'price_range': f"${y_true[mask].min():,.0f} - ${y_true[mask].max():,.0f}"
            }
    
    # Routing accuracy
    routing_accuracy = accuracy_score(actual_tiers, predicted_tiers)
    results['routing_accuracy'] = float(routing_accuracy)
    
    # Misrouted analysis
    misrouted_mask = actual_tiers != predicted_tiers
    if misrouted_mask.sum() > 0:
        results['misrouted'] = {
            'count': int(misrouted_mask.sum()),
            'mae': float(mean_absolute_error(y_true[misrouted_mask], y_pred[misrouted_mask])),
            'pct_of_total': float(misrouted_mask.mean() * 100)
        }
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def train_pipeline(X: pd.DataFrame, y: pd.Series):
    """Train both baseline and tiered models."""
    
    # Create train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    
    # Create tier labels
    tiers_train, threshold = create_tiers(y_train)
    tiers_test = (y_test >= threshold).astype(int)
    
    # Train baseline
    baseline_model = train_baseline_model(X_train, y_train)
    
    # Train tiered system
    tiered_system = TieredModelSystem()
    tiered_system.threshold = threshold
    tiered_system.train_tier_classifier(X_train, tiers_train)
    tiered_system.train_tier_models(X_train, y_train, tiers_train)
    
    # Save models
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_DIR / 'baseline_model.pkl', 'wb') as f:
        pickle.dump(baseline_model, f)
    
    tiered_system.save(OUTPUT_DIR)
    
    return baseline_model, tiered_system, X_test, y_test, tiers_test, threshold


def evaluate_pipeline(
    baseline_model,
    tiered_system: TieredModelSystem,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    tiers_test: pd.Series
):
    """Evaluate and compare models."""
    
    print("\n")
    print("=" * 70)
    print(" V2.7 TIERED MODEL EVALUATION")
    print("=" * 70)
    
    # Baseline predictions
    baseline_pred = baseline_model.predict(X_test)
    baseline_metrics = evaluate_model(y_test, baseline_pred, "V2.5 Baseline (Single Model)")
    
    # Tiered predictions
    tiered_pred, predicted_tiers = tiered_system.predict(X_test)
    tiered_metrics = evaluate_model(y_test, tiered_pred, "V2.7 Tiered System")
    
    # Print comparison
    print("\n" + "-" * 70)
    print(" OVERALL COMPARISON")
    print("-" * 70)
    print(f"{'Metric':<20} {'Baseline':>20} {'Tiered':>20} {'Improvement':>12}")
    print("-" * 70)
    
    mae_imp = (baseline_metrics['mae'] - tiered_metrics['mae']) / baseline_metrics['mae'] * 100
    rmse_imp = (baseline_metrics['rmse'] - tiered_metrics['rmse']) / baseline_metrics['rmse'] * 100
    r2_imp = (tiered_metrics['r2'] - baseline_metrics['r2']) / baseline_metrics['r2'] * 100
    
    print(f"{'MAE':<20} ${baseline_metrics['mae']:>18,.0f} ${tiered_metrics['mae']:>18,.0f} {mae_imp:>+11.1f}%")
    print(f"{'RMSE':<20} ${baseline_metrics['rmse']:>18,.0f} ${tiered_metrics['rmse']:>18,.0f} {rmse_imp:>+11.1f}%")
    print(f"{'R²':<20} {baseline_metrics['r2']:>20.4f} {tiered_metrics['r2']:>20.4f} {r2_imp:>+11.1f}%")
    print(f"{'MAPE':<20} {baseline_metrics['mape']:>19.2f}% {tiered_metrics['mape']:>19.2f}%")
    
    # Tier-specific analysis
    tier_results = evaluate_by_tier(y_test, tiered_pred, tiers_test, predicted_tiers)
    
    print("\n" + "-" * 70)
    print(" TIERED SYSTEM DETAILS")
    print("-" * 70)
    print(f"Routing Accuracy: {tier_results['routing_accuracy']:.1%}")
    
    print("\nPerformance by Actual Price Tier:")
    for tier_name in ['tier_lower', 'tier_higher']:
        if tier_name in tier_results:
            t = tier_results[tier_name]
            print(f"  {tier_name.replace('tier_', '').title():10}: "
                  f"n={t['count']:4}, MAE=${t['mae']:,.0f}, R²={t['r2']:.4f}")
    
    if 'misrouted' in tier_results:
        m = tier_results['misrouted']
        print(f"\nMisrouted Predictions: {m['count']} ({m['pct_of_total']:.1f}%)")
        print(f"  MAE for misrouted: ${m['mae']:,.0f}")
    
    # Compare baseline vs tiered by tier
    print("\n" + "-" * 70)
    print(" BASELINE VS TIERED BY PRICE TIER")
    print("-" * 70)
    
    for tier in [0, 1]:
        tier_mask = tiers_test == tier
        tier_name = "Lower" if tier == 0 else "Higher"
        
        baseline_tier_mae = mean_absolute_error(y_test[tier_mask], baseline_pred[tier_mask])
        tiered_tier_mae = mean_absolute_error(y_test[tier_mask], tiered_pred[tier_mask])
        improvement = (baseline_tier_mae - tiered_tier_mae) / baseline_tier_mae * 100
        
        print(f"{tier_name} Tier (n={tier_mask.sum()}):")
        print(f"  Baseline MAE: ${baseline_tier_mae:,.0f}")
        print(f"  Tiered MAE:   ${tiered_tier_mae:,.0f} ({improvement:+.1f}%)")
    
    # Summary
    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70)
    
    if mae_imp > 0:
        print(f"\n[SUCCESS] Tiered system improves MAE by {mae_imp:.1f}% (${baseline_metrics['mae'] - tiered_metrics['mae']:,.0f})")
    else:
        print(f"\n[INFO] Tiered system MAE is {-mae_imp:.1f}% worse than baseline")
    
    print("\n" + "=" * 70)
    
    # Return results for saving
    return {
        'timestamp': datetime.now().isoformat(),
        'baseline': baseline_metrics,
        'tiered': tiered_metrics,
        'improvement': {
            'mae_pct': mae_imp,
            'rmse_pct': rmse_imp,
            'r2_pct': r2_imp
        },
        'tier_analysis': tier_results,
        'test_samples': len(y_test)
    }


def main():
    parser = argparse.ArgumentParser(description='V2.7: Two-Tier Model System')
    parser.add_argument('--mode', choices=['train', 'evaluate', 'full'], default='full',
                        help='Mode: train, evaluate, or full (both)')
    args = parser.parse_args()
    
    print("\n")
    print("=" * 70)
    print(" V2.7: TWO-TIER MODEL SYSTEM")
    print("=" * 70)
    print(f" Mode: {args.mode}")
    print("=" * 70)
    
    # Load data
    X, y, feature_names = load_data()
    
    if args.mode in ['train', 'full']:
        baseline_model, tiered_system, X_test, y_test, tiers_test, threshold = train_pipeline(X, y)
        
        # Save test data for later evaluation
        test_data = {
            'X_test': X_test,
            'y_test': y_test,
            'tiers_test': tiers_test,
            'threshold': threshold
        }
        with open(OUTPUT_DIR / 'test_data.pkl', 'wb') as f:
            pickle.dump(test_data, f)
    
    if args.mode in ['evaluate', 'full']:
        if args.mode == 'evaluate':
            # Load saved models and test data
            with open(OUTPUT_DIR / 'baseline_model.pkl', 'rb') as f:
                baseline_model = pickle.load(f)
            
            tiered_system = TieredModelSystem()
            tiered_system.load(OUTPUT_DIR / 'tiered_model_system.pkl')
            
            with open(OUTPUT_DIR / 'test_data.pkl', 'rb') as f:
                test_data = pickle.load(f)
            X_test = test_data['X_test']
            y_test = test_data['y_test']
            tiers_test = test_data['tiers_test']
        
        # Evaluate
        results = evaluate_pipeline(baseline_model, tiered_system, X_test, y_test, tiers_test)
        
        # Save results
        with open(OUTPUT_DIR / f'evaluation_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
            # Convert non-serializable types
            def convert(obj):
                if isinstance(obj, (np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, (np.float64, np.float32)):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert(v) for k, v in obj.items()}
                return obj
            
            json.dump(convert(results), f, indent=2)
        
        logger.info(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
