"""
V2.7b: Hybrid Three-Zone Model System

A confidence-based hybrid approach that:
1. Routes bottom 25% (high confidence low) → Specialized low-tier model
2. Routes middle 50% (uncertain zone) → Baseline generalist model
3. Routes top 25% (high confidence high) → Specialized high-tier model

Rationale (Human-in-the-Loop Insight):
    - Pure 2-tier system has 8.1% misrouting with $119k MAE penalty
    - Misrouted samples are primarily in the "uncertain middle zone"
    - By routing the middle 50% to the baseline model, we:
      * Sequester misrouting risk to the generalist model
      * Let specialized models handle high-confidence extremes
      * Reduce skew/tail errors at price extremes

Architecture:
    Input → Zone Classifier → Low Model (25%) | Baseline (50%) | High Model (25%)

Usage:
    python src/hybrid_tiered_system.py
"""

import sys
import json
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
RANDOM_STATE = 42
DATA_DIR = Path("data")

# Zone boundaries (percentiles)
LOW_ZONE_UPPER = 25    # Bottom 25% → Low-tier model
HIGH_ZONE_LOWER = 75   # Top 25% → High-tier model
# Middle 50% (25th-75th percentile) → Baseline model


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Load and prepare data (matching tiered_model_system.py)."""
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


def create_zones(y: pd.Series) -> Tuple[pd.Series, Dict[str, float]]:
    """
    Create 3-zone labels based on percentile boundaries.
    
    Zones:
        0 = Low (bottom 25%) - High confidence low prices
        1 = Middle (25th-75th percentile) - Uncertain zone
        2 = High (top 25%) - High confidence high prices
    
    Returns:
        zones: Series with zone labels (0, 1, 2)
        thresholds: Dict with 'low' and 'high' threshold values
    """
    low_threshold = y.quantile(LOW_ZONE_UPPER / 100)
    high_threshold = y.quantile(HIGH_ZONE_LOWER / 100)
    
    zones = pd.Series(1, index=y.index)  # Default to middle (1)
    zones[y < low_threshold] = 0          # Low zone
    zones[y >= high_threshold] = 2        # High zone
    
    thresholds = {
        'low': low_threshold,
        'high': high_threshold
    }
    
    logger.info(f"\nZone Configuration:")
    logger.info(f"  Low zone (0):    < ${low_threshold:,.0f} (bottom {LOW_ZONE_UPPER}%)")
    logger.info(f"  Middle zone (1): ${low_threshold:,.0f} - ${high_threshold:,.0f} (middle {HIGH_ZONE_LOWER - LOW_ZONE_UPPER}%)")
    logger.info(f"  High zone (2):   >= ${high_threshold:,.0f} (top {100 - HIGH_ZONE_LOWER}%)")
    logger.info(f"\nZone distribution:")
    logger.info(f"  Zone 0 (Low):    {(zones == 0).sum()} samples ({(zones == 0).mean():.1%})")
    logger.info(f"  Zone 1 (Middle): {(zones == 1).sum()} samples ({(zones == 1).mean():.1%})")
    logger.info(f"  Zone 2 (High):   {(zones == 2).sum()} samples ({(zones == 2).mean():.1%})")
    
    return zones, thresholds


# =============================================================================
# MODEL PARAMETERS
# =============================================================================

def get_v25_xgboost_params() -> Dict[str, Any]:
    """Get the validated V2.5 XGBoost parameters."""
    return {
        'n_estimators': 500,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    }


# =============================================================================
# HYBRID THREE-ZONE SYSTEM
# =============================================================================

class HybridThreeZoneSystem:
    """
    Hybrid prediction system with confidence-based routing.
    
    Routes:
        - Bottom 25% (high confidence low) → Specialized low model
        - Middle 50% (uncertain) → Baseline generalist model
        - Top 25% (high confidence high) → Specialized high model
    """
    
    def __init__(self):
        self.zone_classifier = None
        self.zone_scaler = None
        self.low_model = None      # Zone 0: Bottom 25%
        self.baseline_model = None  # Zone 1: Middle 50%
        self.high_model = None     # Zone 2: Top 25%
        self.thresholds = None
        self.feature_names = None
    
    def train_zone_classifier(self, X: pd.DataFrame, zones: pd.Series) -> float:
        """Train classifier to predict which zone a sample belongs to."""
        logger.info("\nTraining Zone Classifier (3-class)...")
        
        self.zone_scaler = RobustScaler()
        X_scaled = self.zone_scaler.fit_transform(X)
        
        self.zone_classifier = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=RANDOM_STATE
        )
        
        # Cross-validation
        cv_scores = cross_val_score(self.zone_classifier, X_scaled, zones, cv=5, scoring='accuracy')
        cv_accuracy = cv_scores.mean()
        
        # Fit on full data
        self.zone_classifier.fit(X_scaled, zones)
        
        logger.info(f"Zone Classifier CV Accuracy: {cv_accuracy:.1%} +/- {cv_scores.std():.1%}")
        
        # Per-zone accuracy
        y_pred = self.zone_classifier.predict(X_scaled)
        for zone in [0, 1, 2]:
            mask = zones == zone
            zone_acc = accuracy_score(zones[mask], y_pred[mask])
            zone_name = ['Low', 'Middle', 'High'][zone]
            logger.info(f"  Zone {zone} ({zone_name}): {zone_acc:.1%} accuracy")
        
        return cv_accuracy
    
    def train_zone_models(self, X: pd.DataFrame, y: pd.Series, zones: pd.Series):
        """Train specialized models for each zone."""
        params = get_v25_xgboost_params()
        
        # Zone 0: Low prices (bottom 25%)
        mask_low = zones == 0
        logger.info(f"\nTraining Low Zone Model (Zone 0)...")
        logger.info(f"  Samples: {mask_low.sum()}, Price range: ${y[mask_low].min():,.0f} - ${y[mask_low].max():,.0f}")
        self.low_model = xgb.XGBRegressor(**params)
        self.low_model.fit(X[mask_low], y[mask_low])
        
        # Zone 1: Middle prices (middle 50%) - Use baseline model
        mask_middle = zones == 1
        logger.info(f"\nTraining Middle Zone Model (Baseline)...")
        logger.info(f"  Samples: {mask_middle.sum()}, Price range: ${y[mask_middle].min():,.0f} - ${y[mask_middle].max():,.0f}")
        # Train on ALL data for the baseline - it needs to handle any price
        self.baseline_model = xgb.XGBRegressor(**params)
        self.baseline_model.fit(X, y)  # Train on full dataset
        
        # Zone 2: High prices (top 25%)
        mask_high = zones == 2
        logger.info(f"\nTraining High Zone Model (Zone 2)...")
        logger.info(f"  Samples: {mask_high.sum()}, Price range: ${y[mask_high].min():,.0f} - ${y[mask_high].max():,.0f}")
        self.high_model = xgb.XGBRegressor(**params)
        self.high_model.fit(X[mask_high], y[mask_high])
        
        self.feature_names = list(X.columns)
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict prices using zone-based routing.
        
        Returns:
            predictions: Array of predicted prices
            predicted_zones: Array of predicted zone assignments
        """
        X_scaled = self.zone_scaler.transform(X)
        predicted_zones = self.zone_classifier.predict(X_scaled)
        
        predictions = np.zeros(len(X))
        
        # Route to appropriate model
        for zone, model in [(0, self.low_model), (1, self.baseline_model), (2, self.high_model)]:
            mask = predicted_zones == zone
            if mask.sum() > 0:
                predictions[mask] = model.predict(X[mask])
        
        return predictions, predicted_zones
    
    def save(self, filepath: str):
        """Save the hybrid system."""
        artifacts = {
            'zone_classifier': self.zone_classifier,
            'zone_scaler': self.zone_scaler,
            'low_model': self.low_model,
            'baseline_model': self.baseline_model,
            'high_model': self.high_model,
            'thresholds': self.thresholds,
            'feature_names': self.feature_names,
            'config': {
                'low_zone_upper_percentile': LOW_ZONE_UPPER,
                'high_zone_lower_percentile': HIGH_ZONE_LOWER,
                'zones': {
                    0: f'Low (bottom {LOW_ZONE_UPPER}%)',
                    1: f'Middle ({LOW_ZONE_UPPER}-{HIGH_ZONE_LOWER}%)',
                    2: f'High (top {100-HIGH_ZONE_LOWER}%)'
                }
            }
        }
        joblib.dump(artifacts, filepath)
        logger.info(f"Hybrid system saved to {filepath}")


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate evaluation metrics."""
    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred),
        'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }


def evaluate_by_zone(y_true: np.ndarray, y_pred: np.ndarray, 
                     actual_zones: np.ndarray, predicted_zones: np.ndarray) -> Dict:
    """Detailed evaluation by zone with routing analysis."""
    results = {}
    
    zone_names = ['Low', 'Middle', 'High']
    
    for zone in [0, 1, 2]:
        mask = actual_zones == zone
        if mask.sum() > 0:
            results[f'zone_{zone}'] = {
                'name': zone_names[zone],
                'n_samples': int(mask.sum()),
                'mae': float(mean_absolute_error(y_true[mask], y_pred[mask])),
                'r2': float(r2_score(y_true[mask], y_pred[mask])) if mask.sum() > 1 else 0
            }
    
    # Routing analysis
    correct_routing = actual_zones == predicted_zones
    misrouted = ~correct_routing
    
    results['routing'] = {
        'accuracy': float(correct_routing.mean()),
        'n_correct': int(correct_routing.sum()),
        'n_misrouted': int(misrouted.sum()),
        'misrouted_pct': float(misrouted.mean()) * 100
    }
    
    if misrouted.sum() > 0:
        results['routing']['misrouted_mae'] = float(mean_absolute_error(
            y_true[misrouted], y_pred[misrouted]
        ))
        results['routing']['correct_mae'] = float(mean_absolute_error(
            y_true[correct_routing], y_pred[correct_routing]
        ))
    
    # Breakdown of misrouting
    misroute_analysis = {}
    for actual in [0, 1, 2]:
        for predicted in [0, 1, 2]:
            if actual != predicted:
                mask = (actual_zones == actual) & (predicted_zones == predicted)
                if mask.sum() > 0:
                    key = f'{zone_names[actual]}_to_{zone_names[predicted]}'
                    misroute_analysis[key] = {
                        'count': int(mask.sum()),
                        'mae': float(mean_absolute_error(y_true[mask], y_pred[mask]))
                    }
    results['misroute_breakdown'] = misroute_analysis
    
    return results


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print(" V2.7b: HYBRID THREE-ZONE MODEL SYSTEM")
    print("=" * 70)
    print(" Confidence-based routing: Low 25% | Middle 50% | High 25%")
    print("=" * 70)
    
    # Load data
    logger.info("Loading data...")
    X, y = load_data()
    
    # Create zones
    zones, thresholds = create_zones(y)
    
    # Train/test split
    X_train, X_test, y_train, y_test, zones_train, zones_test = train_test_split(
        X, y, zones, test_size=0.2, random_state=RANDOM_STATE
    )
    
    # ==========================================================================
    # Train Pure Baseline (for comparison)
    # ==========================================================================
    logger.info("\n" + "-" * 70)
    logger.info("Training Pure Baseline (Single Model on All Data)...")
    logger.info("-" * 70)
    
    baseline = xgb.XGBRegressor(**get_v25_xgboost_params())
    baseline.fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)
    baseline_metrics = evaluate_model(y_test.values, baseline_pred)
    
    # ==========================================================================
    # Train Hybrid System
    # ==========================================================================
    logger.info("\n" + "-" * 70)
    logger.info("Training Hybrid Three-Zone System...")
    logger.info("-" * 70)
    
    hybrid = HybridThreeZoneSystem()
    hybrid.thresholds = thresholds
    
    # Train zone classifier
    zone_cv_accuracy = hybrid.train_zone_classifier(X_train, zones_train)
    
    # Train zone-specific models
    hybrid.train_zone_models(X_train, y_train, zones_train)
    
    # Make predictions
    hybrid_pred, predicted_zones = hybrid.predict(X_test)
    hybrid_metrics = evaluate_model(y_test.values, hybrid_pred)
    
    # Detailed zone evaluation
    zone_results = evaluate_by_zone(
        y_test.values, hybrid_pred, 
        zones_test.values, predicted_zones
    )
    
    # Save system
    output_dir = Path("output/hybrid_models")
    output_dir.mkdir(parents=True, exist_ok=True)
    hybrid.save(str(output_dir / "hybrid_three_zone_system.pkl"))
    
    # ==========================================================================
    # Results Comparison
    # ==========================================================================
    print("\n" + "=" * 70)
    print(" RESULTS COMPARISON")
    print("=" * 70)
    
    print("\n" + "-" * 70)
    print(" OVERALL METRICS")
    print("-" * 70)
    print(f"{'Metric':<20} {'Baseline':>15} {'Hybrid':>15} {'Improvement':>12}")
    print("-" * 70)
    
    for metric, label in [('mae', 'MAE'), ('rmse', 'RMSE'), ('r2', 'R²'), ('mape', 'MAPE')]:
        b_val = baseline_metrics[metric]
        h_val = hybrid_metrics[metric]
        
        if metric in ['mae', 'rmse']:
            improvement = (b_val - h_val) / b_val * 100
            print(f"{label:<20} ${b_val:>14,.0f} ${h_val:>14,.0f} {improvement:>+11.1f}%")
        elif metric == 'r2':
            improvement = (h_val - b_val) / b_val * 100
            print(f"{label:<20} {b_val:>15.4f} {h_val:>15.4f} {improvement:>+11.1f}%")
        else:  # mape
            print(f"{label:<20} {b_val:>14.2f}% {h_val:>14.2f}%")
    
    print("\n" + "-" * 70)
    print(" ZONE ROUTING ANALYSIS")
    print("-" * 70)
    routing = zone_results['routing']
    print(f"Zone Classification Accuracy: {routing['accuracy']:.1%}")
    print(f"Correctly Routed: {routing['n_correct']} samples")
    print(f"Misrouted: {routing['n_misrouted']} samples ({routing['misrouted_pct']:.1f}%)")
    
    if routing['n_misrouted'] > 0:
        print(f"\nMAE for correctly routed: ${routing['correct_mae']:,.0f}")
        print(f"MAE for misrouted: ${routing['misrouted_mae']:,.0f}")
    
    print("\n" + "-" * 70)
    print(" PERFORMANCE BY ACTUAL ZONE")
    print("-" * 70)
    
    # Calculate baseline metrics by zone for comparison
    baseline_by_zone = {}
    for zone in [0, 1, 2]:
        mask = zones_test == zone
        if mask.sum() > 0:
            baseline_by_zone[zone] = {
                'mae': mean_absolute_error(y_test[mask], baseline_pred[mask]),
                'r2': r2_score(y_test[mask], baseline_pred[mask])
            }
    
    zone_names = ['Low (bottom 25%)', 'Middle (25-75%)', 'High (top 25%)']
    print(f"{'Zone':<20} {'n':>6} {'Baseline MAE':>14} {'Hybrid MAE':>14} {'Δ':>10}")
    print("-" * 70)
    
    for zone in [0, 1, 2]:
        if f'zone_{zone}' in zone_results:
            z = zone_results[f'zone_{zone}']
            b_mae = baseline_by_zone[zone]['mae']
            h_mae = z['mae']
            improvement = (b_mae - h_mae) / b_mae * 100
            print(f"{zone_names[zone]:<20} {z['n_samples']:>6} ${b_mae:>13,.0f} ${h_mae:>13,.0f} {improvement:>+9.1f}%")
    
    print("\n" + "-" * 70)
    print(" MISROUTING BREAKDOWN")
    print("-" * 70)
    if zone_results['misroute_breakdown']:
        for route, data in sorted(zone_results['misroute_breakdown'].items()):
            print(f"  {route}: {data['count']} samples, MAE=${data['mae']:,.0f}")
    else:
        print("  No misrouting occurred!")
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70)
    
    mae_improvement = (baseline_metrics['mae'] - hybrid_metrics['mae']) / baseline_metrics['mae'] * 100
    
    if mae_improvement > 0:
        print(f"\n✅ HYBRID SYSTEM WINS: {mae_improvement:.1f}% lower MAE")
        print(f"   Baseline MAE: ${baseline_metrics['mae']:,.0f}")
        print(f"   Hybrid MAE:   ${hybrid_metrics['mae']:,.0f}")
        print(f"   Savings:      ${baseline_metrics['mae'] - hybrid_metrics['mae']:,.0f} per prediction")
    else:
        print(f"\n❌ BASELINE STILL BETTER: Hybrid is {-mae_improvement:.1f}% worse")
        print(f"   Baseline MAE: ${baseline_metrics['mae']:,.0f}")
        print(f"   Hybrid MAE:   ${hybrid_metrics['mae']:,.0f}")
    
    print("\n" + "=" * 70)
    
    # Save detailed results
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'low_zone_percentile': LOW_ZONE_UPPER,
            'high_zone_percentile': HIGH_ZONE_LOWER,
            'thresholds': {k: float(v) for k, v in thresholds.items()}
        },
        'baseline_metrics': baseline_metrics,
        'hybrid_metrics': hybrid_metrics,
        'zone_results': zone_results,
        'baseline_by_zone': {str(k): v for k, v in baseline_by_zone.items()},
        'improvement_pct': mae_improvement
    }
    
    results_file = output_dir / f"hybrid_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\nResults saved to {results_file}")
    
    return results


if __name__ == "__main__":
    main()
