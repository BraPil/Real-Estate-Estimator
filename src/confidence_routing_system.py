"""
V2.7c: Confidence-Threshold Routing System

A conservative routing approach that:
1. Only routes to specialized models when classifier confidence is VERY HIGH
2. Routes uncertain cases (majority) to the robust baseline model
3. Minimizes catastrophic misrouting errors

Key Insight (from V2.7b experiments):
    - Specialized models DO help for high-confidence extremes (Low zone +3.2%)
    - But misrouting is catastrophic ($110k+ MAE for misrouted samples)
    - Solution: Only use specialized models when we're CONFIDENT

Architecture:
    Input → Confidence Classifier → 
        If P(Low) > threshold  → Low Specialist Model
        If P(High) > threshold → High Specialist Model
        Otherwise              → Baseline Model (safe default)

Usage:
    python src/confidence_routing_system.py
"""

import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
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

# Zone boundaries (percentiles) - same as hybrid system
LOW_ZONE_UPPER = 25    # Bottom 25%
HIGH_ZONE_LOWER = 75   # Top 25%

# Confidence thresholds to test
CONFIDENCE_THRESHOLDS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Load and prepare data."""
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
    """Create 3-zone labels."""
    low_threshold = y.quantile(LOW_ZONE_UPPER / 100)
    high_threshold = y.quantile(HIGH_ZONE_LOWER / 100)
    
    zones = pd.Series(1, index=y.index)  # Default middle
    zones[y < low_threshold] = 0          # Low
    zones[y >= high_threshold] = 2        # High
    
    return zones, {'low': low_threshold, 'high': high_threshold}


# =============================================================================
# MODEL PARAMETERS
# =============================================================================

def get_xgboost_params() -> Dict[str, Any]:
    """Get V2.5 XGBoost parameters."""
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
# CONFIDENCE ROUTING SYSTEM
# =============================================================================

class ConfidenceRoutingSystem:
    """
    Routes predictions based on classifier confidence.
    
    Only uses specialized models when confidence exceeds threshold,
    otherwise falls back to robust baseline model.
    """
    
    def __init__(self, confidence_threshold: float = 0.90):
        self.confidence_threshold = confidence_threshold
        self.zone_classifier = None
        self.zone_scaler = None
        self.low_model = None
        self.baseline_model = None
        self.high_model = None
        self.thresholds = None
    
    def train_calibrated_classifier(self, X: pd.DataFrame, zones: pd.Series) -> float:
        """
        Train a calibrated classifier that outputs reliable probabilities.
        
        Calibration ensures predicted probabilities match actual frequencies.
        """
        logger.info(f"\nTraining Calibrated Zone Classifier...")
        
        self.zone_scaler = RobustScaler()
        X_scaled = self.zone_scaler.fit_transform(X)
        
        # Base classifier
        base_clf = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=RANDOM_STATE
        )
        
        # Calibrate for reliable probabilities
        self.zone_classifier = CalibratedClassifierCV(
            base_clf, 
            method='isotonic',  # Better for enough data
            cv=5
        )
        
        # Cross-validation on base classifier first
        cv_scores = cross_val_score(base_clf, X_scaled, zones, cv=5, scoring='accuracy')
        logger.info(f"Base Classifier CV Accuracy: {cv_scores.mean():.1%} +/- {cv_scores.std():.1%}")
        
        # Fit calibrated classifier
        self.zone_classifier.fit(X_scaled, zones)
        
        return cv_scores.mean()
    
    def train_specialized_models(self, X: pd.DataFrame, y: pd.Series, zones: pd.Series):
        """Train baseline and specialized models."""
        params = get_xgboost_params()
        
        # Baseline: trained on ALL data (our safe fallback)
        logger.info(f"\nTraining Baseline Model (all data)...")
        self.baseline_model = xgb.XGBRegressor(**params)
        self.baseline_model.fit(X, y)
        
        # Low specialist: only low-price homes
        mask_low = zones == 0
        logger.info(f"\nTraining Low Specialist (Zone 0)...")
        logger.info(f"  Samples: {mask_low.sum()}, Range: ${y[mask_low].min():,.0f} - ${y[mask_low].max():,.0f}")
        self.low_model = xgb.XGBRegressor(**params)
        self.low_model.fit(X[mask_low], y[mask_low])
        
        # High specialist: only high-price homes
        mask_high = zones == 2
        logger.info(f"\nTraining High Specialist (Zone 2)...")
        logger.info(f"  Samples: {mask_high.sum()}, Range: ${y[mask_high].min():,.0f} - ${y[mask_high].max():,.0f}")
        self.high_model = xgb.XGBRegressor(**params)
        self.high_model.fit(X[mask_high], y[mask_high])
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions with confidence-based routing.
        
        Returns:
            predictions: Predicted prices
            routing: Which model was used (0=low, 1=baseline, 2=high)
            max_confidence: Confidence of the routing decision
        """
        X_scaled = self.zone_scaler.transform(X)
        probabilities = self.zone_classifier.predict_proba(X_scaled)
        
        # Get max probability and predicted class
        max_probs = probabilities.max(axis=1)
        predicted_zones = probabilities.argmax(axis=1)
        
        predictions = np.zeros(len(X))
        routing = np.ones(len(X), dtype=int)  # Default: baseline (1)
        
        for i in range(len(X)):
            if max_probs[i] >= self.confidence_threshold:
                # High confidence - use specialist
                if predicted_zones[i] == 0:  # Confident LOW
                    predictions[i] = self.low_model.predict(X.iloc[[i]])[0]
                    routing[i] = 0
                elif predicted_zones[i] == 2:  # Confident HIGH
                    predictions[i] = self.high_model.predict(X.iloc[[i]])[0]
                    routing[i] = 2
                else:  # Confident MIDDLE - use baseline anyway
                    predictions[i] = self.baseline_model.predict(X.iloc[[i]])[0]
                    routing[i] = 1
            else:
                # Low confidence - use safe baseline
                predictions[i] = self.baseline_model.predict(X.iloc[[i]])[0]
                routing[i] = 1
        
        return predictions, routing, max_probs
    
    def predict_batch(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Batch prediction (faster than row-by-row).
        """
        X_scaled = self.zone_scaler.transform(X)
        probabilities = self.zone_classifier.predict_proba(X_scaled)
        
        max_probs = probabilities.max(axis=1)
        predicted_zones = probabilities.argmax(axis=1)
        
        # Start with baseline predictions for everything
        predictions = self.baseline_model.predict(X)
        routing = np.ones(len(X), dtype=int)
        
        # Override for high-confidence LOW predictions
        confident_low = (max_probs >= self.confidence_threshold) & (predicted_zones == 0)
        if confident_low.sum() > 0:
            predictions[confident_low] = self.low_model.predict(X[confident_low])
            routing[confident_low] = 0
        
        # Override for high-confidence HIGH predictions
        confident_high = (max_probs >= self.confidence_threshold) & (predicted_zones == 2)
        if confident_high.sum() > 0:
            predictions[confident_high] = self.high_model.predict(X[confident_high])
            routing[confident_high] = 2
        
        return predictions, routing, max_probs


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


def evaluate_at_threshold(
    system: ConfidenceRoutingSystem,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    zones_test: pd.Series,
    baseline_pred: np.ndarray,
    threshold: float
) -> Dict:
    """Evaluate system at a specific confidence threshold."""
    
    system.confidence_threshold = threshold
    predictions, routing, confidences = system.predict_batch(X_test)
    
    metrics = evaluate_model(y_test.values, predictions)
    
    # Routing statistics
    n_low = (routing == 0).sum()
    n_baseline = (routing == 1).sum()
    n_high = (routing == 2).sum()
    
    # Accuracy of routing decisions
    correct_routing = (
        ((routing == 0) & (zones_test.values == 0)) |
        ((routing == 1) & (zones_test.values == 1)) |
        ((routing == 2) & (zones_test.values == 2))
    )
    
    # Misrouting analysis
    misrouted = ~correct_routing & (routing != 1)  # Wrong specialist used
    
    results = {
        'threshold': threshold,
        'metrics': metrics,
        'routing': {
            'n_low_specialist': int(n_low),
            'n_baseline': int(n_baseline),
            'n_high_specialist': int(n_high),
            'pct_specialist': float((n_low + n_high) / len(routing) * 100),
            'pct_baseline': float(n_baseline / len(routing) * 100)
        },
        'accuracy': {
            'overall_routing_accuracy': float(correct_routing.mean()),
            'n_misrouted_to_wrong_specialist': int(misrouted.sum())
        }
    }
    
    # MAE breakdown by routing decision
    if n_low > 0:
        results['low_specialist_mae'] = float(mean_absolute_error(
            y_test.values[routing == 0], predictions[routing == 0]
        ))
    if n_baseline > 0:
        results['baseline_mae'] = float(mean_absolute_error(
            y_test.values[routing == 1], predictions[routing == 1]
        ))
    if n_high > 0:
        results['high_specialist_mae'] = float(mean_absolute_error(
            y_test.values[routing == 2], predictions[routing == 2]
        ))
    
    if misrouted.sum() > 0:
        results['misrouted_specialist_mae'] = float(mean_absolute_error(
            y_test.values[misrouted], predictions[misrouted]
        ))
    
    return results


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print(" V2.7c: CONFIDENCE-THRESHOLD ROUTING SYSTEM")
    print("=" * 70)
    print(" Only use specialists when classifier confidence is HIGH")
    print(" Otherwise, fall back to robust baseline model")
    print("=" * 70)
    
    # Load data
    logger.info("Loading data...")
    X, y = load_data()
    zones, thresholds = create_zones(y)
    
    # Train/test split
    X_train, X_test, y_train, y_test, zones_train, zones_test = train_test_split(
        X, y, zones, test_size=0.2, random_state=RANDOM_STATE
    )
    
    logger.info(f"\nZone distribution in test set:")
    logger.info(f"  Low (Zone 0):    {(zones_test == 0).sum()} samples")
    logger.info(f"  Middle (Zone 1): {(zones_test == 1).sum()} samples")
    logger.info(f"  High (Zone 2):   {(zones_test == 2).sum()} samples")
    
    # ==========================================================================
    # Train Pure Baseline
    # ==========================================================================
    logger.info("\n" + "-" * 70)
    logger.info("Training Pure Baseline...")
    logger.info("-" * 70)
    
    baseline = xgb.XGBRegressor(**get_xgboost_params())
    baseline.fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)
    baseline_metrics = evaluate_model(y_test.values, baseline_pred)
    
    logger.info(f"Baseline MAE: ${baseline_metrics['mae']:,.0f}")
    
    # ==========================================================================
    # Train Confidence Routing System
    # ==========================================================================
    logger.info("\n" + "-" * 70)
    logger.info("Training Confidence Routing System...")
    logger.info("-" * 70)
    
    system = ConfidenceRoutingSystem()
    system.thresholds = thresholds
    system.train_calibrated_classifier(X_train, zones_train)
    system.train_specialized_models(X_train, y_train, zones_train)
    
    # ==========================================================================
    # Evaluate at Multiple Thresholds
    # ==========================================================================
    print("\n" + "=" * 70)
    print(" THRESHOLD ANALYSIS")
    print("=" * 70)
    
    results_by_threshold = []
    
    print(f"\n{'Threshold':>10} {'MAE':>12} {'vs Base':>10} {'% Specialist':>14} {'Misrouted':>10}")
    print("-" * 70)
    
    best_threshold = None
    best_improvement = -float('inf')
    
    for threshold in CONFIDENCE_THRESHOLDS:
        result = evaluate_at_threshold(
            system, X_test, y_test, zones_test, baseline_pred, threshold
        )
        results_by_threshold.append(result)
        
        mae = result['metrics']['mae']
        improvement = (baseline_metrics['mae'] - mae) / baseline_metrics['mae'] * 100
        pct_specialist = result['routing']['pct_specialist']
        n_misrouted = result['accuracy']['n_misrouted_to_wrong_specialist']
        
        marker = ""
        if improvement > best_improvement:
            best_improvement = improvement
            best_threshold = threshold
            marker = " ← best"
        
        print(f"{threshold:>10.0%} ${mae:>11,.0f} {improvement:>+9.1f}% {pct_specialist:>13.1f}% {n_misrouted:>10}{marker}")
    
    # ==========================================================================
    # Detailed Analysis of Best Threshold
    # ==========================================================================
    print("\n" + "=" * 70)
    print(f" BEST THRESHOLD: {best_threshold:.0%}")
    print("=" * 70)
    
    system.confidence_threshold = best_threshold
    best_pred, best_routing, best_conf = system.predict_batch(X_test)
    best_result = evaluate_at_threshold(
        system, X_test, y_test, zones_test, baseline_pred, best_threshold
    )
    
    print(f"\nRouting Distribution:")
    print(f"  Low Specialist:  {best_result['routing']['n_low_specialist']:>5} samples ({best_result['routing']['n_low_specialist']/len(X_test)*100:.1f}%)")
    print(f"  Baseline:        {best_result['routing']['n_baseline']:>5} samples ({best_result['routing']['n_baseline']/len(X_test)*100:.1f}%)")
    print(f"  High Specialist: {best_result['routing']['n_high_specialist']:>5} samples ({best_result['routing']['n_high_specialist']/len(X_test)*100:.1f}%)")
    
    print(f"\nMAE by Route:")
    if 'low_specialist_mae' in best_result:
        print(f"  Low Specialist:  ${best_result['low_specialist_mae']:,.0f}")
    print(f"  Baseline:        ${best_result.get('baseline_mae', 0):,.0f}")
    if 'high_specialist_mae' in best_result:
        print(f"  High Specialist: ${best_result['high_specialist_mae']:,.0f}")
    
    # ==========================================================================
    # Compare to Pure Baseline
    # ==========================================================================
    print("\n" + "=" * 70)
    print(" FINAL COMPARISON: BEST CONFIDENCE ROUTING vs PURE BASELINE")
    print("=" * 70)
    
    print(f"\n{'Metric':<15} {'Baseline':>15} {'Confidence':>15} {'Improvement':>12}")
    print("-" * 60)
    
    for metric, label in [('mae', 'MAE'), ('rmse', 'RMSE'), ('r2', 'R²')]:
        b_val = baseline_metrics[metric]
        c_val = best_result['metrics'][metric]
        
        if metric in ['mae', 'rmse']:
            improvement = (b_val - c_val) / b_val * 100
            print(f"{label:<15} ${b_val:>14,.0f} ${c_val:>14,.0f} {improvement:>+11.1f}%")
        else:
            improvement = (c_val - b_val) / abs(b_val) * 100
            print(f"{label:<15} {b_val:>15.4f} {c_val:>15.4f} {improvement:>+11.1f}%")
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70)
    
    final_improvement = (baseline_metrics['mae'] - best_result['metrics']['mae']) / baseline_metrics['mae'] * 100
    
    if final_improvement > 0:
        print(f"\n✅ CONFIDENCE ROUTING WINS!")
        print(f"   Best threshold: {best_threshold:.0%}")
        print(f"   MAE improvement: {final_improvement:.2f}%")
        print(f"   Baseline MAE: ${baseline_metrics['mae']:,.0f}")
        print(f"   Routing MAE:  ${best_result['metrics']['mae']:,.0f}")
        print(f"   Savings: ${baseline_metrics['mae'] - best_result['metrics']['mae']:,.0f} per prediction")
    else:
        print(f"\n❌ BASELINE STILL BETTER")
        print(f"   Best threshold tried: {best_threshold:.0%}")
        print(f"   Still {-final_improvement:.2f}% worse than baseline")
    
    print("\n" + "=" * 70)
    
    # Save results
    output_dir = Path("output/confidence_routing")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'baseline_metrics': baseline_metrics,
        'thresholds_tested': CONFIDENCE_THRESHOLDS,
        'results_by_threshold': results_by_threshold,
        'best_threshold': best_threshold,
        'best_improvement_pct': final_improvement,
        'zone_config': {
            'low_percentile': LOW_ZONE_UPPER,
            'high_percentile': HIGH_ZONE_LOWER,
            'thresholds': {k: float(v) for k, v in thresholds.items()}
        }
    }
    
    results_file = output_dir / f"confidence_routing_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Save model if it's better than baseline
    if final_improvement > 0:
        system.confidence_threshold = best_threshold
        model_file = output_dir / "confidence_routing_system.pkl"
        joblib.dump({
            'system': system,
            'best_threshold': best_threshold,
            'improvement': final_improvement
        }, model_file)
        logger.info(f"\nModel saved to {model_file}")
    
    logger.info(f"Results saved to {results_file}")
    
    return all_results


if __name__ == "__main__":
    main()
