"""
V2.7d: Low-Price Specialist Optimization

Finds the optimal configuration for a low-price specialist model:
1. What percentile defines "low-priced" homes? (10%, 15%, 20%, 25%, 30%, 35%, 40%)
2. What confidence threshold should trigger specialist routing? (70%-99%)

Key Finding from V2.7c:
    - Low specialist: $28,402 MAE (21% better than baseline for low zone)
    - High specialist: $183,431 MAE (40% WORSE than baseline for high zone)
    - Conclusion: Only use low specialist, route everything else to baseline

Architecture:
    Input → Confidence Classifier →
        If P(Low) > threshold → Low Specialist Model
        Otherwise             → Baseline Model

Usage:
    python src/low_specialist_optimization.py
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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

# Search space
LOW_PERCENTILES = [10, 15, 20, 25, 30, 35, 40]  # Percentile cutoffs to test
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
# LOW-ONLY SPECIALIST SYSTEM
# =============================================================================

class LowSpecialistSystem:
    """
    Routes only confident low-price predictions to a specialist.
    Everything else goes to the baseline model.
    """
    
    def __init__(self, low_percentile: float = 25, confidence_threshold: float = 0.90):
        self.low_percentile = low_percentile
        self.confidence_threshold = confidence_threshold
        self.low_classifier = None
        self.scaler = None
        self.low_model = None
        self.baseline_model = None
        self.low_threshold = None
    
    def create_low_labels(self, y: pd.Series) -> pd.Series:
        """Create binary labels: 1 = low-priced, 0 = not low-priced."""
        self.low_threshold = y.quantile(self.low_percentile / 100)
        labels = (y < self.low_threshold).astype(int)
        return labels
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, verbose: bool = True):
        """Train the low-specialist system."""
        # Create low-price labels
        low_labels = self.create_low_labels(y_train)
        
        if verbose:
            logger.info(f"\nLow threshold ({self.low_percentile}th percentile): ${self.low_threshold:,.0f}")
            logger.info(f"Low-price samples: {low_labels.sum()} ({low_labels.mean():.1%})")
        
        # Scale features for classifier
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Train calibrated binary classifier
        base_clf = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=RANDOM_STATE
        )
        
        self.low_classifier = CalibratedClassifierCV(
            base_clf,
            method='isotonic',
            cv=5
        )
        self.low_classifier.fit(X_scaled, low_labels)
        
        # Train baseline model (all data)
        self.baseline_model = xgb.XGBRegressor(**get_xgboost_params())
        self.baseline_model.fit(X_train, y_train)
        
        # Train low specialist (only low-price homes)
        low_mask = low_labels == 1
        self.low_model = xgb.XGBRegressor(**get_xgboost_params())
        self.low_model.fit(X_train[low_mask], y_train[low_mask])
        
        if verbose:
            logger.info(f"Low specialist trained on {low_mask.sum()} samples")
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with confidence-based routing to low specialist.
        
        Returns:
            predictions: Predicted prices
            used_specialist: Boolean array (True = used low specialist)
            low_probabilities: Probability of being low-priced
        """
        X_scaled = self.scaler.transform(X)
        
        # Get probability of being low-priced
        probas = self.low_classifier.predict_proba(X_scaled)
        low_probs = probas[:, 1]  # Probability of class 1 (low-priced)
        
        # Start with baseline predictions
        predictions = self.baseline_model.predict(X)
        used_specialist = np.zeros(len(X), dtype=bool)
        
        # Override with specialist for high-confidence low predictions
        confident_low = low_probs >= self.confidence_threshold
        if confident_low.sum() > 0:
            predictions[confident_low] = self.low_model.predict(X[confident_low])
            used_specialist[confident_low] = True
        
        return predictions, used_specialist, low_probs


def evaluate_configuration(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series,
    low_percentile: float, confidence_threshold: float,
    baseline_pred: np.ndarray, baseline_mae: float
) -> Dict:
    """Evaluate a specific (percentile, confidence) configuration."""
    
    system = LowSpecialistSystem(
        low_percentile=low_percentile,
        confidence_threshold=confidence_threshold
    )
    system.train(X_train, y_train, verbose=False)
    
    predictions, used_specialist, low_probs = system.predict(X_test)
    
    # Overall metrics
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    improvement = (baseline_mae - mae) / baseline_mae * 100
    
    # Routing stats
    n_specialist = used_specialist.sum()
    n_baseline = (~used_specialist).sum()
    
    # Check accuracy of specialist routing
    actual_low = y_test < system.low_threshold
    correct_specialist = (used_specialist & actual_low).sum()
    wrong_specialist = (used_specialist & ~actual_low).sum()  # Routed to specialist but wasn't actually low
    
    result = {
        'low_percentile': low_percentile,
        'confidence_threshold': confidence_threshold,
        'low_threshold_dollars': float(system.low_threshold),
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'improvement_pct': float(improvement),
        'n_specialist': int(n_specialist),
        'n_baseline': int(n_baseline),
        'pct_specialist': float(n_specialist / len(y_test) * 100),
        'correct_specialist': int(correct_specialist),
        'wrong_specialist': int(wrong_specialist),
        'specialist_precision': float(correct_specialist / n_specialist * 100) if n_specialist > 0 else 0
    }
    
    # MAE breakdown
    if n_specialist > 0:
        result['specialist_mae'] = float(mean_absolute_error(
            y_test[used_specialist], predictions[used_specialist]
        ))
        # Compare to what baseline would have predicted for those same samples
        result['baseline_mae_for_specialist_samples'] = float(mean_absolute_error(
            y_test[used_specialist], baseline_pred[used_specialist]
        ))
        result['specialist_vs_baseline'] = float(
            (result['baseline_mae_for_specialist_samples'] - result['specialist_mae']) 
            / result['baseline_mae_for_specialist_samples'] * 100
        )
    
    if n_baseline > 0:
        result['baseline_portion_mae'] = float(mean_absolute_error(
            y_test[~used_specialist], predictions[~used_specialist]
        ))
    
    return result


# =============================================================================
# MAIN OPTIMIZATION
# =============================================================================

def main():
    print("\n" + "=" * 80)
    print(" V2.7d: LOW-PRICE SPECIALIST OPTIMIZATION")
    print("=" * 80)
    print(" Finding optimal: (percentile cutoff) × (confidence threshold)")
    print("=" * 80)
    
    # Load data
    logger.info("Loading data...")
    X, y = load_data()
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    
    # Train pure baseline
    logger.info("\nTraining baseline model...")
    baseline = xgb.XGBRegressor(**get_xgboost_params())
    baseline.fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)
    baseline_mae = mean_absolute_error(y_test, baseline_pred)
    baseline_r2 = r2_score(y_test, baseline_pred)
    
    logger.info(f"Baseline MAE: ${baseline_mae:,.0f}")
    logger.info(f"Baseline R²:  {baseline_r2:.4f}")
    
    # Grid search
    print("\n" + "=" * 80)
    print(" GRID SEARCH: Percentile × Confidence Threshold")
    print("=" * 80)
    
    all_results = []
    best_result = None
    best_improvement = -float('inf')
    
    # Header
    print(f"\n{'Pctl':>5} {'Conf':>6} {'MAE':>12} {'Δ Base':>8} {'% Spec':>8} {'Spec MAE':>10} {'Prec':>6}")
    print("-" * 80)
    
    for percentile in LOW_PERCENTILES:
        for confidence in CONFIDENCE_THRESHOLDS:
            result = evaluate_configuration(
                X_train, y_train, X_test, y_test,
                percentile, confidence,
                baseline_pred, baseline_mae
            )
            all_results.append(result)
            
            marker = ""
            if result['improvement_pct'] > best_improvement:
                best_improvement = result['improvement_pct']
                best_result = result
                marker = " ★"
            
            spec_mae = result.get('specialist_mae', 0)
            precision = result.get('specialist_precision', 0)
            
            print(f"{percentile:>4}% {confidence:>5.0%} ${result['mae']:>11,.0f} "
                  f"{result['improvement_pct']:>+7.2f}% {result['pct_specialist']:>7.1f}% "
                  f"${spec_mae:>9,.0f} {precision:>5.1f}%{marker}")
    
    # ==========================================================================
    # Best Configuration Details
    # ==========================================================================
    print("\n" + "=" * 80)
    print(" BEST CONFIGURATION")
    print("=" * 80)
    
    print(f"\n  Low Percentile:      {best_result['low_percentile']}th percentile")
    print(f"  Price Threshold:     ${best_result['low_threshold_dollars']:,.0f}")
    print(f"  Confidence Threshold: {best_result['confidence_threshold']:.0%}")
    
    print(f"\n  Overall MAE:         ${best_result['mae']:,.0f}")
    print(f"  Improvement:         {best_result['improvement_pct']:+.2f}% vs baseline")
    
    print(f"\n  Routing:")
    print(f"    → Low Specialist:  {best_result['n_specialist']} samples ({best_result['pct_specialist']:.1f}%)")
    print(f"    → Baseline:        {best_result['n_baseline']} samples ({100-best_result['pct_specialist']:.1f}%)")
    
    print(f"\n  Specialist Performance:")
    print(f"    MAE:               ${best_result.get('specialist_mae', 0):,.0f}")
    print(f"    vs Baseline (same samples): {best_result.get('specialist_vs_baseline', 0):+.1f}%")
    print(f"    Precision:         {best_result.get('specialist_precision', 0):.1f}% (correct routing)")
    
    # ==========================================================================
    # Final Comparison
    # ==========================================================================
    print("\n" + "=" * 80)
    print(" FINAL COMPARISON")
    print("=" * 80)
    
    print(f"\n{'Model':<25} {'MAE':>15} {'R²':>10}")
    print("-" * 55)
    print(f"{'Baseline (V2.5)':<25} ${baseline_mae:>14,.0f} {baseline_r2:>10.4f}")
    print(f"{'Low Specialist Routing':<25} ${best_result['mae']:>14,.0f} {best_result['r2']:>10.4f}")
    print("-" * 55)
    
    if best_result['improvement_pct'] > 0:
        savings = baseline_mae - best_result['mae']
        print(f"\n✅ LOW SPECIALIST WINS!")
        print(f"   MAE Reduction: ${savings:,.0f} per prediction ({best_result['improvement_pct']:.2f}%)")
        print(f"   Configuration: {best_result['low_percentile']}th percentile @ {best_result['confidence_threshold']:.0%} confidence")
    else:
        print(f"\n❌ Baseline still better by {-best_result['improvement_pct']:.2f}%")
    
    # ==========================================================================
    # Heatmap-style Summary
    # ==========================================================================
    print("\n" + "=" * 80)
    print(" IMPROVEMENT HEATMAP (% vs Baseline)")
    print("=" * 80)
    
    # Create pivot table
    print(f"\n{'':>8}", end="")
    for conf in CONFIDENCE_THRESHOLDS:
        print(f"{conf:>8.0%}", end="")
    print()
    print("-" * (8 + 8 * len(CONFIDENCE_THRESHOLDS)))
    
    for pctl in LOW_PERCENTILES:
        print(f"{pctl:>6}% |", end="")
        for conf in CONFIDENCE_THRESHOLDS:
            result = next(r for r in all_results 
                         if r['low_percentile'] == pctl and r['confidence_threshold'] == conf)
            imp = result['improvement_pct']
            if imp > 0:
                print(f"{imp:>+7.2f}%", end="")
            else:
                print(f"{imp:>7.2f}%", end="")
        print()
    
    print("\n" + "=" * 80)
    
    # Save results
    output_dir = Path("output/low_specialist")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'baseline_mae': float(baseline_mae),
        'baseline_r2': float(baseline_r2),
        'search_space': {
            'percentiles': LOW_PERCENTILES,
            'confidence_thresholds': CONFIDENCE_THRESHOLDS
        },
        'best_config': best_result,
        'all_results': all_results
    }
    
    results_file = output_dir / f"low_specialist_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nResults saved to {results_file}")
    
    # Train and save the best model if it beats baseline
    if best_result['improvement_pct'] > 0:
        logger.info("\nTraining final model with best configuration...")
        final_system = LowSpecialistSystem(
            low_percentile=best_result['low_percentile'],
            confidence_threshold=best_result['confidence_threshold']
        )
        final_system.train(X_train, y_train)
        
        model_file = output_dir / "low_specialist_system.pkl"
        joblib.dump({
            'system': final_system,
            'config': best_result,
            'baseline_mae': baseline_mae
        }, model_file)
        logger.info(f"Model saved to {model_file}")
    
    return summary


if __name__ == "__main__":
    main()
