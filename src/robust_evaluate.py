"""
V2.5: Robust Model Evaluation

Implements statistically rigorous evaluation methods:
1. K-Fold Cross-Validation with mean ± std
2. Bootstrap confidence intervals (95% CI)
3. Log target transformation experiment
4. Residual analysis and visualization

Usage:
    python src/robust_evaluate.py
    python src/robust_evaluate.py --k-folds 10
    python src/robust_evaluate.py --bootstrap-samples 1000
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
from scipy import stats
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
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
LOGS_DIR = Path("logs")
OUTPUT_DIR = Path("output")

# Feature selection (same as V2.4)
SALES_COLUMNS = [
    'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'sqft_above', 'sqft_basement', 'waterfront', 'view', 'condition', 'grade',
    'yr_built', 'yr_renovated', 'lat', 'long', 'sqft_living15', 'sqft_lot15', 'zipcode'
]

RANDOM_STATE = 42


def load_data() -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Load and prepare data."""
    logger.info("Loading data...")
    sales = pd.read_csv(DATA_DIR / "kc_house_data.csv", usecols=SALES_COLUMNS)
    demographics = pd.read_csv(DATA_DIR / "zipcode_demographics.csv")
    merged = sales.merge(demographics, on='zipcode', how='inner')
    
    y = merged['price']
    X = merged.drop(columns=['price', 'zipcode'])
    
    logger.info(f"Loaded {len(X)} samples, {len(X.columns)} features")
    return X, y, list(X.columns)


def load_model() -> Pipeline:
    """Load the production model."""
    model_path = MODEL_DIR / "model.pkl"
    logger.info(f"Loading model from {model_path}")
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def get_xgboost_params() -> dict:
    """Get the V2.4.1 tuned XGBoost parameters."""
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


# =============================================================================
# 1. K-FOLD CROSS-VALIDATION
# =============================================================================

def kfold_evaluation(
    X: pd.DataFrame,
    y: pd.Series,
    k_folds: int = 5
) -> Dict:
    """
    Perform K-Fold Cross-Validation.
    
    Args:
        X: Features
        y: Target
        k_folds: Number of folds
        
    Returns:
        Dictionary with CV metrics
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"K-FOLD CROSS-VALIDATION (k={k_folds})")
    logger.info(f"{'='*60}")
    
    # Create pipeline with same config as V2.4.1
    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('model', XGBRegressor(**get_xgboost_params()))
    ])
    
    # K-Fold splitter
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=RANDOM_STATE)
    
    # Multiple metrics
    mae_scores = []
    r2_scores = []
    rmse_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Fit and predict
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)
        
        # Calculate metrics
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        
        mae_scores.append(mae)
        r2_scores.append(r2)
        rmse_scores.append(rmse)
        
        logger.info(f"Fold {fold}: MAE=${mae:,.0f}, R²={r2:.4f}, RMSE=${rmse:,.0f}")
    
    results = {
        'k_folds': k_folds,
        'mae': {
            'mean': np.mean(mae_scores),
            'std': np.std(mae_scores),
            'min': np.min(mae_scores),
            'max': np.max(mae_scores),
            'all_folds': mae_scores
        },
        'r2': {
            'mean': np.mean(r2_scores),
            'std': np.std(r2_scores),
            'min': np.min(r2_scores),
            'max': np.max(r2_scores),
            'all_folds': r2_scores
        },
        'rmse': {
            'mean': np.mean(rmse_scores),
            'std': np.std(rmse_scores),
            'min': np.min(rmse_scores),
            'max': np.max(rmse_scores),
            'all_folds': rmse_scores
        }
    }
    
    logger.info(f"\nSUMMARY ({k_folds}-Fold CV):")
    logger.info(f"  MAE:  ${results['mae']['mean']:,.0f} +/- ${results['mae']['std']:,.0f}")
    logger.info(f"  R²:   {results['r2']['mean']:.4f} +/- {results['r2']['std']:.4f}")
    logger.info(f"  RMSE: ${results['rmse']['mean']:,.0f} +/- ${results['rmse']['std']:,.0f}")
    
    return results


# =============================================================================
# 2. BOOTSTRAP CONFIDENCE INTERVALS
# =============================================================================

def bootstrap_confidence_intervals(
    X: pd.DataFrame,
    y: pd.Series,
    n_bootstrap: int = 500,
    confidence: float = 0.95,
    test_size: float = 0.2
) -> Dict:
    """
    Calculate bootstrap confidence intervals for metrics.
    
    Args:
        X: Features
        y: Target
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (default 0.95 for 95% CI)
        test_size: Fraction for test set
        
    Returns:
        Dictionary with confidence intervals
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"BOOTSTRAP CONFIDENCE INTERVALS ({int(confidence*100)}% CI)")
    logger.info(f"{'='*60}")
    logger.info(f"Bootstrap samples: {n_bootstrap}")
    
    # Split data once
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE
    )
    
    mae_samples = []
    r2_samples = []
    rmse_samples = []
    
    for i in range(n_bootstrap):
        # Bootstrap sample from test set
        idx = np.random.choice(len(X_test), size=len(X_test), replace=True)
        X_boot = X_test.iloc[idx]
        y_boot = y_test.iloc[idx]
        
        # Train model (only on first iteration for efficiency)
        if i == 0:
            pipeline = Pipeline([
                ('scaler', RobustScaler()),
                ('model', XGBRegressor(**get_xgboost_params()))
            ])
            pipeline.fit(X_train, y_train)
        
        # Predict on bootstrap sample
        y_pred = pipeline.predict(X_boot)
        
        # Calculate metrics
        mae_samples.append(mean_absolute_error(y_boot, y_pred))
        r2_samples.append(r2_score(y_boot, y_pred))
        rmse_samples.append(np.sqrt(mean_squared_error(y_boot, y_pred)))
        
        if (i + 1) % 100 == 0:
            logger.info(f"  Completed {i+1}/{n_bootstrap} bootstrap samples")
    
    # Calculate confidence intervals
    alpha = 1 - confidence
    
    def get_ci(samples):
        return {
            'mean': np.mean(samples),
            'std': np.std(samples),
            'ci_lower': np.percentile(samples, alpha/2 * 100),
            'ci_upper': np.percentile(samples, (1 - alpha/2) * 100),
            'median': np.median(samples)
        }
    
    results = {
        'n_bootstrap': n_bootstrap,
        'confidence_level': confidence,
        'mae': get_ci(mae_samples),
        'r2': get_ci(r2_samples),
        'rmse': get_ci(rmse_samples)
    }
    
    logger.info(f"\nRESULTS ({int(confidence*100)}% Confidence Intervals):")
    logger.info(f"  MAE:  ${results['mae']['mean']:,.0f} [{results['mae']['ci_lower']:,.0f}, {results['mae']['ci_upper']:,.0f}]")
    logger.info(f"  R²:   {results['r2']['mean']:.4f} [{results['r2']['ci_lower']:.4f}, {results['r2']['ci_upper']:.4f}]")
    logger.info(f"  RMSE: ${results['rmse']['mean']:,.0f} [{results['rmse']['ci_lower']:,.0f}, {results['rmse']['ci_upper']:,.0f}]")
    
    return results


# =============================================================================
# 3. LOG TARGET TRANSFORMATION
# =============================================================================

def log_transform_experiment(
    X: pd.DataFrame,
    y: pd.Series,
    k_folds: int = 5
) -> Dict:
    """
    Compare model performance with and without log transformation.
    
    Args:
        X: Features
        y: Target (prices)
        k_folds: Number of CV folds
        
    Returns:
        Dictionary with comparison results
    """
    logger.info(f"\n{'='*60}")
    logger.info("LOG TARGET TRANSFORMATION EXPERIMENT")
    logger.info(f"{'='*60}")
    
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=RANDOM_STATE)
    
    # Results storage
    normal_mae = []
    log_mae = []
    normal_high_mae = []  # For homes > $1M
    log_high_mae = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # --- Normal (no transform) ---
        pipeline_normal = Pipeline([
            ('scaler', RobustScaler()),
            ('model', XGBRegressor(**get_xgboost_params()))
        ])
        pipeline_normal.fit(X_train, y_train)
        y_pred_normal = pipeline_normal.predict(X_val)
        
        # --- Log transform ---
        y_train_log = np.log(y_train)
        pipeline_log = Pipeline([
            ('scaler', RobustScaler()),
            ('model', XGBRegressor(**get_xgboost_params()))
        ])
        pipeline_log.fit(X_train, y_train_log)
        y_pred_log = np.exp(pipeline_log.predict(X_val))  # Transform back
        
        # Calculate MAE
        normal_mae.append(mean_absolute_error(y_val, y_pred_normal))
        log_mae.append(mean_absolute_error(y_val, y_pred_log))
        
        # High-value homes (> $1M)
        high_mask = y_val > 1_000_000
        if high_mask.sum() > 0:
            normal_high_mae.append(mean_absolute_error(y_val[high_mask], y_pred_normal[high_mask]))
            log_high_mae.append(mean_absolute_error(y_val[high_mask], y_pred_log[high_mask]))
        
        logger.info(f"Fold {fold}: Normal MAE=${normal_mae[-1]:,.0f}, Log MAE=${log_mae[-1]:,.0f}")
    
    results = {
        'normal': {
            'mae_mean': np.mean(normal_mae),
            'mae_std': np.std(normal_mae),
            'high_value_mae_mean': np.mean(normal_high_mae) if normal_high_mae else None
        },
        'log_transform': {
            'mae_mean': np.mean(log_mae),
            'mae_std': np.std(log_mae),
            'high_value_mae_mean': np.mean(log_high_mae) if log_high_mae else None
        },
        'comparison': {
            'mae_difference': np.mean(log_mae) - np.mean(normal_mae),
            'mae_improvement_pct': (np.mean(normal_mae) - np.mean(log_mae)) / np.mean(normal_mae) * 100,
            'high_value_improvement_pct': None
        }
    }
    
    if normal_high_mae and log_high_mae:
        results['comparison']['high_value_improvement_pct'] = (
            (np.mean(normal_high_mae) - np.mean(log_high_mae)) / np.mean(normal_high_mae) * 100
        )
    
    logger.info(f"\nCOMPARISON RESULTS:")
    logger.info(f"  Normal:        MAE ${results['normal']['mae_mean']:,.0f} +/- ${results['normal']['mae_std']:,.0f}")
    logger.info(f"  Log Transform: MAE ${results['log_transform']['mae_mean']:,.0f} +/- ${results['log_transform']['mae_std']:,.0f}")
    
    if results['comparison']['mae_improvement_pct'] > 0:
        logger.info(f"  --> Log transform is {results['comparison']['mae_improvement_pct']:.1f}% BETTER")
    else:
        logger.info(f"  --> Normal is {-results['comparison']['mae_improvement_pct']:.1f}% BETTER")
    
    if results['normal']['high_value_mae_mean']:
        logger.info(f"\n  HIGH-VALUE HOMES (>$1M):")
        logger.info(f"    Normal:        MAE ${results['normal']['high_value_mae_mean']:,.0f}")
        logger.info(f"    Log Transform: MAE ${results['log_transform']['high_value_mae_mean']:,.0f}")
        if results['comparison']['high_value_improvement_pct']:
            if results['comparison']['high_value_improvement_pct'] > 0:
                logger.info(f"    --> Log transform is {results['comparison']['high_value_improvement_pct']:.1f}% BETTER on high-value homes")
            else:
                logger.info(f"    --> Normal is {-results['comparison']['high_value_improvement_pct']:.1f}% BETTER on high-value homes")
    
    return results


# =============================================================================
# 4. RESIDUAL ANALYSIS
# =============================================================================

def residual_analysis(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2
) -> Dict:
    """
    Perform comprehensive residual analysis.
    
    Args:
        X: Features
        y: Target
        test_size: Fraction for test set
        
    Returns:
        Dictionary with residual analysis results
    """
    logger.info(f"\n{'='*60}")
    logger.info("RESIDUAL ANALYSIS")
    logger.info(f"{'='*60}")
    
    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE
    )
    
    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('model', XGBRegressor(**get_xgboost_params()))
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    # Calculate residuals
    residuals = y_test.values - y_pred
    pct_errors = (residuals / y_test.values) * 100
    
    # Basic statistics
    results = {
        'basic_stats': {
            'mean_residual': float(np.mean(residuals)),
            'median_residual': float(np.median(residuals)),
            'std_residual': float(np.std(residuals)),
            'skewness': float(stats.skew(residuals)),
            'kurtosis': float(stats.kurtosis(residuals)),
            'mean_pct_error': float(np.mean(pct_errors)),
            'median_pct_error': float(np.median(pct_errors))
        },
        'percentiles': {
            '5th': float(np.percentile(residuals, 5)),
            '25th': float(np.percentile(residuals, 25)),
            '50th': float(np.percentile(residuals, 50)),
            '75th': float(np.percentile(residuals, 75)),
            '95th': float(np.percentile(residuals, 95))
        },
        'error_distribution': {
            'within_10k': float(np.mean(np.abs(residuals) < 10000) * 100),
            'within_25k': float(np.mean(np.abs(residuals) < 25000) * 100),
            'within_50k': float(np.mean(np.abs(residuals) < 50000) * 100),
            'within_100k': float(np.mean(np.abs(residuals) < 100000) * 100),
            'over_200k': float(np.mean(np.abs(residuals) > 200000) * 100)
        }
    }
    
    # Error by price range
    price_ranges = [
        (0, 300000, 'under_300k'),
        (300000, 500000, '300k_to_500k'),
        (500000, 750000, '500k_to_750k'),
        (750000, 1000000, '750k_to_1m'),
        (1000000, float('inf'), 'over_1m')
    ]
    
    results['by_price_range'] = {}
    for low, high, name in price_ranges:
        mask = (y_test >= low) & (y_test < high)
        if mask.sum() > 0:
            range_residuals = residuals[mask]
            range_pct = pct_errors[mask]
            results['by_price_range'][name] = {
                'count': int(mask.sum()),
                'mae': float(np.mean(np.abs(range_residuals))),
                'mean_residual': float(np.mean(range_residuals)),
                'std_residual': float(np.std(range_residuals)),
                'mean_pct_error': float(np.mean(range_pct)),
                'bias': 'overpredict' if np.mean(range_residuals) < 0 else 'underpredict'
            }
    
    # Normality test
    _, normality_p = stats.normaltest(residuals)
    results['normality_test'] = {
        'test': "D'Agostino-Pearson",
        'p_value': float(normality_p),
        'is_normal': normality_p > 0.05
    }
    
    # Heteroscedasticity check (variance by predicted value)
    pred_quartiles = pd.qcut(y_pred, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    variance_by_quartile = {}
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        mask = pred_quartiles == q
        variance_by_quartile[q] = float(np.var(residuals[mask]))
    
    results['heteroscedasticity'] = {
        'variance_by_quartile': variance_by_quartile,
        'variance_ratio': max(variance_by_quartile.values()) / min(variance_by_quartile.values()),
        'is_heteroscedastic': max(variance_by_quartile.values()) / min(variance_by_quartile.values()) > 2
    }
    
    # Print results
    logger.info("\nBASIC STATISTICS:")
    logger.info(f"  Mean Residual:   ${results['basic_stats']['mean_residual']:,.0f}")
    logger.info(f"  Median Residual: ${results['basic_stats']['median_residual']:,.0f}")
    logger.info(f"  Std Residual:    ${results['basic_stats']['std_residual']:,.0f}")
    logger.info(f"  Skewness:        {results['basic_stats']['skewness']:.2f}")
    logger.info(f"  Kurtosis:        {results['basic_stats']['kurtosis']:.2f}")
    
    logger.info("\nERROR DISTRIBUTION:")
    logger.info(f"  Within $10k:  {results['error_distribution']['within_10k']:.1f}%")
    logger.info(f"  Within $25k:  {results['error_distribution']['within_25k']:.1f}%")
    logger.info(f"  Within $50k:  {results['error_distribution']['within_50k']:.1f}%")
    logger.info(f"  Within $100k: {results['error_distribution']['within_100k']:.1f}%")
    logger.info(f"  Over $200k:   {results['error_distribution']['over_200k']:.1f}%")
    
    logger.info("\nBY PRICE RANGE:")
    for name, data in results['by_price_range'].items():
        logger.info(f"  {name:15}: n={data['count']:4}, MAE=${data['mae']:,.0f}, Bias={data['bias']}")
    
    logger.info("\nDIAGNOSTICS:")
    logger.info(f"  Normality Test: p={results['normality_test']['p_value']:.4f} "
                f"({'Normal' if results['normality_test']['is_normal'] else 'Not Normal'})")
    logger.info(f"  Heteroscedasticity: Variance Ratio={results['heteroscedasticity']['variance_ratio']:.2f} "
                f"({'Yes' if results['heteroscedasticity']['is_heteroscedastic'] else 'No'})")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def save_results(results: Dict, filename: str):
    """Save results to JSON."""
    LOGS_DIR.mkdir(exist_ok=True)
    filepath = LOGS_DIR / filename
    
    # Convert numpy types to Python types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj
    
    with open(filepath, 'w') as f:
        json.dump(convert(results), f, indent=2)
    
    logger.info(f"Results saved to {filepath}")


def main():
    parser = argparse.ArgumentParser(description='V2.5: Robust Model Evaluation')
    parser.add_argument('--k-folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--bootstrap-samples', type=int, default=500, help='Bootstrap samples')
    parser.add_argument('--confidence', type=float, default=0.95, help='Confidence level')
    parser.add_argument('--skip-bootstrap', action='store_true', help='Skip bootstrap (slow)')
    args = parser.parse_args()
    
    print("\n")
    print("=" * 70)
    print(" V2.5: ROBUST MODEL EVALUATION")
    print("=" * 70)
    print(f" K-Folds: {args.k_folds}")
    print(f" Bootstrap Samples: {args.bootstrap_samples}")
    print(f" Confidence Level: {args.confidence * 100}%")
    print("=" * 70)
    
    # Load data
    X, y, feature_names = load_data()
    
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'version': 'v2.5',
        'data': {
            'n_samples': len(X),
            'n_features': len(feature_names)
        }
    }
    
    # 1. K-Fold Cross-Validation
    kfold_results = kfold_evaluation(X, y, k_folds=args.k_folds)
    all_results['kfold_cv'] = kfold_results
    
    # 2. Bootstrap Confidence Intervals
    if not args.skip_bootstrap:
        bootstrap_results = bootstrap_confidence_intervals(
            X, y, 
            n_bootstrap=args.bootstrap_samples,
            confidence=args.confidence
        )
        all_results['bootstrap_ci'] = bootstrap_results
    
    # 3. Log Transform Experiment
    log_results = log_transform_experiment(X, y, k_folds=args.k_folds)
    all_results['log_transform'] = log_results
    
    # 4. Residual Analysis
    residual_results = residual_analysis(X, y)
    all_results['residual_analysis'] = residual_results
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_results(all_results, f'v2.5_robust_evaluation_{timestamp}.json')
    
    # Print summary
    print("\n")
    print("=" * 70)
    print(" V2.5 EVALUATION SUMMARY")
    print("=" * 70)
    
    print(f"\n{args.k_folds}-FOLD CROSS-VALIDATION:")
    print(f"  MAE:  ${kfold_results['mae']['mean']:,.0f} +/- ${kfold_results['mae']['std']:,.0f}")
    print(f"  R²:   {kfold_results['r2']['mean']:.4f} +/- {kfold_results['r2']['std']:.4f}")
    
    if not args.skip_bootstrap:
        print(f"\n95% CONFIDENCE INTERVALS:")
        print(f"  MAE:  ${bootstrap_results['mae']['mean']:,.0f} "
              f"[{bootstrap_results['mae']['ci_lower']:,.0f}, {bootstrap_results['mae']['ci_upper']:,.0f}]")
        print(f"  R²:   {bootstrap_results['r2']['mean']:.4f} "
              f"[{bootstrap_results['r2']['ci_lower']:.4f}, {bootstrap_results['r2']['ci_upper']:.4f}]")
    
    print(f"\nLOG TRANSFORM COMPARISON:")
    print(f"  Normal MAE:        ${log_results['normal']['mae_mean']:,.0f}")
    print(f"  Log Transform MAE: ${log_results['log_transform']['mae_mean']:,.0f}")
    winner = "Log Transform" if log_results['comparison']['mae_improvement_pct'] > 0 else "Normal"
    print(f"  Winner: {winner}")
    
    print(f"\nRESIDUAL DIAGNOSTICS:")
    print(f"  Mean Bias: ${residual_results['basic_stats']['mean_residual']:,.0f}")
    print(f"  Normality: {'Yes' if residual_results['normality_test']['is_normal'] else 'No'}")
    print(f"  Heteroscedastic: {'Yes' if residual_results['heteroscedasticity']['is_heteroscedastic'] else 'No'}")
    
    print("\n" + "=" * 70)
    print(" RECOMMENDATIONS")
    print("=" * 70)
    
    # Generate recommendations
    recommendations = []
    
    if log_results['comparison']['mae_improvement_pct'] > 2:
        recommendations.append("Consider using log transform for production model")
    elif log_results['comparison']['mae_improvement_pct'] < -2:
        recommendations.append("Keep current model (no log transform)")
    else:
        recommendations.append("Log transform shows minimal difference - keep current model for simplicity")
    
    if residual_results['heteroscedasticity']['is_heteroscedastic']:
        recommendations.append("Model shows heteroscedasticity - errors vary by price range")
        recommendations.append("Consider separate models for different price tiers")
    
    if abs(residual_results['basic_stats']['mean_residual']) > 5000:
        bias_dir = "under" if residual_results['basic_stats']['mean_residual'] > 0 else "over"
        recommendations.append(f"Model has systematic bias ({bias_dir}predicting by ${abs(residual_results['basic_stats']['mean_residual']):,.0f})")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")
    
    print("\n" + "=" * 70)
    
    return all_results


if __name__ == "__main__":
    main()
