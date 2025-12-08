"""
V2.4: Model Alternatives Comparison

Compare KNN (current production model) against tree-based and linear models
to find the best performer for real estate price prediction.

Models Evaluated:
1. KNN (baseline) - V2.3 tuned: n_neighbors=7, weights=distance, metric=manhattan
2. Random Forest - Handles non-linearity, provides feature importance
3. XGBoost - Often best on tabular data
4. LightGBM - Fast, handles large data
5. Ridge Regression - Simple linear baseline

Methodology:
- Use same data loading/split as V2.3 (test_size=0.2, random_state=42)
- 5-fold cross-validation on training set
- Final evaluation on held-out test set
- Feature importance analysis for tree models

Usage:
    python src/compare_models.py
    python src/compare_models.py --experiment-name real-estate-v2.4-comparison
"""

import argparse
import json
import logging
import pickle
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

# Suppress sklearn/lightgbm feature name warnings
warnings.filterwarnings('ignore', message='X does not have valid feature names')
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# Optional imports for gradient boosting
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[WARNING] XGBoost not installed. Run: pip install xgboost")

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("[WARNING] LightGBM not installed. Run: pip install lightgbm")

# Optional MLflow import
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("[WARNING] MLflow not available. Results will only be saved locally.")

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
SALES_PATH = DATA_DIR / "kc_house_data.csv"
DEMOGRAPHICS_PATH = DATA_DIR / "zipcode_demographics.csv"

# Feature selection (V2.1 features - 17 home + 26 demographic = 43 total)
SALES_COLUMN_SELECTION = [
    'price',
    # V1 structural features (7)
    'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'sqft_above', 'sqft_basement',
    # V2.1 property characteristics (4)
    'waterfront', 'view', 'condition', 'grade',
    # V2.1 age features (2)
    'yr_built', 'yr_renovated',
    # V2.1 spatial features (2)
    'lat', 'long',
    # V2.1 neighborhood context (2)
    'sqft_living15', 'sqft_lot15',
    # Join key
    'zipcode'
]

# Constants
RANDOM_STATE = 42
CV_FOLDS = 5
TEST_SIZE = 0.2


def load_and_prepare_data() -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Load and merge sales and demographics data.
    
    Returns:
        X: Feature DataFrame
        y: Target Series (price)
        feature_names: List of feature names
    """
    logger.info("Loading data...")
    
    # Load sales data
    sales_df = pd.read_csv(SALES_PATH, usecols=SALES_COLUMN_SELECTION)
    logger.info(f"Loaded {len(sales_df)} sales records")
    
    # Load demographics
    demographics_df = pd.read_csv(DEMOGRAPHICS_PATH)
    logger.info(f"Loaded {len(demographics_df)} zipcode demographics")
    
    # Merge (inner join to match V2.3 tune.py behavior)
    merged_df = sales_df.merge(demographics_df, on='zipcode', how='inner')
    logger.info(f"Merged dataset: {len(merged_df)} records")
    
    # Separate features and target
    y = merged_df['price']
    X = merged_df.drop(columns=['price', 'zipcode'])
    
    feature_names = list(X.columns)
    logger.info(f"Features: {len(feature_names)}")
    
    return X, y, feature_names


def create_model_configs() -> Dict[str, Dict[str, Any]]:
    """Create configurations for all models to compare.
    
    Returns:
        Dictionary mapping model name to config dict with 'model' and 'params'
    """
    models = {}
    
    # 1. KNN - V2.3 tuned baseline
    models['KNN (V2.3 Baseline)'] = {
        'model': KNeighborsRegressor(
            n_neighbors=7,
            weights='distance',
            metric='manhattan'
        ),
        'params': {
            'n_neighbors': 7,
            'weights': 'distance',
            'metric': 'manhattan'
        },
        'supports_feature_importance': False
    }
    
    # 2. Random Forest
    models['Random Forest'] = {
        'model': RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        'params': {
            'n_estimators': 100,
            'max_depth': 20,
            'min_samples_split': 5,
            'min_samples_leaf': 2
        },
        'supports_feature_importance': True
    }
    
    # 3. XGBoost
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = {
            'model': XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=0
            ),
            'params': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8
            },
            'supports_feature_importance': True
        }
    
    # 4. LightGBM
    if LIGHTGBM_AVAILABLE:
        models['LightGBM'] = {
            'model': LGBMRegressor(
                n_estimators=100,
                max_depth=10,
                learning_rate=0.1,
                num_leaves=31,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1
            ),
            'params': {
                'n_estimators': 100,
                'max_depth': 10,
                'learning_rate': 0.1,
                'num_leaves': 31
            },
            'supports_feature_importance': True
        }
    
    # 5. Ridge Regression (linear baseline)
    models['Ridge Regression'] = {
        'model': Ridge(alpha=1.0, random_state=RANDOM_STATE),
        'params': {
            'alpha': 1.0
        },
        'supports_feature_importance': True  # Has coef_
    }
    
    return models


def create_pipeline(model) -> Pipeline:
    """Create a sklearn pipeline with RobustScaler and the model.
    
    Args:
        model: sklearn-compatible model
        
    Returns:
        Pipeline with scaler and model
    """
    return Pipeline([
        ('scaler', RobustScaler()),
        ('model', model)
    ])


def evaluate_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv_folds: int = CV_FOLDS
) -> Dict[str, float]:
    """Evaluate a model with cross-validation and test set.
    
    Args:
        pipeline: sklearn Pipeline
        X_train, y_train: Training data
        X_test, y_test: Test data
        cv_folds: Number of CV folds
        
    Returns:
        Dictionary of metrics
    """
    # Cross-validation on training set
    cv_mae_scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=cv_folds,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )
    cv_r2_scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=cv_folds,
        scoring='r2',
        n_jobs=-1
    )
    
    # Fit on full training set
    pipeline.fit(X_train, y_train)
    
    # Predict on train and test
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    metrics = {
        # Cross-validation metrics (mean ± std)
        'cv_mae_mean': -cv_mae_scores.mean(),
        'cv_mae_std': cv_mae_scores.std(),
        'cv_r2_mean': cv_r2_scores.mean(),
        'cv_r2_std': cv_r2_scores.std(),
        
        # Training metrics
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'train_r2': r2_score(y_train, y_train_pred),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        
        # Test metrics
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'test_r2': r2_score(y_test, y_test_pred),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
    }
    
    # Overfitting check
    metrics['overfitting_gap_mae'] = metrics['train_mae'] - metrics['test_mae']
    metrics['overfitting_gap_r2'] = metrics['train_r2'] - metrics['test_r2']
    
    return metrics


def get_feature_importance(
    pipeline: Pipeline,
    feature_names: List[str],
    model_name: str
) -> pd.DataFrame:
    """Extract feature importance from a fitted pipeline.
    
    Args:
        pipeline: Fitted pipeline
        feature_names: List of feature names
        model_name: Name of the model
        
    Returns:
        DataFrame with feature importances sorted descending
    """
    model = pipeline.named_steps['model']
    
    if hasattr(model, 'feature_importances_'):
        # Tree-based models
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # Linear models - use absolute coefficient values
        importance = np.abs(model.coef_)
    else:
        return None
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('importance', ascending=False)
    importance_df['rank'] = range(1, len(importance_df) + 1)
    importance_df['model'] = model_name
    
    return importance_df


def run_comparison(
    experiment_name: str = "real-estate-v2.4-comparison"
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Run the full model comparison.
    
    Args:
        experiment_name: MLflow experiment name
        
    Returns:
        results_df: DataFrame with all model results
        importance_dict: Dictionary of feature importance DataFrames
    """
    logger.info("=" * 70)
    logger.info("V2.4 MODEL ALTERNATIVES COMPARISON")
    logger.info("=" * 70)
    
    # Load data
    X, y, feature_names = load_and_prepare_data()
    
    # Split data (same as V2.3)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )
    logger.info(f"Train set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    # Get model configs
    model_configs = create_model_configs()
    
    # Results storage
    results = []
    importance_dict = {}
    pipelines = {}
    
    # Evaluate each model
    for model_name, config in model_configs.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {model_name}")
        logger.info(f"{'='*60}")
        
        start_time = time.time()
        
        # Create pipeline
        pipeline = create_pipeline(config['model'])
        
        # Evaluate
        metrics = evaluate_model(
            pipeline, X_train, y_train, X_test, y_test, cv_folds=CV_FOLDS
        )
        
        elapsed_time = time.time() - start_time
        metrics['training_time_sec'] = elapsed_time
        
        # Add model info
        metrics['model_name'] = model_name
        metrics['params'] = json.dumps(config['params'])
        
        results.append(metrics)
        pipelines[model_name] = pipeline
        
        # Feature importance
        if config['supports_feature_importance']:
            importance_df = get_feature_importance(pipeline, feature_names, model_name)
            if importance_df is not None:
                importance_dict[model_name] = importance_df
        
        logger.info(f"  CV MAE: ${metrics['cv_mae_mean']:,.0f} ± ${metrics['cv_mae_std']:,.0f}")
        logger.info(f"  Test MAE: ${metrics['test_mae']:,.0f}")
        logger.info(f"  Test R²: {metrics['test_r2']:.4f}")
        logger.info(f"  Time: {elapsed_time:.1f}s")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by test MAE (lower is better)
    results_df = results_df.sort_values('test_mae')
    
    return results_df, importance_dict, pipelines, feature_names


def print_comparison_table(results_df: pd.DataFrame):
    """Print a formatted comparison table.
    
    Args:
        results_df: DataFrame with model results
    """
    print("\n")
    print("=" * 90)
    print(" MODEL COMPARISON RESULTS (sorted by Test MAE)")
    print("=" * 90)
    
    # Header
    print(f"\n{'Model':<25} {'CV MAE':>15} {'Test MAE':>12} {'Test R²':>10} {'Test RMSE':>12} {'Time':>8}")
    print("-" * 90)
    
    for _, row in results_df.iterrows():
        model_name = row['model_name']
        cv_mae = f"${row['cv_mae_mean']:,.0f}±{row['cv_mae_std']:,.0f}"
        test_mae = f"${row['test_mae']:,.0f}"
        test_r2 = f"{row['test_r2']:.4f}"
        test_rmse = f"${row['test_rmse']:,.0f}"
        time_s = f"{row['training_time_sec']:.1f}s"
        
        print(f"{model_name:<25} {cv_mae:>15} {test_mae:>12} {test_r2:>10} {test_rmse:>12} {time_s:>8}")
    
    print("-" * 90)
    
    # Best model
    best_model = results_df.iloc[0]
    print(f"\n🏆 BEST MODEL: {best_model['model_name']}")
    print(f"   Test MAE: ${best_model['test_mae']:,.0f}")
    print(f"   Test R²: {best_model['test_r2']:.4f}")
    
    # Comparison to KNN baseline
    knn_row = results_df[results_df['model_name'].str.contains('KNN')]
    if not knn_row.empty and best_model['model_name'] != knn_row.iloc[0]['model_name']:
        knn_mae = knn_row.iloc[0]['test_mae']
        improvement = knn_mae - best_model['test_mae']
        improvement_pct = (improvement / knn_mae) * 100
        print(f"\n📊 Improvement over KNN baseline: ${improvement:,.0f} ({improvement_pct:.1f}%)")
    
    print("=" * 90)


def print_feature_importance(importance_dict: Dict[str, pd.DataFrame], top_n: int = 15):
    """Print top feature importances from tree models.
    
    Args:
        importance_dict: Dictionary of feature importance DataFrames
        top_n: Number of top features to show
    """
    print("\n")
    print("=" * 70)
    print(f" TOP {top_n} FEATURES BY MODEL (importance)")
    print("=" * 70)
    
    for model_name, importance_df in importance_dict.items():
        if 'Ridge' in model_name:
            continue  # Skip linear model for cleaner output
            
        print(f"\n{model_name}:")
        print("-" * 50)
        
        top_features = importance_df.head(top_n)
        for _, row in top_features.iterrows():
            bar = "█" * int(row['importance'] * 50 / importance_df['importance'].max())
            print(f"  {row['rank']:2}. {row['feature']:<25} {row['importance']:.4f} {bar}")
    
    print("=" * 70)


def save_results(
    results_df: pd.DataFrame,
    importance_dict: Dict[str, pd.DataFrame],
    best_pipeline: Pipeline,
    feature_names: List[str],
    experiment_name: str
):
    """Save comparison results.
    
    Args:
        results_df: DataFrame with model results
        importance_dict: Dictionary of feature importance DataFrames
        best_pipeline: Pipeline of the best model
        feature_names: List of feature names
        experiment_name: MLflow experiment name
    """
    logger.info("Saving results...")
    
    # Create output directories
    MODEL_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save comparison results CSV
    results_path = LOGS_DIR / f"v2.4_model_comparison_{timestamp}.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"Saved comparison results to {results_path}")
    
    # Save feature importance CSV (combined)
    if importance_dict:
        all_importance = pd.concat(importance_dict.values(), ignore_index=True)
        importance_path = LOGS_DIR / f"v2.4_feature_importance_{timestamp}.csv"
        all_importance.to_csv(importance_path, index=False)
        logger.info(f"Saved feature importance to {importance_path}")
    
    # Save summary JSON
    best_model = results_df.iloc[0]
    summary = {
        'timestamp': datetime.now().isoformat(),
        'version': 'v2.4',
        'best_model': best_model['model_name'],
        'best_model_params': json.loads(best_model['params']),
        'test_mae': float(best_model['test_mae']),
        'test_r2': float(best_model['test_r2']),
        'test_rmse': float(best_model['test_rmse']),
        'cv_mae': float(best_model['cv_mae_mean']),
        'knn_baseline': {
            'test_mae': float(results_df[results_df['model_name'].str.contains('KNN')].iloc[0]['test_mae'])
        },
        'all_models': results_df[['model_name', 'test_mae', 'test_r2']].to_dict(orient='records'),
        'n_features': len(feature_names),
        'feature_names': feature_names,
    }
    
    summary_path = MODEL_DIR / "comparison_results.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Saved summary to {summary_path}")
    
    # Log to MLflow if available
    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_experiment(experiment_name)
            
            with mlflow.start_run(run_name=f"v2.4-comparison-{timestamp}"):
                # Log best model metrics
                mlflow.log_metric("best_test_mae", best_model['test_mae'])
                mlflow.log_metric("best_test_r2", best_model['test_r2'])
                mlflow.log_metric("best_cv_mae", best_model['cv_mae_mean'])
                mlflow.log_param("best_model", best_model['model_name'])
                
                # Log artifacts
                mlflow.log_artifact(str(results_path))
                mlflow.log_artifact(str(summary_path))
                if importance_dict:
                    mlflow.log_artifact(str(importance_path))
                
            logger.info("Results logged to MLflow")
        except Exception as e:
            logger.warning(f"MLflow logging failed: {e}")
    
    return summary


def generate_recommendation(results_df: pd.DataFrame) -> str:
    """Generate a recommendation based on results.
    
    Args:
        results_df: DataFrame with model results
        
    Returns:
        Recommendation string
    """
    best = results_df.iloc[0]
    best_name = best['model_name']
    best_mae = best['test_mae']
    best_r2 = best['test_r2']
    
    # Get KNN baseline
    knn = results_df[results_df['model_name'].str.contains('KNN')].iloc[0]
    knn_mae = knn['test_mae']
    
    improvement = knn_mae - best_mae
    improvement_pct = (improvement / knn_mae) * 100
    
    lines = [
        "\n" + "=" * 70,
        " V2.4 RECOMMENDATION",
        "=" * 70,
        "",
    ]
    
    if 'KNN' in best_name:
        lines.append("✅ RECOMMENDATION: Keep KNN (V2.3) as production model")
        lines.append("")
        lines.append("The tuned KNN model remains competitive with tree-based alternatives.")
        lines.append("Benefits of keeping KNN:")
        lines.append("  - Simpler model, easier to debug")
        lines.append("  - No additional dependencies")
        lines.append("  - Already tuned and validated")
    else:
        lines.append(f"🔄 RECOMMENDATION: Consider switching to {best_name}")
        lines.append("")
        lines.append(f"Improvement over KNN: ${improvement:,.0f} ({improvement_pct:.1f}%)")
        lines.append(f"New Test MAE: ${best_mae:,.0f}")
        lines.append(f"New Test R²: {best_r2:.4f}")
        lines.append("")
        
        if improvement_pct < 5:
            lines.append("⚠️  However, improvement is < 5%. Consider if added complexity is worth it.")
            lines.append("    KNN has benefits: simpler, no gradient boosting dependencies.")
        elif improvement_pct < 10:
            lines.append("ℹ️  Moderate improvement (5-10%). Worth considering for production.")
        else:
            lines.append("✅ Significant improvement (>10%). Strongly recommend switching.")
        
        lines.append("")
        lines.append("Next steps if switching:")
        lines.append("  1. Run hyperparameter tuning on best model")
        lines.append("  2. Update model.pkl with new model")
        lines.append("  3. Update API if model type changes")
        lines.append("  4. Validate on future_unseen_examples.csv")
    
    lines.append("")
    lines.append("=" * 70)
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='V2.4 Model Alternatives Comparison')
    parser.add_argument(
        '--experiment-name',
        default='real-estate-v2.4-comparison',
        help='MLflow experiment name'
    )
    args = parser.parse_args()
    
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " V2.4: MODEL ALTERNATIVES COMPARISON ".center(68) + "║")
    print("║" + " Real Estate Price Predictor ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    # Check for missing libraries
    if not XGBOOST_AVAILABLE:
        print("⚠️  XGBoost not available. Install with: pip install xgboost")
    if not LIGHTGBM_AVAILABLE:
        print("⚠️  LightGBM not available. Install with: pip install lightgbm")
    print()
    
    # Run comparison
    results_df, importance_dict, pipelines, feature_names = run_comparison(
        experiment_name=args.experiment_name
    )
    
    # Print results
    print_comparison_table(results_df)
    
    # Print feature importance
    if importance_dict:
        print_feature_importance(importance_dict)
    
    # Save results
    best_model_name = results_df.iloc[0]['model_name']
    best_pipeline = pipelines[best_model_name]
    
    save_results(
        results_df, importance_dict, best_pipeline,
        feature_names, args.experiment_name
    )
    
    # Generate and print recommendation
    recommendation = generate_recommendation(results_df)
    print(recommendation)
    
    # Return for programmatic use
    return results_df, importance_dict


if __name__ == "__main__":
    main()
