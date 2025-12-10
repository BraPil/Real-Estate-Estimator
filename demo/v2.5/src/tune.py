"""
V2.3: Hyperparameter Tuning with GridSearchCV

This script finds the optimal KNN hyperparameters using cross-validation,
avoiding the test-set peeking issues from V2.1.2.

Methodology:
1. Load training data (same as train.py)
2. Use GridSearchCV with 5-fold cross-validation
3. Tune: n_neighbors, weights, metric
4. Compare CV score to held-out test set
5. Log results to MLflow

Usage:
    python src/tune.py
    python src/tune.py --experiment-name real-estate-v2.3-tuning
"""

import argparse
import json
import logging
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

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


def load_and_prepare_data():
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
    
    # Merge
    merged_df = sales_df.merge(demographics_df, on='zipcode', how='inner')
    logger.info(f"Merged dataset: {len(merged_df)} records")
    
    # Separate features and target
    y = merged_df['price']
    X = merged_df.drop(columns=['price', 'zipcode'])
    
    feature_names = list(X.columns)
    logger.info(f"Features: {len(feature_names)}")
    
    return X, y, feature_names


def create_param_grid():
    """Create parameter grid for GridSearchCV.
    
    Returns:
        Dictionary of parameters to search
    """
    return {
        'knn__n_neighbors': [3, 5, 7, 10, 15, 20, 30],
        'knn__weights': ['uniform', 'distance'],
        'knn__metric': ['euclidean', 'manhattan', 'minkowski'],
        'knn__p': [1, 2, 3],  # Only used with minkowski
    }


def run_grid_search(X_train, y_train, cv_folds=5):
    """Run GridSearchCV to find optimal hyperparameters.
    
    Args:
        X_train: Training features
        y_train: Training target
        cv_folds: Number of cross-validation folds
        
    Returns:
        grid_search: Fitted GridSearchCV object
        results_df: DataFrame with all CV results
    """
    logger.info("=" * 60)
    logger.info("Starting GridSearchCV...")
    logger.info(f"CV Folds: {cv_folds}")
    logger.info("=" * 60)
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('knn', KNeighborsRegressor())
    ])
    
    # Parameter grid
    param_grid = create_param_grid()
    
    # Calculate total combinations
    total_combos = (
        len(param_grid['knn__n_neighbors']) *
        len(param_grid['knn__weights']) *
        len(param_grid['knn__metric']) *
        len(param_grid['knn__p'])
    )
    logger.info(f"Total parameter combinations: {total_combos}")
    logger.info(f"Total fits: {total_combos * cv_folds}")
    
    # GridSearchCV
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv_folds,
        scoring='neg_mean_absolute_error',  # Optimize for MAE
        n_jobs=-1,  # Use all cores
        verbose=2,
        return_train_score=True,
        error_score='raise'
    )
    
    # Fit
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    
    logger.info(f"GridSearchCV completed in {elapsed_time:.1f} seconds")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(grid_search.cv_results_)
    
    return grid_search, results_df


def evaluate_best_model(grid_search, X_test, y_test):
    """Evaluate the best model on held-out test set.
    
    Args:
        grid_search: Fitted GridSearchCV object
        X_test: Test features
        y_test: Test target
        
    Returns:
        metrics: Dictionary of test metrics
    """
    logger.info("Evaluating best model on test set...")
    
    # Get predictions
    y_pred = grid_search.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'test_r2': r2_score(y_test, y_pred),
        'test_mae': mean_absolute_error(y_test, y_pred),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'cv_mae': -grid_search.best_score_,  # GridSearchCV uses negative MAE
    }
    
    # Calculate overfitting gap
    metrics['overfitting_gap'] = metrics['cv_mae'] - metrics['test_mae']
    metrics['overfitting_gap_pct'] = (metrics['overfitting_gap'] / metrics['cv_mae']) * 100
    
    return metrics


def save_results(grid_search, results_df, test_metrics, feature_names, experiment_name):
    """Save tuning results and best model.
    
    Args:
        grid_search: Fitted GridSearchCV object
        results_df: DataFrame with all CV results
        test_metrics: Dictionary of test metrics
        feature_names: List of feature names
        experiment_name: MLflow experiment name
    """
    logger.info("Saving results...")
    
    # Create output directories
    MODEL_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)
    
    # Best parameters
    best_params = grid_search.best_params_
    
    # Summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'version': 'v2.3',
        'best_params': best_params,
        'cv_mae': -grid_search.best_score_,
        'test_metrics': test_metrics,
        'n_features': len(feature_names),
        'feature_names': feature_names,
    }
    
    # Save summary JSON
    summary_path = MODEL_DIR / "tuning_results.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Saved tuning summary to {summary_path}")
    
    # Save detailed results CSV
    results_path = LOGS_DIR / "v2.3_grid_search_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"Saved detailed results to {results_path}")
    
    # Save best model
    model_path = MODEL_DIR / "model_v2.3_tuned.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(grid_search.best_estimator_, f)
    logger.info(f"Saved tuned model to {model_path}")
    
    # Log to MLflow if available
    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_experiment(experiment_name)
            
            with mlflow.start_run(run_name=f"v2.3-tuning-{datetime.now().strftime('%Y%m%d-%H%M%S')}"):
                # Log parameters
                for param_name, param_value in best_params.items():
                    mlflow.log_param(param_name.replace('knn__', ''), param_value)
                
                # Log metrics
                mlflow.log_metric("cv_mae", -grid_search.best_score_)
                mlflow.log_metric("test_r2", test_metrics['test_r2'])
                mlflow.log_metric("test_mae", test_metrics['test_mae'])
                mlflow.log_metric("test_rmse", test_metrics['test_rmse'])
                mlflow.log_metric("overfitting_gap", test_metrics['overfitting_gap'])
                
                # Log artifacts
                mlflow.log_artifact(str(summary_path))
                mlflow.log_artifact(str(results_path))
                
                # Log model
                mlflow.sklearn.log_model(
                    grid_search.best_estimator_,
                    "model",
                    registered_model_name="real-estate-price-predictor-v2.3"
                )
                
            logger.info("Results logged to MLflow")
        except Exception as e:
            logger.warning(f"MLflow logging failed: {e}")
    
    return summary


def print_results_summary(grid_search, test_metrics, baseline_mae=89769):
    """Print a formatted summary of tuning results.
    
    Args:
        grid_search: Fitted GridSearchCV object
        test_metrics: Dictionary of test metrics
        baseline_mae: V2.1 baseline MAE for comparison
    """
    print("\n" + "=" * 70)
    print(" V2.3 HYPERPARAMETER TUNING RESULTS")
    print("=" * 70)
    
    print("\n📊 BEST HYPERPARAMETERS:")
    print("-" * 40)
    for param, value in grid_search.best_params_.items():
        print(f"  {param.replace('knn__', ''):15} = {value}")
    
    print("\n📈 PERFORMANCE METRICS:")
    print("-" * 40)
    print(f"  CV MAE (5-fold):   ${test_metrics['cv_mae']:>12,.2f}")
    print(f"  Test MAE:          ${test_metrics['test_mae']:>12,.2f}")
    print(f"  Test R²:           {test_metrics['test_r2']:>12.4f}")
    print(f"  Test RMSE:         ${test_metrics['test_rmse']:>12,.2f}")
    
    print("\n🔄 OVERFITTING CHECK:")
    print("-" * 40)
    print(f"  CV - Test Gap:     ${test_metrics['overfitting_gap']:>12,.2f}")
    print(f"  Gap %:             {test_metrics['overfitting_gap_pct']:>12.1f}%")
    
    if abs(test_metrics['overfitting_gap_pct']) < 5:
        print("  Status:            ✅ Minimal overfitting")
    elif abs(test_metrics['overfitting_gap_pct']) < 10:
        print("  Status:            ⚠️ Moderate overfitting")
    else:
        print("  Status:            ❌ Significant overfitting")
    
    print("\n📊 COMPARISON TO V2.1 BASELINE:")
    print("-" * 40)
    print(f"  V2.1 Test MAE:     ${baseline_mae:>12,.2f}")
    print(f"  V2.3 Test MAE:     ${test_metrics['test_mae']:>12,.2f}")
    
    improvement = baseline_mae - test_metrics['test_mae']
    improvement_pct = (improvement / baseline_mae) * 100
    
    if improvement > 0:
        print(f"  Improvement:       ${improvement:>12,.2f} ({improvement_pct:.1f}%) ✅")
    else:
        print(f"  Regression:        ${-improvement:>12,.2f} ({-improvement_pct:.1f}%) ❌")
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description='V2.3 Hyperparameter Tuning')
    parser.add_argument(
        '--experiment-name',
        default='real-estate-v2.3-tuning',
        help='MLflow experiment name'
    )
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='Number of cross-validation folds'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Test set proportion'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random state for reproducibility'
    )
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("V2.3 HYPERPARAMETER TUNING")
    logger.info("=" * 60)
    
    # Load data
    X, y, feature_names = load_and_prepare_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state
    )
    logger.info(f"Train set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    # Run grid search
    grid_search, results_df = run_grid_search(
        X_train, y_train,
        cv_folds=args.cv_folds
    )
    
    # Evaluate on test set
    test_metrics = evaluate_best_model(grid_search, X_test, y_test)
    
    # Save results
    summary = save_results(
        grid_search, results_df, test_metrics,
        feature_names, args.experiment_name
    )
    
    # Print summary
    print_results_summary(grid_search, test_metrics)
    
    # Return for programmatic use
    return grid_search, test_metrics


if __name__ == "__main__":
    main()
