"""
V2.4.1: Light Hyperparameter Tuning for Top 3 Models

Uses RandomizedSearchCV for quick optimization of:
1. XGBoost (current leader: $68,092)
2. LightGBM ($68,934)
3. Random Forest ($70,645)

Methodology:
- RandomizedSearchCV with 20 iterations per model
- 5-fold cross-validation
- Focused parameter spaces based on defaults that worked well

Usage:
    python src/tune_top_models.py
    python src/tune_top_models.py --n-iter 30  # More iterations
    python src/tune_top_models.py --models xgboost lightgbm  # Specific models
"""

import argparse
import json
import logging
import pickle
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform

# Suppress sklearn/lightgbm feature name warnings
warnings.filterwarnings('ignore', message='X does not have valid feature names')
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Optional MLflow
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

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

# Feature selection (same as compare_models.py)
SALES_COLUMN_SELECTION = [
    'price',
    'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'sqft_above', 'sqft_basement',
    'waterfront', 'view', 'condition', 'grade',
    'yr_built', 'yr_renovated',
    'lat', 'long',
    'sqft_living15', 'sqft_lot15',
    'zipcode'
]

RANDOM_STATE = 42
CV_FOLDS = 5
TEST_SIZE = 0.2


def load_data() -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Load and prepare data (same as compare_models.py)."""
    logger.info("Loading data...")
    
    sales_df = pd.read_csv(SALES_PATH, usecols=SALES_COLUMN_SELECTION)
    demographics_df = pd.read_csv(DEMOGRAPHICS_PATH)
    merged_df = sales_df.merge(demographics_df, on='zipcode', how='inner')
    
    y = merged_df['price']
    X = merged_df.drop(columns=['price', 'zipcode'])
    
    logger.info(f"Loaded {len(X)} samples, {len(X.columns)} features")
    return X, y, list(X.columns)


def get_param_distributions() -> Dict[str, Dict]:
    """Get parameter distributions for RandomizedSearchCV.
    
    These are focused ranges around the defaults that performed well.
    """
    return {
        'XGBoost': {
            'model__n_estimators': randint(50, 300),
            'model__max_depth': randint(3, 10),
            'model__learning_rate': uniform(0.01, 0.29),  # 0.01 to 0.3
            'model__subsample': uniform(0.6, 0.4),  # 0.6 to 1.0
            'model__colsample_bytree': uniform(0.6, 0.4),  # 0.6 to 1.0
            'model__min_child_weight': randint(1, 10),
            'model__gamma': uniform(0, 0.5),
            'model__reg_alpha': uniform(0, 1),  # L1 regularization
            'model__reg_lambda': uniform(0, 2),  # L2 regularization
        },
        'LightGBM': {
            'model__n_estimators': randint(50, 300),
            'model__max_depth': randint(3, 15),
            'model__learning_rate': uniform(0.01, 0.29),
            'model__num_leaves': randint(15, 100),
            'model__subsample': uniform(0.6, 0.4),
            'model__colsample_bytree': uniform(0.6, 0.4),
            'model__min_child_samples': randint(5, 50),
            'model__reg_alpha': uniform(0, 1),
            'model__reg_lambda': uniform(0, 2),
        },
        'RandomForest': {
            'model__n_estimators': randint(50, 300),
            'model__max_depth': randint(10, 40),
            'model__min_samples_split': randint(2, 20),
            'model__min_samples_leaf': randint(1, 10),
            'model__max_features': ['sqrt', 'log2', 0.5, 0.7, None],
            'model__bootstrap': [True, False],
        }
    }


def create_base_models() -> Dict[str, Any]:
    """Create base model instances."""
    return {
        'XGBoost': XGBRegressor(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0
        ),
        'LightGBM': LGBMRegressor(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1
        ),
        'RandomForest': RandomForestRegressor(
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    }


def tune_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_iter: int = 20
) -> Dict:
    """Tune a single model with RandomizedSearchCV.
    
    Args:
        model_name: Name of model to tune
        X_train, y_train: Training data
        X_test, y_test: Test data
        n_iter: Number of random iterations
        
    Returns:
        Dictionary with results
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Tuning: {model_name}")
    logger.info(f"{'='*60}")
    
    # Get model and params
    base_models = create_base_models()
    param_dists = get_param_distributions()
    
    if model_name not in base_models:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('model', base_models[model_name])
    ])
    
    # RandomizedSearchCV
    logger.info(f"Running RandomizedSearchCV with {n_iter} iterations, {CV_FOLDS}-fold CV...")
    
    search = RandomizedSearchCV(
        pipeline,
        param_dists[model_name],
        n_iter=n_iter,
        cv=CV_FOLDS,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
        return_train_score=True
    )
    
    start_time = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - start_time
    
    # Evaluate on test set
    y_pred = search.predict(X_test)
    
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    cv_mae = -search.best_score_
    
    # Extract best params (remove 'model__' prefix)
    best_params = {
        k.replace('model__', ''): v 
        for k, v in search.best_params_.items()
    }
    
    results = {
        'model_name': model_name,
        'cv_mae': cv_mae,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'best_params': best_params,
        'tuning_time_sec': elapsed,
        'n_iter': n_iter,
        'cv_folds': CV_FOLDS,
        'best_pipeline': search.best_estimator_
    }
    
    logger.info(f"Best CV MAE: ${cv_mae:,.0f}")
    logger.info(f"Test MAE: ${test_mae:,.0f}")
    logger.info(f"Test R²: {test_r2:.4f}")
    logger.info(f"Time: {elapsed:.1f}s")
    
    return results


def run_tuning(
    models_to_tune: List[str] = None,
    n_iter: int = 20
) -> pd.DataFrame:
    """Run tuning on specified models.
    
    Args:
        models_to_tune: List of model names (default: all 3)
        n_iter: Number of random search iterations per model
        
    Returns:
        DataFrame with all results
    """
    if models_to_tune is None:
        models_to_tune = ['XGBoost', 'LightGBM', 'RandomForest']
    
    # Load data
    X, y, feature_names = load_data()
    
    # Split (same as compare_models.py)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Tune each model
    all_results = []
    best_pipelines = {}
    
    for model_name in models_to_tune:
        try:
            results = tune_model(
                model_name, X_train, y_train, X_test, y_test, n_iter
            )
            best_pipelines[model_name] = results.pop('best_pipeline')
            all_results.append(results)
        except Exception as e:
            logger.error(f"Failed to tune {model_name}: {e}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('test_mae')
    
    return results_df, best_pipelines, feature_names


def print_results(results_df: pd.DataFrame, baseline_mae: float = 84494):
    """Print formatted results table."""
    print("\n")
    print("=" * 90)
    print(" V2.4.1 TUNED MODEL COMPARISON (sorted by Test MAE)")
    print("=" * 90)
    
    # Header
    print(f"\n{'Model':<15} {'CV MAE':>12} {'Test MAE':>12} {'Test R²':>10} {'Test RMSE':>12} {'Time':>8}")
    print("-" * 90)
    
    for _, row in results_df.iterrows():
        print(f"{row['model_name']:<15} "
              f"${row['cv_mae']:>10,.0f} "
              f"${row['test_mae']:>10,.0f} "
              f"{row['test_r2']:>10.4f} "
              f"${row['test_rmse']:>10,.0f} "
              f"{row['tuning_time_sec']:>6.1f}s")
    
    print("-" * 90)
    
    # Best model
    best = results_df.iloc[0]
    print(f"\n🏆 BEST TUNED MODEL: {best['model_name']}")
    print(f"   Test MAE: ${best['test_mae']:,.0f}")
    print(f"   Test R²: {best['test_r2']:.4f}")
    
    # Improvement over KNN baseline
    improvement = baseline_mae - best['test_mae']
    improvement_pct = (improvement / baseline_mae) * 100
    print(f"\n📊 Improvement over KNN (V2.3): ${improvement:,.0f} ({improvement_pct:.1f}%)")
    
    # Best params
    print(f"\n🔧 Best Hyperparameters for {best['model_name']}:")
    for k, v in best['best_params'].items():
        if isinstance(v, float):
            print(f"   {k}: {v:.4f}")
        else:
            print(f"   {k}: {v}")
    
    print("=" * 90)


def save_results(
    results_df: pd.DataFrame,
    best_pipelines: Dict,
    feature_names: List[str]
):
    """Save tuning results and best model."""
    MODEL_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save results CSV
    results_path = LOGS_DIR / f"v2.4.1_tuning_results_{timestamp}.csv"
    # Convert best_params dict to string for CSV
    results_df_csv = results_df.copy()
    results_df_csv['best_params'] = results_df_csv['best_params'].apply(json.dumps)
    results_df_csv.to_csv(results_path, index=False)
    logger.info(f"Saved results to {results_path}")
    
    # Save best model
    best_model_name = results_df.iloc[0]['model_name']
    best_pipeline = best_pipelines[best_model_name]
    
    model_path = MODEL_DIR / f"model_v2.4.1_{best_model_name.lower()}_tuned.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(best_pipeline, f)
    logger.info(f"Saved best model to {model_path}")
    
    # Save summary JSON
    best = results_df.iloc[0]
    summary = {
        'timestamp': datetime.now().isoformat(),
        'version': 'v2.4.1',
        'best_model': best_model_name,
        'best_params': best['best_params'],
        'test_mae': float(best['test_mae']),
        'test_r2': float(best['test_r2']),
        'test_rmse': float(best['test_rmse']),
        'cv_mae': float(best['cv_mae']),
        'improvement_over_knn_pct': float((84494 - best['test_mae']) / 84494 * 100),
        'all_models': [
            {
                'model': r['model_name'],
                'test_mae': float(r['test_mae']),
                'test_r2': float(r['test_r2'])
            }
            for _, r in results_df.iterrows()
        ],
        'n_features': len(feature_names)
    }
    
    summary_path = MODEL_DIR / "tuning_results_v2.4.1.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary to {summary_path}")
    
    # Log to MLflow
    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_experiment("real-estate-v2.4.1-tuning")
            with mlflow.start_run(run_name=f"v2.4.1-{best_model_name}-{timestamp}"):
                mlflow.log_metric("test_mae", best['test_mae'])
                mlflow.log_metric("test_r2", best['test_r2'])
                mlflow.log_metric("cv_mae", best['cv_mae'])
                mlflow.log_params(best['best_params'])
                mlflow.log_artifact(str(results_path))
                mlflow.sklearn.log_model(best_pipeline, "model")
            logger.info("Logged to MLflow")
        except Exception as e:
            logger.warning(f"MLflow logging failed: {e}")
    
    return model_path


def main():
    parser = argparse.ArgumentParser(
        description='V2.4.1: Light tuning for top 3 models'
    )
    parser.add_argument(
        '--n-iter', type=int, default=20,
        help='Number of random search iterations per model (default: 20)'
    )
    parser.add_argument(
        '--models', nargs='+', 
        choices=['XGBoost', 'LightGBM', 'RandomForest'],
        default=['XGBoost', 'LightGBM', 'RandomForest'],
        help='Models to tune (default: all 3)'
    )
    args = parser.parse_args()
    
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " V2.4.1: LIGHT HYPERPARAMETER TUNING ".center(68) + "║")
    print("║" + f" RandomizedSearchCV ({args.n_iter} iterations per model) ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    # Run tuning
    results_df, best_pipelines, feature_names = run_tuning(
        models_to_tune=args.models,
        n_iter=args.n_iter
    )
    
    # Print results
    print_results(results_df)
    
    # Save results
    model_path = save_results(results_df, best_pipelines, feature_names)
    
    # Final recommendation
    best = results_df.iloc[0]
    print("\n" + "=" * 70)
    print(" RECOMMENDATION")
    print("=" * 70)
    print(f"\n✅ Use {best['model_name']} as the production model")
    print(f"   Tuned model saved to: {model_path}")
    print(f"\nNext steps:")
    print(f"   1. Copy tuned model to model/model.pkl")
    print(f"   2. Update model/metrics.json with new performance")
    print(f"   3. Test API with new model")
    print(f"   4. Validate on future_unseen_examples.csv")
    print("=" * 70 + "\n")
    
    return results_df, best_pipelines


if __name__ == "__main__":
    main()
