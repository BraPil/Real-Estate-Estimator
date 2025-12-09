"""
MLflow-Integrated Training Script

This script trains the real estate price prediction model with full
MLflow experiment tracking. Every run is logged with:
- Parameters (hyperparameters, data info)
- Metrics (MAE, RMSE, R2, MAPE)
- Artifacts (model file, feature importance plot)
- Tags (version, environment info)

Usage:
    # Basic training run
    python src/train_with_mlflow.py

    # Training with custom parameters
    python src/train_with_mlflow.py --n-estimators 500 --max-depth 6

    # Training with a custom run name
    python src/train_with_mlflow.py --run-name "experiment-v1"

MLflow Tracking Explained:
--------------------------
When you run this script, MLflow will:

1. START A RUN
   - Creates a unique run_id
   - Records start time
   - Associates with experiment

2. LOG PARAMETERS
   - All hyperparameters (n_estimators, max_depth, etc.)
   - Data info (n_samples, n_features, test_size)
   - Model type

3. LOG METRICS
   - MAE (Mean Absolute Error)
   - RMSE (Root Mean Squared Error)
   - R2 (Coefficient of Determination)
   - MAPE (Mean Absolute Percentage Error)

4. LOG ARTIFACTS
   - The trained model (as MLflow model format)
   - Feature importance plot
   - Evaluation report

5. END THE RUN
   - Records end time
   - Calculates duration
   - Marks as FINISHED or FAILED
"""

import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor

import mlflow

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from mlflow_config import MLFLOW_MODEL_NAME, setup_mlflow

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_DIR = Path(__file__).parent.parent / "model"
RANDOM_STATE = 42

# Data source options
DATA_SOURCES = {
    "original": "kc_house_data.csv",  # 2014-2015 data (21K records)
    "fresh": "assessment_2020_plus_v4.csv",  # 2020+ data (143K records)
}
DEFAULT_DATA_SOURCE = "fresh"  # V3.3: Use fresh data by default

# Default XGBoost parameters (V3.3 Optuna-tuned)
DEFAULT_PARAMS = {
    "n_estimators": 355,
    "max_depth": 10,
    "learning_rate": 0.1134,
    "subsample": 0.7456,
    "colsample_bytree": 0.9479,
    "min_child_weight": 3,
    "gamma": 0.0493,
    "reg_alpha": 0.4073,
    "reg_lambda": 1.5574,
}


# =============================================================================
# DATA LOADING
# =============================================================================


def load_data(data_source: str = DEFAULT_DATA_SOURCE) -> tuple[pd.DataFrame, pd.Series, list]:
    """
    Load and prepare training data.

    Args:
        data_source: Either "original" (2014-15) or "fresh" (2020+)

    Returns:
        X: Feature DataFrame
        y: Target Series (prices)
        feature_names: List of feature column names
    """
    print(f"Loading data (source: {data_source})...")

    # Base feature columns (used by both data sources)
    base_features = [
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "floors",
        "sqft_above",
        "sqft_basement",
        "waterfront",
        "view",
        "condition",
        "grade",
        "yr_built",
        "yr_renovated",
        "lat",
        "long",
        "sqft_living15",
        "sqft_lot15",
    ]

    # V3.3 temporal features (fresh data only)
    temporal_features = ["sale_year", "sale_month", "sale_quarter", "sale_dow"]

    # Load data based on source
    data_file = DATA_DIR / DATA_SOURCES[data_source]
    df = pd.read_csv(data_file)
    demographics = pd.read_csv(DATA_DIR / "zipcode_demographics.csv")

    # Merge with demographics
    df["zipcode"] = df["zipcode"].astype(str).str.strip()
    demographics["zipcode"] = demographics["zipcode"].astype(str).str.strip()
    merged = df.merge(demographics, on="zipcode", how="inner")

    # Build feature list
    feature_cols = base_features.copy()

    # Add temporal features if available (V3.3+)
    for col in temporal_features:
        if col in merged.columns:
            feature_cols.append(col)

    # Add demographic columns
    demo_cols = [c for c in demographics.columns if c != "zipcode"]
    feature_cols.extend(demo_cols)

    # Handle missing values
    for col in feature_cols:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0)
        else:
            merged[col] = 0

    # Separate features and target
    y = merged["price"]
    X = merged[feature_cols]

    print(f"Loaded {len(X)} samples, {len(X.columns)} features")

    return X, y, list(X.columns)


# =============================================================================
# TRAINING
# =============================================================================


def train_model(X_train: pd.DataFrame, y_train: pd.Series, params: dict[str, Any]) -> Pipeline:
    """
    Train the XGBoost model with given parameters.

    Args:
        X_train: Training features
        y_train: Training targets
        params: XGBoost hyperparameters

    Returns:
        Trained sklearn Pipeline (scaler + model)
    """
    print("Training model...")

    # Create model
    model = XGBRegressor(**params, random_state=RANDOM_STATE, n_jobs=-1)

    # Create pipeline with scaler
    pipeline = Pipeline([("scaler", RobustScaler()), ("model", model)])

    # Fit
    pipeline.fit(X_train, y_train)

    return pipeline


def evaluate_model(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """
    Evaluate model and return metrics.

    Args:
        pipeline: Trained model pipeline
        X_test: Test features
        y_test: Test targets

    Returns:
        Dictionary of metrics
    """
    print("Evaluating model...")

    # Predictions
    y_pred = pipeline.predict(X_test)

    # Calculate metrics
    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2": r2_score(y_test, y_pred),
        "mape": np.mean(np.abs((y_test - y_pred) / y_test)) * 100,
    }

    return metrics


def cross_validate(
    X: pd.DataFrame, y: pd.Series, params: dict[str, Any], cv: int = 5
) -> dict[str, float]:
    """
    Perform cross-validation.

    Returns:
        Dictionary with CV metrics
    """
    print(f"Running {cv}-fold cross-validation...")

    model = XGBRegressor(**params, random_state=RANDOM_STATE, n_jobs=-1)
    pipeline = Pipeline([("scaler", RobustScaler()), ("model", model)])

    # MAE scores (negated because sklearn convention)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1)

    return {"cv_mae_mean": -scores.mean(), "cv_mae_std": scores.std()}


# =============================================================================
# MLFLOW TRAINING PIPELINE
# =============================================================================


def train_with_tracking(
    run_name: str = None,
    params: dict[str, Any] = None,
    test_size: float = 0.2,
    register_model: bool = False,
    data_source: str = DEFAULT_DATA_SOURCE,
) -> str:
    """
    Full training pipeline with MLflow tracking.

    This is the main function that orchestrates:
    1. Data loading
    2. Train/test split
    3. Model training
    4. Evaluation
    5. MLflow logging

    Args:
        run_name: Optional name for this run
        params: Model hyperparameters (uses defaults if None)
        test_size: Fraction of data for testing
        register_model: Whether to register in model registry
        data_source: "original" (2014-15) or "fresh" (2020+)

    Returns:
        run_id: The MLflow run ID
    """
    # Setup MLflow
    setup_mlflow()

    # Use default params if not provided
    if params is None:
        params = DEFAULT_PARAMS.copy()

    # Generate run name if not provided
    if run_name is None:
        run_name = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print("\n" + "=" * 60)
    print(f" MLflow Training Run: {run_name}")
    print(f" Data Source: {data_source}")
    print("=" * 60)

    # Load data
    X, y, feature_names = load_data(data_source=data_source)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE
    )

    # Start MLflow run
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        print(f"\nMLflow Run ID: {run_id}")

        # ----- LOG PARAMETERS -----
        # These are the "inputs" to your experiment
        print("\n[1/5] Logging parameters...")

        # Log all model hyperparameters
        for key, value in params.items():
            mlflow.log_param(key, value)

        # Log data info
        mlflow.log_param("data_source", data_source)
        mlflow.log_param("n_samples", len(X))
        mlflow.log_param("n_features", len(feature_names))
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", RANDOM_STATE)

        # Log tags (metadata, not parameters)
        mlflow.set_tag("model_type", "XGBoost")
        mlflow.set_tag("version", "v3.3")
        mlflow.set_tag("stage", "development")

        # ----- TRAIN MODEL -----
        print("\n[2/5] Training model...")
        pipeline = train_model(X_train, y_train, params)

        # ----- EVALUATE -----
        print("\n[3/5] Evaluating model...")

        # Test set metrics
        test_metrics = evaluate_model(pipeline, X_test, y_test)

        # Cross-validation metrics
        cv_metrics = cross_validate(X, y, params)

        # ----- LOG METRICS -----
        # These are the "outputs" - what we're trying to optimize
        print("\n[4/5] Logging metrics...")

        # Test metrics
        mlflow.log_metric("test_mae", test_metrics["mae"])
        mlflow.log_metric("test_rmse", test_metrics["rmse"])
        mlflow.log_metric("test_r2", test_metrics["r2"])
        mlflow.log_metric("test_mape", test_metrics["mape"])

        # CV metrics
        mlflow.log_metric("cv_mae_mean", cv_metrics["cv_mae_mean"])
        mlflow.log_metric("cv_mae_std", cv_metrics["cv_mae_std"])

        # ----- LOG ARTIFACTS -----
        # These are files associated with the run
        print("\n[5/5] Logging artifacts...")

        # Log the model using MLflow's model logging
        # This creates a standardized model format that can be loaded later
        mlflow.sklearn.log_model(
            pipeline, "model", registered_model_name=MLFLOW_MODEL_NAME if register_model else None
        )

        # Log feature names as JSON
        feature_info = {"feature_names": feature_names, "n_features": len(feature_names)}
        mlflow.log_dict(feature_info, "feature_info.json")

        # Log evaluation report
        report = {
            "run_id": run_id,
            "run_name": run_name,
            "timestamp": datetime.now().isoformat(),
            "test_metrics": test_metrics,
            "cv_metrics": cv_metrics,
            "parameters": params,
        }
        mlflow.log_dict(report, "evaluation_report.json")

        # ----- PRINT RESULTS -----
        print("\n" + "=" * 60)
        print(" Training Complete!")
        print("=" * 60)
        print(f"\nRun ID: {run_id}")
        print("\nTest Metrics:")
        print(f"  MAE:  ${test_metrics['mae']:,.0f}")
        print(f"  RMSE: ${test_metrics['rmse']:,.0f}")
        print(f"  R2:   {test_metrics['r2']:.4f}")
        print(f"  MAPE: {test_metrics['mape']:.2f}%")
        print("\nCross-Validation:")
        print(f"  MAE:  ${cv_metrics['cv_mae_mean']:,.0f} +/- ${cv_metrics['cv_mae_std']:,.0f}")

        if register_model:
            print(f"\nModel registered as: {MLFLOW_MODEL_NAME}")

        print("\n" + "=" * 60)

        return run_id


# =============================================================================
# CLI INTERFACE
# =============================================================================


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train real estate model with MLflow tracking")

    parser.add_argument("--run-name", type=str, default=None, help="Name for this training run")

    parser.add_argument(
        "--data-source",
        type=str,
        default=DEFAULT_DATA_SOURCE,
        choices=["original", "fresh"],
        help="Data source: 'original' (2014-15) or 'fresh' (2020+)",
    )

    parser.add_argument(
        "--n-estimators", type=int, default=DEFAULT_PARAMS["n_estimators"], help="Number of trees"
    )

    parser.add_argument(
        "--max-depth", type=int, default=DEFAULT_PARAMS["max_depth"], help="Maximum tree depth"
    )

    parser.add_argument(
        "--learning-rate", type=float, default=DEFAULT_PARAMS["learning_rate"], help="Learning rate"
    )

    parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction")

    parser.add_argument("--register", action="store_true", help="Register model in MLflow registry")

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Build params from args
    params = DEFAULT_PARAMS.copy()
    params["n_estimators"] = args.n_estimators
    params["max_depth"] = args.max_depth
    params["learning_rate"] = args.learning_rate

    # Run training
    run_id = train_with_tracking(
        run_name=args.run_name,
        params=params,
        test_size=args.test_size,
        register_model=args.register,
        data_source=args.data_source,
    )

    print("\nTo view this run in MLflow UI:")
    print("  mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db")
    print("  Then open: http://localhost:5000")

    return run_id


if __name__ == "__main__":
    main()
