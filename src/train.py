"""
Training script for the Real Estate Price Predictor.

This script trains a KNeighborsRegressor model with RobustScaler preprocessing
on King County (Seattle) housing data merged with demographic information.

Features:
- MLflow experiment tracking (parameters, metrics, artifacts)
- Comprehensive evaluation metrics (R², MAE, RMSE)
- Train/test split with overfitting detection
- Artifact generation (model.pkl, model_features.json, metrics.json)

Usage:
    python src/train.py
    python src/train.py --k-neighbors 7 --test-size 0.3
    python src/train.py --experiment-name "real-estate-v2" --run-name "v2.1-all-features"

Version History:
    V1: 7 home features + 26 demographics = 33 total (R² = 0.728)
    V2.1: 18 home features + 26 demographics = 44 total (expanded feature set)

Bug Fix Applied:
    - Corrected DEMOGRAPHICS_PATH from kc_house_data.csv to zipcode_demographics.csv
    (Original bug in Reference_Docs/mle-project-challenge-2/create_model.py line 14)

V2.1 Feature Expansion:
    Added features: lat, long, waterfront, view, condition, grade,
    yr_built, yr_renovated, sqft_living15, sqft_lot15
"""

import argparse
import json
import pathlib
import pickle
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn import model_selection, neighbors, pipeline, preprocessing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Try to import MLflow; gracefully degrade if not available
try:
    import mlflow.sklearn

    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("Warning: MLflow not installed. Training will proceed without experiment tracking.")

# ==============================================================================
# CONFIGURATION - Bug fix applied: DEMOGRAPHICS_PATH corrected
# ==============================================================================

SALES_PATH = "data/kc_house_data.csv"  # Path to CSV with home sale data
DEMOGRAPHICS_PATH = "data/zipcode_demographics.csv"  # FIXED: was incorrectly kc_house_data.csv

# ==============================================================================
# V2.1 FEATURE EXPANSION: All 18 columns from kc_house_data.csv
# ==============================================================================
# V1 used: bedrooms, bathrooms, sqft_living, sqft_lot, floors, sqft_above, sqft_basement
# V2.1 adds: lat, long, waterfront, view, condition, grade, yr_built, yr_renovated,
#            sqft_living15, sqft_lot15
#
# Why these features matter:
# - waterfront: Waterfront homes in Seattle can be 50-100% more expensive
# - grade: Construction quality (1-13) is highly predictive
# - view: View quality premium is significant in hilly Seattle
# - lat/long: Captures location value beyond zipcode demographics
# - condition: Well-maintained homes fetch higher prices
# - sqft_living15/sqft_lot15: Neighborhood context ("big/small for the area")
# - yr_built/yr_renovated: Age and renovation status affect value
# ==============================================================================

SALES_COLUMN_SELECTION = [
    # Target variable
    "price",
    # V1 features (structural)
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "sqft_above",
    "sqft_basement",
    # V2.1 NEW: Property characteristics (high predictive value)
    "waterfront",  # Binary: 0/1 - waterfront premium
    "view",  # Ordinal: 0-4 - view quality
    "condition",  # Ordinal: 1-5 - maintenance state
    "grade",  # Ordinal: 1-13 - construction quality
    # V2.1 NEW: Age features
    "yr_built",  # Year house was built
    "yr_renovated",  # Year of last renovation (0 if never)
    # V2.1 NEW: Spatial features
    "lat",  # Latitude
    "long",  # Longitude
    # V2.1 NEW: Neighborhood context
    "sqft_living15",  # Avg living sqft of 15 nearest neighbors
    "sqft_lot15",  # Avg lot sqft of 15 nearest neighbors
    # Join key (dropped after merge)
    "zipcode",
]

OUTPUT_DIR = "model"  # Directory for output artifacts
RANDOM_STATE = 42  # For reproducibility

# Model Registry Name (for MLflow)
MODEL_REGISTRY_NAME = "real-estate-price-predictor"

# ==============================================================================
# DATA LOADING
# ==============================================================================


def load_data(
    sales_path: str, demographics_path: str, sales_column_selection: list[str]
) -> tuple[pd.DataFrame, pd.Series]:
    """Load and merge sales data with demographics.

    Args:
        sales_path: Path to CSV file with home sale data
        demographics_path: Path to CSV file with zipcode demographics
        sales_column_selection: List of columns to select from sales data

    Returns:
        Tuple of (features DataFrame, target Series)
    """
    # Load sales data with selected columns
    sales = pd.read_csv(sales_path, usecols=sales_column_selection, dtype={"zipcode": str})

    # Load demographics data
    demographics = pd.read_csv(demographics_path, dtype={"zipcode": str})

    # Merge on zipcode and drop the join key
    merged = sales.merge(demographics, how="left", on="zipcode").drop(columns="zipcode")

    # Separate target from features
    y = merged.pop("price")
    X = merged

    return X, y


# ==============================================================================
# MODEL TRAINING
# ==============================================================================


def create_model(k_neighbors: int = 5) -> pipeline.Pipeline:
    """Create the ML pipeline with scaling and KNN regressor.

    Args:
        k_neighbors: Number of neighbors for KNN (default 5)

    Returns:
        Unfitted sklearn Pipeline
    """
    return pipeline.make_pipeline(
        preprocessing.RobustScaler(), neighbors.KNeighborsRegressor(n_neighbors=k_neighbors)
    )


def evaluate_model(
    model: pipeline.Pipeline, X: pd.DataFrame, y: pd.Series, prefix: str = ""
) -> dict:
    """Calculate comprehensive evaluation metrics.

    Args:
        model: Fitted model
        X: Feature matrix
        y: True target values
        prefix: Prefix for metric names (e.g., "train_" or "test_")

    Returns:
        Dictionary of metrics
    """
    predictions = model.predict(X)

    r2 = r2_score(y, predictions)
    mae = mean_absolute_error(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))

    return {
        f"{prefix}r2": round(r2, 4),
        f"{prefix}mae": round(mae, 2),
        f"{prefix}rmse": round(rmse, 2),
    }


def train_model(
    k_neighbors: int = 5,
    test_size: float = 0.25,
    experiment_name: str = "real-estate-v2",
    run_name: str = None,
) -> tuple[pipeline.Pipeline, dict]:
    """Train the model with MLflow tracking.

    Args:
        k_neighbors: Number of neighbors for KNN
        test_size: Fraction of data for testing
        experiment_name: MLflow experiment name
        run_name: MLflow run name (auto-generated if None)

    Returns:
        Tuple of (trained model, metrics dictionary)
    """
    # Generate run name if not provided
    if run_name is None:
        run_name = f"knn-k{k_neighbors}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # Load data
    print(f"Loading data from {SALES_PATH} and {DEMOGRAPHICS_PATH}...")
    X, y = load_data(SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION)
    print(f"Loaded {len(X)} samples with {len(X.columns)} features")

    # Train/test split
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE
    )
    print(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")

    # Prepare output directory
    output_dir = pathlib.Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    # MLflow tracking context (or no-op if unavailable)
    mlflow_context = _get_mlflow_context(experiment_name, run_name)

    with mlflow_context:
        # Log parameters
        params = {
            "k_neighbors": k_neighbors,
            "test_size": test_size,
            "scaler": "RobustScaler",
            "model_type": "KNeighborsRegressor",
            "random_state": RANDOM_STATE,
            "n_features": len(X.columns),
            "n_samples": len(X),
            "data_source": "King County (Seattle) 2014-2015",
        }
        _log_params(params)
        print(f"Training with parameters: k={k_neighbors}, test_size={test_size}")

        # Create and train model
        model = create_model(k_neighbors)
        model.fit(X_train, y_train)
        print("Model training complete.")

        # Evaluate on train and test sets
        train_metrics = evaluate_model(model, X_train, y_train, prefix="train_")
        test_metrics = evaluate_model(model, X_test, y_test, prefix="test_")

        # Calculate overfitting gap
        overfitting_gap = round(train_metrics["train_r2"] - test_metrics["test_r2"], 4)

        # Combine all metrics
        all_metrics = {
            **train_metrics,
            **test_metrics,
            "overfitting_gap": overfitting_gap,
        }
        _log_metrics(all_metrics)

        # Print metrics
        print("\n" + "=" * 60)
        print("MODEL EVALUATION RESULTS")
        print("=" * 60)
        print(f"Train R²:  {train_metrics['train_r2']:.4f}")
        print(f"Test R²:   {test_metrics['test_r2']:.4f}")
        print(f"Overfitting Gap: {overfitting_gap:.4f}")
        print("-" * 60)
        print(f"Train MAE:  ${train_metrics['train_mae']:,.2f}")
        print(f"Test MAE:   ${test_metrics['test_mae']:,.2f}")
        print(f"Train RMSE: ${train_metrics['train_rmse']:,.2f}")
        print(f"Test RMSE:  ${test_metrics['test_rmse']:,.2f}")
        print("=" * 60 + "\n")

        # Save artifacts locally
        model_path = output_dir / "model.pkl"
        features_path = output_dir / "model_features.json"
        metrics_path = output_dir / "metrics.json"

        # Save model
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"Model saved to {model_path}")

        # Save feature names (order matters for prediction)
        feature_names = list(X_train.columns)
        with open(features_path, "w") as f:
            json.dump(feature_names, f, indent=2)
        print(f"Feature names saved to {features_path}")

        # Save metrics for CI/CD
        metrics_output = {
            **all_metrics,
            "timestamp": datetime.now().isoformat(),
            "model_version": "v2.1",  # V2.1 = expanded features
            "k_neighbors": k_neighbors,
            "test_size": test_size,
            "n_samples": len(X),
            "n_features": len(feature_names),
            "n_home_features": 18,  # All columns from kc_house_data.csv
            "n_demographic_features": len(feature_names) - 18,
            "data_vintage": "2014-2015",
            "data_location": "King County (Seattle), WA",
            "features_added_in_v2.1": [
                "lat",
                "long",
                "waterfront",
                "view",
                "condition",
                "grade",
                "yr_built",
                "yr_renovated",
                "sqft_living15",
                "sqft_lot15",
            ],
        }
        with open(metrics_path, "w") as f:
            json.dump(metrics_output, f, indent=2)
        print(f"Metrics saved to {metrics_path}")

        # Log artifacts to MLflow
        _log_artifact(str(model_path))
        _log_artifact(str(features_path))
        _log_artifact(str(metrics_path))

        # Log model to MLflow Model Registry
        _log_sklearn_model(model, "model", feature_names)

    return model, all_metrics


# ==============================================================================
# MLFLOW HELPERS (graceful degradation if not available)
# ==============================================================================


class _NoOpContext:
    """No-op context manager when MLflow is unavailable."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def _get_mlflow_context(experiment_name: str, run_name: str):
    """Get MLflow run context or no-op context."""
    if not MLFLOW_AVAILABLE:
        return _NoOpContext()

    # Set or create experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

    return mlflow.start_run(run_name=run_name)


def _log_params(params: dict):
    """Log parameters to MLflow if available."""
    if MLFLOW_AVAILABLE:
        mlflow.log_params(params)


def _log_metrics(metrics: dict):
    """Log metrics to MLflow if available."""
    if MLFLOW_AVAILABLE:
        mlflow.log_metrics(metrics)


def _log_artifact(path: str):
    """Log artifact to MLflow if available."""
    if MLFLOW_AVAILABLE:
        mlflow.log_artifact(path)


def _log_sklearn_model(model, artifact_path: str, feature_names: list[str]):
    """Log sklearn model to MLflow if available."""
    if MLFLOW_AVAILABLE:
        # Create input signature
        try:
            from mlflow.models.signature import infer_signature

            # Create dummy input for signature
            dummy_input = pd.DataFrame([[0.0] * len(feature_names)], columns=feature_names)
            signature = infer_signature(dummy_input, model.predict(dummy_input))

            mlflow.sklearn.log_model(
                model,
                artifact_path=artifact_path,
                signature=signature,
                registered_model_name=MODEL_REGISTRY_NAME,
            )
            print(f"Model logged to MLflow registry as '{MODEL_REGISTRY_NAME}'")
        except Exception as e:
            # Fall back to logging without signature
            print(f"Warning: Could not infer signature ({e}). Logging model without signature.")
            mlflow.sklearn.log_model(model, artifact_path=artifact_path)


# ==============================================================================
# CLI ENTRY POINT
# ==============================================================================


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train the Real Estate Price Predictor model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--k-neighbors", "-k", type=int, default=5, help="Number of neighbors for KNN algorithm"
    )
    parser.add_argument(
        "--test-size",
        "-t",
        type=float,
        default=0.25,
        help="Fraction of data to use for testing (0.0-1.0)",
    )
    parser.add_argument(
        "--experiment-name", "-e", type=str, default="real-estate-v2", help="MLflow experiment name"
    )
    parser.add_argument(
        "--run-name",
        "-r",
        type=str,
        default=None,
        help="MLflow run name (auto-generated if not provided)",
    )
    return parser.parse_args()


def main():
    """Main entry point for training."""
    args = parse_args()

    print("\n" + "=" * 60)
    print("REAL ESTATE PRICE PREDICTOR - MODEL TRAINING")
    print("=" * 60)
    print("Data: King County (Seattle) housing sales 2014-2015")
    print("Model: KNeighborsRegressor with RobustScaler")
    print(f"MLflow tracking: {'Enabled' if MLFLOW_AVAILABLE else 'Disabled'}")
    print("=" * 60 + "\n")

    try:
        model, metrics = train_model(
            k_neighbors=args.k_neighbors,
            test_size=args.test_size,
            experiment_name=args.experiment_name,
            run_name=args.run_name,
        )

        print("\n[SUCCESS] Training completed successfully!")
        print(f"          Model artifacts saved to: {OUTPUT_DIR}/")

        # Return success
        return 0

    except FileNotFoundError as e:
        print(f"\n[ERROR] Data file not found: {e}")
        print("        Make sure data files are in the 'data/' directory.")
        return 1

    except Exception as e:
        print(f"\n[ERROR] Training failed with error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
