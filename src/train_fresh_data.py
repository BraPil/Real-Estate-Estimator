#!/usr/bin/env python3
"""
Train Model on Fresh 2020+ Assessment Data
V3.2: Fresh Data Integration

This script trains a new XGBoost model on King County 2020+ sales data
using the same pipeline as V2.5 but with fresh data.
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# Sklearn imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# XGBoost
from xgboost import XGBRegressor

# MLflow
import mlflow
import mlflow.sklearn
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from mlflow_config import setup_mlflow, MLFLOW_MODEL_NAME

# Paths
DATA_DIR = Path("data")
MODEL_DIR = Path("model")
DEMOGRAPHICS_PATH = DATA_DIR / "zipcode_demographics.csv"
FRESH_DATA_PATH = DATA_DIR / "assessment_2020_plus_v2.csv"  # V2 with improved neighbor features

# V2.5 tuned hyperparameters (use as starting point)
DEFAULT_PARAMS = {
    "n_estimators": 239,
    "max_depth": 7,
    "learning_rate": 0.0863,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
}


def load_fresh_data():
    """Load and prepare fresh assessment data."""
    print("Loading fresh 2020+ data...")
    
    # Load fresh data
    df = pd.read_csv(FRESH_DATA_PATH)
    print(f"  Records: {len(df):,}")
    
    # Load demographics
    demographics = pd.read_csv(DEMOGRAPHICS_PATH, dtype={"zipcode": str})
    print(f"  Zipcodes in demographics: {len(demographics)}")
    
    # Join with demographics
    df["zipcode"] = df["zipcode"].astype(str).str.strip()
    demographics["zipcode"] = demographics["zipcode"].astype(str).str.strip()
    df = df.merge(demographics, on="zipcode", how="left")
    
    # Define feature columns (same as V2.5)
    feature_cols = [
        "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
        "waterfront", "view", "condition", "grade", "sqft_above",
        "sqft_basement", "yr_built", "yr_renovated",
        "lat", "long", "sqft_living15", "sqft_lot15",
    ]
    
    # Add demographic columns
    demo_cols = [col for col in demographics.columns if col != "zipcode"]
    feature_cols.extend(demo_cols)
    
    # Handle missing values
    for col in feature_cols:
        if col not in df.columns:
            if col == "lat":
                df[col] = 47.5  # King County center
            elif col == "long":
                df[col] = -122.3
            elif col == "sqft_living15":
                df[col] = df["sqft_living"]
            elif col == "sqft_lot15":
                df[col] = df["sqft_lot"]
            else:
                df[col] = 0
        else:
            if df[col].dtype == "object":
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(0)
    
    X = df[feature_cols]
    y = df["price"]
    
    print(f"  Features: {len(feature_cols)}")
    print(f"  Target median: ${y.median():,.0f}")
    
    return X, y, feature_cols


def train_model(X, y, params=None):
    """Train XGBoost model with cross-validation."""
    if params is None:
        params = DEFAULT_PARAMS.copy()
    
    print("\nTraining XGBoost model...")
    print(f"  Parameters: n_estimators={params['n_estimators']}, "
          f"max_depth={params['max_depth']}, lr={params['learning_rate']}")
    
    # Create pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", XGBRegressor(**params))
    ])
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train
    pipeline.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = pipeline.predict(X_test)
    
    test_metrics = {
        "test_mae": mean_absolute_error(y_test, y_pred),
        "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "test_r2": r2_score(y_test, y_pred),
        "test_mape": np.mean(np.abs((y_test - y_pred) / y_test)) * 100,
    }
    
    print(f"\n  Test Set Performance:")
    print(f"    MAE:  ${test_metrics['test_mae']:,.0f}")
    print(f"    RMSE: ${test_metrics['test_rmse']:,.0f}")
    print(f"    R2:   {test_metrics['test_r2']:.4f}")
    print(f"    MAPE: {test_metrics['test_mape']:.1f}%")
    
    # Cross-validation
    print("\n  Running 5-fold cross-validation...")
    cv_scores = cross_val_score(
        pipeline, X, y, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1
    )
    cv_mae = -cv_scores.mean()
    cv_mae_std = cv_scores.std()
    
    cv_metrics = {
        "cv_mae": cv_mae,
        "cv_mae_std": cv_mae_std,
    }
    
    print(f"    CV MAE: ${cv_mae:,.0f} (+/- ${cv_mae_std:,.0f})")
    
    return pipeline, {**test_metrics, **cv_metrics}


def train_with_mlflow(run_name=None):
    """Train model with MLflow tracking."""
    
    # Setup MLflow
    setup_mlflow()
    
    if run_name is None:
        run_name = f"v3.2_fresh_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print("\n" + "=" * 60)
    print(f" MLflow Training Run: {run_name}")
    print("=" * 60)
    
    # Load data
    X, y, feature_names = load_fresh_data()
    
    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_param("data_source", "King County Assessment 2020+")
        mlflow.log_param("n_samples", len(X))
        mlflow.log_param("n_features", len(feature_names))
        for param, value in DEFAULT_PARAMS.items():
            mlflow.log_param(param, value)
        
        # Train model
        pipeline, metrics = train_model(X, y, DEFAULT_PARAMS)
        
        # Log metrics
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)
        
        # Log model
        mlflow.sklearn.log_model(pipeline, "model")
        
        # Log feature info
        feature_info = {
            "feature_names": feature_names,
            "n_features": len(feature_names),
            "data_source": "assessment_2020_plus_full.csv",
        }
        mlflow.log_dict(feature_info, "feature_info.json")
        
        # Log evaluation report
        eval_report = {
            "model_version": "V3.2",
            "data_source": "King County Assessment 2020+",
            "n_samples": len(X),
            "metrics": metrics,
            "comparison_to_v2.5": {
                "v2.5_cv_mae": 63529,
                "v3.2_cv_mae": metrics["cv_mae"],
                "improvement_pct": (63529 - metrics["cv_mae"]) / 63529 * 100,
            }
        }
        mlflow.log_dict(eval_report, "evaluation_report.json")
        
        run_id = mlflow.active_run().info.run_id
        
    print(f"\n  MLflow Run ID: {run_id}")
    
    return pipeline, metrics, run_id


def save_model(pipeline, output_path=None):
    """Save model to disk."""
    if output_path is None:
        output_path = MODEL_DIR / "model.pkl"
    
    MODEL_DIR.mkdir(exist_ok=True)
    
    with open(output_path, "wb") as f:
        pickle.dump(pipeline, f)
    
    print(f"\nModel saved to: {output_path}")
    print(f"Model size: {output_path.stat().st_size / 1024:.1f} KB")


def main():
    """Main training function."""
    print("=" * 80)
    print("V3.2 FRESH DATA TRAINING")
    print("=" * 80)
    
    # Train with MLflow tracking
    pipeline, metrics, run_id = train_with_mlflow()
    
    # Save model
    save_model(pipeline)
    
    # Summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nV3.2 Model Performance:")
    print(f"  CV MAE:  ${metrics['cv_mae']:,.0f}")
    print(f"  Test R2: {metrics['test_r2']:.4f}")
    print(f"  MAPE:    {metrics['test_mape']:.1f}%")
    
    print(f"\nComparison to V2.5:")
    print(f"  V2.5 CV MAE: $63,529")
    print(f"  V3.2 CV MAE: ${metrics['cv_mae']:,.0f}")
    
    # Note: V3.2 MAE will be higher because prices are higher
    # but MAPE should be similar or better
    
    print(f"\nTo view in MLflow UI:")
    print(f"  mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db")
    
    return pipeline, metrics


if __name__ == "__main__":
    main()
