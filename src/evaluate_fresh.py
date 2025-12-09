#!/usr/bin/env python3
"""
Evaluation script for V3.2 Fresh Data Model.

This script evaluates the model trained on 2020+ King County assessment data.
It uses the same metrics as evaluate.py but with the fresh data source.

Usage:
    python src/evaluate_fresh.py
    python src/evaluate_fresh.py --data data/assessment_2020_plus_v3.csv
"""

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# MLflow
try:
    import mlflow
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from mlflow_config import setup_mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# ==============================================================================
# CONFIGURATION
# ==============================================================================

FRESH_DATA_PATH = Path("data/assessment_2020_plus_v3.csv")
DEMOGRAPHICS_PATH = Path("data/zipcode_demographics.csv")
MODEL_PATH = Path("model/model.pkl")
FEATURES_PATH = Path("model/model_features.json")
OUTPUT_PATH = Path("model/evaluation_fresh_report.json")

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Features from fresh data (17 base + demographics = 43)
BASE_FEATURES = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
    "waterfront", "view", "condition", "grade", "sqft_above",
    "sqft_basement", "yr_built", "yr_renovated", "lat", "long",
    "sqft_living15", "sqft_lot15"
]


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_fresh_data(data_path: Path = FRESH_DATA_PATH, test_size: float = TEST_SIZE):
    """Load fresh 2020+ data and prepare train/test split."""
    print(f"Loading fresh data from {data_path}...")
    
    df = pd.read_csv(data_path)
    print(f"  Total records: {len(df):,}")
    
    # Load demographics
    demographics = pd.read_csv(DEMOGRAPHICS_PATH)
    demo_cols = [c for c in demographics.columns if c != "zipcode"]
    
    # Merge with demographics
    df["zipcode"] = df["zipcode"].astype(str).str[:5]
    demographics["zipcode"] = demographics["zipcode"].astype(str)
    df = df.merge(demographics, on="zipcode", how="left")
    
    # Fill missing demographics with median
    for col in demo_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    # Build feature list
    feature_cols = BASE_FEATURES.copy()
    feature_cols.extend(demo_cols)
    
    # Handle missing values
    for col in feature_cols:
        if col not in df.columns:
            if col == "lat":
                df[col] = 47.5
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
    
    # Same split as training
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE
    )
    
    print(f"  Test set: {len(X_test):,} samples")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Median price: ${y_test.median():,.0f}")
    
    return X_test, y_test, feature_cols


def load_model(model_path: Path = MODEL_PATH):
    """Load trained model."""
    print(f"Loading model from {model_path}...")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


# ==============================================================================
# EVALUATION FUNCTIONS
# ==============================================================================

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate comprehensive evaluation metrics."""
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # MAPE
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    return {
        "r2_score": round(r2, 4),
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape_percent": round(mape, 2),
    }


def analyze_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Analyze prediction residuals."""
    residuals = y_true - y_pred
    
    return {
        "mean_residual": round(float(np.mean(residuals)), 2),
        "std_residual": round(float(np.std(residuals)), 2),
        "median_residual": round(float(np.median(residuals)), 2),
        "skewness": round(float(pd.Series(residuals).skew()), 4),
        "bias": "underpredicting" if np.mean(residuals) > 0 else "overpredicting",
    }


def analyze_by_price_tier(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Analyze errors by price tier (2020+ appropriate ranges)."""
    tiers = {
        "under_500k": (0, 500000),
        "500k_to_750k": (500000, 750000),
        "750k_to_1m": (750000, 1000000),
        "1m_to_1.5m": (1000000, 1500000),
        "over_1.5m": (1500000, float("inf")),
    }
    
    results = {}
    for tier_name, (low, high) in tiers.items():
        mask = (y_true >= low) & (y_true < high)
        if mask.sum() > 0:
            tier_true = y_true[mask]
            tier_pred = y_pred[mask]
            mae = mean_absolute_error(tier_true, tier_pred)
            mape = np.mean(np.abs((tier_true - tier_pred) / tier_true)) * 100
            results[tier_name] = {
                "n_samples": int(mask.sum()),
                "mae": round(mae, 2),
                "mape": round(mape, 2),
            }
    
    return results


# ==============================================================================
# MAIN EVALUATION
# ==============================================================================

def evaluate(data_path: Path = None, log_to_mlflow: bool = True):
    """Run full evaluation."""
    print("=" * 70)
    print("V3.2 FRESH DATA MODEL EVALUATION")
    print("=" * 70)
    
    if data_path is None:
        data_path = FRESH_DATA_PATH
    
    # Load data and model
    X_test, y_test, feature_names = load_fresh_data(data_path)
    model = load_model()
    
    # Generate predictions
    print("\nGenerating predictions...")
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    print("Calculating metrics...")
    metrics = calculate_metrics(y_test.values, y_pred)
    residuals = analyze_residuals(y_test.values, y_pred)
    tier_analysis = analyze_by_price_tier(y_test.values, y_pred)
    
    # Print results
    print("\n" + "-" * 70)
    print("PERFORMANCE METRICS")
    print("-" * 70)
    print(f"R2 Score:  {metrics['r2_score']:.4f}")
    print(f"MAE:       ${metrics['mae']:,.2f}")
    print(f"RMSE:      ${metrics['rmse']:,.2f}")
    print(f"MAPE:      {metrics['mape_percent']:.2f}%")
    
    print("\n" + "-" * 70)
    print("RESIDUAL ANALYSIS")
    print("-" * 70)
    print(f"Mean Residual:   ${residuals['mean_residual']:,.2f}")
    print(f"Std Residual:    ${residuals['std_residual']:,.2f}")
    print(f"Median Residual: ${residuals['median_residual']:,.2f}")
    print(f"Bias:            {residuals['bias']}")
    
    print("\n" + "-" * 70)
    print("PERFORMANCE BY PRICE TIER")
    print("-" * 70)
    for tier, stats in tier_analysis.items():
        print(f"{tier:15s}: n={stats['n_samples']:5d}, MAE=${stats['mae']:,.0f}, MAPE={stats['mape']:.1f}%")
    
    # Build report
    report = {
        "model_version": "V3.2",
        "data_source": str(data_path),
        "evaluation_time": datetime.now().isoformat(),
        "test_samples": len(X_test),
        "metrics": metrics,
        "residuals": residuals,
        "tier_analysis": tier_analysis,
        "comparison_to_v2.5": {
            "note": "V2.5 trained on 2014-2015 data, V3.2 on 2020+ data",
            "v2.5_cv_mae": 63529,
            "v3.2_test_mae": metrics["mae"],
            "price_inflation": "2015 median ~$450K vs 2020+ median ~$845K (+88%)",
        }
    }
    
    # Save report
    with open(OUTPUT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {OUTPUT_PATH}")
    
    # Log to MLflow
    if log_to_mlflow and MLFLOW_AVAILABLE:
        print("\nLogging to MLflow...")
        setup_mlflow()
        
        with mlflow.start_run(run_name=f"v3.2_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_param("model_version", "V3.2")
            mlflow.log_param("data_source", "assessment_2020_plus_v3.csv")
            mlflow.log_param("test_samples", len(X_test))
            
            mlflow.log_metric("eval_r2", metrics["r2_score"])
            mlflow.log_metric("eval_mae", metrics["mae"])
            mlflow.log_metric("eval_rmse", metrics["rmse"])
            mlflow.log_metric("eval_mape", metrics["mape_percent"])
            mlflow.log_metric("eval_mean_residual", residuals["mean_residual"])
            
            mlflow.log_dict(report, "evaluation_fresh_report.json")
            
            run_id = mlflow.active_run().info.run_id
            print(f"  MLflow Run ID: {run_id}")
    
    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    if metrics["r2_score"] >= 0.65:
        print(f"[GOOD] Model explains {metrics['r2_score']*100:.1f}% of price variance")
    elif metrics["r2_score"] >= 0.5:
        print(f"[OK] Model explains {metrics['r2_score']*100:.1f}% of price variance")
    else:
        print(f"[POOR] Model explains only {metrics['r2_score']*100:.1f}% of price variance")
    
    if metrics["mape_percent"] <= 25:
        print(f"[GOOD] Average error is {metrics['mape_percent']:.1f}% of home value")
    elif metrics["mape_percent"] <= 35:
        print(f"[OK] Average error is {metrics['mape_percent']:.1f}% of home value")
    else:
        print(f"[HIGH] Average error is {metrics['mape_percent']:.1f}% of home value")
    
    print("\n[SUCCESS] Fresh data evaluation completed!")
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate V3.2 fresh data model")
    parser.add_argument("--data", type=Path, default=FRESH_DATA_PATH,
                        help="Path to fresh data CSV")
    parser.add_argument("--no-mlflow", action="store_true",
                        help="Skip MLflow logging")
    args = parser.parse_args()
    
    evaluate(data_path=args.data, log_to_mlflow=not args.no_mlflow)


if __name__ == "__main__":
    main()
