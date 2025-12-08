"""
Evaluation script for the Real Estate Price Predictor.

This script evaluates a trained model against test data and logs results to MLflow.
It can be run standalone after training to re-evaluate or compare models.

Features:
- Comprehensive metrics (R2, MAE, RMSE)
- Residual analysis
- Prediction range validation
- MLflow logging
- JSON report generation

Usage:
    python src/evaluate.py
    python src/evaluate.py --model-path model/model.pkl
    python src/evaluate.py --output evaluation_report.json
"""

import argparse
import json
import pathlib
import pickle
import sys
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Try to import MLflow
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# ==============================================================================
# CONFIGURATION
# ==============================================================================

SALES_PATH = "data/kc_house_data.csv"
DEMOGRAPHICS_PATH = "data/zipcode_demographics.csv"
MODEL_PATH = "model/model.pkl"
FEATURES_PATH = "model/model_features.json"
DEFAULT_OUTPUT = "model/evaluation_report.json"

# V2.1+ uses all 17 home features (expanded from 7 in V1)
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

RANDOM_STATE = 42

# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_test_data(test_size: float = 0.20) -> Tuple[pd.DataFrame, pd.Series]:
    """Load and prepare test data using same split as training.

    Args:
        test_size: Fraction used for test set (must match training - V2.3 uses 0.20)

    Returns:
        Tuple of (X_test, y_test)
    """
    from sklearn.model_selection import train_test_split
    
    # Load sales data (must match tune.py exactly)
    sales = pd.read_csv(SALES_PATH, usecols=SALES_COLUMN_SELECTION)
    
    # Load demographics
    demographics = pd.read_csv(DEMOGRAPHICS_PATH)
    
    # Merge and prepare (inner join to match tune.py)
    merged = sales.merge(demographics, on='zipcode', how='inner').drop(columns='zipcode')
    y = merged.pop('price')
    X = merged
    
    # Use same split as training (same random_state ensures same split)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE
    )
    
    return X_test, y_test


def load_model(model_path: str):
    """Load trained model from pickle file."""
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def load_feature_names(features_path: str) -> List[str]:
    """Load expected feature names from JSON."""
    with open(features_path) as f:
        return json.load(f)


# ==============================================================================
# EVALUATION FUNCTIONS
# ==============================================================================

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Calculate comprehensive evaluation metrics.

    Args:
        y_true: Actual target values
        y_pred: Predicted values

    Returns:
        Dictionary of metrics
    """
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Mean Absolute Percentage Error (handle zeros)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    return {
        "r2_score": round(r2, 4),
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape_percent": round(mape, 2),
    }


def analyze_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Analyze prediction residuals.

    Args:
        y_true: Actual target values
        y_pred: Predicted values

    Returns:
        Dictionary of residual statistics
    """
    residuals = y_true - y_pred
    
    return {
        "mean_residual": round(float(np.mean(residuals)), 2),
        "std_residual": round(float(np.std(residuals)), 2),
        "median_residual": round(float(np.median(residuals)), 2),
        "min_residual": round(float(np.min(residuals)), 2),
        "max_residual": round(float(np.max(residuals)), 2),
        "skewness": round(float(pd.Series(residuals).skew()), 4),
    }


def analyze_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Analyze prediction distribution.

    Args:
        y_true: Actual target values
        y_pred: Predicted values

    Returns:
        Dictionary of prediction analysis
    """
    return {
        "actual_min": round(float(y_true.min()), 2),
        "actual_max": round(float(y_true.max()), 2),
        "actual_mean": round(float(y_true.mean()), 2),
        "predicted_min": round(float(y_pred.min()), 2),
        "predicted_max": round(float(y_pred.max()), 2),
        "predicted_mean": round(float(y_pred.mean()), 2),
        "correlation": round(float(np.corrcoef(y_true, y_pred)[0, 1]), 4),
    }


def analyze_error_by_price_range(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict:
    """Analyze errors by price range.

    Args:
        y_true: Actual target values
        y_pred: Predicted values

    Returns:
        Dictionary of error analysis by price range
    """
    # Define price buckets
    buckets = [
        (0, 300000, "under_300k"),
        (300000, 500000, "300k_to_500k"),
        (500000, 750000, "500k_to_750k"),
        (750000, 1000000, "750k_to_1m"),
        (1000000, float('inf'), "over_1m"),
    ]
    
    results = {}
    for low, high, label in buckets:
        mask = (y_true >= low) & (y_true < high)
        if mask.sum() > 0:
            bucket_mae = mean_absolute_error(y_true[mask], y_pred[mask])
            bucket_rmse = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
            results[label] = {
                "count": int(mask.sum()),
                "mae": round(bucket_mae, 2),
                "rmse": round(bucket_rmse, 2),
            }
    
    return results


# ==============================================================================
# MAIN EVALUATION
# ==============================================================================

def evaluate_model(
    model_path: str = MODEL_PATH,
    features_path: str = FEATURES_PATH,
    test_size: float = 0.25,
    output_path: str = None,
    experiment_name: str = "real-estate-v1"
) -> Dict:
    """Run comprehensive model evaluation.

    Args:
        model_path: Path to trained model pickle
        features_path: Path to feature names JSON
        test_size: Test set fraction (must match training)
        output_path: Path to save evaluation report
        experiment_name: MLflow experiment name for logging

    Returns:
        Complete evaluation report dictionary
    """
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    # Load model and features
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    features = load_feature_names(features_path)
    print(f"Model loaded with {len(features)} features")
    
    # Load test data
    print(f"Loading test data (test_size={test_size})...")
    X_test, y_test = load_test_data(test_size)
    print(f"Test set: {len(X_test)} samples")
    
    # Ensure feature order matches
    X_test = X_test[features]
    
    # Generate predictions
    print("Generating predictions...")
    y_pred = model.predict(X_test)
    y_true = y_test.values
    
    # Calculate all metrics
    print("Calculating metrics...")
    metrics = calculate_metrics(y_true, y_pred)
    residuals = analyze_residuals(y_true, y_pred)
    predictions = analyze_predictions(y_true, y_pred)
    error_by_range = analyze_error_by_price_range(y_true, y_pred)
    
    # Build comprehensive report
    report = {
        "evaluation_timestamp": datetime.now().isoformat(),
        "model_path": model_path,
        "test_samples": len(X_test),
        "n_features": len(features),
        "metrics": metrics,
        "residual_analysis": residuals,
        "prediction_analysis": predictions,
        "error_by_price_range": error_by_range,
        "data_info": {
            "source": "King County (Seattle), WA",
            "vintage": "2014-2015",
            "test_size": test_size,
        },
        "model_info": {
            "type": "KNeighborsRegressor",
            "scaler": "RobustScaler",
        }
    }
    
    # Print results
    print("\n" + "-" * 60)
    print("PERFORMANCE METRICS")
    print("-" * 60)
    print(f"R2 Score:  {metrics['r2_score']:.4f}")
    print(f"MAE:       ${metrics['mae']:,.2f}")
    print(f"RMSE:      ${metrics['rmse']:,.2f}")
    print(f"MAPE:      {metrics['mape_percent']:.2f}%")
    
    print("\n" + "-" * 60)
    print("RESIDUAL ANALYSIS")
    print("-" * 60)
    print(f"Mean Residual:   ${residuals['mean_residual']:,.2f}")
    print(f"Std Residual:    ${residuals['std_residual']:,.2f}")
    print(f"Median Residual: ${residuals['median_residual']:,.2f}")
    print(f"Skewness:        {residuals['skewness']:.4f}")
    
    print("\n" + "-" * 60)
    print("PREDICTION RANGE")
    print("-" * 60)
    print(f"Actual Range:    ${predictions['actual_min']:,.0f} - ${predictions['actual_max']:,.0f}")
    print(f"Predicted Range: ${predictions['predicted_min']:,.0f} - ${predictions['predicted_max']:,.0f}")
    print(f"Correlation:     {predictions['correlation']:.4f}")
    
    print("\n" + "-" * 60)
    print("ERROR BY PRICE RANGE")
    print("-" * 60)
    for range_name, stats in error_by_range.items():
        print(f"{range_name:15s}: n={stats['count']:4d}, MAE=${stats['mae']:,.0f}, RMSE=${stats['rmse']:,.0f}")
    
    print("=" * 60)
    
    # Save report
    if output_path:
        output_file = pathlib.Path(output_path)
        output_file.parent.mkdir(exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nEvaluation report saved to: {output_path}")
    
    # Log to MLflow
    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_experiment(experiment_name)
            with mlflow.start_run(run_name=f"evaluation-{datetime.now().strftime('%Y%m%d-%H%M%S')}"):
                mlflow.log_metrics(metrics)
                mlflow.log_metrics({f"residual_{k}": v for k, v in residuals.items()})
                if output_path:
                    mlflow.log_artifact(output_path)
            print("Metrics logged to MLflow")
        except Exception as e:
            print(f"Warning: Could not log to MLflow: {e}")
    
    return report


# ==============================================================================
# CLI ENTRY POINT
# ==============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate the Real Estate Price Predictor model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model-path", "-m",
        type=str,
        default=MODEL_PATH,
        help="Path to trained model pickle file"
    )
    parser.add_argument(
        "--features-path", "-f",
        type=str,
        default=FEATURES_PATH,
        help="Path to feature names JSON file"
    )
    parser.add_argument(
        "--test-size", "-t",
        type=float,
        default=0.25,
        help="Test set fraction (must match training split)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=DEFAULT_OUTPUT,
        help="Output path for evaluation report JSON"
    )
    parser.add_argument(
        "--experiment-name", "-e",
        type=str,
        default="real-estate-v1",
        help="MLflow experiment name for logging"
    )
    return parser.parse_args()


def main():
    """Main entry point for evaluation."""
    args = parse_args()
    
    print("\n" + "=" * 60)
    print("REAL ESTATE PRICE PREDICTOR - MODEL EVALUATION")
    print("=" * 60)
    print(f"MLflow tracking: {'Enabled' if MLFLOW_AVAILABLE else 'Disabled'}")
    
    try:
        report = evaluate_model(
            model_path=args.model_path,
            features_path=args.features_path,
            test_size=args.test_size,
            output_path=args.output,
            experiment_name=args.experiment_name
        )
        
        # Provide interpretation
        r2 = report['metrics']['r2_score']
        mae = report['metrics']['mae']
        
        print("\n" + "=" * 60)
        print("INTERPRETATION")
        print("=" * 60)
        
        if r2 >= 0.8:
            print(f"[EXCELLENT] Model explains {r2*100:.1f}% of price variance")
        elif r2 >= 0.6:
            print(f"[GOOD] Model explains {r2*100:.1f}% of price variance")
        elif r2 >= 0.4:
            print(f"[FAIR] Model explains {r2*100:.1f}% of price variance")
        else:
            print(f"[POOR] Model explains only {r2*100:.1f}% of price variance")
        
        print(f"        Average prediction error: ${mae:,.0f}")
        
        # Check for bias
        mean_residual = report['residual_analysis']['mean_residual']
        if abs(mean_residual) > 10000:
            print(f"[WARNING] Bias detected: Mean residual is ${mean_residual:,.0f}")
        else:
            print(f"[OK] No significant bias (mean residual ${mean_residual:,.0f})")
        
        print("\n[SUCCESS] Evaluation completed successfully!")
        return 0
        
    except FileNotFoundError as e:
        print(f"\n[ERROR] File not found: {e}")
        print("        Make sure to run training first (python src/train.py)")
        return 1
        
    except Exception as e:
        print(f"\n[ERROR] Evaluation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
