#!/usr/bin/env python3
"""
V3.3: Optuna Hyperparameter Tuning for Fresh Data Model

Uses Optuna for efficient Bayesian optimization of XGBoost parameters
on the 2020+ assessment data.

Usage:
    python src/tune_v33.py
    python src/tune_v33.py --n-trials 50
    python src/tune_v33.py --n-trials 100 --timeout 600
"""

import argparse
import json
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# Optional Optuna import
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("WARNING: Optuna not installed. Run: pip install optuna")

# Paths
DATA_DIR = Path("data")
MODEL_DIR = Path("model")
DEMOGRAPHICS_PATH = DATA_DIR / "zipcode_demographics.csv"
FRESH_DATA_PATH = DATA_DIR / "assessment_2020_plus_v4.csv"

RANDOM_STATE = 42


def load_data():
    """Load and prepare fresh assessment data with demographics."""
    print("Loading data...")
    
    # Load fresh data
    df = pd.read_csv(FRESH_DATA_PATH)
    print(f"  Records: {len(df):,}")
    
    # Load demographics
    demographics = pd.read_csv(DEMOGRAPHICS_PATH, dtype={"zipcode": str})
    
    # Join with demographics
    df["zipcode"] = df["zipcode"].astype(str).str.strip()
    demographics["zipcode"] = demographics["zipcode"].astype(str).str.strip()
    df = df.merge(demographics, on="zipcode", how="left")
    
    # Define feature columns
    feature_cols = [
        "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
        "waterfront", "view", "condition", "grade", "sqft_above",
        "sqft_basement", "yr_built", "yr_renovated",
        "lat", "long", "sqft_living15", "sqft_lot15",
    ]
    
    # Add temporal features
    temporal_cols = ["sale_year", "sale_month", "sale_quarter", "sale_dow"]
    for col in temporal_cols:
        if col in df.columns:
            feature_cols.append(col)
    
    # Add demographic columns
    demo_cols = [col for col in demographics.columns if col != "zipcode"]
    feature_cols.extend(demo_cols)
    
    # Handle missing values
    for col in feature_cols:
        if col not in df.columns:
            print(f"  WARNING: Missing column {col}, using 0")
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


def create_objective(X_train, y_train):
    """Create Optuna objective function."""
    
    def objective(trial):
        # Sample hyperparameters
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.95),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 0.5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 2.0),
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        }
        
        # Create pipeline
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", XGBRegressor(**params))
        ])
        
        # Cross-validation
        scores = cross_val_score(
            pipeline, X_train, y_train,
            cv=5, scoring="neg_mean_absolute_error", n_jobs=-1
        )
        
        return -scores.mean()  # Minimize MAE
    
    return objective


def main():
    parser = argparse.ArgumentParser(description="V3.3 Optuna Hyperparameter Tuning")
    parser.add_argument("--n-trials", type=int, default=30, help="Number of trials")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout in seconds")
    args = parser.parse_args()
    
    if not OPTUNA_AVAILABLE:
        print("ERROR: Optuna not available. Install with: pip install optuna")
        return
    
    print("\n" + "=" * 70)
    print(" V3.3: OPTUNA HYPERPARAMETER TUNING")
    print(f" Trials: {args.n_trials}, Timeout: {args.timeout or 'None'}")
    print("=" * 70 + "\n")
    
    # Load data
    X, y, feature_cols = load_data()
    
    # Split for final evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"\nTrain: {len(X_train):,}, Test: {len(X_test):,}")
    
    # Create study
    sampler = TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name="v33_xgboost_tuning"
    )
    
    # Optimize
    print(f"\nStarting optimization ({args.n_trials} trials)...")
    objective = create_objective(X_train, y_train)
    
    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=True,
        n_jobs=1,  # Sequential trials (parallel CV inside)
    )
    
    # Best parameters
    print("\n" + "=" * 70)
    print(" OPTIMIZATION RESULTS")
    print("=" * 70)
    
    best_params = study.best_params
    best_mae = study.best_value
    
    print(f"\nBest CV MAE: ${best_mae:,.0f}")
    print("\nBest Parameters:")
    for k, v in sorted(best_params.items()):
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    # Train final model with best params
    print("\n" + "-" * 70)
    print(" TRAINING FINAL MODEL WITH BEST PARAMETERS")
    print("-" * 70)
    
    final_params = {**best_params, "random_state": RANDOM_STATE, "n_jobs": -1}
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", XGBRegressor(**final_params))
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = pipeline.predict(X_test)
    
    test_mae = mean_absolute_error(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_r2 = r2_score(y_test, y_pred)
    test_mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    print(f"\nTest Set Performance:")
    print(f"  MAE:  ${test_mae:,.0f}")
    print(f"  RMSE: ${test_rmse:,.0f}")
    print(f"  R2:   {test_r2:.4f}")
    print(f"  MAPE: {test_mape:.1f}%")
    
    # Compare to current V3.3 baseline
    print("\n" + "-" * 70)
    print(" COMPARISON TO V3.3 BASELINE")
    print("-" * 70)
    print(f"  V3.3 Baseline CV MAE: $121,928")
    print(f"  Tuned CV MAE:         ${best_mae:,.0f}")
    improvement = (121928 - best_mae) / 121928 * 100
    print(f"  Improvement:          {improvement:+.1f}%")
    
    # Save if better
    if best_mae < 121928:
        print("\n  [IMPROVED] Saving new model...")
        
        # Retrain on full data
        full_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", XGBRegressor(**final_params))
        ])
        full_pipeline.fit(X, y)
        
        # Save model
        model_path = MODEL_DIR / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(full_pipeline, f)
        print(f"  Model saved: {model_path}")
        
        # Save features
        features_path = MODEL_DIR / "model_features.json"
        with open(features_path, "w") as f:
            json.dump(feature_cols, f, indent=2)
        print(f"  Features saved: {features_path}")
        
        # Save best params
        params_path = MODEL_DIR / "best_params.json"
        with open(params_path, "w") as f:
            json.dump({
                "params": best_params,
                "cv_mae": best_mae,
                "test_mae": test_mae,
                "test_r2": test_r2,
                "test_mape": test_mape,
                "n_trials": args.n_trials,
                "timestamp": datetime.now().isoformat(),
            }, f, indent=2)
        print(f"  Parameters saved: {params_path}")
    else:
        print("\n  [NO IMPROVEMENT] Keeping existing model.")
    
    print("\n" + "=" * 70)
    print(" TUNING COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
