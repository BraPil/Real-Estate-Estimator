"""
V2.4.1: XGBoost Hyperparameter Tuning (Quick)

Focused tuning for XGBoost only - the V2.4 winner.
Uses constrained parameter ranges for faster execution.

Usage:
    python src/tune_xgboost.py
    python src/tune_xgboost.py --n-iter 50
"""

import argparse
import json
import logging
import pickle
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path("data")
MODEL_DIR = Path("model")
LOGS_DIR = Path("logs")

# Feature selection
SALES_COLUMNS = [
    "price",
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
    "zipcode",
]

RANDOM_STATE = 42


def load_data():
    """Load and prepare data."""
    sales = pd.read_csv(DATA_DIR / "kc_house_data.csv", usecols=SALES_COLUMNS)
    demographics = pd.read_csv(DATA_DIR / "zipcode_demographics.csv")
    merged = sales.merge(demographics, on="zipcode", how="inner")

    y = merged["price"]
    X = merged.drop(columns=["price", "zipcode"])
    return X, y, list(X.columns)


def get_xgboost_params():
    """Focused XGBoost parameter space - constrained for speed."""
    return {
        "model__n_estimators": randint(100, 250),  # Focused range
        "model__max_depth": randint(4, 8),  # Constrained (was 3-10)
        "model__learning_rate": uniform(0.05, 0.15),  # 0.05-0.20 (focused)
        "model__subsample": uniform(0.7, 0.25),  # 0.7-0.95
        "model__colsample_bytree": uniform(0.7, 0.25),  # 0.7-0.95
        "model__min_child_weight": randint(1, 7),  # Constrained
        "model__gamma": uniform(0, 0.3),  # Constrained
        "model__reg_alpha": uniform(0, 0.5),  # L1
        "model__reg_lambda": uniform(0.5, 1.0),  # L2
    }


def main():
    parser = argparse.ArgumentParser(description="Quick XGBoost tuning")
    parser.add_argument("--n-iter", type=int, default=30, help="Iterations (default: 30)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print(" V2.4.1: XGBOOST HYPERPARAMETER TUNING")
    print(f" RandomizedSearchCV ({args.n_iter} iterations)")
    print("=" * 60 + "\n")

    # Load data
    logger.info("Loading data...")
    X, y, feature_names = load_data()

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}, Features: {len(feature_names)}")

    # Create pipeline
    pipeline = Pipeline(
        [
            ("scaler", RobustScaler()),
            ("model", XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)),
        ]
    )

    # RandomizedSearchCV
    logger.info("Running RandomizedSearchCV...")
    search = RandomizedSearchCV(
        pipeline,
        get_xgboost_params(),
        n_iter=args.n_iter,
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
    )

    start = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - start

    # Evaluate
    y_pred = search.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    cv_mae = -search.best_score_

    # Best params
    best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}

    # Print results
    print("\n" + "=" * 60)
    print(" RESULTS")
    print("=" * 60)
    print("\n🏆 TUNED XGBOOST PERFORMANCE:")
    print(f"   CV MAE:    ${cv_mae:,.0f}")
    print(f"   Test MAE:  ${test_mae:,.0f}")
    print(f"   Test R²:   {test_r2:.4f}")
    print(f"   Test RMSE: ${test_rmse:,.0f}")
    print(f"   Time:      {elapsed:.1f}s")

    # Comparison
    knn_mae = 84494
    improvement = knn_mae - test_mae
    improvement_pct = (improvement / knn_mae) * 100
    print(f"\n📊 vs KNN (V2.3): ${improvement:,.0f} better ({improvement_pct:.1f}%)")

    print("\n🔧 Best Hyperparameters:")
    for k, v in best_params.items():
        print(f"   {k}: {v:.4f}" if isinstance(v, float) else f"   {k}: {v}")

    # Save model
    MODEL_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)

    model_path = MODEL_DIR / "model_v2.4.1_xgboost_tuned.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(search.best_estimator_, f)
    logger.info(f"Saved model to {model_path}")

    # Save metrics
    metrics = {
        "version": "v2.4.1",
        "timestamp": datetime.now().isoformat(),
        "model_type": "XGBRegressor (tuned)",
        "hyperparameters": best_params,
        "test_metrics": {
            "r2_score": round(test_r2, 4),
            "mae": round(test_mae, 2),
            "rmse": round(test_rmse, 2),
        },
        "cv_mae": round(cv_mae, 2),
        "data": {
            "total_samples": len(X),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "n_features": len(feature_names),
        },
        "comparison": {
            "knn_v2.3_mae": knn_mae,
            "improvement_dollars": round(improvement, 2),
            "improvement_pct": round(improvement_pct, 2),
        },
    }

    metrics_path = MODEL_DIR / "metrics_v2.4.1.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")

    # Feature importance
    model = search.best_estimator_.named_steps["model"]
    importance = pd.DataFrame(
        {"feature": feature_names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    print("\n📈 Top 10 Features:")
    for i, (_, row) in enumerate(importance.head(10).iterrows(), 1):
        bar = "█" * int(row["importance"] * 40 / importance["importance"].max())
        print(f"   {i:2}. {row['feature']:<22} {row['importance']:.4f} {bar}")

    importance.to_csv(LOGS_DIR / "v2.4.1_xgboost_feature_importance.csv", index=False)

    print("\n" + "=" * 60)
    print(" NEXT STEPS")
    print("=" * 60)
    print("\n1. Copy model to production:")
    print("   copy model\\model_v2.4.1_xgboost_tuned.pkl model\\model.pkl")
    print("\n2. Update metrics.json")
    print("\n3. Test API: python scripts/test_api.py")
    print("=" * 60 + "\n")

    return search.best_estimator_, metrics


if __name__ == "__main__":
    main()
