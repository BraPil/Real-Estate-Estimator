"""
V1 MVP - Real Estate Price Predictor Training Script

This is the V1 baseline that uses only 7 home features + demographics.
Matches the original create_model.py specification with bug fix.

Features Used (7 home + 26 demographic = 33 total):
- bedrooms, bathrooms, sqft_living, sqft_lot, floors, sqft_above, sqft_basement
- Plus all 26 demographic features from zipcode lookup
"""

import json
import pathlib
import pickle
from typing import List, Tuple

import pandas as pd
from sklearn import model_selection, neighbors, pipeline, preprocessing

# Configuration
SALES_PATH = "data/kc_house_data.csv"
DEMOGRAPHICS_PATH = "data/zipcode_demographics.csv"  # Bug fix from original

# V1: Only 7 home features (matches original create_model.py)
SALES_COLUMN_SELECTION = [
    'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'sqft_above', 'sqft_basement', 'zipcode'
]

OUTPUT_DIR = "model"
RANDOM_STATE = 42


def load_data(
    sales_path: str,
    demographics_path: str,
    sales_column_selection: List[str]
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load and merge sales data with demographics."""
    sales_data = pd.read_csv(sales_path, usecols=sales_column_selection)
    demographics_data = pd.read_csv(demographics_path)
    
    merged_data = sales_data.merge(demographics_data, on="zipcode")
    merged_data = merged_data.drop(columns=["zipcode"])
    
    X = merged_data.drop(columns=["price"])
    y = merged_data["price"]
    
    return X, y


def main():
    print("=" * 60)
    print("V1 MVP - Real Estate Price Predictor")
    print("=" * 60)
    print()
    
    # Load data
    print("Loading data...")
    X, y = load_data(SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION)
    print(f"  Samples: {len(X)}")
    print(f"  Features: {len(X.columns)} (7 home + {len(X.columns) - 7} demographic)")
    
    # Split data
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    
    # Create and train model (matches original)
    print("\nTraining KNeighborsRegressor...")
    model = pipeline.Pipeline([
        ("scaler", preprocessing.RobustScaler()),
        ("regressor", neighbors.KNeighborsRegressor(n_neighbors=5))
    ])
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"\nResults:")
    print(f"  Train R²: {train_score:.4f}")
    print(f"  Test R²:  {test_score:.4f}")
    print(f"  Overfitting Gap: {train_score - test_score:.4f}")
    
    # Save artifacts
    output_path = pathlib.Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)
    
    with open(output_path / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    with open(output_path / "model_features.json", "w") as f:
        json.dump(list(X.columns), f, indent=2)
    
    with open(output_path / "metrics.json", "w") as f:
        json.dump({
            "model_type": "KNeighborsRegressor",
            "model_version": "1.0.0",
            "n_neighbors": 5,
            "train_r2": train_score,
            "test_r2": test_score,
            "n_features": len(X.columns),
            "home_features": 7,
            "demographic_features": len(X.columns) - 7
        }, f, indent=2)
    
    print(f"\nModel saved to {OUTPUT_DIR}/")
    print("\n[SUCCESS] V1 Training complete!")


if __name__ == "__main__":
    main()
