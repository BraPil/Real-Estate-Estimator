#!/usr/bin/env python3
"""
King County Assessment Data Transformation Pipeline
V3.2: Fresh Data Integration

This script transforms King County Assessment data into the format
expected by the V2.5 model (trained on 2014-2015 kc_house_data.csv).

Output format matches: data/kc_house_data.csv columns
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import warnings

warnings.filterwarnings("ignore")

# Paths
ASSESSMENT_DIR = Path("Reference_Docs/King_County_Assessment_data_ALL")
OUTPUT_DIR = Path("data")

# Feature mapping: Original → Assessment columns
FEATURE_MAPPING = {
    # Target
    "price": "SalePrice",
    # From EXTR_ResBldg
    "bedrooms": "Bedrooms",
    "sqft_living": "SqFtTotLiving",
    "sqft_basement": "SqFtTotBasement",
    "yr_built": "YrBuilt",
    "yr_renovated": "YrRenovated",
    "zipcode": "ZipCode",
    "floors": "Stories",
    "condition": "Condition",
    "grade": "BldgGrade",
    # From EXTR_Parcel
    "sqft_lot": "SqFtLot",
}


def load_assessment_data(
    min_year: int = 2020,
    min_price: int = 50000,
    max_price: int = 10000000,
    sample_size: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load and merge assessment data files.

    Args:
        min_year: Minimum sale year to include
        min_price: Minimum sale price (filters out $0 and nominal sales)
        max_price: Maximum sale price (filters out extreme outliers)
        sample_size: If set, return random sample of this size

    Returns:
        DataFrame with merged sales, building, and parcel data
    """
    print(f"Loading assessment data (sales from {min_year}+)...")

    # Load residential buildings
    print("  Loading EXTR_ResBldg.csv...")
    resbldg = pd.read_csv(
        ASSESSMENT_DIR / "EXTR_ResBldg.csv",
        encoding="latin1",
        low_memory=False,
        dtype={"Major": str, "Minor": str},
    )
    print(f"    Buildings: {len(resbldg):,}")

    # Load sales
    print("  Loading EXTR_RPSale.csv...")
    sales = pd.read_csv(
        ASSESSMENT_DIR / "EXTR_RPSale.csv",
        encoding="latin1",
        low_memory=False,
        dtype={"Major": str, "Minor": str},
    )
    sales["DocumentDate"] = pd.to_datetime(sales["DocumentDate"], errors="coerce")
    sales["Year"] = sales["DocumentDate"].dt.year

    # Filter sales
    sales = sales[
        (sales["Year"] >= min_year)
        & (sales["SalePrice"] >= min_price)
        & (sales["SalePrice"] <= max_price)
    ]
    print(f"    Sales ({min_year}+, ${min_price:,}-${max_price:,}): {len(sales):,}")

    # Load parcels
    print("  Loading EXTR_Parcel.csv...")
    parcel = pd.read_csv(
        ASSESSMENT_DIR / "EXTR_Parcel.csv",
        encoding="latin1",
        low_memory=False,
        usecols=["Major", "Minor", "SqFtLot", "WfntLocation", "PropType"],
        dtype={"Major": str, "Minor": str},
    )
    print(f"    Parcels: {len(parcel):,}")

    # Merge: Sales + Buildings
    print("  Merging sales with buildings...")
    merged = sales.merge(resbldg, on=["Major", "Minor"], how="inner")
    print(f"    After building join: {len(merged):,}")

    # Merge: + Parcels
    print("  Merging with parcels...")
    merged = merged.merge(parcel, on=["Major", "Minor"], how="inner")
    print(f"    After parcel join: {len(merged):,}")

    if sample_size and len(merged) > sample_size:
        merged = merged.sample(n=sample_size, random_state=42)
        print(f"    Random sample: {len(merged):,}")

    return merged


def calculate_bathrooms(df: pd.DataFrame) -> pd.Series:
    """
    Calculate total bathrooms from component counts.

    Original formula: Full + 0.75*3/4 + 0.5*Half
    """
    full = df["BathFullCount"].fillna(0)
    three_quarter = df["Bath3qtrCount"].fillna(0)
    half = df["BathHalfCount"].fillna(0)

    return full + (0.75 * three_quarter) + (0.5 * half)


def calculate_sqft_above(df: pd.DataFrame) -> pd.Series:
    """
    Calculate above-ground square footage.

    Sum of 1st floor, 2nd floor, and upper floors.
    """
    first = df["SqFt1stFloor"].fillna(0)
    second = df["SqFt2ndFloor"].fillna(0)
    upper = df["SqFtUpperFloor"].fillna(0)
    half = df["SqFtHalfFloor"].fillna(0)

    return first + second + upper + half


def calculate_waterfront(df: pd.DataFrame) -> pd.Series:
    """
    Derive waterfront flag from WfntLocation.

    WfntLocation > 0 indicates waterfront property.
    """
    return (df["WfntLocation"].fillna(0) > 0).astype(int)


def calculate_view(df: pd.DataFrame) -> pd.Series:
    """
    Derive view score from ViewUtilization or other view columns.

    ViewUtilization in assessment data is a letter code.
    We'll convert to 0-4 scale to match original data.
    """
    # ViewUtilization appears to be a category - map to numeric
    view_map = {"N": 0, "F": 1, "A": 2, "G": 3, "E": 4, "": 0}
    view = df["ViewUtilization"].fillna("N").map(view_map).fillna(0)
    return view.astype(int)


def transform_to_model_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform assessment data to model input format.

    Returns DataFrame matching kc_house_data.csv columns.
    """
    print("\nTransforming to model format...")

    result = pd.DataFrame()

    # Direct mappings
    result["price"] = df["SalePrice"]
    result["bedrooms"] = df["Bedrooms"].fillna(0).astype(int)
    result["sqft_living"] = df["SqFtTotLiving"].fillna(0).astype(int)
    result["sqft_lot"] = df["SqFtLot"].fillna(0).astype(int)
    result["floors"] = df["Stories"].fillna(1).astype(float)
    result["condition"] = df["Condition"].fillna(3).astype(int)
    result["grade"] = df["BldgGrade"].fillna(7).astype(int)
    result["sqft_basement"] = df["SqFtTotBasement"].fillna(0).astype(int)
    result["yr_built"] = df["YrBuilt"].fillna(1970).astype(int)
    result["yr_renovated"] = df["YrRenovated"].fillna(0).astype(int)
    result["zipcode"] = df["ZipCode"].fillna(0).astype(str)

    # Calculated fields
    result["bathrooms"] = calculate_bathrooms(df)
    result["sqft_above"] = calculate_sqft_above(df)
    result["waterfront"] = calculate_waterfront(df)
    result["view"] = calculate_view(df)

    # Fields we can't derive (set to reasonable defaults or null)
    result["lat"] = np.nan  # Would need geocoding
    result["long"] = np.nan  # Would need geocoding
    result["sqft_living15"] = result["sqft_living"]  # Default to self
    result["sqft_lot15"] = result["sqft_lot"]  # Default to self

    # Add metadata
    result["id"] = df["Major"].astype(str) + df["Minor"].astype(str)
    result["date"] = pd.to_datetime(df["DocumentDate"]).dt.strftime("%Y%m%dT000000")

    print(f"  Output records: {len(result):,}")
    print(f"  Output columns: {len(result.columns)}")

    return result


def validate_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Validate and clean transformed data.

    Returns cleaned DataFrame and validation report.
    """
    print("\nValidating data...")
    report = {}

    initial_count = len(df)
    report["initial_count"] = initial_count

    # Remove rows with missing critical fields
    critical_fields = ["price", "sqft_living", "bedrooms"]
    for field in critical_fields:
        missing = df[field].isna().sum()
        report[f"missing_{field}"] = missing

    # Remove invalid values
    df = df[df["sqft_living"] > 0]
    df = df[df["price"] > 0]
    df = df[df["bedrooms"] >= 0]
    df = df[df["bathrooms"] >= 0]

    # Remove extreme outliers
    df = df[df["bedrooms"] <= 20]
    df = df[df["bathrooms"] <= 15]
    df = df[df["sqft_living"] <= 20000]
    df = df[df["sqft_lot"] <= 1000000]

    report["final_count"] = len(df)
    report["removed"] = initial_count - len(df)
    report["removal_pct"] = (initial_count - len(df)) / initial_count * 100

    print(f"  Removed {report['removed']:,} records ({report['removal_pct']:.1f}%)")
    print(f"  Final count: {report['final_count']:,}")

    return df, report


def main(
    output_file: str = "assessment_2020_plus.csv",
    min_year: int = 2020,
    sample_size: Optional[int] = None,
):
    """
    Main pipeline: Load, transform, validate, and save.
    """
    print("=" * 80)
    print("KING COUNTY ASSESSMENT DATA TRANSFORMATION PIPELINE")
    print("=" * 80)

    # Load
    raw_df = load_assessment_data(min_year=min_year, sample_size=sample_size)

    # Transform
    transformed_df = transform_to_model_format(raw_df)

    # Validate
    clean_df, report = validate_data(transformed_df)

    # Save
    output_path = OUTPUT_DIR / output_file
    clean_df.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Input: {report['initial_count']:,} records from {min_year}+")
    print(f"Output: {report['final_count']:,} records")
    print(f"Median price: ${clean_df['price'].median():,.0f}")
    print(f"Mean sqft: {clean_df['sqft_living'].mean():,.0f}")
    print(f"Mean bedrooms: {clean_df['bedrooms'].mean():.1f}")
    print(f"Mean bathrooms: {clean_df['bathrooms'].mean():.1f}")

    return clean_df, report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Transform assessment data")
    parser.add_argument(
        "--output", default="assessment_2020_plus.csv", help="Output filename"
    )
    parser.add_argument("--min-year", type=int, default=2020, help="Minimum sale year")
    parser.add_argument(
        "--sample", type=int, default=None, help="Sample size (optional)"
    )

    args = parser.parse_args()
    main(output_file=args.output, min_year=args.min_year, sample_size=args.sample)
