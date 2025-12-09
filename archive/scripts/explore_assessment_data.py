#!/usr/bin/env python3
"""
King County Assessment Data Exploration Script
V3.2: Fresh Data Integration

This script explores the King County Assessment data to understand:
1. Schema mapping from assessment DB to original features
2. Data quality (missing values, outliers, date ranges)
3. Join strategy between tables
4. Feature availability for model prediction
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
DATA_DIR = Path("data")
ASSESSMENT_DIR = Path("Reference_Docs/King_County_Assessment_data_ALL")

print("=" * 80)
print("KING COUNTY ASSESSMENT DATA EXPLORATION")
print("=" * 80)

# ============================================================================
# 1. LOAD SAMPLE DATA
# ============================================================================

print("\n1. Loading sample data from assessment files...")

sales = pd.read_csv(ASSESSMENT_DIR / "EXTR_RPSale.csv", nrows=1000, encoding='latin1')
resbldg = pd.read_csv(ASSESSMENT_DIR / "EXTR_ResBldg.csv", nrows=1000, encoding='latin1')
parcel = pd.read_csv(ASSESSMENT_DIR / "EXTR_Parcel.csv", nrows=1000, encoding='latin1')

print(f"   - Sales records: {len(sales):,}")
print(f"   - Residential buildings: {len(resbldg):,}")
print(f"   - Parcels: {len(parcel):,}")

# ============================================================================
# 2. SCHEMA MAPPING: Assessment → Original Features
# ============================================================================

print("\n2. Schema Mapping (Assessment → Original Features):")
print("-" * 80)

mapping = {
    "Original Feature": [
        "price",
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "floors",
        "waterfront",
        "view",
        "condition",
        "grade",
        "sqft_above",
        "sqft_basement",
        "yr_built",
        "yr_renovated",
        "zipcode",
        "lat",
        "long",
        "sqft_living15",
        "sqft_lot15",
    ],
    "Assessment Source": [
        "EXTR_RPSale.SalePrice",
        "EXTR_ResBldg.Bedrooms",
        "EXTR_ResBldg.BathFullCount + BathHalfCount*0.5 + Bath3qtrCount*0.75",
        "EXTR_ResBldg.SqFtTotLiving",
        "EXTR_Parcel.SqFtLot",
        "EXTR_ResBldg.Stories",
        "EXTR_Parcel.WfntLocation (derived: >0 = True)",
        "EXTR_ResBldg.ViewUtilization OR multiple view columns in Parcel",
        "EXTR_ResBldg.Condition",
        "EXTR_ResBldg.BldgGrade",
        "SqFt1stFloor + SqFt2ndFloor + SqFtUpperFloor",
        "EXTR_ResBldg.SqFtTotBasement",
        "EXTR_ResBldg.YrBuilt",
        "EXTR_ResBldg.YrRenovated",
        "EXTR_ResBldg.ZipCode",
        "NOT AVAILABLE (need geocoding)",
        "NOT AVAILABLE (need geocoding)",
        "NOT AVAILABLE (neighbor avg - complex)",
        "NOT AVAILABLE (neighbor avg - complex)",
    ],
}

mapping_df = pd.DataFrame(mapping)
print(mapping_df.to_string(index=False))

# ============================================================================
# 3. DATA QUALITY CHECKS
# ============================================================================

print("\n\n3. Data Quality Analysis:")
print("-" * 80)

print("\na) Sales Data (EXTR_RPSale):")
print(f"   Date range: {sales['DocumentDate'].min()} to {sales['DocumentDate'].max()}")
print(f"   Price range: ${sales['SalePrice'].min():,.0f} to ${sales['SalePrice'].max():,.0f}")
print(f"   Median price: ${sales['SalePrice'].median():,.0f}")
print(f"   Missing prices: {sales['SalePrice'].isna().sum()}")
print(f"   Zero prices: {(sales['SalePrice'] == 0).sum()}")

print("\nb) Residential Building Data (EXTR_ResBldg):")
print(f"   Bedrooms range: {resbldg['Bedrooms'].min()} to {resbldg['Bedrooms'].max()}")
print(f"   SqFtTotLiving range: {resbldg['SqFtTotLiving'].min():,} to {resbldg['SqFtTotLiving'].max():,}")
print(f"   Missing SqFtTotLiving: {resbldg['SqFtTotLiving'].isna().sum()}")
print(f"   YrBuilt range: {resbldg['YrBuilt'].min()} to {resbldg['YrBuilt'].max()}")
print(f"   Missing ZipCode: {resbldg['ZipCode'].isna().sum()}")

print("\nc) Parcel Data (EXTR_Parcel):")
print(f"   SqFtLot range: {parcel['SqFtLot'].min():,} to {parcel['SqFtLot'].max():,}")
print(f"   Missing SqFtLot: {parcel['SqFtLot'].isna().sum()}")
print(f"   Waterfront parcels: {(parcel['WfntLocation'] > 0).sum()}")

# ============================================================================
# 4. JOIN STRATEGY
# ============================================================================

print("\n\n4. Join Strategy:")
print("-" * 80)
print("   Key columns:")
print("   - EXTR_RPSale: Major + Minor (parcel ID)")
print("   - EXTR_ResBldg: Major + Minor (+ BldgNbr for multi-building parcels)")
print("   - EXTR_Parcel: Major + Minor")
print()
print("   Join sequence:")
print("   1. Sales → ResBldg (on Major+Minor, keep most recent building if multiple)")
print("   2. Result → Parcel (on Major+Minor)")
print()
print("   Potential issues:")
print("   - Multiple buildings per parcel (need aggregation strategy)")
print("   - Sales without building records (new construction?)")
print("   - Missing zipcodes (need to derive from other location data)")

# ============================================================================
# 5. SAMPLE JOIN TEST
# ============================================================================

print("\n\n5. Sample Join Test:")
print("-" * 80)

# Merge sales with buildings
merged = sales.merge(
    resbldg,
    on=["Major", "Minor"],
    how="inner",
    suffixes=("_sale", "_bldg")
)

print(f"   Sales records: {len(sales):,}")
print(f"   After joining with buildings: {len(merged):,} ({len(merged)/len(sales)*100:.1f}%)")

# Further merge with parcel
merged = merged.merge(
    parcel,
    on=["Major", "Minor"],
    how="inner",
    suffixes=("", "_parcel")
)

print(f"   After joining with parcels: {len(merged):,} ({len(merged)/len(sales)*100:.1f}%)")

# ============================================================================
# 6. DATE RANGE ANALYSIS
# ============================================================================

print("\n\n6. Date Range Analysis:")
print("-" * 80)

# Load more sales data to check date range
try:
    sales_full = pd.read_csv(ASSESSMENT_DIR / "EXTR_RPSale.csv", nrows=50000, encoding='latin1')
    sales_full['DocumentDate'] = pd.to_datetime(sales_full['DocumentDate'], errors='coerce')
    sales_full['Year'] = sales_full['DocumentDate'].dt.year

    print(f"   Total sales loaded: {len(sales_full):,}")
    print(f"   Date range: {sales_full['DocumentDate'].min()} to {sales_full['DocumentDate'].max()}")
    print()
    print("   Sales by year:")
    year_counts = sales_full['Year'].value_counts().sort_index()
    for year, count in year_counts.items():
        median_price = sales_full[sales_full['Year'] == year]['SalePrice'].median()
        print(f"     {year}: {count:>6,} sales, median ${median_price:>10,.0f}")
except Exception as e:
    print(f"   Error loading full sales data: {e}")
    print("   Using sample data for date analysis...")

# ============================================================================
# 7. FEATURE COMPARISON
# ============================================================================

print("\n\n7. Feature Comparison Summary:")
print("-" * 80)
print("   ✅ Available in assessment data:")
print("      - price, bedrooms, bathrooms, sqft_living, sqft_lot")
print("      - floors, view, condition, grade")
print("      - sqft_above, sqft_basement, yr_built, yr_renovated")
print("      - zipcode, waterfront (derived)")
print()
print("   ❌ NOT available (would need geocoding/aggregation):")
print("      - lat, long (addresses available, need geocoding)")
print("      - sqft_living15, sqft_lot15 (neighbor averages)")
print()
print("   📊 Impact on model:")
print("      - Can train on ~15 of 17 original features")
print("      - Missing lat/long may reduce geographic precision")
print("      - Missing neighbor features may reduce local market effects")
print("      - Consider adding new assessment-specific features")

print("\n" + "=" * 80)
print("EXPLORATION COMPLETE")
print("=" * 80)
print("\nNext steps:")
print("1. Create data transformation pipeline (Assessment → Model format)")
print("2. Filter to residential sales (2020+) for training")
print("3. Handle missing values and outliers")
print("4. Train new model on fresh data")
print("5. Compare V2.5 (2014 data) vs V3.2 (2020+ data) performance")
