#!/bin/bash
# Setup Data Directory
# This script sets up the data directory and files needed for training

set -e

DATA_DIR="data"
ASSESSMENT_DATA_DIR="Reference_Docs/King_County_Assessment_data_ALL"

echo "Setting up data directories..."

# Create data directory if it doesn't exist
mkdir -p "$DATA_DIR"

# Check if original data exists
if [ -f "$DATA_DIR/kc_house_data.csv" ] && [ -f "$DATA_DIR/zipcode_demographics.csv" ]; then
    echo "✓ Original data files found at $DATA_DIR/"
else
    echo "⚠ Original data files not found at $DATA_DIR/"
    echo "  Expected files:"
    echo "    - $DATA_DIR/kc_house_data.csv"
    echo "    - $DATA_DIR/zipcode_demographics.csv"
    echo "    - $DATA_DIR/future_unseen_examples.csv"
    echo ""
    echo "  Please copy these files from C:\\Experiments\\Real-Estate-Estimator\\data"
fi

# Check if King County Assessment data exists
if [ -d "$ASSESSMENT_DATA_DIR" ]; then
    echo "✓ King County Assessment data found at $ASSESSMENT_DATA_DIR/"
    ls -lh "$ASSESSMENT_DATA_DIR"/ | head -10
else
    echo "⚠ King County Assessment data not found at $ASSESSMENT_DATA_DIR/"
    echo "  Please copy this directory from C:\\Experiments\\Real-Estate-Estimator\\Reference_Docs\\King_County_Assessment_data_ALL"
fi

echo ""
echo "Data setup check complete."
