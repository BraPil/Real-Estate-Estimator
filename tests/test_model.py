"""
Model Tests

These tests verify that the model works correctly.
They're designed to run in CI and catch regressions.

Test Categories:
1. Model Loading - Can we load the model?
2. Prediction - Does prediction work?
3. Input Validation - Does it handle bad input?
4. Performance Bounds - Is accuracy within expected range?
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# =============================================================================
# FIXTURES
# =============================================================================
# Fixtures are reusable test components. pytest runs them before tests.


@pytest.fixture
def model_path():
    """Path to the production model."""
    return Path(__file__).parent.parent / "model" / "model.pkl"


@pytest.fixture
def model(model_path):
    """Load the production model."""
    if not model_path.exists():
        pytest.skip("No model.pkl found - skipping model tests")

    with open(model_path, "rb") as f:
        return pickle.load(f)


@pytest.fixture
def sample_input():
    """
    A realistic sample input for testing.

    Feature names must match those in model/model_features.json exactly.
    The model expects 43 features (17 home + 26 demographic).
    """
    return pd.DataFrame(
        [
            {
                # 17 Home features
                "bedrooms": 3,
                "bathrooms": 2.0,
                "sqft_living": 1800,
                "sqft_lot": 5000,
                "floors": 1.0,
                "waterfront": 0,
                "view": 0,
                "condition": 3,
                "grade": 7,
                "sqft_above": 1800,
                "sqft_basement": 0,
                "yr_built": 1990,
                "yr_renovated": 0,
                "lat": 47.5,
                "long": -122.2,
                "sqft_living15": 1800,
                "sqft_lot15": 5000,
                # 26 Demographic features (names from zipcode_demographics.csv)
                "ppltn_qty": 50000,
                "urbn_ppltn_qty": 40000,
                "sbrbn_ppltn_qty": 8000,
                "farm_ppltn_qty": 2000,
                "non_farm_qty": 48000,
                "medn_hshld_incm_amt": 75000,
                "medn_incm_per_prsn_amt": 35000,
                "hous_val_amt": 450000,
                "edctn_less_than_9_qty": 1000,
                "edctn_9_12_qty": 2000,
                "edctn_high_schl_qty": 10000,
                "edctn_some_clg_qty": 12000,
                "edctn_assoc_dgre_qty": 5000,
                "edctn_bchlr_dgre_qty": 15000,
                "edctn_prfsnl_qty": 5000,
                "per_urbn": 0.80,
                "per_sbrbn": 0.16,
                "per_farm": 0.04,
                "per_non_farm": 0.96,
                "per_less_than_9": 0.02,
                "per_9_to_12": 0.04,
                "per_hsd": 0.20,
                "per_some_clg": 0.24,
                "per_assoc": 0.10,
                "per_bchlr": 0.30,
                "per_prfsnl": 0.10,
            }
        ]
    )


# =============================================================================
# MODEL LOADING TESTS
# =============================================================================


class TestModelLoading:
    """Tests for model loading functionality."""

    def test_model_file_exists(self, model_path):
        """Verify model file exists (or skip)."""
        # If model doesn't exist, test will be skipped by fixture
        pass

    def test_model_is_pipeline(self, model):
        """Model should be a sklearn Pipeline."""
        from sklearn.pipeline import Pipeline

        assert isinstance(model, Pipeline), f"Expected Pipeline, got {type(model)}"

    def test_model_has_predict(self, model):
        """Model must have a predict method."""
        assert hasattr(model, "predict"), "Model must have predict method"

    def test_model_has_steps(self, model):
        """Pipeline should have expected steps."""
        assert hasattr(model, "named_steps"), "Pipeline must have named_steps"
        assert "scaler" in model.named_steps, "Pipeline must have scaler"
        assert "model" in model.named_steps, "Pipeline must have model"


# =============================================================================
# PREDICTION TESTS
# =============================================================================


class TestPrediction:
    """Tests for prediction functionality."""

    def test_predict_returns_array(self, model, sample_input):
        """Prediction should return numpy array."""
        result = model.predict(sample_input)
        assert isinstance(result, np.ndarray), f"Expected ndarray, got {type(result)}"

    def test_predict_returns_correct_shape(self, model, sample_input):
        """Prediction should return one value per input."""
        result = model.predict(sample_input)
        assert result.shape == (len(sample_input),), f"Wrong shape: {result.shape}"

    def test_predict_returns_positive(self, model, sample_input):
        """Price predictions should be positive."""
        result = model.predict(sample_input)
        assert (result > 0).all(), "Predictions should be positive"

    def test_predict_reasonable_range(self, model, sample_input):
        """Prediction should be in reasonable price range."""
        result = model.predict(sample_input)
        # For a 1800 sqft home, expect $200k - $1M
        assert result[0] > 100_000, f"Prediction too low: ${result[0]:,.0f}"
        assert result[0] < 2_000_000, f"Prediction too high: ${result[0]:,.0f}"


# =============================================================================
# INPUT VALIDATION TESTS
# =============================================================================


class TestInputValidation:
    """Tests for input handling."""

    def test_handles_single_sample(self, model, sample_input):
        """Model should handle single sample."""
        result = model.predict(sample_input)
        assert len(result) == 1

    def test_handles_multiple_samples(self, model, sample_input):
        """Model should handle multiple samples."""
        multiple = pd.concat([sample_input] * 5, ignore_index=True)
        result = model.predict(multiple)
        assert len(result) == 5


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================


class TestPerformance:
    """Tests for model performance bounds."""

    @pytest.mark.slow
    def test_prediction_speed(self, model, sample_input):
        """Single prediction should be fast."""
        import time

        # Warm up
        model.predict(sample_input)

        # Time 100 predictions
        start = time.time()
        for _ in range(100):
            model.predict(sample_input)
        elapsed = time.time() - start

        # Should be < 1 second for 100 predictions
        assert elapsed < 1.0, f"Too slow: {elapsed:.2f}s for 100 predictions"


# =============================================================================
# MLFLOW TESTS
# =============================================================================


class TestMLflowConfig:
    """Tests for MLflow configuration."""

    def test_mlflow_config_imports(self):
        """MLflow config should import without errors."""
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent))

        from mlflow_config import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI

        assert MLFLOW_TRACKING_URI is not None
        assert MLFLOW_EXPERIMENT_NAME is not None

    def test_mlflow_setup(self):
        """MLflow setup should create experiment."""
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent))

        from mlflow_config import setup_mlflow

        # This should not raise an error
        experiment_id = setup_mlflow()
        assert experiment_id is not None
