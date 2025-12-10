"""
Model service for loading and using the price prediction model.

This service handles:
- Loading the model from local pickle file or MLflow registry
- Making predictions with proper feature ordering
- Managing model lifecycle (reload, version tracking)

Supports both local file-based loading and MLflow Model Registry.
"""

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Settings, get_settings

# Optional MLflow import
try:
    import mlflow.sklearn

    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)


class ModelService:
    """Service for model loading and prediction.

    This service loads the trained sklearn model (Pipeline with scaler + regressor)
    and provides methods for making predictions.

    Attributes:
        model: The loaded sklearn Pipeline
        feature_names: Ordered list of feature names expected by the model
        model_version: Version string for the loaded model (e.g., "v2.4.1")
        model_type: Type of model (e.g., "XGBRegressor (tuned)")
        is_loaded: Whether the model is currently loaded
    """

    def __init__(self, settings: Settings):
        """Initialize the model service.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self.model = None
        self.feature_names: list[str] = []
        self.model_version: str = "unknown"
        self.model_type: str = "unknown"
        self.is_loaded: bool = False
        self._load_model()

    def _load_model(self) -> None:
        """Load the model from local file or MLflow registry.

        Raises:
            FileNotFoundError: If local model files are not found
            RuntimeError: If model loading fails
        """
        if self.settings.use_mlflow_model and MLFLOW_AVAILABLE:
            self._load_from_mlflow()
        else:
            self._load_from_local()

    def _load_from_local(self) -> None:
        """Load model from local pickle file.

        Raises:
            FileNotFoundError: If model or features file not found
        """
        model_path = Path(self.settings.model_path)
        features_path = Path(self.settings.features_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}. "
                "Run 'python src/train.py' to train the model first."
            )

        if not features_path.exists():
            raise FileNotFoundError(
                f"Features file not found: {features_path}. "
                "Run 'python src/train.py' to train the model first."
            )

        # Load the model
        logger.info("Loading model from local file: %s", model_path)
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        # Load feature names
        logger.info("Loading feature names from: %s", features_path)
        with open(features_path) as f:
            self.feature_names = json.load(f)

        # Detect model version from metrics.json if available
        metrics_path = model_path.parent / "metrics.json"
        if metrics_path.exists():
            try:
                with open(metrics_path) as f:
                    metrics = json.load(f)
                self.model_version = metrics.get("version", "unknown")
                self.model_type = metrics.get("model_type", "unknown")
                logger.info(
                    "Loaded model info from metrics.json: %s (%s)",
                    self.model_version,
                    self.model_type,
                )
            except Exception as e:
                logger.warning("Could not read metrics.json: %s", e)
                self.model_version = "v2.1" if len(self.feature_names) >= 40 else "v1"
        else:
            # Fallback: detect version based on feature count
            # V1: 33 features (7 home + 26 demographic)
            # V2.1+: 43 features (17 home + 26 demographic)
            if len(self.feature_names) >= 40:
                self.model_version = "v2.1"
            else:
                self.model_version = "v1"

        self.is_loaded = True
        logger.info(
            "Model loaded successfully. Version: %s, Features: %d",
            self.model_version,
            len(self.feature_names),
        )

    def _load_from_mlflow(self) -> None:
        """Load model from MLflow Model Registry.

        Raises:
            RuntimeError: If MLflow loading fails
        """
        if not MLFLOW_AVAILABLE:
            raise RuntimeError("MLflow is not installed. Cannot load from registry.")

        if self.settings.mlflow_tracking_uri:
            mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)

        model_name = self.settings.mlflow_model_name
        stage = self.settings.mlflow_model_stage

        # Construct model URI
        if stage:
            model_uri = f"models:/{model_name}/{stage}"
        else:
            model_uri = f"models:/{model_name}/latest"

        logger.info("Loading model from MLflow: %s", model_uri)

        try:
            self.model = mlflow.sklearn.load_model(model_uri)

            # Try to get feature names from model metadata
            # If not available, fall back to local features file
            features_path = Path(self.settings.features_path)
            if features_path.exists():
                with open(features_path) as f:
                    self.feature_names = json.load(f)
            else:
                logger.warning("Features file not found. Feature ordering may be incorrect.")

            self.model_version = stage or "latest"
            self.is_loaded = True
            logger.info("Model loaded from MLflow successfully.")

        except Exception as e:
            logger.error("Failed to load model from MLflow: %s", str(e))
            logger.info("Falling back to local model file...")
            self._load_from_local()

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Make predictions using the loaded model.

        Args:
            features: DataFrame with feature columns matching self.feature_names

        Returns:
            Array of predicted prices

        Raises:
            RuntimeError: If model is not loaded
            ValueError: If feature columns don't match expected features
        """
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded. Call _load_model() first.")

        # Ensure feature order matches training
        missing_features = set(self.feature_names) - set(features.columns)
        if missing_features:
            raise ValueError(
                f"Missing features: {missing_features}. " f"Expected: {self.feature_names}"
            )

        # Reorder columns to match training feature order
        features_ordered = features[self.feature_names]

        # Make prediction
        predictions = self.model.predict(features_ordered)

        return predictions

    def predict_single(self, features_dict: dict) -> float:
        """Make a single prediction from a feature dictionary.

        Args:
            features_dict: Dictionary of feature_name -> value

        Returns:
            Predicted price as float
        """
        df = pd.DataFrame([features_dict])
        predictions = self.predict(df)
        return float(predictions[0])

    def reload(self) -> None:
        """Reload the model (e.g., after a new version is deployed).

        This can be used for zero-downtime model updates in a
        blue/green or A/B deployment pattern.
        """
        logger.info("Reloading model...")
        self.is_loaded = False
        self._load_model()
        logger.info("Model reloaded successfully.")

    def get_status(self) -> dict:
        """Get the current status of the model service.

        Returns:
            Dictionary with model status information
        """
        return {
            "is_loaded": self.is_loaded,
            "model_version": self.model_version,
            "model_type": self.model_type,
            "feature_count": len(self.feature_names),
            "feature_names": self.feature_names,
        }


# Singleton instance
_model_service: ModelService | None = None


def get_model_service() -> ModelService:
    """Get the singleton ModelService instance.

    Returns:
        ModelService: The model service singleton

    Raises:
        RuntimeError: If model loading fails
    """
    global _model_service
    if _model_service is None:
        settings = get_settings()
        _model_service = ModelService(settings)
    return _model_service


def reset_model_service() -> None:
    """Reset the model service singleton.

    Useful for testing or forcing a model reload.
    """
    global _model_service
    _model_service = None
