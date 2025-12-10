"""
V1 MVP Model Service - Loads and serves predictions.
"""

import json
import logging
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


class ModelService:
    """V1 Model loading and prediction service."""
    
    def __init__(self, model_path: str = "model/model.pkl", features_path: str = "model/model_features.json"):
        self.model = None
        self.feature_names: List[str] = []
        self.model_version = "1.0.0"
        self._load_model(model_path, features_path)
    
    def _load_model(self, model_path: str, features_path: str) -> None:
        """Load model and feature names."""
        logger.info("Loading V1 model from: %s", model_path)
        
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        
        with open(features_path, "r") as f:
            self.feature_names = json.load(f)
        
        # Try to get version from metrics
        metrics_path = Path(model_path).parent / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
                self.model_version = metrics.get("model_version", "1.0.0")
        
        logger.info("V1 Model loaded. Features: %d", len(self.feature_names))
    
    def predict_single(self, features: Dict[str, float]) -> float:
        """Make a single prediction."""
        # Build feature vector in correct order
        feature_vector = []
        for name in self.feature_names:
            if name in features:
                feature_vector.append(features[name])
            else:
                raise ValueError(f"Missing feature: {name}")
        
        X = np.array([feature_vector])
        prediction = self.model.predict(X)[0]
        
        return float(prediction)
    
    def get_status(self) -> Dict:
        """Return model status."""
        return {
            "is_loaded": self.model is not None,
            "model_version": self.model_version,
            "n_features": len(self.feature_names)
        }


_model_service = None


@lru_cache
def get_model_service() -> ModelService:
    global _model_service
    if _model_service is None:
        _model_service = ModelService()
    return _model_service


def reset_model_service() -> None:
    global _model_service
    _model_service = None
    get_model_service.cache_clear()
