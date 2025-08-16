"""
Inference Pipeline for Real-time Churn Prediction

This module provides a production-ready inference pipeline that processes
raw user event data through feature engineering and model prediction.

Key Components:
- RealTimeFeatureEngine: Handles feature engineering for single users
- BatchInferenceEngine: Processes multiple users efficiently
- PredictionPipeline: End-to-end pipeline orchestrator
- Caching and optimization for production performance

Usage:
    pipeline = PredictionPipeline.from_model_path("models/production_churn_model.joblib")
    prediction = pipeline.predict_user_churn(user_events)
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from ..features.feature_store import FeatureConfig, FeatureStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealTimeFeatureEngine:
    """
    Real-time feature engineering engine for single user inference.

    Optimized for low-latency single user predictions with caching.
    """

    def __init__(self, feature_config: FeatureConfig, enable_caching: bool = True):
        self.feature_config = feature_config
        self.feature_store = FeatureStore(feature_config)
        self.enable_caching = enable_caching
        self._feature_cache = {}

        logger.info(
            f"Initialized RealTimeFeatureEngine with {feature_config.feature_version}"
        )

    def extract_features(self, events_df: pd.DataFrame, user_id: str) -> pd.DataFrame:
        """
        Extract features for a single user from their event history.

        Args:
            events_df: User's event data
            user_id: User identifier

        Returns:
            DataFrame with computed features for the user
        """
        start_time = time.time()

        # Validate input
        if len(events_df) == 0:
            raise ValueError("No events provided for feature extraction")

        if events_df["userId"].nunique() > 1:
            raise ValueError("Multiple users found, expected single user")

        # Check cache if enabled
        cache_key = None
        if self.enable_caching:
            # Create a simple cache key based on events
            cache_key = f"{user_id}_{len(events_df)}_{events_df['ts'].max()}"
            if cache_key in self._feature_cache:
                logger.debug(f"Cache hit for user {user_id}")
                return self._feature_cache[cache_key]

        try:
            # Compute features using FeatureStore
            features_df, validation = self.feature_store.compute_features(events_df)

            if not validation.get("passed", False):
                logger.warning(
                    f"Feature validation failed for user {user_id}: {validation.get('warnings', [])}"
                )

            # Cache the result
            if self.enable_caching and cache_key:
                self._feature_cache[cache_key] = features_df

                # Limit cache size
                if len(self._feature_cache) > 1000:
                    # Remove oldest entries
                    oldest_keys = list(self._feature_cache.keys())[:100]
                    for key in oldest_keys:
                        del self._feature_cache[key]

            processing_time = time.time() - start_time
            logger.debug(
                f"Feature extraction completed for user {user_id} in {processing_time:.3f}s"
            )

            return features_df

        except Exception as e:
            logger.error(f"Feature extraction failed for user {user_id}: {e!s}")
            raise


class BatchInferenceEngine:
    """
    Batch inference engine for processing multiple users efficiently.

    Optimized for throughput when processing multiple users simultaneously.
    """

    def __init__(self, feature_config: FeatureConfig):
        self.feature_config = feature_config
        self.feature_store = FeatureStore(feature_config)

        logger.info(
            f"Initialized BatchInferenceEngine with {feature_config.feature_version}"
        )

    def extract_features_batch(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features for multiple users from their combined event history.

        Args:
            events_df: Combined event data for multiple users

        Returns:
            DataFrame with computed features for all users
        """
        start_time = time.time()

        if len(events_df) == 0:
            raise ValueError("No events provided for batch feature extraction")

        n_users = events_df["userId"].nunique()
        logger.info(f"Starting batch feature extraction for {n_users} users")

        try:
            # Compute features using FeatureStore
            features_df, validation = self.feature_store.compute_features(events_df)

            if not validation.get("passed", False):
                logger.warning(
                    f"Batch feature validation issues: {validation.get('warnings', [])}"
                )

            processing_time = time.time() - start_time
            logger.info(
                f"Batch feature extraction completed for {n_users} users in {processing_time:.3f}s"
            )

            return features_df

        except Exception as e:
            logger.error(f"Batch feature extraction failed: {e!s}")
            raise


class PredictionPipeline:
    """
    End-to-end prediction pipeline for churn prediction.

    Handles feature engineering, model inference, and post-processing.
    """

    def __init__(
        self,
        model: Any,
        feature_columns: list[str],
        feature_config: FeatureConfig,
        preprocessing_config: dict | None = None,
        model_metadata: dict | None = None,
    ):
        self.model = model
        self.feature_columns = feature_columns
        self.feature_config = feature_config
        self.preprocessing_config = preprocessing_config or {}
        self.model_metadata = model_metadata or {}

        # Initialize feature engines
        self.realtime_engine = RealTimeFeatureEngine(feature_config)
        self.batch_engine = BatchInferenceEngine(feature_config)

        # Prediction statistics
        self.prediction_count = 0
        self.total_processing_time = 0.0

        logger.info(
            f"Initialized PredictionPipeline with {len(feature_columns)} features"
        )

    @classmethod
    def from_model_path(cls, model_path: str) -> "PredictionPipeline":
        """
        Create pipeline from saved model and metadata files.

        Args:
            model_path: Path to the saved model file

        Returns:
            Configured PredictionPipeline instance
        """
        logger.info(f"Loading prediction pipeline from {model_path}")

        # Load model
        model = joblib.load(model_path)

        # Load metadata
        metadata_path = str(model_path).replace(".joblib", "_metadata.json")
        with open(metadata_path) as f:
            metadata = json.load(f)

        # Extract configuration
        feature_columns = metadata["feature_columns"]
        feature_config = FeatureConfig(**metadata["feature_config"])
        preprocessing_config = metadata.get("preprocessing", {})

        logger.info(f"Loaded model with {len(feature_columns)} features")

        return cls(
            model=model,
            feature_columns=feature_columns,
            feature_config=feature_config,
            preprocessing_config=preprocessing_config,
            model_metadata=metadata,
        )

    def _preprocess_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply preprocessing to extracted features.

        Args:
            features_df: Raw features from feature engineering

        Returns:
            Preprocessed features ready for model input
        """
        # Ensure all required features are present
        missing_features = set(self.feature_columns) - set(features_df.columns)
        if missing_features:
            logger.warning(f"Missing features detected: {list(missing_features)}")
            for feature in missing_features:
                features_df[feature] = 0

        # Select and order features
        x = features_df[self.feature_columns].copy()

        # Apply preprocessing based on configuration
        preprocessing = self.preprocessing_config

        if preprocessing.get("fillna_strategy") == "median":
            # Use cached median values if available, otherwise compute
            if not hasattr(self, "_feature_medians"):
                self._feature_medians = x.median()
            x = x.fillna(self._feature_medians)
        elif preprocessing.get("fillna_strategy") == "mean":
            if not hasattr(self, "_feature_means"):
                self._feature_means = x.mean()
            x = x.fillna(self._feature_means)
        else:
            # Default: fill with 0
            x = x.fillna(0)

        return x

    def _postprocess_prediction(
        self, prediction_proba: float, user_id: str
    ) -> dict[str, Any]:
        """
        Post-process model prediction into final result.

        Args:
            prediction_proba: Raw prediction probability from model
            user_id: User identifier

        Returns:
            Dictionary with processed prediction result
        """
        # Determine risk level
        if prediction_proba >= 0.7:
            risk_level = "HIGH"
        elif prediction_proba >= 0.3:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return {
            "user_id": user_id,
            "churn_probability": float(prediction_proba),
            "risk_level": risk_level,
            "model_version": self.feature_config.feature_version,
            "prediction_timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def predict_user_churn(
        self, events_df: pd.DataFrame, user_id: str
    ) -> dict[str, Any]:
        """
        Predict churn for a single user.

        Args:
            events_df: User's event history
            user_id: User identifier

        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()

        try:
            # Extract features
            features_df = self.realtime_engine.extract_features(events_df, user_id)

            # Preprocess features
            x = self._preprocess_features(features_df)

            # Make prediction
            prediction_proba = self.model.predict_proba(x)[0, 1]

            # Post-process result
            result = self._postprocess_prediction(prediction_proba, user_id)

            # Update statistics
            processing_time = time.time() - start_time
            self.prediction_count += 1
            self.total_processing_time += processing_time

            result["processing_time_seconds"] = processing_time

            logger.debug(
                f"Prediction completed for user {user_id}: {prediction_proba:.4f}"
            )

            return result

        except Exception as e:
            logger.error(f"Prediction failed for user {user_id}: {e!s}")
            raise

    def predict_batch_churn(self, events_df: pd.DataFrame) -> list[dict[str, Any]]:
        """
        Predict churn for multiple users.

        Args:
            events_df: Combined event history for multiple users

        Returns:
            List of prediction results for each user
        """
        start_time = time.time()

        user_ids = events_df["userId"].unique()
        n_users = len(user_ids)

        logger.info(f"Starting batch prediction for {n_users} users")

        try:
            # Extract features for all users
            features_df = self.batch_engine.extract_features_batch(events_df)

            # Preprocess features
            x = self._preprocess_features(features_df)

            # Make predictions for all users
            prediction_probas = self.model.predict_proba(x)[:, 1]

            # Post-process results
            results = []
            for i, user_id in enumerate(features_df["userId"]):
                result = self._postprocess_prediction(prediction_probas[i], user_id)
                results.append(result)

            # Update statistics
            processing_time = time.time() - start_time
            self.prediction_count += n_users
            self.total_processing_time += processing_time

            logger.info(
                f"Batch prediction completed for {n_users} users in {processing_time:.3f}s"
            )

            return results

        except Exception as e:
            logger.error(f"Batch prediction failed: {e!s}")
            raise

    def get_pipeline_stats(self) -> dict[str, Any]:
        """
        Get pipeline performance statistics.

        Returns:
            Dictionary with pipeline statistics
        """
        avg_processing_time = (
            self.total_processing_time / self.prediction_count
            if self.prediction_count > 0
            else 0
        )

        return {
            "total_predictions": self.prediction_count,
            "total_processing_time_seconds": self.total_processing_time,
            "average_processing_time_seconds": avg_processing_time,
            "model_version": self.feature_config.feature_version,
            "feature_count": len(self.feature_columns),
            "pipeline_initialized_at": getattr(self, "_init_time", None),
        }


# Utility functions for pipeline management
def create_sample_events(user_id: str, n_events: int = 100) -> pd.DataFrame:
    """
    Create sample events for testing the pipeline.

    Args:
        user_id: User identifier
        n_events: Number of events to generate

    Returns:
        DataFrame with sample events
    """
    import secrets

    events = []
    base_time = int(datetime.now().timestamp() * 1000)

    pages = ["NextSong", "Home", "Settings", "Playlist", "Logout", "Cancel"]
    levels = ["paid", "free"]

    for i in range(n_events):
        event = {
            "ts": base_time + (i * 1000 * 60),  # Events every minute
            "userId": user_id,
            "sessionId": f"session_{user_id}_{i//10}",
            "page": secrets.choice(pages),
            "level": secrets.choice(levels),
            "itemInSession": i % 10,
        }
        events.append(event)

    return pd.DataFrame(events)


if __name__ == "__main__":
    """Example usage of the inference pipeline."""
    print("Inference Pipeline for Churn Prediction")
    print("=======================================")

    # Check if model exists
    model_path = "models/production_churn_model.joblib"
    if Path(model_path).exists():
        print(f"Loading pipeline from {model_path}")

        # Create pipeline
        pipeline = PredictionPipeline.from_model_path(model_path)

        # Create sample events
        sample_events = create_sample_events("test_user_123", 50)
        print(f"Created {len(sample_events)} sample events")

        # Make prediction
        result = pipeline.predict_user_churn(sample_events, "test_user_123")
        print(f"Prediction result: {result}")

        # Show pipeline stats
        stats = pipeline.get_pipeline_stats()
        print(f"Pipeline stats: {stats}")

    else:
        print(f"Model not found at {model_path}")
        print("Please train a model first using scripts/train_production_model.py")
