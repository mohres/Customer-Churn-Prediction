"""
Feature Store Module

This module provides centralized feature computation, storage, and management
for the churn prediction system. It handles:
- Feature computation orchestration
- Time-window based feature generation
- Feature caching and versioning
- Feature validation and monitoring
- Incremental feature updates

Key classes:
- FeatureStore: Main orchestration class for feature operations
- FeatureConfig: Configuration for feature computation
- FeatureValidator: Feature quality validation
- FeatureCache: Caching mechanism for computed features

Key functions:
- compute_features(): Main feature computation pipeline
- validate_features(): Feature quality checks
- save_features(): Persist features with versioning
- load_features(): Load cached features
"""

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .base_features import (
    compute_all_base_features,
)
from .behavioral_features import compute_all_behavioral_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature computation."""

    # Time windows for activity features
    activity_windows: list[int] = None

    # Time windows for trend analysis
    trend_windows: list[int] = None

    # Reference date for time-based calculations
    reference_date: datetime | None = None

    # Feature types to compute
    include_base_features: bool = True
    include_behavioral_features: bool = True

    # Validation settings
    validate_features: bool = True
    validation_threshold: float = 0.95  # Minimum data quality threshold

    # Caching settings
    enable_caching: bool = True
    cache_dir: str = "cache/features"

    # Versioning
    feature_version: str = "1.0"

    def __post_init__(self) -> None:
        if self.activity_windows is None:
            self.activity_windows = [7, 14, 30]

        if self.trend_windows is None:
            self.trend_windows = [7, 14, 21]

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        config_dict = asdict(self)
        if self.reference_date:
            config_dict["reference_date"] = self.reference_date.isoformat()
        return config_dict

    @classmethod
    def from_dict(cls, config_dict: dict) -> "FeatureConfig":
        """Create config from dictionary."""
        if config_dict.get("reference_date"):
            config_dict["reference_date"] = datetime.fromisoformat(
                config_dict["reference_date"]
            )
        return cls(**config_dict)


class FeatureValidator:
    """Feature quality validation and monitoring."""

    def __init__(self, config: FeatureConfig) -> None:
        self.config = config
        self.validation_rules = self._setup_validation_rules()

    def _setup_validation_rules(self) -> dict:
        """Define validation rules for features."""
        return {
            "missing_threshold": 0.05,  # Max 5% missing values
            "infinite_threshold": 0.0,  # No infinite values allowed
            "min_variance_threshold": 1e-8,  # Minimum variance for numeric features
            "correlation_threshold": 0.95,  # Max correlation between features
            "feature_range_checks": {
                "rate_features": (0, 1),  # Rate features should be [0,1]
                "count_features": (0, float("inf")),  # Count features >= 0
                "ratio_features": (0, 1),  # Ratio features should be [0,1]
            },
        }

    def validate_feature_set(self, features_df: pd.DataFrame) -> dict:
        """
        Comprehensive feature validation.

        Args:
            features_df: DataFrame with computed features

        Returns:
            Dictionary with validation results
        """
        logger.info(
            f"Validating feature set with {len(features_df.columns)-1} features"
        )

        validation_results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_features": len(features_df.columns) - 1,
            "total_users": len(features_df),
            "passed": True,
            "warnings": [],
            "errors": [],
            "feature_quality": {},
            "summary": {},
        }

        # Check for missing values
        missing_stats = self._check_missing_values(features_df)
        validation_results["missing_values"] = missing_stats

        # Check for infinite/NaN values
        infinite_stats = self._check_infinite_values(features_df)
        validation_results["infinite_values"] = infinite_stats

        # Check feature variance
        variance_stats = self._check_feature_variance(features_df)
        validation_results["variance_stats"] = variance_stats

        # Check feature correlations
        correlation_stats = self._check_feature_correlations(features_df)
        validation_results["correlation_stats"] = correlation_stats

        # Check feature ranges
        range_stats = self._check_feature_ranges(features_df)
        validation_results["range_stats"] = range_stats

        # Generate summary
        validation_results["summary"] = self._generate_validation_summary(
            validation_results
        )

        # Determine overall pass/fail status
        validation_results["passed"] = len(validation_results["errors"]) == 0

        if validation_results["passed"]:
            logger.info("Feature validation PASSED")
        else:
            logger.warning(
                f"Feature validation FAILED with {len(validation_results['errors'])} errors"
            )

        return validation_results

    def _check_missing_values(self, df: pd.DataFrame) -> dict:
        """Check for missing values in features."""
        missing_stats = {}
        threshold = self.validation_rules["missing_threshold"]

        for col in df.columns:
            if col == "userId":
                continue

            missing_count = df[col].isnull().sum()
            missing_rate = missing_count / len(df)

            missing_stats[col] = {
                "missing_count": int(missing_count),
                "missing_rate": float(missing_rate),
                "exceeds_threshold": missing_rate > threshold,
            }

        return missing_stats

    def _check_infinite_values(self, df: pd.DataFrame) -> dict:
        """Check for infinite values in features."""
        infinite_stats = {}

        for col in df.columns:
            if col == "userId" or df[col].dtype not in ["float64", "int64"]:
                continue

            infinite_count = np.isinf(df[col]).sum()
            infinite_rate = infinite_count / len(df)

            infinite_stats[col] = {
                "infinite_count": int(infinite_count),
                "infinite_rate": float(infinite_rate),
                "has_infinite": infinite_count > 0,
            }

        return infinite_stats

    def _check_feature_variance(self, df: pd.DataFrame) -> dict:
        """Check feature variance to identify constant features."""
        variance_stats = {}
        threshold = self.validation_rules["min_variance_threshold"]

        for col in df.columns:
            if col == "userId" or df[col].dtype not in ["float64", "int64"]:
                continue

            variance = df[col].var()
            is_constant = variance < threshold

            variance_stats[col] = {
                "variance": float(variance) if not pd.isna(variance) else 0,
                "is_constant": bool(is_constant),
                "unique_values": int(df[col].nunique()),
            }

        return variance_stats

    def _check_feature_correlations(self, df: pd.DataFrame) -> dict:
        """Check for highly correlated features."""
        correlation_stats = {}
        threshold = self.validation_rules["correlation_threshold"]

        # Get numeric columns only
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
        numeric_cols = [col for col in numeric_cols if col != "userId"]

        if len(numeric_cols) > 1:
            correlation_matrix = df[numeric_cols].corr().abs()

            # Find highly correlated pairs
            high_correlations = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i + 1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if corr_value > threshold:
                        high_correlations.append(
                            {
                                "feature_1": correlation_matrix.columns[i],
                                "feature_2": correlation_matrix.columns[j],
                                "correlation": float(corr_value),
                            }
                        )

            correlation_stats = {
                "high_correlation_pairs": high_correlations,
                "max_correlation": float(
                    correlation_matrix.max().max() - 1
                ),  # -1 to exclude self-correlation
                "mean_correlation": float(correlation_matrix.mean().mean()),
            }

        return correlation_stats

    def _check_feature_ranges(self, df: pd.DataFrame) -> dict:
        """Check if features are within expected ranges."""
        range_stats = {}

        for col in df.columns:
            if col == "userId":
                continue

            range_info = {
                "min": (
                    float(df[col].min())
                    if df[col].dtype in ["float64", "int64"]
                    else None
                ),
                "max": (
                    float(df[col].max())
                    if df[col].dtype in ["float64", "int64"]
                    else None
                ),
                "range_violations": [],
            }

            # Check specific range rules
            if "rate" in col.lower() and df[col].dtype in ["float64", "int64"]:
                violations = ((df[col] < 0) | (df[col] > 1)).sum()
                if violations > 0:
                    range_info["range_violations"].append(
                        f"Rate feature outside [0,1]: {violations} values"
                    )

            if "count" in col.lower() and df[col].dtype in ["float64", "int64"]:
                violations = (df[col] < 0).sum()
                if violations > 0:
                    range_info["range_violations"].append(
                        f"Count feature negative: {violations} values"
                    )

            range_stats[col] = range_info

        return range_stats

    def _generate_validation_summary(self, validation_results: dict) -> dict:
        """Generate validation summary."""
        summary = {
            "total_features_validated": validation_results["total_features"],
            "features_with_missing_values": 0,
            "features_with_infinite_values": 0,
            "constant_features": 0,
            "highly_correlated_pairs": 0,
            "features_with_range_violations": 0,
        }

        # Count issues
        for col_stats in validation_results["missing_values"].values():
            if col_stats["missing_count"] > 0:
                summary["features_with_missing_values"] += 1

        for col_stats in validation_results["infinite_values"].values():
            if col_stats["infinite_count"] > 0:
                summary["features_with_infinite_values"] += 1

        for col_stats in validation_results["variance_stats"].values():
            if col_stats["is_constant"]:
                summary["constant_features"] += 1

        if "high_correlation_pairs" in validation_results["correlation_stats"]:
            summary["highly_correlated_pairs"] = len(
                validation_results["correlation_stats"]["high_correlation_pairs"]
            )

        for col_stats in validation_results["range_stats"].values():
            if col_stats["range_violations"]:
                summary["features_with_range_violations"] += 1

        return summary


class FeatureCache:
    """Feature caching mechanism."""

    def __init__(self, cache_dir: str) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _generate_cache_key(self, df_hash: str, config: FeatureConfig) -> str:
        """Generate cache key based on data and config."""
        config_str = json.dumps(config.to_dict(), sort_keys=True)
        combined = f"{df_hash}_{config_str}"
        return hashlib.md5(combined.encode(), usedforsecurity=False).hexdigest()

    def _get_data_hash(self, df: pd.DataFrame) -> str:
        """Generate hash of input dataframe."""
        # Use a sample of the data for hashing to be efficient
        sample_size = min(1000, len(df))
        sample_df = df.sample(n=sample_size, random_state=42).sort_index()

        # Create hash based on content
        content_str = f"{len(df)}_{df.columns.tolist()}_{sample_df.to_string()}"
        return hashlib.md5(content_str.encode(), usedforsecurity=False).hexdigest()[:16]

    def get_cached_features(
        self, df: pd.DataFrame, config: FeatureConfig
    ) -> pd.DataFrame | None:
        """Load cached features if available."""
        if not config.enable_caching:
            return None

        try:
            data_hash = self._get_data_hash(df)
            cache_key = self._generate_cache_key(data_hash, config)
            cache_file = self.cache_dir / f"features_{cache_key}.joblib"

            if cache_file.exists():
                logger.info(f"Loading cached features from {cache_file}")
                cached_data = joblib.load(cache_file)
                return cached_data["features"]

        except Exception as e:
            logger.warning(f"Failed to load cached features: {e}")

        return None

    def save_cached_features(
        self,
        df: pd.DataFrame,
        config: FeatureConfig,
        features: pd.DataFrame,
        validation_results: dict,
    ) -> None:
        """Save features to cache."""
        if not config.enable_caching:
            return

        try:
            data_hash = self._get_data_hash(df)
            cache_key = self._generate_cache_key(data_hash, config)
            cache_file = self.cache_dir / f"features_{cache_key}.joblib"

            cache_data = {
                "features": features,
                "config": config.to_dict(),
                "validation_results": validation_results,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data_hash": data_hash,
            }

            joblib.dump(cache_data, cache_file)
            logger.info(f"Saved features to cache: {cache_file}")

        except Exception as e:
            logger.warning(f"Failed to save features to cache: {e}")


class FeatureStore:
    """Main feature store class for orchestrating feature operations."""

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config or FeatureConfig()
        self.validator = FeatureValidator(self.config)
        self.cache = FeatureCache(self.config.cache_dir)

        logger.info(
            f"Initialized FeatureStore with version {self.config.feature_version}"
        )

    def compute_features(
        self, df: pd.DataFrame, use_cache: bool = True
    ) -> tuple[pd.DataFrame, dict]:
        """
        Main feature computation pipeline.

        Args:
            df: Input event log dataframe
            use_cache: Whether to use cached features if available

        Returns:
            Tuple of (features_dataframe, validation_results)
        """
        logger.info(
            f"Computing features for {len(df)} events from {df['userId'].nunique()} users"
        )

        # Check cache first
        if use_cache:
            cached_features = self.cache.get_cached_features(df, self.config)
            if cached_features is not None:
                logger.info("Using cached features")
                # Still validate cached features
                validation_results = self.validator.validate_feature_set(
                    cached_features
                )
                return cached_features, validation_results

        # Set reference date if not provided
        if self.config.reference_date is None:
            if df["ts"].dtype == "int64":
                self.config.reference_date = pd.to_datetime(df["ts"], unit="ms").max()
            else:
                self.config.reference_date = pd.to_datetime(df["ts"]).max()

        feature_sets = []

        # Compute base features
        if self.config.include_base_features:
            logger.info("Computing base features...")
            base_features = compute_all_base_features(df, self.config.activity_windows)
            feature_sets.append(base_features)

        # Compute behavioral features
        if self.config.include_behavioral_features:
            logger.info("Computing behavioral features...")
            behavioral_features = compute_all_behavioral_features(
                df, self.config.trend_windows
            )
            feature_sets.append(behavioral_features)

        # Merge all feature sets
        if len(feature_sets) == 0:
            raise ValueError("No feature sets enabled in configuration")

        combined_features = feature_sets[0]
        for feature_df in feature_sets[1:]:
            combined_features = combined_features.merge(
                feature_df, on="userId", how="outer"
            )

        # Add metadata
        combined_features["feature_computation_timestamp"] = datetime.now(timezone.utc)
        combined_features["feature_version"] = self.config.feature_version

        logger.info(f"Generated {len(combined_features.columns)-1} total features")

        # Validate features
        validation_results = {}
        if self.config.validate_features:
            validation_results = self.validator.validate_feature_set(combined_features)

        # Cache results
        if self.config.enable_caching:
            self.cache.save_cached_features(
                df, self.config, combined_features, validation_results
            )

        return combined_features, validation_results

    def compute_incremental_features(
        self, df: pd.DataFrame, existing_features: pd.DataFrame, cutoff_date: datetime
    ) -> pd.DataFrame:
        """
        Compute features incrementally for new data.

        Args:
            df: Full event log dataframe
            existing_features: Previously computed features
            cutoff_date: Date after which to compute new features

        Returns:
            Updated feature dataframe
        """
        logger.info(f"Computing incremental features from {cutoff_date}")

        # Filter for new/updated data
        if df["ts"].dtype == "int64":
            df["datetime"] = pd.to_datetime(df["ts"], unit="ms")
        else:
            df["datetime"] = pd.to_datetime(df["ts"])

        # Users with activity after cutoff date
        updated_users = df[df["datetime"] >= cutoff_date]["userId"].unique()

        if len(updated_users) == 0:
            logger.info("No users with new activity - returning existing features")
            return existing_features

        logger.info(
            f"Recomputing features for {len(updated_users)} users with new activity"
        )

        # Recompute features for affected users
        updated_user_data = df[df["userId"].isin(updated_users)]
        new_features, _ = self.compute_features(updated_user_data, use_cache=False)

        # Merge with existing features
        updated_features = existing_features[
            ~existing_features["userId"].isin(updated_users)
        ]
        updated_features = pd.concat(
            [updated_features, new_features], ignore_index=True
        )

        return updated_features

    def save_features(
        self, features_df: pd.DataFrame, output_path: str, include_metadata: bool = True
    ) -> None:
        """
        Save features to file with metadata.

        Args:
            features_df: Feature dataframe to save
            output_path: Output file path
            include_metadata: Whether to include computation metadata
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save features
        if output_path.suffix == ".csv":
            features_df.to_csv(output_path, index=False)
        elif output_path.suffix == ".parquet":
            features_df.to_parquet(output_path, index=False)
        else:
            # Default to joblib
            joblib.dump(features_df, output_path)

        logger.info(f"Saved {len(features_df)} feature records to {output_path}")

        # Save metadata if requested
        if include_metadata:
            metadata_path = output_path.with_suffix(".metadata.json")
            metadata = {
                "feature_count": len(features_df.columns) - 1,
                "user_count": len(features_df),
                "feature_version": self.config.feature_version,
                "computation_timestamp": datetime.now(timezone.utc).isoformat(),
                "config": self.config.to_dict(),
                "feature_columns": features_df.columns.tolist(),
            }

            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Saved feature metadata to {metadata_path}")

    def load_features(self, input_path: str) -> pd.DataFrame:
        """
        Load features from file.

        Args:
            input_path: Input file path

        Returns:
            Feature dataframe
        """
        input_path = Path(input_path)

        if input_path.suffix == ".csv":
            features_df = pd.read_csv(input_path)
        elif input_path.suffix == ".parquet":
            features_df = pd.read_parquet(input_path)
        else:
            # Default to joblib
            features_df = joblib.load(input_path)

        logger.info(f"Loaded {len(features_df)} feature records from {input_path}")
        return features_df

    def get_feature_importance_analysis(
        self, features_df: pd.DataFrame, target_column: str = "is_churned"
    ) -> dict:
        """
        Analyze feature importance using correlation and basic statistics.

        Args:
            features_df: Feature dataframe
            target_column: Target variable column name

        Returns:
            Dictionary with feature importance analysis
        """
        if target_column not in features_df.columns:
            logger.warning(f"Target column '{target_column}' not found in features")
            return {}

        # Get numeric features
        numeric_features = features_df.select_dtypes(
            include=["float64", "int64"]
        ).columns
        numeric_features = [
            col for col in numeric_features if col not in ["userId", target_column]
        ]

        importance_analysis = {}

        # Correlation with target
        correlations = {}
        for feature in numeric_features:
            corr = features_df[feature].corr(features_df[target_column])
            if not pd.isna(corr):
                correlations[feature] = abs(corr)

        # Sort by importance
        sorted_correlations = dict(
            sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        )

        importance_analysis = {
            "correlation_importance": sorted_correlations,
            "top_10_features": list(sorted_correlations.keys())[:10],
            "feature_statistics": {},
        }

        # Feature statistics for top features
        for feature in importance_analysis["top_10_features"]:
            importance_analysis["feature_statistics"][feature] = {
                "correlation_with_target": correlations[feature],
                "mean": features_df[feature].mean(),
                "std": features_df[feature].std(),
                "missing_rate": features_df[feature].isnull().mean(),
            }

        logger.info(
            f"Generated feature importance analysis for {len(numeric_features)} features"
        )
        return importance_analysis


if __name__ == "__main__":
    # Example usage
    print("Feature Store Module - Centralized Feature Management")
    print(
        "This module provides comprehensive feature computation, validation, and caching."
    )
    print("Use FeatureStore class to orchestrate all feature operations.")
