"""
Data Preparation Module for Churn Prediction

This module provides utilities for preparing datasets for machine learning training,
combining feature engineering with churn label generation and temporal validation.

Key functions:
- prepare_training_dataset(): Create complete dataset with features and labels
- create_churn_labels(): Generate churn labels based on defined criteria
- add_reference_dates(): Add reference dates for temporal validation
- validate_dataset(): Validate prepared dataset for model training

Usage:
    from models.data_preparation import prepare_training_dataset

    dataset = prepare_training_dataset(events_df)
    train_df, val_df, test_df = split_temporal_dataset(dataset)
"""

import logging
import os
import sys
from datetime import datetime, timedelta

import pandas as pd

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.features import FeatureConfig, FeatureStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_churn_labels(
    events_df: pd.DataFrame,
    reference_date: datetime | None = None,
    inactivity_days: int = 21,
    explicit_churn_events: list[str] | None = None,
) -> pd.DataFrame:
    """
    Create churn labels based on inactivity and explicit churn events.

    Args:
        events_df: Raw event log dataframe
        reference_date: Reference date for churn calculation (defaults to max date)
        inactivity_days: Days of inactivity to consider as churn
        explicit_churn_events: List of page events that indicate explicit churn

    Returns:
        DataFrame with userId, is_churned, churn_reason, and days_since_last_activity
    """
    logger.info("Creating churn labels")

    if explicit_churn_events is None:
        explicit_churn_events = [
            "Cancel",
            "Cancellation Confirmation",
            "Submit Downgrade",
            "Downgrade",
        ]

    # Convert timestamp to datetime if needed
    df = events_df.copy()
    if df["ts"].dtype == "int64":
        df["datetime"] = pd.to_datetime(df["ts"], unit="ms")
    else:
        df["datetime"] = pd.to_datetime(df["ts"])

    # Use max date as reference if not provided
    if reference_date is None:
        reference_date = df["datetime"].max()

    logger.info(f"Using reference date: {reference_date}")
    logger.info(f"Inactivity threshold: {inactivity_days} days")
    logger.info(f"Explicit churn events: {explicit_churn_events}")

    churn_labels = []

    for user_id in df["userId"].unique():
        user_df = df[df["userId"] == user_id]

        # Calculate days since last activity
        last_activity = user_df["datetime"].max()
        days_since_last = (reference_date - last_activity).days

        # Check for explicit churn events
        has_explicit_churn = (
            len(user_df[user_df["page"].isin(explicit_churn_events)]) > 0
        )

        # Determine churn status
        is_churned_inactivity = days_since_last >= inactivity_days
        is_churned = is_churned_inactivity or has_explicit_churn

        # Determine churn reason
        if has_explicit_churn and is_churned_inactivity:
            churn_reason = "explicit_and_inactive"
        elif has_explicit_churn:
            churn_reason = "explicit_churn"
        elif is_churned_inactivity:
            churn_reason = "inactivity"
        else:
            churn_reason = "active"

        churn_labels.append(
            {
                "userId": user_id,
                "is_churned": is_churned,
                "churn_reason": churn_reason,
                "days_since_last_activity": days_since_last,
                "last_activity_date": last_activity,
                "has_explicit_churn": has_explicit_churn,
            }
        )

    labels_df = pd.DataFrame(churn_labels)

    # Log summary statistics
    churn_rate = labels_df["is_churned"].mean()
    churn_reasons = labels_df["churn_reason"].value_counts()

    logger.info(f"Churn labels created for {len(labels_df)} users")
    logger.info(f"Overall churn rate: {churn_rate:.1%}")
    logger.info("Churn reasons breakdown:")
    for reason, count in churn_reasons.items():
        logger.info(f"  {reason}: {count} users ({count/len(labels_df):.1%})")

    return labels_df


def add_reference_dates(
    events_df: pd.DataFrame, method: str = "sliding_window", window_days: int = 30
) -> pd.DataFrame:
    """
    Add reference dates for temporal feature computation and validation.

    Args:
        events_df: Event log dataframe
        method: Method for reference date assignment ('sliding_window' or 'fixed_date')
        window_days: Days before max date for feature computation window

    Returns:
        DataFrame with reference_date column added
    """
    logger.info(f"Adding reference dates using method: {method}")

    df = events_df.copy()
    if df["ts"].dtype == "int64":
        df["datetime"] = pd.to_datetime(df["ts"], unit="ms")
    else:
        df["datetime"] = pd.to_datetime(df["ts"])

    max_date = df["datetime"].max()

    if method == "fixed_date":
        # Use fixed reference date (max_date - window_days)
        reference_date = max_date - timedelta(days=window_days)
        df["reference_date"] = reference_date

    elif method == "sliding_window":
        # Use sliding window approach - reference date is the latest date
        df["reference_date"] = max_date

    else:
        raise ValueError(f"Unknown reference date method: {method}")

    logger.info(
        f"Reference date range: {df['reference_date'].min()} to {df['reference_date'].max()}"
    )

    return df


def prepare_training_dataset(
    events_df: pd.DataFrame,
    feature_config: FeatureConfig | None = None,
    reference_date: datetime | None = None,
    churn_inactivity_days: int = 21,
) -> pd.DataFrame:
    """
    Prepare complete training dataset with features and churn labels.

    Args:
        events_df: Raw event log dataframe
        feature_config: Configuration for feature computation
        reference_date: Reference date for churn and feature calculation
        churn_inactivity_days: Days of inactivity to consider as churn

    Returns:
        DataFrame with features, churn labels, and metadata
    """
    logger.info("Preparing complete training dataset")
    logger.info(
        f"Input data: {len(events_df)} events from {events_df['userId'].nunique()} users"
    )

    # Default feature configuration
    if feature_config is None:
        feature_config = FeatureConfig(
            activity_windows=[7, 14, 30],
            trend_windows=[7, 14, 21],
            include_base_features=True,
            include_behavioral_features=True,
            enable_caching=True,
            feature_version="baseline_v1.0",
        )

    # Add reference dates for temporal consistency
    events_with_ref = add_reference_dates(events_df, method="sliding_window")

    # Use specified reference date or max date from data
    if reference_date is None:
        reference_date = events_with_ref["datetime"].max()

    # Create churn labels
    churn_labels = create_churn_labels(
        events_df, reference_date=reference_date, inactivity_days=churn_inactivity_days
    )

    # Compute features using the feature store
    logger.info("Computing features using FeatureStore")
    feature_store = FeatureStore(feature_config)
    features_df, validation_results = feature_store.compute_features(events_with_ref)

    # Add reference_date back to features (feature store doesn't preserve it)
    # Use a single reference date per user (the max date from the events)
    user_reference_dates = (
        events_with_ref.groupby("userId")["reference_date"].first().reset_index()
    )
    features_with_ref = features_df.merge(user_reference_dates, on="userId", how="left")

    # Merge features with churn labels
    dataset = features_with_ref.merge(churn_labels, on="userId", how="inner")

    # Add dataset metadata
    dataset["dataset_reference_date"] = reference_date
    dataset["churn_inactivity_threshold"] = churn_inactivity_days

    # Validation summary
    logger.info("Dataset preparation completed")
    logger.info(
        f"Final dataset: {len(dataset)} users with {len(dataset.columns)} total columns"
    )
    logger.info(
        f"Feature columns: {len([c for c in dataset.columns if c not in ['userId', 'is_churned', 'churn_reason', 'days_since_last_activity', 'last_activity_date', 'has_explicit_churn', 'reference_date', 'computation_date', 'feature_version', 'dataset_reference_date', 'churn_inactivity_threshold']])}"
    )
    logger.info(f"Churn rate: {dataset['is_churned'].mean():.1%}")

    if validation_results.get("warnings"):
        logger.warning(
            f"Feature validation warnings: {len(validation_results['warnings'])}"
        )

    return dataset


def validate_dataset(dataset_df: pd.DataFrame) -> dict:
    """
    Validate the prepared dataset for model training.

    Args:
        dataset_df: Complete dataset with features and labels

    Returns:
        Dictionary with validation results
    """
    logger.info("Validating prepared dataset")

    validation_results = {
        "total_users": len(dataset_df),
        "total_columns": len(dataset_df.columns),
        "missing_values": {},
        "data_quality_issues": [],
        "target_distribution": {},
        "feature_statistics": {},
        "passed": True,
    }

    # Check for required columns
    required_cols = ["userId", "is_churned"]
    missing_required = [col for col in required_cols if col not in dataset_df.columns]
    if missing_required:
        validation_results["data_quality_issues"].append(
            f"Missing required columns: {missing_required}"
        )
        validation_results["passed"] = False

    # Check target distribution
    if "is_churned" in dataset_df.columns:
        target_dist = dataset_df["is_churned"].value_counts()
        validation_results["target_distribution"] = target_dist.to_dict()

        churn_rate = dataset_df["is_churned"].mean()
        if churn_rate < 0.05 or churn_rate > 0.95:
            validation_results["data_quality_issues"].append(
                f"Extreme class imbalance: {churn_rate:.1%} churn rate"
            )

    # Check for missing values in features
    feature_cols = [
        col
        for col in dataset_df.columns
        if col
        not in [
            "userId",
            "is_churned",
            "churn_reason",
            "days_since_last_activity",
            "last_activity_date",
            "has_explicit_churn",
            "reference_date",
            "computation_date",
            "feature_version",
            "dataset_reference_date",
            "churn_inactivity_threshold",
        ]
    ]

    for col in feature_cols:
        missing_count = dataset_df[col].isnull().sum()
        if missing_count > 0:
            validation_results["missing_values"][col] = missing_count

        # Basic feature statistics
        if dataset_df[col].dtype in ["float64", "int64"]:
            validation_results["feature_statistics"][col] = {
                "min": float(dataset_df[col].min()),
                "max": float(dataset_df[col].max()),
                "mean": float(dataset_df[col].mean()),
                "std": float(dataset_df[col].std()),
            }

    # Check for duplicate users
    duplicate_users = dataset_df["userId"].duplicated().sum()
    if duplicate_users > 0:
        validation_results["data_quality_issues"].append(
            f"Found {duplicate_users} duplicate users"
        )
        validation_results["passed"] = False

    # Summary
    n_feature_cols = len(feature_cols)
    n_missing_features = len(validation_results["missing_values"])
    n_quality_issues = len(validation_results["data_quality_issues"])

    logger.info("Dataset validation completed:")
    logger.info(f"  Users: {validation_results['total_users']}")
    logger.info(f"  Feature columns: {n_feature_cols}")
    logger.info(f"  Features with missing values: {n_missing_features}")
    logger.info(f"  Data quality issues: {n_quality_issues}")
    logger.info(f"  Validation passed: {validation_results['passed']}")

    return validation_results


def split_temporal_dataset(
    dataset_df: pd.DataFrame,
    reference_date_col: str = "reference_date",
    test_days: int = 14,
    val_days: int = 14,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into temporal train/validation/test sets.

    Args:
        dataset_df: Complete dataset with reference dates
        reference_date_col: Column name containing reference dates
        test_days: Days to reserve for test set (most recent)
        val_days: Days to reserve for validation set

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    logger.info("Creating temporal train/validation/test splits")

    # Ensure reference date is datetime
    if dataset_df[reference_date_col].dtype == "object":
        dataset_df[reference_date_col] = pd.to_datetime(dataset_df[reference_date_col])

    # Calculate split dates
    max_date = dataset_df[reference_date_col].max()
    test_start = max_date - timedelta(days=test_days)
    val_start = test_start - timedelta(days=val_days)

    # Create splits
    train_df = dataset_df[dataset_df[reference_date_col] < val_start].copy()
    val_df = dataset_df[
        (dataset_df[reference_date_col] >= val_start)
        & (dataset_df[reference_date_col] < test_start)
    ].copy()
    test_df = dataset_df[dataset_df[reference_date_col] >= test_start].copy()

    logger.info("Temporal splits created:")
    logger.info(f"  Train: {len(train_df)} users (until {val_start.date()})")
    logger.info(
        f"  Validation: {len(val_df)} users ({val_start.date()} to {test_start.date()})"
    )
    logger.info(f"  Test: {len(test_df)} users (from {test_start.date()})")

    # Check for empty splits
    if len(train_df) == 0:
        logger.warning("Training set is empty! Check your temporal split parameters.")
    if len(val_df) == 0:
        logger.warning("Validation set is empty! Check your temporal split parameters.")
    if len(test_df) == 0:
        logger.warning("Test set is empty! Check your temporal split parameters.")

    return train_df, val_df, test_df


if __name__ == "__main__":
    print("Data Preparation Module for Churn Prediction")
    print("This module prepares complete datasets for machine learning training.")
    print(
        "Use prepare_training_dataset() to create feature-rich datasets with churn labels."
    )
