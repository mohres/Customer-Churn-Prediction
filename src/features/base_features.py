"""
Base Feature Engineering Module

This module contains core feature engineering functions that transform raw event logs
into meaningful predictive signals for churn prediction. Features include:
- Activity volume metrics (song counts, session counts, active days)
- Engagement metrics (playlist additions, thumbs up/down ratios, skip rates)
- Temporal patterns (usage by hour/day, session duration distributions)

Key functions:
- compute_activity_features(): Basic activity volume metrics
- compute_engagement_features(): User engagement patterns
- compute_temporal_features(): Time-based usage patterns
- compute_all_base_features(): Complete base feature computation
"""

import logging
from datetime import timedelta

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_activity_features(
    df: pd.DataFrame, window_days: list[int] | None = None
) -> pd.DataFrame:
    """
    Compute basic activity volume features for each user.

    Args:
        df: Event log dataframe with columns: userId, ts, page, sessionId
        window_days: List of time windows to compute features for

    Returns:
        DataFrame with activity features per user
    """
    logger.info(f"Computing activity features for {df['userId'].nunique()} users")
    if window_days is None:
        window_days = [7, 14, 30]
    # Convert timestamp to datetime if needed
    df = df.copy()  # Avoid SettingWithCopyWarning
    if df["ts"].dtype == "int64":
        df["datetime"] = pd.to_datetime(df["ts"], unit="ms")
    else:
        df["datetime"] = pd.to_datetime(df["ts"])

    # Get the latest timestamp for reference point
    reference_date = df["datetime"].max()

    features_list = []

    for user_id in df["userId"].unique():
        user_df = df[df["userId"] == user_id].copy()
        user_features = {"userId": user_id}

        # Overall activity metrics (entire period)
        user_features.update(
            {
                "total_events": len(user_df),
                "total_sessions": user_df["sessionId"].nunique(),
                "total_active_days": user_df["datetime"].dt.date.nunique(),
                "activity_span_days": (
                    user_df["datetime"].max() - user_df["datetime"].min()
                ).days
                + 1,
                "avg_events_per_session": len(user_df) / user_df["sessionId"].nunique(),
                "avg_events_per_day": len(user_df)
                / user_df["datetime"].dt.date.nunique(),
            }
        )

        # Music-specific activity
        music_events = user_df[user_df["page"] == "NextSong"]
        user_features.update(
            {
                "total_songs": len(music_events),
                "avg_songs_per_session": (
                    len(music_events) / user_df["sessionId"].nunique()
                    if user_df["sessionId"].nunique() > 0
                    else 0
                ),
                "music_event_ratio": (
                    len(music_events) / len(user_df) if len(user_df) > 0 else 0
                ),
            }
        )

        # Time-windowed features
        for window in window_days:
            window_start = reference_date - timedelta(days=window)
            window_df = user_df[user_df["datetime"] >= window_start]
            window_music = window_df[window_df["page"] == "NextSong"]

            user_features.update(
                {
                    f"events_last_{window}d": len(window_df),
                    f"sessions_last_{window}d": window_df["sessionId"].nunique(),
                    f"active_days_last_{window}d": window_df[
                        "datetime"
                    ].dt.date.nunique(),
                    f"songs_last_{window}d": len(window_music),
                    f"avg_events_per_day_last_{window}d": (
                        len(window_df) / window if window > 0 else 0
                    ),
                    f"session_frequency_last_{window}d": (
                        window_df["sessionId"].nunique() / window if window > 0 else 0
                    ),
                }
            )

        # Recent activity trends
        user_features["days_since_last_activity"] = (
            reference_date - user_df["datetime"].max()
        ).days

        features_list.append(user_features)

    result_df = pd.DataFrame(features_list)
    logger.info(
        f"Generated {len(result_df.columns)-1} activity features for {len(result_df)} users"
    )
    return result_df


def compute_engagement_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute user engagement features based on interaction patterns.

    Args:
        df: Event log dataframe

    Returns:
        DataFrame with engagement features per user
    """
    logger.info("Computing engagement features")

    features_list = []

    for user_id in df["userId"].unique():
        user_df = df[df["userId"] == user_id].copy()
        user_features = {"userId": user_id}

        # Engagement actions
        thumbs_up = len(user_df[user_df["page"] == "Thumbs Up"])
        thumbs_down = len(user_df[user_df["page"] == "Thumbs Down"])
        playlist_adds = len(user_df[user_df["page"] == "Add to Playlist"])
        friend_adds = len(user_df[user_df["page"] == "Add Friend"])

        total_engagement = thumbs_up + thumbs_down + playlist_adds + friend_adds
        total_events = len(user_df)

        user_features.update(
            {
                "thumbs_up_count": thumbs_up,
                "thumbs_down_count": thumbs_down,
                "playlist_adds_count": playlist_adds,
                "friend_adds_count": friend_adds,
                "total_engagement_actions": total_engagement,
                "engagement_rate": (
                    total_engagement / total_events if total_events > 0 else 0
                ),
                "thumbs_up_rate": thumbs_up / total_events if total_events > 0 else 0,
                "thumbs_down_rate": (
                    thumbs_down / total_events if total_events > 0 else 0
                ),
                "playlist_add_rate": (
                    playlist_adds / total_events if total_events > 0 else 0
                ),
            }
        )

        # Engagement ratios and preferences
        total_thumbs = thumbs_up + thumbs_down
        if total_thumbs > 0:
            user_features.update(
                {
                    "thumbs_up_ratio": thumbs_up / total_thumbs,
                    "thumbs_down_ratio": thumbs_down / total_thumbs,
                    "positive_engagement_ratio": thumbs_up / total_thumbs,
                }
            )
        else:
            user_features.update(
                {
                    "thumbs_up_ratio": 0,
                    "thumbs_down_ratio": 0,
                    "positive_engagement_ratio": 0.5,  # neutral when no thumbs activity
                }
            )

        # Page diversity (how many different types of pages user visits)
        unique_pages = user_df["page"].nunique()
        total_page_types = df["page"].nunique()

        user_features.update(
            {
                "page_diversity": unique_pages,
                "page_exploration_ratio": unique_pages / total_page_types,
            }
        )

        # Navigation patterns
        home_visits = len(user_df[user_df["page"] == "Home"])
        settings_visits = len(user_df[user_df["page"] == "Settings"])
        help_visits = len(user_df[user_df["page"] == "Help"])

        user_features.update(
            {
                "home_visits": home_visits,
                "settings_visits": settings_visits,
                "help_visits": help_visits,
                "home_visit_rate": (
                    home_visits / total_events if total_events > 0 else 0
                ),
                "settings_engagement": settings_visits > 0,  # Boolean feature
                "help_seeking_behavior": help_visits > 0,  # Boolean feature
            }
        )

        features_list.append(user_features)

    result_df = pd.DataFrame(features_list)
    logger.info(
        f"Generated {len(result_df.columns)-1} engagement features for {len(result_df)} users"
    )
    return result_df


def compute_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute temporal usage patterns and session-based features.

    Args:
        df: Event log dataframe with timestamp information

    Returns:
        DataFrame with temporal features per user
    """
    logger.info("Computing temporal features")

    # Convert timestamp to datetime if needed
    if df["ts"].dtype == "int64":
        df["datetime"] = pd.to_datetime(df["ts"], unit="ms")
    else:
        df["datetime"] = pd.to_datetime(df["ts"])

    # Add time components
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek  # 0=Monday, 6=Sunday
    df["is_weekend"] = df["day_of_week"].isin([5, 6])

    features_list = []

    for user_id in df["userId"].unique():
        user_df = df[df["userId"] == user_id].copy()
        user_features = {"userId": user_id}

        # Time-of-day patterns
        hour_dist = user_df["hour"].value_counts().sort_index()
        peak_hour = hour_dist.idxmax() if len(hour_dist) > 0 else 12

        # Define time periods
        morning_hours = range(6, 12)
        afternoon_hours = range(12, 18)
        evening_hours = range(18, 24)
        night_hours = list(range(0, 6))

        morning_events = len(user_df[user_df["hour"].isin(morning_hours)])
        afternoon_events = len(user_df[user_df["hour"].isin(afternoon_hours)])
        evening_events = len(user_df[user_df["hour"].isin(evening_hours)])
        night_events = len(user_df[user_df["hour"].isin(night_hours)])

        total_events = len(user_df)

        user_features.update(
            {
                "peak_usage_hour": peak_hour,
                "morning_usage_rate": (
                    morning_events / total_events if total_events > 0 else 0
                ),
                "afternoon_usage_rate": (
                    afternoon_events / total_events if total_events > 0 else 0
                ),
                "evening_usage_rate": (
                    evening_events / total_events if total_events > 0 else 0
                ),
                "night_usage_rate": (
                    night_events / total_events if total_events > 0 else 0
                ),
                "hour_diversity": user_df["hour"].nunique(),
                "usage_time_spread": user_df["hour"].std() if len(user_df) > 1 else 0,
            }
        )

        # Day-of-week patterns
        weekday_events = len(user_df[~user_df["is_weekend"]])
        weekend_events = len(user_df[user_df["is_weekend"]])

        user_features.update(
            {
                "weekday_usage_rate": (
                    weekday_events / total_events if total_events > 0 else 0
                ),
                "weekend_usage_rate": (
                    weekend_events / total_events if total_events > 0 else 0
                ),
                "weekend_preference": weekend_events > weekday_events,
                "day_diversity": user_df["day_of_week"].nunique(),
            }
        )

        # Session patterns
        session_stats = []
        for session_id in user_df["sessionId"].unique():
            session_df = user_df[user_df["sessionId"] == session_id]
            session_duration = (
                session_df["datetime"].max() - session_df["datetime"].min()
            ).total_seconds() / 60  # minutes
            session_stats.append(
                {
                    "duration": session_duration,
                    "events": len(session_df),
                    "songs": len(session_df[session_df["page"] == "NextSong"]),
                }
            )

        session_df_stats = pd.DataFrame(session_stats)

        if len(session_df_stats) > 0:
            user_features.update(
                {
                    "avg_session_duration_min": session_df_stats["duration"].mean(),
                    "median_session_duration_min": session_df_stats[
                        "duration"
                    ].median(),
                    "max_session_duration_min": session_df_stats["duration"].max(),
                    "session_duration_std": session_df_stats["duration"].std(),
                    "avg_session_events": session_df_stats["events"].mean(),
                    "avg_session_songs": session_df_stats["songs"].mean(),
                    "short_sessions_rate": (session_df_stats["duration"] < 5).sum()
                    / len(session_df_stats),  # < 5 min
                    "long_sessions_rate": (session_df_stats["duration"] > 60).sum()
                    / len(session_df_stats),  # > 1 hour
                }
            )
        else:
            # Handle edge case with no sessions
            user_features.update(
                {
                    "avg_session_duration_min": 0,
                    "median_session_duration_min": 0,
                    "max_session_duration_min": 0,
                    "session_duration_std": 0,
                    "avg_session_events": 0,
                    "avg_session_songs": 0,
                    "short_sessions_rate": 0,
                    "long_sessions_rate": 0,
                }
            )

        # Activity consistency
        daily_events = user_df.groupby(user_df["datetime"].dt.date).size()
        user_features.update(
            {
                "daily_activity_std": (
                    daily_events.std() if len(daily_events) > 1 else 0
                ),
                "daily_activity_consistency": (
                    1 - (daily_events.std() / daily_events.mean())
                    if daily_events.mean() > 0
                    else 0
                ),
                "active_days_streak_max": _calculate_max_streak(daily_events.index),
            }
        )

        features_list.append(user_features)

    result_df = pd.DataFrame(features_list)
    logger.info(
        f"Generated {len(result_df.columns)-1} temporal features for {len(result_df)} users"
    )
    return result_df


def _calculate_max_streak(dates: pd.Index) -> int:
    """
    Calculate maximum consecutive days streak.

    Args:
        dates: Index of dates

    Returns:
        Maximum streak length in days
    """
    if len(dates) == 0:
        return 0

    dates_sorted = sorted(dates)
    max_streak = 1
    current_streak = 1

    for i in range(1, len(dates_sorted)):
        if (dates_sorted[i] - dates_sorted[i - 1]).days == 1:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 1

    return max_streak


def compute_subscription_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute subscription-related features.

    Args:
        df: Event log dataframe

    Returns:
        DataFrame with subscription features per user
    """
    logger.info("Computing subscription features")

    features_list = []

    for user_id in df["userId"].unique():
        user_df = df[df["userId"] == user_id].copy()
        user_features = {"userId": user_id}

        # Current subscription level (most recent)
        if "datetime" not in user_df.columns:
            user_df["datetime"] = pd.to_datetime(user_df["ts"], unit="ms")

        latest_entry = user_df.loc[user_df["datetime"].idxmax()]
        current_level = latest_entry["level"]

        user_features["current_subscription_level"] = current_level
        user_features["is_paid_user"] = current_level == "paid"

        # Subscription changes
        level_changes = user_df["level"].nunique()
        user_features["subscription_level_changes"] = (
            level_changes - 1
        )  # -1 because nunique includes the initial level

        # Upgrade/downgrade events
        upgrade_events = len(user_df[user_df["page"] == "Upgrade"])
        downgrade_events = len(user_df[user_df["page"] == "Downgrade"])
        submit_upgrade_events = len(user_df[user_df["page"] == "Submit Upgrade"])
        submit_downgrade_events = len(user_df[user_df["page"] == "Submit Downgrade"])

        user_features.update(
            {
                "upgrade_events": upgrade_events,
                "downgrade_events": downgrade_events,
                "submit_upgrade_events": submit_upgrade_events,
                "submit_downgrade_events": submit_downgrade_events,
                "has_upgrade_activity": upgrade_events > 0 or submit_upgrade_events > 0,
                "has_downgrade_activity": downgrade_events > 0
                or submit_downgrade_events > 0,
                "net_upgrade_intent": upgrade_events - downgrade_events,
            }
        )

        # Ads exposure (for free users)
        ad_events = len(user_df[user_df["page"] == "Roll Advert"])
        total_events = len(user_df)

        user_features.update(
            {
                "ad_exposure_count": ad_events,
                "ad_exposure_rate": ad_events / total_events if total_events > 0 else 0,
            }
        )

        features_list.append(user_features)

    result_df = pd.DataFrame(features_list)
    logger.info(
        f"Generated {len(result_df.columns)-1} subscription features for {len(result_df)} users"
    )
    return result_df


def compute_all_base_features(
    df: pd.DataFrame, window_days: list[int] | None = None
) -> pd.DataFrame:
    """
    Compute all base features and merge them into a single dataframe.

    Args:
        df: Event log dataframe
        window_days: List of time windows for activity features

    Returns:
        DataFrame with all base features per user
    """
    logger.info("Computing all base features")

    if window_days is None:
        window_days = [7, 14, 30]
    # Compute each feature set
    activity_features = compute_activity_features(df, window_days)
    engagement_features = compute_engagement_features(df)
    temporal_features = compute_temporal_features(df)
    subscription_features = compute_subscription_features(df)

    # Merge all features
    base_features = activity_features

    for feature_df in [engagement_features, temporal_features, subscription_features]:
        base_features = base_features.merge(feature_df, on="userId", how="outer")

    logger.info(
        f"Complete base feature set: {len(base_features.columns)-1} features for {len(base_features)} users"
    )

    return base_features


def validate_features(features_df: pd.DataFrame) -> dict:
    """
    Validate computed features for quality and consistency.

    Args:
        features_df: DataFrame with computed features

    Returns:
        Dictionary with validation results
    """
    validation_results = {
        "total_features": len(features_df.columns) - 1,  # -1 for userId
        "total_users": len(features_df),
        "missing_values": {},
        "infinite_values": {},
        "feature_ranges": {},
        "warnings": [],
    }

    for col in features_df.columns:
        if col == "userId":
            continue

        # Check for missing values
        missing_count = features_df[col].isnull().sum()
        if missing_count > 0:
            validation_results["missing_values"][col] = missing_count

        # Check for infinite values
        if features_df[col].dtype in ["float64", "int64"]:
            inf_count = np.isinf(features_df[col]).sum()
            if inf_count > 0:
                validation_results["infinite_values"][col] = inf_count

            # Store feature ranges
            validation_results["feature_ranges"][col] = {
                "min": features_df[col].min(),
                "max": features_df[col].max(),
                "mean": features_df[col].mean(),
            }

    # Generate warnings
    if validation_results["missing_values"]:
        validation_results["warnings"].append(
            f"Missing values found in {len(validation_results['missing_values'])} features"
        )

    if validation_results["infinite_values"]:
        validation_results["warnings"].append(
            f"Infinite values found in {len(validation_results['infinite_values'])} features"
        )

    logger.info(
        f"Feature validation complete: {validation_results['total_features']} features validated"
    )
    return validation_results


if __name__ == "__main__":
    # Example usage - this would typically be called from a script or notebook
    print("Base Features Module - Feature Engineering for Churn Prediction")
    print(
        "This module provides functions to compute activity, engagement, temporal, and subscription features."
    )
    print("Use compute_all_base_features() to generate the complete feature set.")
