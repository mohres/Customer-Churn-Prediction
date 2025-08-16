"""
Behavioral Feature Engineering Module

This module computes advanced behavioral signals that capture subtle user patterns
indicative of churn risk. Features include:
- Content preferences (genre diversity, artist loyalty, discovery rates)
- Usage trends (activity trend analysis, seasonality detection)
- Interaction depth (session complexity, engagement evolution)
- Risk indicators (churning behavior patterns)

Key functions:
- compute_content_preference_features(): Music content and preference patterns
- compute_usage_trend_features(): Temporal usage trend analysis
- compute_interaction_depth_features(): Deep engagement analysis
- compute_risk_indicator_features(): Churn risk behavioral signals
- compute_all_behavioral_features(): Complete behavioral feature computation
"""

import logging
from collections import Counter
from datetime import timedelta

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_content_preference_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute content preference and music consumption pattern features.

    Args:
        df: Event log dataframe with music event information

    Returns:
        DataFrame with content preference features per user
    """
    logger.info("Computing content preference features")

    features_list = []

    # Get music events only
    music_df = df[df["page"] == "NextSong"].copy()

    for user_id in df["userId"].unique():

        user_music = music_df[music_df["userId"] == user_id].copy()
        user_features = {"userId": user_id}

        if len(user_music) == 0:
            # User with no music events
            user_features.update(
                {
                    "unique_artists": 0,
                    "unique_songs": 0,
                    "artist_diversity": 0,
                    "song_diversity": 0,
                    "avg_song_length": 0,
                    "total_listening_time": 0,
                    "artist_loyalty_score": 0,
                    "top_artist_dominance": 0,
                    "song_repetition_rate": 0,
                    "music_discovery_rate": 0,
                    "has_music_activity": False,
                }
            )
        else:
            # Artists and songs analysis
            unique_artists = user_music["artist"].nunique()
            unique_songs = user_music["song"].nunique()
            total_songs = len(user_music)

            user_features.update(
                {
                    "unique_artists": unique_artists,
                    "unique_songs": unique_songs,
                    "artist_diversity": (
                        unique_artists / total_songs if total_songs > 0 else 0
                    ),
                    "song_diversity": (
                        unique_songs / total_songs if total_songs > 0 else 0
                    ),
                    "has_music_activity": True,
                }
            )

            # Song length and listening time analysis
            valid_lengths = user_music["length"].dropna()
            if len(valid_lengths) > 0:
                avg_song_length = valid_lengths.mean()
                total_listening_time = valid_lengths.sum() / 60  # Convert to minutes

                user_features.update(
                    {
                        "avg_song_length": avg_song_length,
                        "total_listening_time": total_listening_time,
                        "song_length_std": valid_lengths.std(),
                        "short_song_preference": (
                            valid_lengths < 180
                        ).mean(),  # < 3 minutes
                        "long_song_preference": (
                            valid_lengths > 300
                        ).mean(),  # > 5 minutes
                    }
                )
            else:
                user_features.update(
                    {
                        "avg_song_length": 0,
                        "total_listening_time": 0,
                        "song_length_std": 0,
                        "short_song_preference": 0,
                        "long_song_preference": 0,
                    }
                )

            # Artist loyalty analysis
            artist_counts = user_music["artist"].value_counts()
            top_artist_plays = artist_counts.iloc[0] if len(artist_counts) > 0 else 0

            user_features.update(
                {
                    "artist_loyalty_score": (
                        top_artist_plays / total_songs if total_songs > 0 else 0
                    ),
                    "top_artist_dominance": (
                        top_artist_plays / total_songs if total_songs > 0 else 0
                    ),
                    "artist_concentration": _calculate_gini_coefficient(
                        artist_counts.values
                    ),
                }
            )

            # Song repetition analysis
            song_counts = user_music["song"].value_counts()
            repeated_songs = (song_counts > 1).sum()

            user_features.update(
                {
                    "song_repetition_rate": (
                        repeated_songs / unique_songs if unique_songs > 0 else 0
                    ),
                    "most_played_song_count": (
                        song_counts.iloc[0] if len(song_counts) > 0 else 0
                    ),
                    "song_concentration": _calculate_gini_coefficient(
                        song_counts.values
                    ),
                }
            )

            # Music discovery patterns (based on temporal ordering)
            if "datetime" not in user_music.columns:
                user_music["datetime"] = pd.to_datetime(user_music["ts"], unit="ms")

            user_music_sorted = user_music.sort_values("datetime")

            # Calculate discovery rate as rate of new artists/songs over time
            seen_artists = set()
            seen_songs = set()
            discovery_points = []

            for _, row in user_music_sorted.iterrows():
                artist_new = row["artist"] not in seen_artists
                song_new = row["song"] not in seen_songs

                seen_artists.add(row["artist"])
                seen_songs.add(row["song"])

                discovery_points.append(
                    {
                        "new_artist": artist_new,
                        "new_song": song_new,
                        "discovery": artist_new or song_new,
                    }
                )

            discovery_df = pd.DataFrame(discovery_points)

            user_features.update(
                {
                    "music_discovery_rate": (
                        discovery_df["discovery"].mean() if len(discovery_df) > 0 else 0
                    ),
                    "new_artist_rate": (
                        discovery_df["new_artist"].mean()
                        if len(discovery_df) > 0
                        else 0
                    ),
                    "new_song_rate": (
                        discovery_df["new_song"].mean() if len(discovery_df) > 0 else 0
                    ),
                }
            )

        features_list.append(user_features)

    result_df = pd.DataFrame(features_list)
    logger.info(
        f"Generated {len(result_df.columns)-1} content preference features for {len(result_df)} users"
    )
    return result_df


def compute_usage_trend_features(
    df: pd.DataFrame, trend_windows: list[int] | None = None
) -> pd.DataFrame:
    """
    Compute usage trend and activity evolution features.

    Args:
        df: Event log dataframe
        trend_windows: List of windows (in days) to compute trends over

    Returns:
        DataFrame with usage trend features per user
    """
    logger.info("Computing usage trend features")
    if trend_windows is None:
        trend_windows = [7, 14, 21]
    if df["ts"].dtype == "int64":
        df["datetime"] = pd.to_datetime(df["ts"], unit="ms")
    else:
        df["datetime"] = pd.to_datetime(df["ts"])

    features_list = []
    reference_date = df["datetime"].max()

    for user_id in df["userId"].unique():
        user_df = df[df["userId"] == user_id].copy()
        user_features = {"userId": user_id}

        # Daily activity over time
        daily_activity = user_df.groupby(user_df["datetime"].dt.date).size()

        if len(daily_activity) < 3:
            # Not enough data for trend analysis
            for window in trend_windows:
                user_features.update(
                    {
                        f"activity_trend_{window}d": 0,
                        f"activity_trend_strength_{window}d": 0,
                        f"activity_volatility_{window}d": 0,
                    }
                )
            user_features.update(
                {
                    "overall_activity_trend": 0,
                    "recent_activity_change": 0,
                    "activity_decline_rate": 0,
                    "days_since_peak_activity": 0,
                    "activity_consistency_score": 0,
                }
            )
        else:
            # Compute trends for different windows
            for window in trend_windows:
                window_start = reference_date - timedelta(days=window)
                window_activity = daily_activity[
                    daily_activity.index >= window_start.date()
                ]

                if len(window_activity) >= 3:
                    # Linear trend analysis
                    x = np.arange(len(window_activity))
                    y = window_activity.values
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

                    user_features.update(
                        {
                            f"activity_trend_{window}d": slope,
                            f"activity_trend_strength_{window}d": abs(r_value),
                            f"activity_volatility_{window}d": (
                                np.std(y) / np.mean(y)
                                if np.mean(y) > 0
                                and not np.isnan(np.mean(y))
                                and not np.isnan(np.std(y))
                                else 0
                            ),
                        }
                    )
                else:
                    user_features.update(
                        {
                            f"activity_trend_{window}d": 0,
                            f"activity_trend_strength_{window}d": 0,
                            f"activity_volatility_{window}d": 0,
                        }
                    )

            # Overall trend analysis
            x = np.arange(len(daily_activity))
            y = daily_activity.values
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

            user_features.update(
                {
                    "overall_activity_trend": slope,
                    "overall_trend_strength": abs(r_value),
                    "overall_activity_volatility": (
                        np.std(y) / np.mean(y)
                        if np.mean(y) > 0
                        and not np.isnan(np.mean(y))
                        and not np.isnan(np.std(y))
                        else 0
                    ),
                }
            )

            # Recent vs historical comparison
            mid_point = len(daily_activity) // 2
            if mid_point > 0:
                recent_avg = daily_activity.iloc[mid_point:].mean()
                historical_avg = daily_activity.iloc[:mid_point].mean()
                activity_change = (
                    (recent_avg - historical_avg) / historical_avg
                    if historical_avg > 0
                    else 0
                )

                user_features["recent_activity_change"] = activity_change
            else:
                user_features["recent_activity_change"] = 0

            # Activity decline analysis
            peak_date = daily_activity.idxmax()
            days_since_peak = (reference_date.date() - peak_date).days

            # Calculate decline rate from peak
            activity_after_peak = daily_activity[daily_activity.index >= peak_date]
            if len(activity_after_peak) > 1:
                decline_slope, _, _, _, _ = stats.linregress(
                    np.arange(len(activity_after_peak)), activity_after_peak.values
                )
                user_features["activity_decline_rate"] = min(
                    0, decline_slope
                )  # Only negative slopes
            else:
                user_features["activity_decline_rate"] = 0

            user_features["days_since_peak_activity"] = days_since_peak

            # Activity consistency
            # Measures how consistent daily activity is (lower CV indicates more consistency)
            if np.mean(y) > 0 and not np.isnan(np.mean(y)) and not np.isnan(np.std(y)):
                activity_cv = np.std(y) / np.mean(y)
            else:
                activity_cv = 1
            user_features["activity_consistency_score"] = 1 / (
                1 + activity_cv
            )  # Normalize to [0,1]

        # Session-level trend analysis
        if "sessionId" in user_df.columns:
            session_stats = []
            for session_id in user_df["sessionId"].unique():
                session_df = user_df[user_df["sessionId"] == session_id]
                session_date = session_df["datetime"].min().date()
                session_stats.append(
                    {
                        "date": session_date,
                        "events": len(session_df),
                        "duration": (
                            session_df["datetime"].max() - session_df["datetime"].min()
                        ).total_seconds()
                        / 60,
                    }
                )

            if len(session_stats) >= 3:
                session_df_trends = pd.DataFrame(session_stats).sort_values("date")

                # Session frequency trend
                daily_sessions = session_df_trends.groupby("date").size()
                if len(daily_sessions) >= 3:
                    x = np.arange(len(daily_sessions))
                    slope, _, r_value, _, _ = stats.linregress(x, daily_sessions.values)
                    user_features.update(
                        {
                            "session_frequency_trend": slope,
                            "session_frequency_trend_strength": abs(r_value),
                        }
                    )
                else:
                    user_features.update(
                        {
                            "session_frequency_trend": 0,
                            "session_frequency_trend_strength": 0,
                        }
                    )

                # Session duration trend
                if len(session_df_trends) >= 3:
                    x = np.arange(len(session_df_trends))
                    slope, _, r_value, _, _ = stats.linregress(
                        x, session_df_trends["duration"].values
                    )
                    user_features.update(
                        {
                            "session_duration_trend": slope,
                            "session_duration_trend_strength": abs(r_value),
                        }
                    )
                else:
                    user_features.update(
                        {
                            "session_duration_trend": 0,
                            "session_duration_trend_strength": 0,
                        }
                    )
            else:
                user_features.update(
                    {
                        "session_frequency_trend": 0,
                        "session_frequency_trend_strength": 0,
                        "session_duration_trend": 0,
                        "session_duration_trend_strength": 0,
                    }
                )

        features_list.append(user_features)

    result_df = pd.DataFrame(features_list)
    logger.info(
        f"Generated {len(result_df.columns)-1} usage trend features for {len(result_df)} users"
    )
    return result_df


def compute_interaction_depth_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute deep interaction and engagement complexity features.

    Args:
        df: Event log dataframe

    Returns:
        DataFrame with interaction depth features per user
    """
    logger.info("Computing interaction depth features")

    features_list = []

    for user_id in df["userId"].unique():
        user_df = df[df["userId"] == user_id].copy()
        user_features = {"userId": user_id}

        # Session complexity analysis
        if "sessionId" in user_df.columns:
            session_complexities = []

            for session_id in user_df["sessionId"].unique():
                session_df = user_df[user_df["sessionId"] == session_id]

                # Complexity metrics for each session
                unique_pages = session_df["page"].nunique()
                total_events = len(session_df)

                # Calculate page transition entropy (measure of navigation complexity)
                page_sequence = session_df["page"].tolist()
                if len(page_sequence) > 1:
                    transitions = [
                        (page_sequence[i], page_sequence[i + 1])
                        for i in range(len(page_sequence) - 1)
                    ]
                    transition_entropy = _calculate_entropy(transitions)
                else:
                    transition_entropy = 0

                session_complexities.append(
                    {
                        "unique_pages": unique_pages,
                        "page_diversity_ratio": (
                            unique_pages / total_events if total_events > 0 else 0
                        ),
                        "transition_entropy": transition_entropy,
                        "events_count": total_events,
                    }
                )

            complexity_df = pd.DataFrame(session_complexities)

            if len(complexity_df) > 0:
                user_features.update(
                    {
                        "avg_session_page_diversity": complexity_df[
                            "page_diversity_ratio"
                        ].mean(),
                        "avg_session_transition_entropy": complexity_df[
                            "transition_entropy"
                        ].mean(),
                        "max_session_complexity": complexity_df["unique_pages"].max(),
                        "session_complexity_std": complexity_df["unique_pages"].std(),
                        "complex_session_rate": (
                            complexity_df["unique_pages"] >= 5
                        ).mean(),
                    }
                )
            else:
                user_features.update(
                    {
                        "avg_session_page_diversity": 0,
                        "avg_session_transition_entropy": 0,
                        "max_session_complexity": 0,
                        "session_complexity_std": 0,
                        "complex_session_rate": 0,
                    }
                )
        else:
            user_features.update(
                {
                    "avg_session_page_diversity": 0,
                    "avg_session_transition_entropy": 0,
                    "max_session_complexity": 0,
                    "session_complexity_std": 0,
                    "complex_session_rate": 0,
                }
            )

        # Engagement evolution analysis
        if "datetime" not in user_df.columns:
            user_df["datetime"] = pd.to_datetime(user_df["ts"], unit="ms")

        user_df_sorted = user_df.sort_values("datetime")

        # Analyze engagement actions over time
        engagement_actions = [
            "Thumbs Up",
            "Thumbs Down",
            "Add to Playlist",
            "Add Friend",
        ]
        engagement_events = user_df_sorted[
            user_df_sorted["page"].isin(engagement_actions)
        ]

        if len(engagement_events) > 0:
            # Calculate engagement rate evolution
            total_events = len(user_df_sorted)
            window_size = max(100, total_events // 10)  # Adaptive window size

            engagement_rates = []
            for i in range(0, len(user_df_sorted), window_size):
                window_df = user_df_sorted.iloc[i : i + window_size]
                window_engagement = len(
                    window_df[window_df["page"].isin(engagement_actions)]
                )
                engagement_rate = (
                    window_engagement / len(window_df) if len(window_df) > 0 else 0
                )
                engagement_rates.append(engagement_rate)

            if len(engagement_rates) >= 2:
                # Engagement trend
                x = np.arange(len(engagement_rates))
                slope, _, r_value, _, _ = stats.linregress(x, engagement_rates)

                user_features.update(
                    {
                        "engagement_evolution_trend": slope,
                        "engagement_evolution_strength": abs(r_value),
                        "engagement_rate_volatility": np.std(engagement_rates),
                    }
                )
            else:
                user_features.update(
                    {
                        "engagement_evolution_trend": 0,
                        "engagement_evolution_strength": 0,
                        "engagement_rate_volatility": 0,
                    }
                )

            # Early vs late engagement comparison
            mid_point = len(engagement_events) // 2
            first_half_size = len(user_df_sorted) // 2
            second_half_size = len(user_df_sorted) - first_half_size

            if mid_point > 0 and first_half_size > 0 and second_half_size > 0:
                early_engagement_rate = (
                    len(engagement_events[:mid_point]) / first_half_size
                )
                late_engagement_rate = (
                    len(engagement_events[mid_point:]) / second_half_size
                )

                user_features["engagement_rate_change"] = (
                    late_engagement_rate - early_engagement_rate
                )
            else:
                user_features["engagement_rate_change"] = 0
        else:
            user_features.update(
                {
                    "engagement_evolution_trend": 0,
                    "engagement_evolution_strength": 0,
                    "engagement_rate_volatility": 0,
                    "engagement_rate_change": 0,
                }
            )

        # Feature interaction analysis
        page_types = user_df["page"].unique()
        feature_interactions = []

        for page_type in page_types:
            page_events = user_df[user_df["page"] == page_type]
            feature_interactions.append(
                {
                    "page": page_type,
                    "count": len(page_events),
                    "rate": len(page_events) / len(user_df),
                }
            )

        interaction_df = pd.DataFrame(feature_interactions)

        # Calculate feature usage entropy (diversity of feature usage)
        if len(interaction_df) > 0:
            feature_entropy = _calculate_entropy(interaction_df["count"].tolist())
            user_features["feature_usage_entropy"] = feature_entropy
            user_features["feature_usage_concentration"] = _calculate_gini_coefficient(
                interaction_df["count"].values
            )
        else:
            user_features["feature_usage_entropy"] = 0
            user_features["feature_usage_concentration"] = 1

        features_list.append(user_features)

    result_df = pd.DataFrame(features_list)
    logger.info(
        f"Generated {len(result_df.columns)-1} interaction depth features for {len(result_df)} users"
    )
    return result_df


def compute_risk_indicator_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute churn risk indicator features based on behavioral patterns.

    Args:
        df: Event log dataframe

    Returns:
        DataFrame with risk indicator features per user
    """
    logger.info("Computing risk indicator features")

    features_list = []

    # Define risk-associated events
    explicit_churn_events = ["Cancel", "Cancellation Confirmation", "Submit Downgrade"]
    negative_events = ["Error", "Thumbs Down"]
    support_events = ["Help", "Settings"]

    for user_id in df["userId"].unique():
        user_df = df[df["userId"] == user_id].copy()
        user_features = {"userId": user_id}

        total_events = len(user_df)

        # Explicit churn risk signals
        churn_event_count = len(user_df[user_df["page"].isin(explicit_churn_events)])
        user_features.update(
            {
                "explicit_churn_events": churn_event_count,
                "explicit_churn_rate": (
                    churn_event_count / total_events if total_events > 0 else 0
                ),
                "has_explicit_churn_signal": churn_event_count > 0,
            }
        )

        # Negative sentiment indicators
        negative_event_count = len(user_df[user_df["page"].isin(negative_events)])
        error_count = len(user_df[user_df["page"] == "Error"])
        thumbs_down_count = len(user_df[user_df["page"] == "Thumbs Down"])

        user_features.update(
            {
                "negative_events": negative_event_count,
                "negative_event_rate": (
                    negative_event_count / total_events if total_events > 0 else 0
                ),
                "error_events": error_count,
                "error_rate": error_count / total_events if total_events > 0 else 0,
                "thumbs_down_events": thumbs_down_count,
                "dissatisfaction_indicator": thumbs_down_count > 0,
            }
        )

        # Support-seeking behavior
        support_event_count = len(user_df[user_df["page"].isin(support_events)])
        help_count = len(user_df[user_df["page"] == "Help"])
        settings_count = len(user_df[user_df["page"] == "Settings"])

        user_features.update(
            {
                "support_events": support_event_count,
                "support_seeking_rate": (
                    support_event_count / total_events if total_events > 0 else 0
                ),
                "help_events": help_count,
                "settings_events": settings_count,
                "has_support_seeking_behavior": support_event_count > 0,
            }
        )

        # Subscription-related risk indicators
        downgrade_events = len(user_df[user_df["page"] == "Downgrade"])
        downgrade_submit_events = len(user_df[user_df["page"] == "Submit Downgrade"])
        upgrade_events = len(user_df[user_df["page"] == "Upgrade"])

        user_features.update(
            {
                "downgrade_events": downgrade_events,
                "downgrade_submit_events": downgrade_submit_events,
                "upgrade_events": upgrade_events,
                "subscription_instability": downgrade_events
                + downgrade_submit_events
                + upgrade_events,
                "downgrade_risk": downgrade_events > 0 or downgrade_submit_events > 0,
                "net_subscription_intent": upgrade_events
                - (downgrade_events + downgrade_submit_events),
            }
        )

        # Logout pattern analysis (potential disengagement)
        logout_count = len(user_df[user_df["page"] == "Logout"])
        login_count = len(user_df[user_df["page"] == "Login"])

        user_features.update(
            {
                "logout_events": logout_count,
                "login_events": login_count,
                "logout_rate": logout_count / total_events if total_events > 0 else 0,
                "logout_login_ratio": (
                    logout_count / login_count if login_count > 0 else 0
                ),
                "frequent_logout_pattern": logout_count
                > np.percentile(
                    df.groupby("userId")["page"].apply(lambda x: (x == "Logout").sum()),
                    75,
                ),
            }
        )

        # Activity pattern risk indicators
        if "datetime" not in user_df.columns:
            user_df["datetime"] = pd.to_datetime(user_df["ts"], unit="ms")

        # Declining activity pattern
        recent_days = 7
        reference_date = user_df["datetime"].max()
        recent_cutoff = reference_date - timedelta(days=recent_days)

        recent_events = len(user_df[user_df["datetime"] >= recent_cutoff])
        total_span_days = (
            user_df["datetime"].max() - user_df["datetime"].min()
        ).days + 1
        expected_recent_events = (recent_days / total_span_days) * total_events

        user_features.update(
            {
                "recent_activity_deficit": max(
                    0, expected_recent_events - recent_events
                ),
                "recent_activity_ratio": (
                    recent_events / expected_recent_events
                    if expected_recent_events > 0
                    else 1
                ),
                "activity_decline_risk": recent_events
                < expected_recent_events * 0.5,  # 50% below expected
            }
        )

        # Session abandonment patterns
        if "sessionId" in user_df.columns:
            session_lengths = user_df.groupby("sessionId").size()
            very_short_sessions = (
                session_lengths <= 2
            ).sum()  # Sessions with 2 or fewer events
            total_sessions = len(session_lengths)

            user_features.update(
                {
                    "short_session_count": very_short_sessions,
                    "short_session_rate": (
                        very_short_sessions / total_sessions
                        if total_sessions > 0
                        else 0
                    ),
                    "session_abandonment_risk": (
                        very_short_sessions / total_sessions > 0.3
                        if total_sessions > 0
                        else False
                    ),
                }
            )
        else:
            user_features.update(
                {
                    "short_session_count": 0,
                    "short_session_rate": 0,
                    "session_abandonment_risk": False,
                }
            )

        # Composite risk score
        risk_factors = [
            user_features["explicit_churn_rate"] * 5,  # Weight explicit churn highly
            user_features["negative_event_rate"] * 2,
            user_features["support_seeking_rate"] * 1.5,
            user_features["logout_rate"] * 1.5,
            (1 - user_features["recent_activity_ratio"])
            * 2,  # Inverse of activity ratio
            user_features["short_session_rate"] * 1,
        ]

        user_features["composite_risk_score"] = sum(risk_factors)
        user_features["high_risk_flag"] = user_features["composite_risk_score"] > 1.0

        features_list.append(user_features)

    result_df = pd.DataFrame(features_list)
    logger.info(
        f"Generated {len(result_df.columns)-1} risk indicator features for {len(result_df)} users"
    )
    return result_df


def compute_all_behavioral_features(
    df: pd.DataFrame, trend_windows: list[int] | None = None
) -> pd.DataFrame:
    """
    Compute all behavioral features and merge them into a single dataframe.

    Args:
        df: Event log dataframe
        trend_windows: List of windows for trend analysis

    Returns:
        DataFrame with all behavioral features per user
    """
    logger.info("Computing all behavioral features")
    if trend_windows is None:
        trend_windows = [7, 14, 21]
    # Compute each behavioral feature set
    content_features = compute_content_preference_features(df)
    trend_features = compute_usage_trend_features(df, trend_windows)
    interaction_features = compute_interaction_depth_features(df)
    risk_features = compute_risk_indicator_features(df)

    # Merge all features
    behavioral_features = content_features

    for feature_df in [trend_features, interaction_features, risk_features]:
        behavioral_features = behavioral_features.merge(
            feature_df, on="userId", how="outer"
        )

    logger.info(
        f"Complete behavioral feature set: {len(behavioral_features.columns)-1} features for {len(behavioral_features)} users"
    )

    return behavioral_features


def _calculate_gini_coefficient(values: np.ndarray) -> float:
    """Calculate Gini coefficient for inequality measurement."""
    if len(values) == 0:
        return 0

    values = np.array(values, dtype=float)
    values = values[values >= 0]  # Remove negative values

    if len(values) == 0 or np.sum(values) == 0:
        return 0

    values = np.sort(values)
    n = len(values)

    # Calculate Gini coefficient
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * values)) / (n * np.sum(values)) - (n + 1) / n

    return gini


def _calculate_entropy(items: list) -> float:
    """Calculate Shannon entropy of a list of items."""
    if not items:
        return 0

    # Count occurrences
    counts = Counter(items)
    total = len(items)

    # Calculate entropy
    entropy = 0
    for count in counts.values():
        if count > 0:
            probability = count / total
            entropy -= probability * np.log2(probability)

    return entropy


if __name__ == "__main__":
    # Example usage
    print(
        "Behavioral Features Module - Advanced Feature Engineering for Churn Prediction"
    )
    print(
        "This module provides functions to compute content preferences, usage trends,"
    )
    print("interaction depth, and risk indicator features.")
    print(
        "Use compute_all_behavioral_features() to generate the complete behavioral feature set."
    )
