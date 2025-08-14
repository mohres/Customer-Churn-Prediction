"""
Features package for churn prediction system.

This package contains modules for feature engineering:
- base_features: Core activity, engagement, and temporal features
- behavioral_features: Advanced behavioral signals and risk indicators
- feature_store: Centralized feature computation and management
"""

from .base_features import (
    compute_activity_features,
    compute_all_base_features,
    compute_engagement_features,
    compute_subscription_features,
    compute_temporal_features,
    validate_features,
)
from .behavioral_features import (
    compute_all_behavioral_features,
    compute_content_preference_features,
    compute_interaction_depth_features,
    compute_risk_indicator_features,
    compute_usage_trend_features,
)
from .feature_store import FeatureCache, FeatureConfig, FeatureStore, FeatureValidator

__version__ = "1.0.0"
__author__ = "Mohammad Fares"

__all__ = [
    "FeatureCache",
    "FeatureConfig",
    "FeatureStore",
    "FeatureValidator",
    "compute_activity_features",
    "compute_all_base_features",
    "compute_all_behavioral_features",
    "compute_content_preference_features",
    "compute_engagement_features",
    "compute_interaction_depth_features",
    "compute_risk_indicator_features",
    "compute_subscription_features",
    "compute_temporal_features",
    "compute_usage_trend_features",
    "validate_features",
]
