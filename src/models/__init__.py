"""
Models Module

This module contains machine learning models for churn prediction including:
- baseline_model: Simple logistic regression baseline
- ensemble_models: Advanced ensemble methods (XGBoost, LightGBM)
- model_utils: Utilities for model training, evaluation, and selection
"""

from .baseline_model import ChurnBaselineModel

__version__ = "1.0.0"

__all__ = ["ChurnBaselineModel"]
