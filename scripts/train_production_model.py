#!/usr/bin/env python3
"""
Production Training Script for Churn Prediction Model

This script trains a churn prediction model using the full customer_churn.json dataset
with complete feature engineering pipeline. It processes raw event data and applies
all feature engineering steps during training.

Usage:
    python scripts/train_production_model.py [--data-path path/to/data.json] [--config-path path/to/config.yaml]

Features:
- Works with raw event data (customer_churn.json)
- Applies complete feature engineering pipeline
- Trains optimized model with hyperparameter tuning
- Saves model artifacts for production deployment
- Generates comprehensive evaluation reports
- Integrates with MLflow for experiment tracking
"""

import argparse
import contextlib
import json
import logging
import os
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

import mlflow
import mlflow.lightgbm
import mlflow.sklearn

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.features.feature_store import FeatureConfig
from src.models.data_preparation import prepare_training_dataset, validate_dataset

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML file."""
    if not os.path.exists(config_path):
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return {}

    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_mlflow():
    """Setup MLflow tracking."""
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
    mlflow.set_tracking_uri(mlflow_uri)

    # Set experiment
    experiment_name = "churn_prediction_production"
    with contextlib.suppress(mlflow.exceptions.MlflowException):
        mlflow.create_experiment(experiment_name)

    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    logger.info(f"MLflow experiment: {experiment_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Train production churn prediction model"
    )
    parser.add_argument(
        "--data-path",
        default="data/customer_churn.json",
        help="Path to raw customer event data",
    )
    parser.add_argument(
        "--config-path",
        default="config/training_config.yaml",
        help="Path to training configuration file",
    )
    parser.add_argument(
        "--output-dir", default="models", help="Directory to save trained models"
    )
    parser.add_argument(
        "--experiment-name", default=None, help="MLflow experiment name override"
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config_path)

    # Setup MLflow
    setup_mlflow()
    if args.experiment_name:
        mlflow.set_experiment(args.experiment_name)

    logger.info("=" * 60)
    logger.info("PRODUCTION CHURN PREDICTION MODEL TRAINING")
    logger.info("=" * 60)
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Config path: {args.config_path}")
    logger.info(f"Output directory: {args.output_dir}")

    # Load raw event data
    logger.info("Loading raw event data...")
    if not os.path.exists(args.data_path):
        logger.error(f"Data file not found: {args.data_path}")
        sys.exit(1)

    try:
        events_df = pd.read_json(args.data_path, lines=True)
        logger.info(
            f"Loaded {len(events_df):,} events from {events_df['userId'].nunique():,} users"
        )

        # Basic data validation
        if len(events_df) == 0:
            logger.error("No events found in data file")
            sys.exit(1)

        if "userId" not in events_df.columns:
            logger.error("Missing required 'userId' column")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        sys.exit(1)

    # Setup feature configuration
    feature_config_params = config.get("feature_engineering", {})
    feature_config = FeatureConfig(
        activity_windows=feature_config_params.get("activity_windows", [7, 14, 30]),
        trend_windows=feature_config_params.get("trend_windows", [7, 14, 21]),
        include_base_features=feature_config_params.get("include_base_features", True),
        include_behavioral_features=feature_config_params.get(
            "include_behavioral_features", True
        ),
        enable_caching=feature_config_params.get("enable_caching", True),
        feature_version=f"production_v{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )

    logger.info("Feature engineering configuration:")
    logger.info(f"  Activity windows: {feature_config.activity_windows}")
    logger.info(f"  Trend windows: {feature_config.trend_windows}")
    logger.info(f"  Base features: {feature_config.include_base_features}")
    logger.info(f"  Behavioral features: {feature_config.include_behavioral_features}")
    logger.info(f"  Version: {feature_config.feature_version}")

    # Start MLflow run
    with mlflow.start_run() as run:
        logger.info(f"Started MLflow run: {run.info.run_id}")

        # Log configuration
        mlflow.log_params(
            {
                "data_source": args.data_path,
                "total_events": len(events_df),
                "total_users": events_df["userId"].nunique(),
                **feature_config_params,
            }
        )

        # Prepare training dataset with feature engineering
        logger.info("Preparing training dataset with feature engineering...")
        dataset = prepare_training_dataset(
            events_df=events_df,
            feature_config=feature_config,
            churn_inactivity_days=config.get("churn_definition", {}).get(
                "inactivity_days", 21
            ),
        )

        # Validate dataset
        validation_results = validate_dataset(dataset)
        if not validation_results["passed"]:
            logger.error("Dataset validation failed:")
            for issue in validation_results["data_quality_issues"]:
                logger.error(f"  - {issue}")
            sys.exit(1)

        logger.info("Dataset validation passed")

        # Log dataset metrics
        mlflow.log_metrics(
            {
                "dataset_users": validation_results["total_users"],
                "dataset_features": len(
                    [
                        c
                        for c in dataset.columns
                        if c
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
                ),
                "churn_rate": dataset["is_churned"].mean(),
            }
        )

        # Prepare features and target
        feature_columns = [
            c
            for c in dataset.columns
            if c
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
                "feature_computation_timestamp",
            ]
        ]

        x = dataset[feature_columns]
        y = dataset["is_churned"]

        logger.info(f"Training features: {len(feature_columns)}")
        logger.info(f"Training samples: {len(x)}")
        logger.info(f"Churn rate: {y.mean():.1%}")

        # Handle missing values (simple imputation for production)
        # Separate numeric and categorical columns
        numeric_columns = x.select_dtypes(include=[np.number]).columns
        categorical_columns = x.select_dtypes(include=["object"]).columns

        x_filled = x.copy()

        # Fill numeric columns with median
        if len(numeric_columns) > 0:
            x_filled[numeric_columns] = x_filled[numeric_columns].fillna(
                x[numeric_columns].median()
            )

        # Fill categorical columns with mode or 'unknown'
        if len(categorical_columns) > 0:
            for col in categorical_columns:
                mode_value = x[col].mode()
                fill_value = mode_value[0] if len(mode_value) > 0 else "unknown"
                x_filled[col] = x_filled[col].fillna(fill_value)

        # Encode categorical features
        from sklearn.preprocessing import LabelEncoder

        label_encoders = {}

        for col in categorical_columns:
            le = LabelEncoder()
            x_filled[col] = le.fit_transform(x_filled[col].astype(str))
            label_encoders[col] = le

        logger.info(
            f"Processed {len(numeric_columns)} numeric and {len(categorical_columns)} categorical features"
        )

        # Model configuration
        model_config = config.get("model", {})
        model_params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": model_config.get("num_leaves", 31),
            "learning_rate": model_config.get("learning_rate", 0.1),
            "feature_fraction": model_config.get("feature_fraction", 0.9),
            "bagging_fraction": model_config.get("bagging_fraction", 0.8),
            "bagging_freq": model_config.get("bagging_freq", 5),
            "min_child_samples": model_config.get("min_child_samples", 20),
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }

        # Cross-validation for model selection
        logger.info("Performing cross-validation...")
        cv_folds = config.get("validation", {}).get("cv_folds", 5)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        model = LGBMClassifier(**model_params)
        cv_scores = cross_val_score(
            model, x_filled, y, cv=cv, scoring="roc_auc", n_jobs=-1
        )

        logger.info(
            f"Cross-validation AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})"
        )

        # Log CV results
        mlflow.log_metrics(
            {
                "cv_auc_mean": cv_scores.mean(),
                "cv_auc_std": cv_scores.std(),
                "cv_auc_min": cv_scores.min(),
                "cv_auc_max": cv_scores.max(),
            }
        )

        # Train final model on full dataset
        logger.info("Training final model...")
        final_model = LGBMClassifier(**model_params)
        final_model.fit(x_filled, y)

        # Generate predictions for evaluation
        y_pred_proba = final_model.predict_proba(x_filled)[:, 1]
        final_auc = roc_auc_score(y, y_pred_proba)

        logger.info(f"Final model training AUC: {final_auc:.4f}")

        # Log model parameters and metrics
        mlflow.log_params(model_params)
        mlflow.log_metrics(
            {
                "final_auc": final_auc,
                "training_samples": len(x),
                "n_features": len(feature_columns),
            }
        )

        # Additional evaluation metrics
        from sklearn.metrics import (
            f1_score,
            precision_score,
            recall_score,
        )

        y_pred = final_model.predict(x_filled)

        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)

        logger.info("Final model metrics:")
        logger.info(f"  AUC: {final_auc:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1-Score: {f1:.4f}")

        # Log additional metrics
        mlflow.log_metrics({"precision": precision, "recall": recall, "f1_score": f1})

        # Save model artifacts
        os.makedirs(args.output_dir, exist_ok=True)

        # Save the model
        model_path = os.path.join(args.output_dir, "production_churn_model.joblib")
        joblib.dump(final_model, model_path)
        logger.info(f"Saved model to: {model_path}")

        # Save feature metadata
        # Convert feature_config to JSON-serializable format
        feature_config_dict = feature_config.__dict__.copy()
        if (
            "reference_date" in feature_config_dict
            and feature_config_dict["reference_date"] is not None
        ):
            feature_config_dict["reference_date"] = feature_config_dict[
                "reference_date"
            ].isoformat()

        feature_metadata = {
            "feature_columns": feature_columns,
            "feature_config": feature_config_dict,
            "preprocessing": {
                "fillna_strategy_numeric": "median",
                "fillna_strategy_categorical": "mode",
                "categorical_encoding": "label_encoder",
                "label_encoders": {
                    col: le.classes_.tolist() for col, le in label_encoders.items()
                },
            },
            "model_type": "LGBMClassifier",
            "model_params": model_params,
            "training_date": datetime.now().isoformat(),
            "data_source": args.data_path,
            "validation_auc": float(cv_scores.mean()),
            "final_auc": float(final_auc),
            "n_training_samples": len(x),
            "churn_rate": float(y.mean()),
            "numeric_features": len(numeric_columns),
            "categorical_features": len(categorical_columns),
        }

        metadata_path = os.path.join(args.output_dir, "production_model_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(feature_metadata, f, indent=2)
        logger.info(f"Saved metadata to: {metadata_path}")

        # Log model to MLflow
        mlflow.lightgbm.log_model(
            final_model,
            "model",
            input_example=x_filled.head(1),
            signature=mlflow.models.infer_signature(x_filled, y_pred_proba),
        )

        # Log artifacts
        mlflow.log_artifact(metadata_path)

        # Generate and save feature importance
        if hasattr(final_model, "feature_importances_"):
            feature_importance = pd.DataFrame(
                {
                    "feature": feature_columns,
                    "importance": final_model.feature_importances_,
                }
            ).sort_values("importance", ascending=False)

            importance_path = os.path.join(args.output_dir, "feature_importance.csv")
            feature_importance.to_csv(importance_path, index=False)
            mlflow.log_artifact(importance_path)

            logger.info("Top 10 important features:")
            for _idx, row in feature_importance.head(10).iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")

        logger.info("=" * 60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Metadata saved to: {metadata_path}")
        logger.info(f"MLflow run ID: {run.info.run_id}")
        logger.info(f"Final model AUC: {final_auc:.4f}")
        logger.info(
            f"Cross-validation AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})"
        )


if __name__ == "__main__":
    main()
