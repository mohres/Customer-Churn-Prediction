#!/usr/bin/env python3
"""
MLflow Training Entry Point

This script serves as the entry point for MLflow Projects, allowing users to
start training runs directly from the MLflow UI with configurable parameters.

The script integrates with the existing training pipeline and provides a
command-line interface for parameter configuration.
"""

import argparse
import sys
from pathlib import Path

import mlflow

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.features.feature_store import FeatureConfig
from src.models.data_preparation import prepare_training_dataset
from src.pipelines.training_pipeline import TrainingPipeline


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="MLflow Training Entry Point")

    # Data parameters
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/customer_churn_mini.json",
        help="Path to training data",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Test set size (0.0-1.0)"
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Random state for reproducibility"
    )

    # XGBoost parameters
    parser.add_argument(
        "--max-depth-xgb", type=int, default=6, help="XGBoost max depth"
    )
    parser.add_argument(
        "--learning-rate-xgb", type=float, default=0.1, help="XGBoost learning rate"
    )
    parser.add_argument(
        "--n-estimators-xgb", type=int, default=200, help="XGBoost number of estimators"
    )

    # LightGBM parameters
    parser.add_argument(
        "--num-leaves-lgb", type=int, default=31, help="LightGBM number of leaves"
    )
    parser.add_argument(
        "--learning-rate-lgb", type=float, default=0.05, help="LightGBM learning rate"
    )
    parser.add_argument(
        "--n-estimators-lgb",
        type=int,
        default=200,
        help="LightGBM number of estimators",
    )

    # Model selection
    parser.add_argument(
        "--enable-xgboost",
        type=str,
        default="true",
        help="Enable XGBoost training (true/false)",
    )
    parser.add_argument(
        "--enable-lightgbm",
        type=str,
        default="true",
        help="Enable LightGBM training (true/false)",
    )

    return parser.parse_args()


def str_to_bool(v):
    """Convert string to boolean."""
    return v.lower() in ("true", "1", "yes", "on")


def create_dynamic_config(args):
    """Create training configuration from command line arguments."""
    config = {
        "data": {
            "raw_data_path": args.data_path,
            "test_size": args.test_size,
            "validation_enabled": True,
        },
        "models": {},
        "validation": {
            "min_auc_threshold": 0.70,
            "min_precision_threshold": 0.65,
            "min_recall_threshold": 0.65,
        },
        "mlflow": {
            "auto_register": True,
            "staging_threshold": 0.80,
            "production_threshold": 0.85,
        },
    }

    # Add XGBoost configuration if enabled
    if str_to_bool(args.enable_xgboost):
        config["models"]["xgboost"] = {
            "enabled": True,
            "params": {
                "max_depth": args.max_depth_xgb,
                "learning_rate": args.learning_rate_xgb,
                "n_estimators": args.n_estimators_xgb,
                "random_state": args.random_state,
                "eval_metric": "auc",
                "objective": "binary:logistic",
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            },
        }

    # Add LightGBM configuration if enabled
    if str_to_bool(args.enable_lightgbm):
        config["models"]["lightgbm"] = {
            "enabled": True,
            "params": {
                "num_leaves": args.num_leaves_lgb,
                "learning_rate": args.learning_rate_lgb,
                "n_estimators": args.n_estimators_lgb,
                "random_state": args.random_state,
                "objective": "binary",
                "metric": "auc",
                "boosting_type": "gbdt",
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "verbose": -1,
            },
        }

    return config


def main():
    """Main training function."""
    args = parse_args()

    # Log all parameters to MLflow
    mlflow.log_params(
        {
            "data_path": args.data_path,
            "test_size": args.test_size,
            "random_state": args.random_state,
            "max_depth_xgb": args.max_depth_xgb,
            "learning_rate_xgb": args.learning_rate_xgb,
            "n_estimators_xgb": args.n_estimators_xgb,
            "num_leaves_lgb": args.num_leaves_lgb,
            "learning_rate_lgb": args.learning_rate_lgb,
            "n_estimators_lgb": args.n_estimators_lgb,
            "enable_xgboost": args.enable_xgboost,
            "enable_lightgbm": args.enable_lightgbm,
        }
    )

    print("🚀 Starting MLflow Training Pipeline...")
    print(f"📊 Data path: {args.data_path}")
    print(f"🔧 XGBoost enabled: {args.enable_xgboost}")
    print(f"🔧 LightGBM enabled: {args.enable_lightgbm}")

    try:
        # Check if we need to prepare features first
        data_path = Path(args.data_path)
        if data_path.suffix == ".json":
            print("📊 Preparing features from raw event data...")

            # Load raw events
            import pandas as pd

            events_df = pd.read_json(str(data_path), lines=True)
            print(
                f"Loaded {len(events_df):,} events from {events_df['userId'].nunique():,} users"
            )

            # Configure feature engineering
            feature_config = FeatureConfig(
                activity_windows=[7, 14, 30],
                trend_windows=[7, 14, 21],
                include_base_features=True,
                include_behavioral_features=True,
                enable_caching=True,
                feature_version=f"mlflow_ui_v{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
            )

            # Prepare training dataset with feature engineering
            dataset = prepare_training_dataset(
                events_df=events_df,
                feature_config=feature_config,
                churn_inactivity_days=21,
            )

            # Save prepared features temporarily
            temp_features_path = "data/processed/temp_features_for_training.csv"
            Path(temp_features_path).parent.mkdir(parents=True, exist_ok=True)
            dataset.to_csv(temp_features_path, index=False)

            # Update config to use prepared features
            config = create_dynamic_config(args)
            config["data"]["features_path"] = temp_features_path
            config["data"]["target_column"] = "is_churned"
        else:
            # Use existing prepared features
            config = create_dynamic_config(args)
            config["data"]["features_path"] = args.data_path
            config["data"]["target_column"] = "is_churned"

        # Create and run training pipeline
        pipeline = TrainingPipeline(config_dict=config)
        results = pipeline.run()

        # Log summary metrics
        if results:
            mlflow.log_metrics(
                {
                    "pipeline_success": 1,
                    "models_trained": len(
                        [m for m in config["models"] if config["models"][m]["enabled"]]
                    ),
                    "total_runtime_seconds": results.get("runtime_seconds", 0),
                }
            )

            # Log best model metrics if available
            if "best_model" in results:
                best_metrics = results["best_model"].get("metrics", {})
                for metric_name, value in best_metrics.items():
                    mlflow.log_metric(f"best_model_{metric_name}", value)

        print("✅ Training pipeline completed successfully!")
        return results

    except Exception as e:
        print(f"❌ Training pipeline failed: {e!s}")
        mlflow.log_metrics({"pipeline_success": 0})
        mlflow.log_param("error_message", str(e))
        raise


if __name__ == "__main__":
    main()
