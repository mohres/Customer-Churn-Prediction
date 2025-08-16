"""
Automated Training Pipeline for Churn Prediction

This module provides a comprehensive, automated training pipeline that can be
configured via YAML files and executed with different parameters. It includes:

- Data validation and preprocessing
- Feature engineering and selection
- Model training with multiple algorithms
- Model evaluation and comparison
- Automated model registration based on performance thresholds
- Pipeline scheduling and monitoring capabilities

Key features:
- Configurable via YAML files
- Support for multiple model types
- Automated hyperparameter optimization
- MLflow integration for experiment tracking
- Data quality validation gates
- Model performance validation gates
- Automated artifact generation and storage

Usage:
    from pipelines.training_pipeline import TrainingPipeline

    pipeline = TrainingPipeline("config/training_config.yaml")
    results = pipeline.run()
"""

import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data.data_validator import DataValidator
from features.feature_store import FeatureStore
from models.mlflow_ensemble_models import (
    ExperimentManager,
    MLflowLightGBMChurnModel,
    MLflowModelComparison,
    MLflowXGBoostChurnModel,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PipelineConfig:
    """Configuration management for training pipeline."""

    def __init__(self, config_path: str):
        """Load configuration from YAML file."""
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()

    def _load_config(self) -> dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {self.config_path} not found. Using defaults.")
            return self._get_default_config()

    def _get_default_config(self) -> dict[str, Any]:
        """Get default configuration."""
        return {
            "data": {
                "features_path": "data/processed/features_selected.csv",
                "target_column": "is_churned",
                "test_size": 0.2,
                "random_state": 42,
                "validation_enabled": True,
            },
            "models": {
                "xgboost": {
                    "enabled": True,
                    "params": {
                        "max_depth": 6,
                        "learning_rate": 0.1,
                        "n_estimators": 200,
                        "subsample": 0.8,
                        "colsample_bytree": 0.8,
                    },
                },
                "lightgbm": {
                    "enabled": True,
                    "params": {
                        "num_leaves": 31,
                        "learning_rate": 0.05,
                        "n_estimators": 200,
                        "feature_fraction": 0.9,
                        "bagging_fraction": 0.8,
                    },
                },
            },
            "validation": {
                "min_auc_threshold": 0.75,
                "min_precision_threshold": 0.70,
                "min_recall_threshold": 0.70,
                "data_quality_checks": True,
            },
            "output": {
                "models_dir": "models",
                "reports_dir": "reports",
                "plots_dir": "plots",
                "save_artifacts": True,
            },
            "mlflow": {
                "experiment_name": "churn_prediction_pipeline",
                "auto_register": True,
                "staging_threshold": 0.85,
                "production_threshold": 0.90,
            },
        }

    def _validate_config(self):
        """Validate configuration."""
        required_sections = ["data", "models", "validation", "output", "mlflow"]
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required config section: {section}")

        # Validate at least one model is enabled
        models_enabled = any(
            self.config["models"].get(model, {}).get("enabled", False)
            for model in ["xgboost", "lightgbm"]
        )
        if not models_enabled:
            raise ValueError("At least one model must be enabled")

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        keys = key.split(".")
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value


class TrainingPipeline:
    """Automated training pipeline for churn prediction."""

    def __init__(self, config_path: str = "config/training_config.yaml"):
        """Initialize training pipeline with configuration."""
        self.config = PipelineConfig(config_path)
        self.experiment_manager = ExperimentManager()
        self.feature_store = FeatureStore()
        self.data_validator = DataValidator()
        self.results = {}

        # Set up directories
        self._setup_directories()

        logger.info(f"Initialized training pipeline with config: {config_path}")

    def _setup_directories(self):
        """Create necessary directories."""
        directories = [
            self.config.get("output.models_dir"),
            self.config.get("output.reports_dir"),
            self.config.get("output.plots_dir"),
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def _load_and_validate_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """Load and validate feature data."""
        logger.info("🔄 Loading and validating data...")

        features_path = self.config.get("data.features_path")
        target_column = self.config.get("data.target_column")

        # Load features
        try:
            features_df = self.feature_store.load_features(features_path)
            logger.info(
                f"✅ Loaded {len(features_df)} feature records from {features_path}"
            )
        except FileNotFoundError as err:
            raise FileNotFoundError(f"Feature file not found: {features_path}") from err

        # Validate data quality if enabled
        if self.config.get("validation.data_quality_checks"):
            logger.info("🔍 Running data quality checks...")
            validation_results = self.data_validator.validate_dataset(features_df)

            # Check if there are errors or warnings
            summary = validation_results.get("summary", {})
            error_count = summary.get("error_count", 0)
            warning_count = summary.get("warning_count", 0)

            if error_count > 0:
                logger.warning(
                    f"⚠️  Data quality issues detected: {error_count} errors, {warning_count} warnings"
                )
                # Optionally fail pipeline on data quality issues
                # raise ValueError("Data quality validation failed")
            else:
                logger.info(
                    f"✅ Data quality validation passed ({warning_count} warnings)"
                )

        # Prepare features and target
        if target_column not in features_df.columns:
            raise ValueError(f"Target column '{target_column}' not found in features")

        x = features_df.drop([target_column], axis=1)
        y = features_df[target_column]

        logger.info(f"📊 Dataset shape: {x.shape}")
        logger.info(f"📊 Churn rate: {y.mean():.2%}")

        return x, y

    def _split_data(
        self, x: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into training and test sets."""
        test_size = self.config.get("data.test_size")
        random_state = self.config.get("data.random_state")

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=random_state, stratify=y
        )

        logger.info(f"📊 Training set: {x_train.shape[0]} samples")
        logger.info(f"📊 Test set: {x_test.shape[0]} samples")

        return x_train, x_test, y_train, y_test

    def _train_models(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> dict[str, Any]:
        """Train configured models."""
        logger.info("🚀 Training models...")

        models_config = self.config.get("models")
        comparison = MLflowModelComparison(self.experiment_manager)
        results = {}

        # Train XGBoost if enabled
        if models_config.get("xgboost", {}).get("enabled", False):
            logger.info("🔧 Training XGBoost model...")
            xgb_params = models_config.get("xgboost", {}).get("params", {})

            xgb_model = MLflowXGBoostChurnModel(self.experiment_manager, **xgb_params)
            xgb_result = xgb_model.train_and_evaluate(
                x_train,
                y_train,
                x_test,
                y_test,
                run_name=f"xgboost_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            )

            results["xgboost"] = xgb_result
            comparison.add_model_result("XGBoost", xgb_result)
            logger.info(
                f"✅ XGBoost completed - AUC: {xgb_result['metrics']['roc_auc']:.4f}"
            )

        # Train LightGBM if enabled
        if models_config.get("lightgbm", {}).get("enabled", False):
            logger.info("🔧 Training LightGBM model...")
            lgb_params = models_config.get("lightgbm", {}).get("params", {})

            lgb_model = MLflowLightGBMChurnModel(self.experiment_manager, **lgb_params)
            lgb_result = lgb_model.train_and_evaluate(
                x_train,
                y_train,
                x_test,
                y_test,
                run_name=f"lightgbm_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            )

            results["lightgbm"] = lgb_result
            comparison.add_model_result("LightGBM", lgb_result)
            logger.info(
                f"✅ LightGBM completed - AUC: {lgb_result['metrics']['roc_auc']:.4f}"
            )

        # Generate model comparison
        comparison_df = comparison.compare_models()
        results["comparison"] = comparison_df

        return results

    def _validate_model_performance(self, results: dict[str, Any]) -> bool:
        """Validate model performance against thresholds."""
        logger.info("🔍 Validating model performance...")

        min_auc = self.config.get("validation.min_auc_threshold")
        min_precision = self.config.get("validation.min_precision_threshold")
        min_recall = self.config.get("validation.min_recall_threshold")

        validation_passed = True

        for model_name, result in results.items():
            if model_name == "comparison":
                continue

            metrics = result.get("metrics", {})
            auc = metrics.get("roc_auc", 0)
            precision = metrics.get("precision", 0)
            recall = metrics.get("recall", 0)

            logger.info(
                f"📊 {model_name}: AUC={auc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}"
            )

            if auc < min_auc:
                logger.warning(
                    f"⚠️  {model_name} AUC ({auc:.4f}) below threshold ({min_auc:.4f})"
                )
                validation_passed = False

            if precision < min_precision:
                logger.warning(
                    f"⚠️  {model_name} Precision ({precision:.4f}) below threshold ({min_precision:.4f})"
                )
                validation_passed = False

            if recall < min_recall:
                logger.warning(
                    f"⚠️  {model_name} Recall ({recall:.4f}) below threshold ({min_recall:.4f})"
                )
                validation_passed = False

        if validation_passed:
            logger.info("✅ All models passed performance validation")
        else:
            logger.warning("❌ Some models failed performance validation")

        return validation_passed

    def _save_pipeline_artifacts(self, results: dict[str, Any]):
        """Save pipeline artifacts."""
        if not self.config.get("output.save_artifacts"):
            return

        logger.info("💾 Saving pipeline artifacts...")

        # Save pipeline configuration
        config_path = os.path.join(
            self.config.get("output.reports_dir"),
            f"pipeline_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml",
        )
        with open(config_path, "w") as f:
            yaml.dump(self.config.config, f, default_flow_style=False, indent=2)

        # Save pipeline results summary
        summary_path = os.path.join(
            self.config.get("output.reports_dir"),
            f"pipeline_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml",
        )

        summary = {
            "pipeline_run": {
                "timestamp": datetime.now().isoformat(),
                "config_path": self.config.config_path,
                "models_trained": list(results.keys()),
                "best_model": (
                    results.get("comparison", pd.DataFrame()).iloc[0]["model"]
                    if "comparison" in results and not results["comparison"].empty
                    else None
                ),
            },
            "performance_summary": {},
        }

        for model_name, result in results.items():
            if model_name != "comparison":
                summary["performance_summary"][model_name] = result.get("metrics", {})

        with open(summary_path, "w") as f:
            yaml.dump(summary, f, default_flow_style=False, indent=2)

        logger.info(
            f"✅ Pipeline artifacts saved to {self.config.get('output.reports_dir')}"
        )

    def run(self) -> dict[str, Any]:
        """Run the complete training pipeline."""
        logger.info("🚀 Starting Training Pipeline")
        logger.info("=" * 50)

        try:
            # Load and validate data
            x, y = self._load_and_validate_data()

            # Split data
            x_train, x_test, y_train, y_test = self._split_data(x, y)

            # Train models
            results = self._train_models(x_train, y_train, x_test, y_test)

            # Validate performance
            performance_passed = self._validate_model_performance(results)

            # Save artifacts
            self._save_pipeline_artifacts(results)

            # Pipeline summary
            pipeline_results = {
                "success": True,
                "performance_validation_passed": performance_passed,
                "models_trained": list(results.keys()),
                "results": results,
                "timestamp": datetime.now().isoformat(),
            }

            logger.info("✅ Training Pipeline Completed Successfully")
            logger.info("=" * 50)

            # Print summary
            if "comparison" in results and not results["comparison"].empty:
                best_model = results["comparison"].iloc[0]
                logger.info(f"🏆 Best Model: {best_model['model']}")
                logger.info(f"📈 Best AUC: {best_model['roc_auc']:.4f}")
                logger.info(
                    f"🏷️  Registered: {'Yes' if best_model['registered'] else 'No'}"
                )

            return pipeline_results

        except Exception as e:
            logger.error(f"❌ Pipeline failed with error: {e!s}")
            logger.error(traceback.format_exc())

            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }


def create_default_training_config(config_path: str = "config/training_config.yaml"):
    """Create a default training configuration file."""
    config = PipelineConfig("")._get_default_config()

    # Create config directory if it doesn't exist
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    # Write config file
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

    logger.info(f"Created default training configuration at {config_path}")


if __name__ == "__main__":
    # Create default config if it doesn't exist
    config_path = "config/training_config.yaml"
    if not os.path.exists(config_path):
        create_default_training_config(config_path)

    # Run pipeline
    pipeline = TrainingPipeline(config_path)
    results = pipeline.run()

    if results["success"]:
        logger.info("🎉 Pipeline completed successfully!")
    else:
        logger.error("💥 Pipeline failed!")
        sys.exit(1)
