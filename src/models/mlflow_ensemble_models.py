"""
MLflow-Integrated Ensemble Models for Churn Prediction

This module extends the ensemble models with comprehensive MLflow experiment tracking,
model registry, and artifact management capabilities.

Features:
- Automatic experiment logging
- Model registry integration
- Artifact management (plots, reports, model files)
- Hyperparameter tracking
- Performance metrics logging
- Model versioning and staging

Key classes:
- MLflowXGBoostChurnModel: XGBoost with MLflow integration
- MLflowLightGBMChurnModel: LightGBM with MLflow integration
- MLflowModelComparison: Compare models with MLflow tracking
- ExperimentManager: Centralized experiment management
"""

import os
import tempfile
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import mlflow.xgboost

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

warnings.filterwarnings("ignore")


class ExperimentManager:
    """Manage MLflow experiments and configuration."""

    def __init__(self, config_path: str = "config/mlflow_config.yaml"):
        """Initialize experiment manager with configuration."""
        self.config = self._load_config(config_path)
        self.setup_mlflow()

    def _load_config(self, config_path: str) -> dict[str, Any]:
        """Load MLflow configuration."""
        try:
            with open(config_path) as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            # Default configuration
            return {
                "mlflow": {
                    "tracking_uri": f"file://{Path.cwd()}/mlruns",
                    "experiment_name": "churn_prediction",
                    "model_registry": {
                        "staging_threshold": 0.85,
                        "production_threshold": 0.90,
                    },
                    "artifacts": {
                        "models_path": "models",
                        "plots_path": "plots",
                        "reports_path": "reports",
                    },
                }
            }

    def setup_mlflow(self):
        """Set up MLflow tracking and experiment."""
        mlflow_config = self.config["mlflow"]

        # Set tracking URI
        mlflow.set_tracking_uri(mlflow_config["tracking_uri"])

        # Create or get experiment
        experiment_name = mlflow_config["experiment_name"]
        try:
            experiment_id = mlflow.create_experiment(experiment_name)
        except mlflow.exceptions.MlflowException:
            # Experiment already exists
            experiment = mlflow.get_experiment_by_name(experiment_name)
            experiment_id = experiment.experiment_id

        # Set the experiment
        mlflow.set_experiment(experiment_name)

        print(f"MLflow experiment: {experiment_name} (ID: {experiment_id})")

    def get_staging_threshold(self) -> float:
        """Get staging threshold from config."""
        return self.config["mlflow"]["model_registry"]["staging_threshold"]

    def get_production_threshold(self) -> float:
        """Get production threshold from config."""
        return self.config["mlflow"]["model_registry"]["production_threshold"]


class MLflowModelBase:
    """Base class for MLflow-integrated models."""

    def __init__(self, experiment_manager: ExperimentManager | None = None):
        """Initialize with experiment manager."""
        self.experiment_manager = experiment_manager or ExperimentManager()
        self.model = None
        self.run_id = None
        self.model_name = None

    def _create_plots(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> dict[str, str]:
        """Create evaluation plots and return paths."""
        plots = {}

        with tempfile.TemporaryDirectory() as temp_dir:
            # ROC Curve
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            plt.figure(figsize=(8, 6))
            plt.plot(
                fpr,
                tpr,
                label=f"ROC Curve (AUC = {roc_auc_score(y_true, y_proba):.3f})",
            )
            plt.plot([0, 1], [0, 1], "k--", label="Random")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend()
            plt.grid(True, alpha=0.3)
            roc_path = os.path.join(temp_dir, "roc_curve.png")
            plt.savefig(roc_path, dpi=300, bbox_inches="tight")
            plt.close()
            plots["roc_curve"] = roc_path

            # Precision-Recall Curve
            precision, recall, _ = precision_recall_curve(y_true, y_proba)
            plt.figure(figsize=(8, 6))
            plt.plot(
                recall,
                precision,
                label=f"PR Curve (F1 = {f1_score(y_true, y_pred):.3f})",
            )
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Precision-Recall Curve")
            plt.legend()
            plt.grid(True, alpha=0.3)
            pr_path = os.path.join(temp_dir, "precision_recall_curve.png")
            plt.savefig(pr_path, dpi=300, bbox_inches="tight")
            plt.close()
            plots["precision_recall_curve"] = pr_path

            # Confusion Matrix
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title("Confusion Matrix")
            plt.ylabel("True Label")
            plt.xlabel("Predicted Label")
            cm_path = os.path.join(temp_dir, "confusion_matrix.png")
            plt.savefig(cm_path, dpi=300, bbox_inches="tight")
            plt.close()
            plots["confusion_matrix"] = cm_path

            # Feature Importance (if available)
            if hasattr(self.model, "feature_importances_") and feature_names:
                importance_df = (
                    pd.DataFrame(
                        {
                            "feature": feature_names,
                            "importance": self.model.feature_importances_,
                        }
                    )
                    .sort_values("importance", ascending=False)
                    .head(20)
                )

                plt.figure(figsize=(10, 8))
                sns.barplot(data=importance_df, x="importance", y="feature")
                plt.title("Top 20 Feature Importances")
                plt.xlabel("Importance")
                plt.tight_layout()
                fi_path = os.path.join(temp_dir, "feature_importance.png")
                plt.savefig(fi_path, dpi=300, bbox_inches="tight")
                plt.close()
                plots["feature_importance"] = fi_path

            # Copy plots to permanent location and return paths
            permanent_plots = {}
            plots_dir = "plots"
            os.makedirs(plots_dir, exist_ok=True)

            for plot_name, temp_path in plots.items():
                permanent_path = os.path.join(
                    plots_dir, f"{self.model_name}_{plot_name}.png"
                )
                import shutil

                shutil.copy2(temp_path, permanent_path)
                permanent_plots[plot_name] = permanent_path

        return permanent_plots

    def _log_metrics_and_artifacts(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        params: dict[str, Any],
        feature_names: list[str] | None = None,
    ):
        """Log metrics and artifacts to MLflow."""
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred),
            "roc_auc": roc_auc_score(y_true, y_proba),
        }

        # Log parameters
        for param, value in params.items():
            mlflow.log_param(param, value)

        # Log metrics
        for metric, value in metrics.items():
            mlflow.log_metric(metric, value)

        # Create and log plots
        plots = self._create_plots(y_true, y_pred, y_proba, feature_names)
        for _plot_name, plot_path in plots.items():
            mlflow.log_artifact(plot_path, "plots")

        # Log classification report
        report_path = f"reports/{self.model_name}_classification_report.txt"
        os.makedirs("reports", exist_ok=True)

        with open(report_path, "w") as f:
            f.write(classification_report(y_true, y_pred))
        mlflow.log_artifact(report_path, "reports")

        return metrics

    def register_model_if_good(self, test_auc: float, model_name: str):
        """Register model if it meets quality thresholds."""
        staging_threshold = self.experiment_manager.get_staging_threshold()
        production_threshold = self.experiment_manager.get_production_threshold()

        if test_auc >= staging_threshold:
            model_uri = f"runs:/{self.run_id}/model"

            if test_auc >= production_threshold:
                stage = "Production"
                print(
                    f"🎯 Model exceeds production threshold ({production_threshold:.3f})"
                )
            else:
                stage = "Staging"
                print(f"🎯 Model meets staging threshold ({staging_threshold:.3f})")

            try:
                # Register model
                registered_model = mlflow.register_model(model_uri, model_name)

                # Try to transition to appropriate stage (may fail due to MLflow issues)
                try:
                    client = mlflow.tracking.MlflowClient()
                    client.transition_model_version_stage(
                        name=model_name, version=registered_model.version, stage=stage
                    )
                    print(
                        f"✅ Registered {model_name} v{registered_model.version} in {stage}"
                    )
                except Exception as e:
                    print(f"⚠️  Model registered but stage transition failed: {e}")
                    print(
                        f"✅ Registered {model_name} v{registered_model.version} (manual stage transition needed)"
                    )

                return registered_model

            except Exception as e:
                print(f"⚠️  Model registration failed: {e}")
                print(f"✅ Model logged successfully with AUC: {test_auc:.4f}")
                return None
        else:
            print(
                f"❌ Model AUC ({test_auc:.3f}) below staging threshold ({staging_threshold:.3f})"
            )
            return None


class MLflowXGBoostChurnModel(MLflowModelBase):
    """XGBoost model with MLflow integration."""

    def __init__(self, experiment_manager: ExperimentManager | None = None, **params):
        """Initialize XGBoost model with MLflow tracking."""
        super().__init__(experiment_manager)
        self.model_name = "xgboost_churn_model"

        # Default parameters
        self.default_params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
        }
        self.params = {**self.default_params, **params}

    def train_and_evaluate(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        run_name: str | None = None,
    ) -> dict[str, Any]:
        """Train and evaluate XGBoost model with MLflow logging."""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not available")

        run_name = run_name or f"xgboost_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        with mlflow.start_run(run_name=run_name):
            self.run_id = mlflow.active_run().info.run_id

            # Train model
            self.model = xgb.XGBClassifier(**self.params)
            self.model.fit(x_train, y_train)

            # Predictions
            y_pred = self.model.predict(x_test)
            y_proba = self.model.predict_proba(x_test)[:, 1]

            # Log model
            mlflow.xgboost.log_model(self.model, "model")

            # Log metrics and artifacts
            metrics = self._log_metrics_and_artifacts(
                y_test, y_pred, y_proba, self.params, x_train.columns.tolist()
            )

            # Register model if it's good enough
            registered_model = self.register_model_if_good(
                metrics["roc_auc"], self.model_name
            )

            results = {
                "model": self.model,
                "metrics": metrics,
                "run_id": self.run_id,
                "registered_model": registered_model,
            }

            print(f"✅ XGBoost training completed - AUC: {metrics['roc_auc']:.4f}")
            return results


class MLflowLightGBMChurnModel(MLflowModelBase):
    """LightGBM model with MLflow integration."""

    def __init__(self, experiment_manager: ExperimentManager | None = None, **params):
        """Initialize LightGBM model with MLflow tracking."""
        super().__init__(experiment_manager)
        self.model_name = "lightgbm_churn_model"

        # Default parameters
        self.default_params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "random_state": 42,
        }
        self.params = {**self.default_params, **params}

    def train_and_evaluate(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        run_name: str | None = None,
    ) -> dict[str, Any]:
        """Train and evaluate LightGBM model with MLflow logging."""
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not available")

        run_name = run_name or f"lightgbm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        with mlflow.start_run(run_name=run_name):
            self.run_id = mlflow.active_run().info.run_id

            # Train model
            self.model = lgb.LGBMClassifier(**self.params)
            self.model.fit(x_train, y_train)

            # Predictions
            y_pred = self.model.predict(x_test)
            y_proba = self.model.predict_proba(x_test)[:, 1]

            # Log model
            mlflow.lightgbm.log_model(self.model, "model")

            # Log metrics and artifacts
            metrics = self._log_metrics_and_artifacts(
                y_test, y_pred, y_proba, self.params, x_train.columns.tolist()
            )

            # Register model if it's good enough
            registered_model = self.register_model_if_good(
                metrics["roc_auc"], self.model_name
            )

            results = {
                "model": self.model,
                "metrics": metrics,
                "run_id": self.run_id,
                "registered_model": registered_model,
            }

            print(f"✅ LightGBM training completed - AUC: {metrics['roc_auc']:.4f}")
            return results


class MLflowModelComparison:
    """Compare multiple models with MLflow tracking."""

    def __init__(self, experiment_manager: ExperimentManager | None = None):
        """Initialize model comparison with MLflow tracking."""
        self.experiment_manager = experiment_manager or ExperimentManager()
        self.results = {}

    def add_model_result(self, model_name: str, result: dict[str, Any]):
        """Add a model result to comparison."""
        self.results[model_name] = result

    def compare_models(self, save_comparison: bool = True) -> pd.DataFrame:
        """Compare all models and create summary."""
        if not self.results:
            print("No models to compare")
            return pd.DataFrame()

        # Create comparison dataframe
        comparison_data = []
        for model_name, result in self.results.items():
            metrics = result["metrics"]
            row = {
                "model": model_name,
                "run_id": result["run_id"][:8],  # Short run ID
                **metrics,
                "registered": result["registered_model"] is not None,
            }
            comparison_data.append(row)

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values("roc_auc", ascending=False)

        if save_comparison:
            # Save comparison to file
            comparison_path = f"reports/model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            os.makedirs("reports", exist_ok=True)
            comparison_df.to_csv(comparison_path, index=False)

            # Log to MLflow
            with mlflow.start_run(
                run_name=f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            ):
                mlflow.log_artifact(comparison_path, "reports")

                # Log best model metrics
                best_model = comparison_df.iloc[0]
                mlflow.log_metric("best_model_auc", best_model["roc_auc"])
                mlflow.log_param("best_model_name", best_model["model"])

                print(f"📊 Model comparison saved to {comparison_path}")

        return comparison_df


def run_model_experiment(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    models_config: dict[str, dict] | None = None,
) -> dict[str, Any]:
    """Run a complete model experiment with multiple algorithms."""

    # Default models configuration
    if models_config is None:
        models_config = {
            "xgboost": {"max_depth": 6, "learning_rate": 0.1, "n_estimators": 100},
            "lightgbm": {"num_leaves": 31, "learning_rate": 0.05, "n_estimators": 100},
        }

    experiment_manager = ExperimentManager()
    comparison = MLflowModelComparison(experiment_manager)

    results = {}

    # Run XGBoost if configured
    if "xgboost" in models_config:
        print("🚀 Training XGBoost model...")
        xgb_model = MLflowXGBoostChurnModel(
            experiment_manager, **models_config["xgboost"]
        )
        xgb_result = xgb_model.train_and_evaluate(x_train, y_train, x_test, y_test)
        results["xgboost"] = xgb_result
        comparison.add_model_result("XGBoost", xgb_result)

    # Run LightGBM if configured
    if "lightgbm" in models_config:
        print("🚀 Training LightGBM model...")
        lgb_model = MLflowLightGBMChurnModel(
            experiment_manager, **models_config["lightgbm"]
        )
        lgb_result = lgb_model.train_and_evaluate(x_train, y_train, x_test, y_test)
        results["lightgbm"] = lgb_result
        comparison.add_model_result("LightGBM", lgb_result)

    # Compare models
    comparison_df = comparison.compare_models()

    print("\n📊 Model Comparison Results:")
    print("=" * 50)
    print(
        comparison_df[
            ["model", "roc_auc", "precision", "recall", "f1_score", "registered"]
        ].round(4)
    )

    # Return results
    experiment_results = {
        "results": results,
        "comparison": comparison_df,
        "best_model": (
            comparison_df.iloc[0]["model"] if not comparison_df.empty else None
        ),
    }

    return experiment_results


if __name__ == "__main__":
    print("MLflow Ensemble Models - Ready for integration!")
    print("Use run_model_experiment() for complete model training with MLflow tracking")
