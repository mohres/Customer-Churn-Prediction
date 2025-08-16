#!/usr/bin/env python3
"""
MLflow Setup and Configuration Module

This module provides utilities for setting up MLflow tracking,
model registry, and experiment management for the churn prediction project.
"""

import os
from pathlib import Path
from typing import Any

import yaml

import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import mlflow.xgboost


class MLflowManager:
    """Centralized MLflow management for the churn prediction project."""

    def __init__(
        self,
        tracking_uri: str | None = None,
        experiment_name: str = "churn_prediction",
    ):
        """
        Initialize MLflow manager.

        Args:
            tracking_uri: MLflow tracking URI. If None, uses local file store.
            experiment_name: Name of the MLflow experiment.
        """
        self.tracking_uri = tracking_uri or f"file://{Path.cwd()}/mlruns"
        self.experiment_name = experiment_name
        self.setup_mlflow()

    def setup_mlflow(self):
        """Set up MLflow tracking and experiment."""
        # Set tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)

        # Create or get experiment
        try:
            experiment_id = mlflow.create_experiment(self.experiment_name)
        except mlflow.exceptions.MlflowException:
            # Experiment already exists
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            experiment_id = experiment.experiment_id

        # Set the experiment
        mlflow.set_experiment(self.experiment_name)

        print(f"MLflow tracking URI: {self.tracking_uri}")
        print(f"Active experiment: {self.experiment_name} (ID: {experiment_id})")

    def log_model_artifacts(
        self,
        model,
        model_type: str,
        model_name: str,
        metrics: dict[str, float],
        params: dict[str, Any],
        artifacts: dict[str, str] | None = None,
    ):
        """
        Log model with artifacts to MLflow.

        Args:
            model: Trained model object
            model_type: Type of model ('sklearn', 'xgboost', 'lightgbm')
            model_name: Name for the model
            metrics: Dictionary of metrics to log
            params: Dictionary of parameters to log
            artifacts: Dictionary of artifact paths to log
        """
        with mlflow.start_run(run_name=model_name):
            # Log parameters
            for param, value in params.items():
                mlflow.log_param(param, value)

            # Log metrics
            for metric, value in metrics.items():
                mlflow.log_metric(metric, value)

            # Log model based on type
            if model_type == "sklearn":
                mlflow.sklearn.log_model(model, "model")
            elif model_type == "xgboost":
                mlflow.xgboost.log_model(model, "model")
            elif model_type == "lightgbm":
                mlflow.lightgbm.log_model(model, "model")
            else:
                # Generic pickle logging
                mlflow.sklearn.log_model(model, "model")

            # Log additional artifacts
            if artifacts:
                for artifact_name, artifact_path in artifacts.items():
                    if os.path.exists(artifact_path):
                        mlflow.log_artifact(artifact_path, artifact_name)

            run_id = mlflow.active_run().info.run_id
            print(f"Logged model {model_name} with run ID: {run_id}")
            return run_id

    def register_model(self, run_id: str, model_name: str, stage: str = "Staging"):
        """
        Register model to MLflow model registry.

        Args:
            run_id: MLflow run ID
            model_name: Name for registered model
            stage: Stage for the model (Staging, Production, Archived)
        """
        model_uri = f"runs:/{run_id}/model"

        # Register model
        registered_model = mlflow.register_model(model_uri, model_name)

        # Transition to specified stage
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name, version=registered_model.version, stage=stage
        )

        print(
            f"Registered model {model_name} version {registered_model.version} in {stage}"
        )
        return registered_model

    def load_model(self, model_name: str, stage: str = "Production"):
        """
        Load model from MLflow model registry.

        Args:
            model_name: Name of registered model
            stage: Stage to load from

        Returns:
            Loaded model
        """
        model_uri = f"models:/{model_name}/{stage}"
        model = mlflow.pyfunc.load_model(model_uri)
        print(f"Loaded model {model_name} from {stage}")
        return model

    def compare_experiments(self, metric_name: str = "test_auc", top_n: int = 10):
        """
        Compare experiments and return top performing runs.

        Args:
            metric_name: Metric to compare by
            top_n: Number of top runs to return

        Returns:
            List of top runs
        """
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric_name} DESC"],
            max_results=top_n,
        )

        print(f"Top {top_n} runs by {metric_name}:")
        for i, run in runs.iterrows():
            print(
                f"{i+1}. Run {run['run_id'][:8]}: {metric_name}={run[f'metrics.{metric_name}']:.4f}"
            )

        return runs


def create_mlflow_config(config_path: str = "config/mlflow_config.yaml"):
    """Create MLflow configuration file."""
    config = {
        "mlflow": {
            "tracking_uri": f"file://{Path.cwd()}/mlruns",
            "experiment_name": "churn_prediction",
            "model_registry": {
                "staging_threshold": 0.85,  # Minimum AUC for staging
                "production_threshold": 0.90,  # Minimum AUC for production
            },
            "artifacts": {
                "models_path": "models",
                "plots_path": "plots",
                "reports_path": "reports",
            },
        }
    }

    # Create config directory if it doesn't exist
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    # Write config file
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

    print(f"Created MLflow configuration at {config_path}")


def start_mlflow_ui(host: str = "127.0.0.1", port: int = 5000):
    """Start MLflow UI server."""
    import subprocess
    import sys

    cmd = [
        sys.executable,
        "-m",
        "mlflow",
        "ui",
        "--host",
        host,
        "--port",
        str(port),
        "--backend-store-uri",
        f"file://{Path.cwd()}/mlruns",
    ]

    print(f"Starting MLflow UI at http://{host}:{port}")
    print("Press Ctrl+C to stop the server")

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nMLflow UI stopped")


if __name__ == "__main__":
    # Create MLflow configuration
    create_mlflow_config()

    # Initialize MLflow manager
    manager = MLflowManager()

    # Start MLflow UI
    start_mlflow_ui()
