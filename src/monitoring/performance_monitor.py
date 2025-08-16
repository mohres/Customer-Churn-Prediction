"""
Model Performance Monitoring System

This module provides real-time monitoring of model performance metrics,
automated alerting, and performance tracking over time.

Key features:
- Real-time performance metric calculation
- Performance threshold monitoring
- Automated alerting for degradation
- Historical performance tracking
- Business impact monitoring

Usage:
    from monitoring.performance_monitor import PerformanceMonitor

    monitor = PerformanceMonitor()
    results = monitor.track_performance(model, data, targets)
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system for churn prediction models.
    """

    def __init__(
        self,
        storage_path: str = "monitoring_data",
        auc_threshold: float = 0.8,
        accuracy_threshold: float = 0.8,
        precision_threshold: float = 0.7,
        alert_cooldown_hours: int = 1,
    ):
        """
        Initialize performance monitor.

        Args:
            storage_path: Path to store monitoring data
            auc_threshold: Minimum acceptable AUC score
            accuracy_threshold: Minimum acceptable accuracy
            precision_threshold: Minimum acceptable precision
            alert_cooldown_hours: Hours to wait between similar alerts
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)

        self.thresholds = {
            "auc": auc_threshold,
            "accuracy": accuracy_threshold,
            "precision": precision_threshold,
        }

        self.alert_cooldown = timedelta(hours=alert_cooldown_hours)
        self.last_alerts = {}

    def track_performance(
        self,
        model: BaseEstimator,
        data: pd.DataFrame,
        targets: pd.Series,
        features: list[str] | None = None,
        model_version: str = "current",
    ) -> dict[str, Any]:
        """
        Track model performance and generate alerts if needed.

        Args:
            model: Trained model to evaluate
            data: Input data for predictions
            targets: True target values
            features: Feature columns to use
            model_version: Version identifier for the model

        Returns:
            Performance tracking results
        """
        timestamp = datetime.now()

        if features is None:
            features = [col for col in data.columns if col != "target"]

        try:
            # Generate predictions
            predictions_proba = model.predict_proba(data[features])[:, 1]
            predictions = (predictions_proba > 0.5).astype(int)

            # Calculate metrics
            metrics = self._calculate_comprehensive_metrics(
                targets, predictions, predictions_proba
            )

            # Check for performance issues
            alerts = self._check_performance_alerts(metrics, timestamp)

            # Create performance record
            performance_record = {
                "timestamp": timestamp.isoformat(),
                "model_version": model_version,
                "sample_count": len(data),
                "metrics": metrics,
                "alerts": alerts,
                "threshold_violations": self._check_threshold_violations(metrics),
            }

            # Store performance data
            self._store_performance_data(performance_record)

            logger.info(
                f"Performance tracking completed: AUC={metrics['auc']:.3f}, "
                f"Accuracy={metrics['accuracy']:.3f}, {len(alerts)} alerts"
            )

            return performance_record

        except Exception as e:
            logger.error(f"Error tracking performance: {e!s}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": timestamp.isoformat(),
            }

    def get_performance_history(
        self, hours: int = 24, model_version: str | None = None
    ) -> list[dict[str, Any]]:
        """
        Get historical performance data.

        Args:
            hours: Number of hours to look back
            model_version: Filter by model version

        Returns:
            List of performance records
        """
        performance_file = self.storage_path / "performance_history.jsonl"

        if not performance_file.exists():
            return []

        cutoff_time = datetime.now() - timedelta(hours=hours)
        history = []

        try:
            with open(performance_file) as f:
                for line in f:
                    record = json.loads(line.strip())
                    record_time = datetime.fromisoformat(record["timestamp"])

                    if record_time > cutoff_time and (
                        model_version is None
                        or record.get("model_version") == model_version
                    ):
                        history.append(record)

            return sorted(history, key=lambda x: x["timestamp"])

        except Exception as e:
            logger.error(f"Error reading performance history: {e!s}")
            return []

    def get_performance_summary(self, hours: int = 24) -> dict[str, Any]:
        """
        Get performance summary for the specified time period.

        Args:
            hours: Number of hours to analyze

        Returns:
            Performance summary statistics
        """
        history = self.get_performance_history(hours)

        if not history:
            return {"status": "no_data", "message": "No performance data available"}

        # Extract metrics
        auc_scores = [record["metrics"]["auc"] for record in history]
        accuracy_scores = [record["metrics"]["accuracy"] for record in history]
        precision_scores = [record["metrics"]["precision"] for record in history]

        # Count alerts
        total_alerts = sum(len(record.get("alerts", [])) for record in history)
        alert_types = {}

        for record in history:
            for alert in record.get("alerts", []):
                alert_type = alert.get("type", "unknown")
                alert_types[alert_type] = alert_types.get(alert_type, 0) + 1

        return {
            "time_period_hours": hours,
            "total_evaluations": len(history),
            "metrics_summary": {
                "auc": {
                    "mean": float(np.mean(auc_scores)),
                    "std": float(np.std(auc_scores)),
                    "min": float(np.min(auc_scores)),
                    "max": float(np.max(auc_scores)),
                    "latest": auc_scores[-1] if auc_scores else 0,
                },
                "accuracy": {
                    "mean": float(np.mean(accuracy_scores)),
                    "std": float(np.std(accuracy_scores)),
                    "min": float(np.min(accuracy_scores)),
                    "max": float(np.max(accuracy_scores)),
                    "latest": accuracy_scores[-1] if accuracy_scores else 0,
                },
                "precision": {
                    "mean": float(np.mean(precision_scores)),
                    "std": float(np.std(precision_scores)),
                    "min": float(np.min(precision_scores)),
                    "max": float(np.max(precision_scores)),
                    "latest": precision_scores[-1] if precision_scores else 0,
                },
            },
            "alert_summary": {
                "total_alerts": total_alerts,
                "alert_types": alert_types,
                "alert_rate": total_alerts / len(history) if history else 0,
            },
        }

    def _calculate_comprehensive_metrics(
        self, y_true: pd.Series, y_pred: np.ndarray, y_pred_proba: np.ndarray
    ) -> dict[str, float]:
        """
        Calculate comprehensive performance metrics.

        Args:
            y_true: True labels
            y_pred: Predicted binary labels
            y_pred_proba: Predicted probabilities

        Returns:
            Dictionary of performance metrics
        """
        try:
            return {
                "auc": float(roc_auc_score(y_true, y_pred_proba)),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "true_positives": int(confusion_matrix(y_true, y_pred)[1, 1]),
                "false_positives": int(confusion_matrix(y_true, y_pred)[0, 1]),
                "true_negatives": int(confusion_matrix(y_true, y_pred)[0, 0]),
                "false_negatives": int(confusion_matrix(y_true, y_pred)[1, 0]),
                "positive_rate": float(np.mean(y_pred)),
                "prediction_mean": float(np.mean(y_pred_proba)),
                "prediction_std": float(np.std(y_pred_proba)),
            }

        except Exception as e:
            logger.error(f"Error calculating metrics: {e!s}")
            return {
                "auc": 0.0,
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "true_positives": 0,
                "false_positives": 0,
                "true_negatives": 0,
                "false_negatives": 0,
                "positive_rate": 0.0,
                "prediction_mean": 0.0,
                "prediction_std": 0.0,
            }

    def _check_performance_alerts(
        self, metrics: dict[str, float], timestamp: datetime
    ) -> list[dict[str, Any]]:
        """
        Check for performance alerts based on thresholds.

        Args:
            metrics: Performance metrics
            timestamp: Current timestamp

        Returns:
            List of alerts to trigger
        """
        alerts = []

        for metric_name, threshold in self.thresholds.items():
            if metric_name in metrics and metrics[metric_name] < threshold:
                # Check cooldown period
                last_alert_key = f"{metric_name}_below_threshold"
                last_alert_time = self.last_alerts.get(last_alert_key)

                if (
                    last_alert_time is None
                    or timestamp - last_alert_time > self.alert_cooldown
                ):
                    severity = (
                        "high" if metrics[metric_name] < threshold * 0.9 else "medium"
                    )

                    alerts.append(
                        {
                            "type": "performance_degradation",
                            "metric": metric_name,
                            "value": metrics[metric_name],
                            "threshold": threshold,
                            "severity": severity,
                            "message": f"{metric_name.upper()} below threshold: {metrics[metric_name]:.3f} < {threshold}",
                            "timestamp": timestamp.isoformat(),
                        }
                    )

                    self.last_alerts[last_alert_key] = timestamp

        return alerts

    def _check_threshold_violations(self, metrics: dict[str, float]) -> dict[str, bool]:
        """
        Check which thresholds are violated.

        Args:
            metrics: Performance metrics

        Returns:
            Dictionary of threshold violations
        """
        violations = {}

        for metric_name, threshold in self.thresholds.items():
            if metric_name in metrics:
                violations[metric_name] = metrics[metric_name] < threshold

        return violations

    def _store_performance_data(self, performance_record: dict[str, Any]) -> None:
        """
        Store performance data to file.

        Args:
            performance_record: Performance record to store
        """
        try:
            performance_file = self.storage_path / "performance_history.jsonl"

            with open(performance_file, "a") as f:
                f.write(json.dumps(performance_record) + "\n")

        except Exception as e:
            logger.error(f"Error storing performance data: {e!s}")


class BusinessImpactMonitor:
    """
    Monitor business impact metrics for churn prediction model.
    """

    def __init__(self, storage_path: str = "monitoring_data"):
        """
        Initialize business impact monitor.

        Args:
            storage_path: Path to store monitoring data
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)

    def track_business_impact(
        self,
        predictions: np.ndarray,
        actual_outcomes: np.ndarray | None = None,
        intervention_cost: float = 10.0,
        revenue_per_customer: float = 50.0,
        timestamp: datetime | None = None,
    ) -> dict[str, Any]:
        """
        Track business impact of model predictions.

        Args:
            predictions: Model predictions (probabilities)
            actual_outcomes: Actual churn outcomes (if available)
            intervention_cost: Cost of intervention per customer
            revenue_per_customer: Average revenue per customer
            timestamp: Timestamp for the measurement

        Returns:
            Business impact metrics
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Calculate prediction-based metrics
        high_risk_customers = np.sum(predictions > 0.5)
        medium_risk_customers = np.sum((predictions > 0.3) & (predictions <= 0.5))
        low_risk_customers = np.sum(predictions <= 0.3)

        total_customers = len(predictions)

        business_metrics = {
            "timestamp": timestamp.isoformat(),
            "total_customers": int(total_customers),
            "high_risk_customers": int(high_risk_customers),
            "medium_risk_customers": int(medium_risk_customers),
            "low_risk_customers": int(low_risk_customers),
            "high_risk_rate": float(high_risk_customers / total_customers),
            "predicted_intervention_cost": float(
                high_risk_customers * intervention_cost
            ),
            "potential_revenue_at_risk": float(
                high_risk_customers * revenue_per_customer
            ),
        }

        # If actual outcomes are available, calculate ROI
        if actual_outcomes is not None:
            true_positives = np.sum((predictions > 0.5) & (actual_outcomes == 1))
            false_positives = np.sum((predictions > 0.5) & (actual_outcomes == 0))

            # Assuming intervention prevents 80% of predicted churners
            prevented_churn = int(true_positives * 0.8)
            wasted_interventions = false_positives

            total_intervention_cost = (
                true_positives + false_positives
            ) * intervention_cost
            revenue_saved = prevented_churn * revenue_per_customer
            net_roi = revenue_saved - total_intervention_cost

            business_metrics.update(
                {
                    "true_positives": int(true_positives),
                    "false_positives": int(false_positives),
                    "prevented_churn_estimated": prevented_churn,
                    "wasted_interventions": int(wasted_interventions),
                    "total_intervention_cost": float(total_intervention_cost),
                    "revenue_saved": float(revenue_saved),
                    "net_roi": float(net_roi),
                    "roi_ratio": (
                        float(net_roi / total_intervention_cost)
                        if total_intervention_cost > 0
                        else 0.0
                    ),
                }
            )

        # Store business metrics
        self._store_business_data(business_metrics)

        return business_metrics

    def _store_business_data(self, business_record: dict[str, Any]) -> None:
        """
        Store business impact data to file.

        Args:
            business_record: Business impact record to store
        """
        try:
            business_file = self.storage_path / "business_impact_history.jsonl"

            with open(business_file, "a") as f:
                f.write(json.dumps(business_record) + "\n")

        except Exception as e:
            logger.error(f"Error storing business data: {e!s}")
