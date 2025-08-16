"""
Model Monitoring and Drift Detection System

This module provides comprehensive monitoring capabilities for churn prediction models,
including data drift detection, concept drift detection, and performance monitoring.

Key features:
- Statistical tests for feature distribution changes (KS test, PSI)
- Model performance drift detection
- Prediction distribution tracking
- Automated alerting for model degradation
- Data quality monitoring

Usage:
    from monitoring.drift_detector import DriftDetector

    detector = DriftDetector()
    drift_report = detector.detect_drift(reference_data, current_data, model)
"""

import logging
import warnings
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DriftDetector:
    """
    Comprehensive drift detection system for monitoring model performance
    and data quality changes over time.
    """

    def __init__(
        self,
        ks_threshold: float = 0.05,
        psi_threshold: float = 0.2,
        performance_threshold: float = 0.05,
        min_samples: int = 100,
    ):
        """
        Initialize drift detector with configurable thresholds.

        Args:
            ks_threshold: P-value threshold for Kolmogorov-Smirnov test
            psi_threshold: Population Stability Index threshold
            performance_threshold: Acceptable performance degradation
            min_samples: Minimum samples required for drift detection
        """
        self.ks_threshold = ks_threshold
        self.psi_threshold = psi_threshold
        self.performance_threshold = performance_threshold
        self.min_samples = min_samples

    def detect_data_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        features: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Detect data drift using statistical tests.

        Args:
            reference_data: Historical reference dataset
            current_data: Current dataset to compare against reference
            features: List of features to check (all numeric if None)

        Returns:
            Dictionary with drift detection results
        """
        if len(current_data) < self.min_samples:
            logger.warning(
                f"Insufficient samples for drift detection: {len(current_data)}"
            )
            return {"status": "insufficient_data", "sample_count": len(current_data)}

        if features is None:
            features = reference_data.select_dtypes(
                include=[np.number]
            ).columns.tolist()

        drift_results = {
            "timestamp": datetime.now().isoformat(),
            "features_tested": len(features),
            "sample_count": len(current_data),
            "drift_detected": False,
            "feature_drift": {},
            "summary": {},
        }

        drifted_features = []

        for feature in features:
            if (
                feature not in reference_data.columns
                or feature not in current_data.columns
            ):
                continue

            # Remove missing values
            ref_values = reference_data[feature].dropna()
            curr_values = current_data[feature].dropna()

            if len(ref_values) == 0 or len(curr_values) == 0:
                continue

            # Kolmogorov-Smirnov test
            ks_stat, ks_pvalue = stats.ks_2samp(ref_values, curr_values)

            # Population Stability Index
            psi = self._calculate_psi(ref_values, curr_values)

            feature_drift = {
                "ks_statistic": float(ks_stat),
                "ks_pvalue": float(ks_pvalue),
                "psi": float(psi),
                "drift_detected": ks_pvalue < self.ks_threshold
                or psi > self.psi_threshold,
            }

            drift_results["feature_drift"][feature] = feature_drift

            if feature_drift["drift_detected"]:
                drifted_features.append(feature)

        drift_results["drift_detected"] = len(drifted_features) > 0
        drift_results["drifted_features"] = drifted_features
        drift_results["drift_percentage"] = (
            len(drifted_features) / len(features) if features else 0
        )

        return drift_results

    def detect_concept_drift(
        self,
        model: BaseEstimator,
        reference_data: pd.DataFrame,
        reference_targets: pd.Series,
        current_data: pd.DataFrame,
        current_targets: pd.Series,
        features: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Detect concept drift by comparing model performance.

        Args:
            model: Trained model to evaluate
            reference_data: Historical reference dataset
            reference_targets: Historical target values
            current_data: Current dataset
            current_targets: Current target values
            features: Features to use for prediction

        Returns:
            Dictionary with concept drift results
        """
        if len(current_data) < self.min_samples:
            logger.warning(
                f"Insufficient samples for concept drift detection: {len(current_data)}"
            )
            return {"status": "insufficient_data", "sample_count": len(current_data)}

        if features is None:
            features = [
                col for col in reference_data.columns if col in current_data.columns
            ]

        try:
            # Calculate reference performance
            ref_predictions = model.predict_proba(reference_data[features])[:, 1]
            ref_metrics = self._calculate_performance_metrics(
                reference_targets, ref_predictions
            )

            # Calculate current performance
            curr_predictions = model.predict_proba(current_data[features])[:, 1]
            curr_metrics = self._calculate_performance_metrics(
                current_targets, curr_predictions
            )

            # Calculate performance degradation
            auc_degradation = ref_metrics["auc"] - curr_metrics["auc"]
            accuracy_degradation = ref_metrics["accuracy"] - curr_metrics["accuracy"]

            concept_drift_detected = (
                auc_degradation > self.performance_threshold
                or accuracy_degradation > self.performance_threshold
            )

            return {
                "timestamp": datetime.now().isoformat(),
                "sample_count": len(current_data),
                "reference_metrics": ref_metrics,
                "current_metrics": curr_metrics,
                "auc_degradation": float(auc_degradation),
                "accuracy_degradation": float(accuracy_degradation),
                "concept_drift_detected": concept_drift_detected,
                "severity": (
                    "high"
                    if auc_degradation > 0.1
                    else "medium" if auc_degradation > 0.05 else "low"
                ),
            }

        except Exception as e:
            logger.error(f"Error detecting concept drift: {e!s}")
            return {"status": "error", "message": str(e)}

    def track_prediction_distribution(
        self,
        model: BaseEstimator,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        features: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Track changes in prediction distributions.

        Args:
            model: Trained model
            reference_data: Historical reference dataset
            current_data: Current dataset
            features: Features to use for prediction

        Returns:
            Dictionary with prediction distribution analysis
        """
        if features is None:
            features = [
                col for col in reference_data.columns if col in current_data.columns
            ]

        try:
            # Get predictions
            ref_predictions = model.predict_proba(reference_data[features])[:, 1]
            curr_predictions = model.predict_proba(current_data[features])[:, 1]

            # Statistical comparison
            ks_stat, ks_pvalue = stats.ks_2samp(ref_predictions, curr_predictions)

            # Distribution statistics
            ref_stats = {
                "mean": float(np.mean(ref_predictions)),
                "std": float(np.std(ref_predictions)),
                "median": float(np.median(ref_predictions)),
                "q25": float(np.percentile(ref_predictions, 25)),
                "q75": float(np.percentile(ref_predictions, 75)),
            }

            curr_stats = {
                "mean": float(np.mean(curr_predictions)),
                "std": float(np.std(curr_predictions)),
                "median": float(np.median(curr_predictions)),
                "q25": float(np.percentile(curr_predictions, 25)),
                "q75": float(np.percentile(curr_predictions, 75)),
            }

            return {
                "timestamp": datetime.now().isoformat(),
                "sample_count": len(current_data),
                "ks_statistic": float(ks_stat),
                "ks_pvalue": float(ks_pvalue),
                "distribution_shift": ks_pvalue < self.ks_threshold,
                "reference_stats": ref_stats,
                "current_stats": curr_stats,
                "mean_shift": abs(curr_stats["mean"] - ref_stats["mean"]),
            }

        except Exception as e:
            logger.error(f"Error tracking prediction distribution: {e!s}")
            return {"status": "error", "message": str(e)}

    def generate_monitoring_report(
        self,
        model: BaseEstimator,
        reference_data: pd.DataFrame,
        reference_targets: pd.Series,
        current_data: pd.DataFrame,
        current_targets: pd.Series | None = None,
        features: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Generate comprehensive monitoring report.

        Args:
            model: Trained model
            reference_data: Historical reference dataset
            reference_targets: Historical target values
            current_data: Current dataset
            current_targets: Current target values (optional)
            features: Features to use

        Returns:
            Comprehensive monitoring report
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "data_drift": self.detect_data_drift(
                reference_data, current_data, features
            ),
            "prediction_drift": self.track_prediction_distribution(
                model, reference_data, current_data, features
            ),
        }

        # Add concept drift if targets available
        if current_targets is not None:
            report["concept_drift"] = self.detect_concept_drift(
                model,
                reference_data,
                reference_targets,
                current_data,
                current_targets,
                features,
            )

        # Overall alert status
        alerts = []
        if report["data_drift"].get("drift_detected", False):
            alerts.append("data_drift")
        if report["prediction_drift"].get("distribution_shift", False):
            alerts.append("prediction_drift")
        if report.get("concept_drift", {}).get("concept_drift_detected", False):
            alerts.append("concept_drift")

        report["alert_status"] = {
            "alerts_triggered": alerts,
            "alert_count": len(alerts),
            "requires_attention": len(alerts) > 0,
        }

        return report

    def _calculate_psi(
        self, reference: pd.Series, current: pd.Series, bins: int = 10
    ) -> float:
        """
        Calculate Population Stability Index (PSI).

        Args:
            reference: Reference distribution
            current: Current distribution
            bins: Number of bins for discretization

        Returns:
            PSI value
        """
        try:
            # Create bins based on reference data
            _, bin_edges = np.histogram(reference, bins=bins)

            # Calculate distributions
            ref_counts, _ = np.histogram(reference, bins=bin_edges)
            curr_counts, _ = np.histogram(current, bins=bin_edges)

            # Convert to percentages
            ref_perc = ref_counts / len(reference)
            curr_perc = curr_counts / len(current)

            # Avoid division by zero
            ref_perc = np.where(ref_perc == 0, 0.0001, ref_perc)
            curr_perc = np.where(curr_perc == 0, 0.0001, curr_perc)

            # Calculate PSI
            psi = np.sum((curr_perc - ref_perc) * np.log(curr_perc / ref_perc))

            return psi

        except Exception as e:
            logger.warning(f"Error calculating PSI: {e!s}")
            return 0.0

    def _calculate_performance_metrics(
        self, y_true: pd.Series, y_pred_proba: np.ndarray
    ) -> dict[str, float]:
        """
        Calculate performance metrics.

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities

        Returns:
            Dictionary of performance metrics
        """
        try:
            y_pred = (y_pred_proba > 0.5).astype(int)

            return {
                "auc": float(roc_auc_score(y_true, y_pred_proba)),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            }

        except Exception as e:
            logger.error(f"Error calculating metrics: {e!s}")
            return {"auc": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0}


class AlertManager:
    """
    Manages alerting for model monitoring events.
    """

    def __init__(self):
        self.alert_history = []

    def check_alerts(self, monitoring_report: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Check monitoring report for alert conditions.

        Args:
            monitoring_report: Report from DriftDetector

        Returns:
            List of alerts to trigger
        """
        alerts = []

        # Data drift alerts
        if monitoring_report["data_drift"].get("drift_detected", False):
            drift_pct = monitoring_report["data_drift"].get("drift_percentage", 0)
            alerts.append(
                {
                    "type": "data_drift",
                    "severity": "high" if drift_pct > 0.5 else "medium",
                    "message": f"Data drift detected in {drift_pct:.1%} of features",
                    "timestamp": datetime.now().isoformat(),
                }
            )

        # Prediction drift alerts
        if monitoring_report["prediction_drift"].get("distribution_shift", False):
            mean_shift = monitoring_report["prediction_drift"].get("mean_shift", 0)
            alerts.append(
                {
                    "type": "prediction_drift",
                    "severity": "high" if mean_shift > 0.1 else "medium",
                    "message": f"Prediction distribution shift detected (mean shift: {mean_shift:.3f})",
                    "timestamp": datetime.now().isoformat(),
                }
            )

        # Concept drift alerts
        concept_drift = monitoring_report.get("concept_drift", {})
        if concept_drift.get("concept_drift_detected", False):
            auc_degradation = concept_drift.get("auc_degradation", 0)
            alerts.append(
                {
                    "type": "concept_drift",
                    "severity": concept_drift.get("severity", "medium"),
                    "message": f"Model performance degradation detected (AUC drop: {auc_degradation:.3f})",
                    "timestamp": datetime.now().isoformat(),
                }
            )

        self.alert_history.extend(alerts)
        return alerts

    def get_alert_summary(self, hours: int = 24) -> dict[str, Any]:
        """
        Get summary of recent alerts.

        Args:
            hours: Number of hours to look back

        Returns:
            Alert summary
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_alerts = [
            alert
            for alert in self.alert_history
            if datetime.fromisoformat(alert["timestamp"]) > cutoff_time
        ]

        return {
            "total_alerts": len(recent_alerts),
            "alert_types": list({alert["type"] for alert in recent_alerts}),
            "severity_counts": {
                "high": len([a for a in recent_alerts if a["severity"] == "high"]),
                "medium": len([a for a in recent_alerts if a["severity"] == "medium"]),
                "low": len([a for a in recent_alerts if a["severity"] == "low"]),
            },
            "recent_alerts": recent_alerts[-10:],  # Last 10 alerts
        }
