"""
Monitoring Module for Churn Prediction Model

This module provides comprehensive monitoring capabilities including:
- Data and concept drift detection
- Model performance monitoring
- Business impact tracking
- Alerting and dashboard generation

Main Components:
- DriftDetector: Detects data and concept drift
- PerformanceMonitor: Tracks model performance metrics
- MonitoringDashboard: Generates monitoring reports and dashboards
- AlertManager: Manages alerting for monitoring events

Usage:
    from monitoring import DriftDetector, PerformanceMonitor, MonitoringDashboard

    # Set up monitoring
    drift_detector = DriftDetector()
    performance_monitor = PerformanceMonitor()
    dashboard = MonitoringDashboard()

    # Monitor model
    drift_report = drift_detector.generate_monitoring_report(
        model, reference_data, reference_targets, current_data, current_targets
    )

    performance_report = performance_monitor.track_performance(
        model, current_data, current_targets
    )

    # Generate dashboard
    dashboard_file = dashboard.generate_dashboard()
"""

from .dashboard import MonitoringDashboard
from .drift_detector import AlertManager, DriftDetector
from .performance_monitor import BusinessImpactMonitor, PerformanceMonitor

__all__ = [
    "AlertManager",
    "BusinessImpactMonitor",
    "DriftDetector",
    "MonitoringDashboard",
    "PerformanceMonitor",
]
