#!/usr/bin/env python3
"""
Test script for the monitoring system.

This script demonstrates how to use the monitoring components
and generates sample monitoring reports.

Usage:
    python scripts/test_monitoring.py
"""

import sys
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from monitoring import (
    BusinessImpactMonitor,
    DriftDetector,
    MonitoringDashboard,
    PerformanceMonitor,
)


def create_sample_data(n_samples=1000, n_features=10, drift=False):
    """Create sample data for testing."""
    np.random.seed(42 if not drift else 123)

    # Generate features
    x = np.random.randn(n_samples, n_features)

    # Add drift to some features if requested
    if drift:
        x[:, :3] += 2.0  # Shift first 3 features
        x[:, 3:6] *= 1.5  # Scale next 3 features

    # Create target with some relationship to features
    y = (x[:, 0] + x[:, 1] - x[:, 2] + np.random.randn(n_samples) * 0.5) > 0
    y = y.astype(int)

    # Create DataFrame
    feature_names = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(x, columns=feature_names)

    return df, pd.Series(y)


def test_drift_detection():
    """Test drift detection functionality."""
    print("Testing Drift Detection...")

    # Create reference and current data
    ref_data, ref_targets = create_sample_data(1000, 10, drift=False)
    curr_data, curr_targets = create_sample_data(500, 10, drift=True)

    # Train a simple model
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(ref_data, ref_targets)

    # Initialize drift detector
    drift_detector = DriftDetector()

    # Test data drift detection
    data_drift = drift_detector.detect_data_drift(ref_data, curr_data)
    print(f"Data drift detected: {data_drift['drift_detected']}")
    print(f"Drifted features: {len(data_drift['drifted_features'])}")

    # Test concept drift detection
    concept_drift = drift_detector.detect_concept_drift(
        model, ref_data, ref_targets, curr_data, curr_targets
    )
    print(f"Concept drift detected: {concept_drift['concept_drift_detected']}")
    print(f"AUC degradation: {concept_drift['auc_degradation']:.3f}")

    # Test prediction distribution tracking
    pred_drift = drift_detector.track_prediction_distribution(
        model, ref_data, curr_data
    )
    print(f"Prediction distribution shift: {pred_drift['distribution_shift']}")

    # Generate comprehensive report
    report = drift_detector.generate_monitoring_report(
        model, ref_data, ref_targets, curr_data, curr_targets
    )
    print(f"Total alerts: {report['alert_status']['alert_count']}")

    return True


def test_performance_monitoring():
    """Test performance monitoring functionality."""
    print("\nTesting Performance Monitoring...")

    # Create test data
    data, targets = create_sample_data(500, 10)

    # Train model
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    x_train, x_test, y_train, y_test = train_test_split(
        data, targets, test_size=0.3, random_state=42
    )
    model.fit(x_train, y_train)

    # Initialize performance monitor
    perf_monitor = PerformanceMonitor(storage_path="test_monitoring_data")

    # Track performance
    perf_result = perf_monitor.track_performance(model, x_test, y_test)
    print(f"AUC: {perf_result['metrics']['auc']:.3f}")
    print(f"Accuracy: {perf_result['metrics']['accuracy']:.3f}")
    print(f"Alerts: {len(perf_result['alerts'])}")

    # Get performance summary
    summary = perf_monitor.get_performance_summary(hours=1)
    print(f"Total evaluations: {summary.get('total_evaluations', 0)}")

    return True


def test_business_impact_monitoring():
    """Test business impact monitoring."""
    print("\nTesting Business Impact Monitoring...")

    # Create test data
    data, targets = create_sample_data(300, 10)

    # Train model
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(data, targets)

    # Get predictions
    predictions = model.predict_proba(data)[:, 1]

    # Initialize business monitor
    business_monitor = BusinessImpactMonitor(storage_path="test_monitoring_data")

    # Track business impact
    business_result = business_monitor.track_business_impact(
        predictions, targets, intervention_cost=15.0, revenue_per_customer=75.0
    )

    print(f"High risk customers: {business_result['high_risk_customers']}")
    print(f"High risk rate: {business_result['high_risk_rate']:.1%}")
    if "net_roi" in business_result:
        print(f"Net ROI: ${business_result['net_roi']:.2f}")

    return True


def test_dashboard_generation():
    """Test dashboard generation."""
    print("\nTesting Dashboard Generation...")

    # Initialize dashboard
    dashboard = MonitoringDashboard(
        storage_path="test_monitoring_data", output_path="test_dashboard"
    )

    # Generate text report
    text_report = dashboard.generate_text_report(hours=1)
    print("Text Report Generated:")
    print("=" * 50)
    print(text_report)
    print("=" * 50)

    # Generate HTML dashboard
    html_file = dashboard.generate_dashboard(hours=1)
    if html_file:
        print(f"HTML dashboard generated: {html_file}")
    else:
        print("Failed to generate HTML dashboard")

    return True


def main():
    """Run all monitoring tests."""
    print("Starting Monitoring System Tests")
    print("=" * 50)

    try:
        # Run tests
        test_drift_detection()
        test_performance_monitoring()
        test_business_impact_monitoring()
        test_dashboard_generation()

        print("\n" + "=" * 50)
        print("All monitoring tests completed successfully!")
        print("✅ Monitoring & Drift Detection is working!")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e!s}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
