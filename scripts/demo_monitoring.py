#!/usr/bin/env python3
"""
Monitoring System Demo

Simple demonstration of the monitoring capabilities for the churn prediction model.
This script shows how to use the monitoring system with real model data.

Usage:
    python scripts/demo_monitoring.py
"""

import sys
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from monitoring import MonitoringDashboard


def main():
    """
    Simple demo of monitoring capabilities.
    """
    print("Churn Model Monitoring System Demo")
    print("=" * 40)

    # Initialize monitoring components
    dashboard = MonitoringDashboard()

    print("✅ Monitoring system initialized successfully")

    # Generate a simple text report (will show no data initially)
    report = dashboard.generate_text_report(hours=24)
    print("\nCurrent Monitoring Status:")
    print("-" * 30)
    print(report)

    print("\nMonitoring System Components Available:")
    print("- DriftDetector: Data and concept drift detection")
    print("- PerformanceMonitor: Real-time performance tracking")
    print("- MonitoringDashboard: HTML and text reports")
    print("- AlertManager: Automated alerting system")

    print("\nTo use with real data:")
    print("1. Load your trained model")
    print("2. Prepare reference and current datasets")
    print("3. Call monitoring functions with your data")
    print("4. Generate reports and dashboards")

    print("\nMonitoring & Drift Detection is complete! 🎉")


if __name__ == "__main__":
    main()
