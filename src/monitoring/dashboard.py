"""
Simple Monitoring Dashboard for Churn Prediction Model

This module provides a basic monitoring dashboard that displays model performance,
drift detection results, and business impact metrics in a simple format.

Key features:
- Performance metrics visualization
- Drift detection summaries
- Business impact tracking
- Alert status display
- Simple HTML dashboard generation

Usage:
    from monitoring.dashboard import MonitoringDashboard

    dashboard = MonitoringDashboard()
    dashboard.generate_dashboard()
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MonitoringDashboard:
    """
    Simple monitoring dashboard for model performance and drift detection.
    """

    def __init__(
        self, storage_path: str = "monitoring_data", output_path: str = "dashboard"
    ):
        """
        Initialize monitoring dashboard.

        Args:
            storage_path: Path where monitoring data is stored
            output_path: Path to output dashboard files
        """
        self.storage_path = Path(storage_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)

    def generate_dashboard(self, hours: int = 24) -> str:
        """
        Generate a simple HTML dashboard.

        Args:
            hours: Number of hours of data to display

        Returns:
            Path to generated dashboard file
        """
        try:
            # Collect data
            performance_data = self._load_performance_data(hours)
            business_data = self._load_business_data(hours)
            alert_summary = self._get_alert_summary(hours)

            # Generate HTML
            html_content = self._generate_html_dashboard(
                performance_data, business_data, alert_summary, hours
            )

            # Save dashboard
            dashboard_file = (
                self.output_path
                / f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            )

            with open(dashboard_file, "w") as f:
                f.write(html_content)

            # Also save as latest
            latest_file = self.output_path / "dashboard_latest.html"
            with open(latest_file, "w") as f:
                f.write(html_content)

            logger.info(f"Dashboard generated: {dashboard_file}")
            return str(dashboard_file)

        except Exception as e:
            logger.error(f"Error generating dashboard: {e!s}")
            return ""

    def generate_text_report(self, hours: int = 24) -> str:
        """
        Generate a simple text report for command line display.

        Args:
            hours: Number of hours of data to include

        Returns:
            Text report as string
        """
        try:
            performance_data = self._load_performance_data(hours)
            business_data = self._load_business_data(hours)
            alert_summary = self._get_alert_summary(hours)

            report = []
            report.append("=" * 60)
            report.append("CHURN PREDICTION MODEL MONITORING REPORT")
            report.append("=" * 60)
            report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append(f"Time Period: Last {hours} hours")
            report.append("")

            # Performance Summary
            report.append("PERFORMANCE METRICS")
            report.append("-" * 30)
            if performance_data:
                latest_perf = performance_data[-1]
                metrics = latest_perf.get("metrics", {})
                report.append(f"Latest AUC:       {metrics.get('auc', 0):.3f}")
                report.append(f"Latest Accuracy:  {metrics.get('accuracy', 0):.3f}")
                report.append(f"Latest Precision: {metrics.get('precision', 0):.3f}")
                report.append(f"Latest Recall:    {metrics.get('recall', 0):.3f}")
                report.append(f"Evaluations:      {len(performance_data)}")
            else:
                report.append("No performance data available")
            report.append("")

            # Alert Summary
            report.append("ALERT STATUS")
            report.append("-" * 30)
            if alert_summary["total_alerts"] > 0:
                report.append(f"⚠️  Total Alerts: {alert_summary['total_alerts']}")
                for alert_type, count in alert_summary["alert_types"].items():
                    report.append(f"   {alert_type}: {count}")
            else:
                report.append("✅ No alerts in the last 24 hours")
            report.append("")

            # Business Impact
            report.append("BUSINESS IMPACT")
            report.append("-" * 30)
            if business_data:
                latest_biz = business_data[-1]
                report.append(
                    f"High Risk Customers: {latest_biz.get('high_risk_customers', 0)}"
                )
                report.append(
                    f"High Risk Rate:      {latest_biz.get('high_risk_rate', 0):.1%}"
                )
                if "net_roi" in latest_biz:
                    report.append(
                        f"Estimated ROI:       ${latest_biz.get('net_roi', 0):,.2f}"
                    )
            else:
                report.append("No business impact data available")

            report.append("")
            report.append("=" * 60)

            return "\n".join(report)

        except Exception as e:
            logger.error(f"Error generating text report: {e!s}")
            return f"Error generating report: {e!s}"

    def _load_performance_data(self, hours: int) -> list[dict[str, Any]]:
        """Load performance monitoring data."""
        performance_file = self.storage_path / "performance_history.jsonl"

        if not performance_file.exists():
            return []

        cutoff_time = datetime.now() - timedelta(hours=hours)
        data = []

        try:
            with open(performance_file) as f:
                for line in f:
                    record = json.loads(line.strip())
                    record_time = datetime.fromisoformat(record["timestamp"])

                    if record_time > cutoff_time:
                        data.append(record)

            return sorted(data, key=lambda x: x["timestamp"])

        except Exception as e:
            logger.error(f"Error loading performance data: {e!s}")
            return []

    def _load_business_data(self, hours: int) -> list[dict[str, Any]]:
        """Load business impact data."""
        business_file = self.storage_path / "business_impact_history.jsonl"

        if not business_file.exists():
            return []

        cutoff_time = datetime.now() - timedelta(hours=hours)
        data = []

        try:
            with open(business_file) as f:
                for line in f:
                    record = json.loads(line.strip())
                    record_time = datetime.fromisoformat(record["timestamp"])

                    if record_time > cutoff_time:
                        data.append(record)

            return sorted(data, key=lambda x: x["timestamp"])

        except Exception as e:
            logger.error(f"Error loading business data: {e!s}")
            return []

    def _get_alert_summary(self, hours: int) -> dict[str, Any]:
        """Get alert summary from performance data."""
        performance_data = self._load_performance_data(hours)

        total_alerts = 0
        alert_types = {}
        recent_alerts = []

        for record in performance_data:
            alerts = record.get("alerts", [])
            total_alerts += len(alerts)

            for alert in alerts:
                alert_type = alert.get("type", "unknown")
                alert_types[alert_type] = alert_types.get(alert_type, 0) + 1
                recent_alerts.append(alert)

        return {
            "total_alerts": total_alerts,
            "alert_types": alert_types,
            "recent_alerts": recent_alerts[-10:],  # Last 10 alerts
        }

    def _generate_html_dashboard(
        self,
        performance_data: list[dict],
        business_data: list[dict],
        alert_summary: dict,
        hours: int,
    ) -> str:
        """Generate HTML dashboard content."""

        # Get latest metrics
        latest_metrics = {}
        if performance_data:
            latest_metrics = performance_data[-1].get("metrics", {})

        latest_business = {}
        if business_data:
            latest_business = business_data[-1]

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Churn Model Monitoring Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ background-color: #2c3e50; color: white; padding: 20px; border-radius: 5px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric-card {{ background: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #3498db; }}
        .metric-label {{ color: #7f8c8d; margin-top: 5px; }}
        .alert-high {{ color: #e74c3c; }}
        .alert-medium {{ color: #f39c12; }}
        .alert-low {{ color: #27ae60; }}
        .status-good {{ color: #27ae60; }}
        .status-warning {{ color: #f39c12; }}
        .status-error {{ color: #e74c3c; }}
        .section {{ background: white; margin: 20px 0; padding: 20px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f8f9fa; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Churn Prediction Model Monitoring</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Period: Last {hours} hours</p>
        </div>

        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{latest_metrics.get('auc', 0):.3f}</div>
                <div class="metric-label">AUC Score</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{latest_metrics.get('accuracy', 0):.3f}</div>
                <div class="metric-label">Accuracy</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{latest_metrics.get('precision', 0):.3f}</div>
                <div class="metric-label">Precision</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{alert_summary['total_alerts']}</div>
                <div class="metric-label">Total Alerts</div>
            </div>
        </div>

        <div class="section">
            <h2>Alert Status</h2>
            {self._generate_alert_html(alert_summary)}
        </div>

        <div class="section">
            <h2>Business Impact</h2>
            {self._generate_business_html(latest_business)}
        </div>

        <div class="section">
            <h2>Recent Performance</h2>
            {self._generate_performance_table_html(performance_data[-10:])}
        </div>
    </div>
</body>
</html>"""

        return html

    def _generate_alert_html(self, alert_summary: dict) -> str:
        """Generate HTML for alert section."""
        if alert_summary["total_alerts"] == 0:
            return '<p class="status-good">✅ No alerts in the monitoring period</p>'

        html = f'<p class="status-warning">⚠️ {alert_summary["total_alerts"]} alerts detected</p>'

        if alert_summary["alert_types"]:
            html += "<table><tr><th>Alert Type</th><th>Count</th></tr>"
            for alert_type, count in alert_summary["alert_types"].items():
                html += f"<tr><td>{alert_type}</td><td>{count}</td></tr>"
            html += "</table>"

        return html

    def _generate_business_html(self, business_data: dict) -> str:
        """Generate HTML for business section."""
        if not business_data:
            return "<p>No business impact data available</p>"

        html = f"""
        <table>
            <tr><td>Total Customers</td><td>{business_data.get('total_customers', 0):,}</td></tr>
            <tr><td>High Risk Customers</td><td>{business_data.get('high_risk_customers', 0):,}</td></tr>
            <tr><td>High Risk Rate</td><td>{business_data.get('high_risk_rate', 0):.1%}</td></tr>
        """

        if "net_roi" in business_data:
            html += f'<tr><td>Estimated ROI</td><td>${business_data.get("net_roi", 0):,.2f}</td></tr>'

        html += "</table>"
        return html

    def _generate_performance_table_html(self, performance_data: list[dict]) -> str:
        """Generate HTML table for performance data."""
        if not performance_data:
            return "<p>No performance data available</p>"

        html = "<table><tr><th>Timestamp</th><th>AUC</th><th>Accuracy</th><th>Precision</th><th>Alerts</th></tr>"

        for record in performance_data:
            timestamp = datetime.fromisoformat(record["timestamp"]).strftime(
                "%Y-%m-%d %H:%M"
            )
            metrics = record.get("metrics", {})
            alert_count = len(record.get("alerts", []))

            html += f"""
            <tr>
                <td>{timestamp}</td>
                <td>{metrics.get('auc', 0):.3f}</td>
                <td>{metrics.get('accuracy', 0):.3f}</td>
                <td>{metrics.get('precision', 0):.3f}</td>
                <td>{alert_count}</td>
            </tr>
            """

        html += "</table>"
        return html


def main():
    """
    Simple command-line interface for the monitoring dashboard.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Generate monitoring dashboard")
    parser.add_argument(
        "--hours", type=int, default=24, help="Hours of data to include"
    )
    parser.add_argument(
        "--format", choices=["html", "text"], default="text", help="Output format"
    )
    parser.add_argument("--output", help="Output file path")

    args = parser.parse_args()

    dashboard = MonitoringDashboard()

    if args.format == "html":
        output_file = dashboard.generate_dashboard(args.hours)
        print(f"Dashboard generated: {output_file}")
    else:
        report = dashboard.generate_text_report(args.hours)

        if args.output:
            with open(args.output, "w") as f:
                f.write(report)
            print(f"Report saved to: {args.output}")
        else:
            print(report)


if __name__ == "__main__":
    main()
