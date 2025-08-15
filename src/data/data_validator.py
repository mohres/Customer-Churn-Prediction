"""
Data validation module for customer churn prediction pipeline.

This module provides comprehensive data quality validation including:
- Schema validation
- Data type checks
- Missing value analysis
- Outlier detection
- Temporal consistency checks
- Data leakage prevention
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd


class DataValidator:
    """Comprehensive data validator for customer churn event logs."""

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize data validator with configuration.

        Args:
            config: Validation configuration dictionary
        """
        self.config = config or {}
        self.validation_results = {}
        self.errors = []
        self.warnings = []

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def validate_dataset(self, data: pd.DataFrame | str | Path) -> dict[str, Any]:
        """
        Run comprehensive validation on dataset.

        Args:
            data: DataFrame or path to data file

        Returns:
            Dictionary containing validation results
        """
        self.logger.info("Starting comprehensive data validation...")

        # Load data if path provided
        df = self._load_data(data) if isinstance(data, str | Path) else data.copy()

        # Reset validation state
        self.validation_results = {}
        self.errors = []
        self.warnings = []

        # Run validation checks
        self.validation_results["schema_validation"] = self._validate_schema(df)
        self.validation_results["data_quality"] = self._validate_data_quality(df)
        self.validation_results["temporal_consistency"] = (
            self._validate_temporal_consistency(df)
        )
        self.validation_results["business_logic"] = self._validate_business_logic(df)
        self.validation_results["data_leakage"] = self._check_data_leakage(df)

        # Compile summary
        self.validation_results["summary"] = self._compile_summary()

        self.logger.info(
            f"Validation completed. Found {len(self.errors)} errors and {len(self.warnings)} warnings."
        )

        return self.validation_results

    def _load_data(self, file_path: str | Path) -> pd.DataFrame:
        """Load data from JSON file."""
        data = []
        try:
            with open(file_path) as f:
                try:
                    content = json.load(f)

                    data = content if isinstance(content, list) else [content]
                except json.JSONDecodeError:
                    f.seek(0)
                    for line in f:
                        line = line.strip()
                        if line:
                            data.append(json.loads(line))
        except Exception as e:
            self.errors.append(f"Failed to load data from {file_path}: {e}")
            return pd.DataFrame()

        return pd.DataFrame(data)

    def _validate_schema(self, df: pd.DataFrame) -> dict[str, Any]:
        """Validate data schema and required columns."""
        schema_results = {
            "required_columns_present": True,
            "data_types_valid": True,
            "missing_columns": [],
            "unexpected_columns": [],
            "type_issues": [],
        }

        # Define expected schema
        expected_columns = {
            "ts": ["int64", "float64"],  # Timestamp
            "userId": ["object", "str"],  # User ID
            "sessionId": ["int64", "float64"],  # Session ID
            "page": ["object", "str"],  # Event type
            "auth": ["object", "str"],  # Authentication status
            "method": ["object", "str"],  # HTTP method
            "status": ["int64", "float64"],  # HTTP status
            "level": ["object", "str"],  # Subscription level
            "location": ["object", "str"],  # User location
            "userAgent": ["object", "str"],  # Browser info
            "firstName": ["object", "str"],  # First name
            "lastName": ["object", "str"],  # Last name
            "gender": ["object", "str"],  # Gender
            "registration": ["int64", "float64"],  # Registration timestamp
        }

        # Check for missing required columns
        required_core_columns = ["ts", "userId", "page"]
        missing_required = [
            col for col in required_core_columns if col not in df.columns
        ]
        if missing_required:
            schema_results["required_columns_present"] = False
            schema_results["missing_columns"] = missing_required
            self.errors.extend(
                [f"Missing required column: {col}" for col in missing_required]
            )

        # Check for unexpected columns
        expected_cols = set(expected_columns.keys())
        actual_cols = set(df.columns)
        unexpected = actual_cols - expected_cols
        if unexpected:
            schema_results["unexpected_columns"] = list(unexpected)
            self.warnings.extend(
                [f"Unexpected column found: {col}" for col in unexpected]
            )

        # Validate data types
        for col, expected_types in expected_columns.items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                if not any(
                    expected_type in actual_type for expected_type in expected_types
                ):
                    schema_results["data_types_valid"] = False
                    schema_results["type_issues"].append(
                        {
                            "column": col,
                            "expected": expected_types,
                            "actual": actual_type,
                        }
                    )
                    self.warnings.append(
                        f"Type mismatch in {col}: expected {expected_types}, got {actual_type}"
                    )

        return schema_results

    def _validate_data_quality(self, df: pd.DataFrame) -> dict[str, Any]:
        """Validate data quality metrics."""
        quality_results = {
            "total_records": len(df),
            "missing_values": {},
            "duplicate_records": 0,
            "null_user_ids": 0,
            "invalid_timestamps": 0,
            "quality_score": 0.0,
        }

        # Missing values analysis
        missing_analysis = {}
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_pct = (missing_count / len(df)) * 100
            missing_analysis[col] = {
                "count": int(missing_count),
                "percentage": float(missing_pct),
            }

            # Flag high missing value columns
            if missing_pct > 20:
                self.warnings.append(
                    f"High missing values in {col}: {missing_pct:.1f}%"
                )
            elif missing_pct > 50:
                self.errors.append(
                    f"Excessive missing values in {col}: {missing_pct:.1f}%"
                )

        quality_results["missing_values"] = missing_analysis

        # Duplicate records
        duplicates = df.duplicated().sum()
        quality_results["duplicate_records"] = int(duplicates)
        if duplicates > 0:
            dup_pct = (duplicates / len(df)) * 100
            if dup_pct > 5:
                self.errors.append(f"High duplicate rate: {dup_pct:.1f}%")
            else:
                self.warnings.append(
                    f"Duplicate records found: {duplicates} ({dup_pct:.1f}%)"
                )

        # Null user IDs
        if "userId" in df.columns:
            null_users = df["userId"].isnull().sum()
            quality_results["null_user_ids"] = int(null_users)
            if null_users > 0:
                self.errors.append(f"Found {null_users} records with null user IDs")

        # Invalid timestamps
        if "ts" in df.columns:
            # Check for reasonable timestamp range (2010-2030)
            min_ts = pd.Timestamp("2010-01-01").timestamp() * 1000
            max_ts = pd.Timestamp("2030-01-01").timestamp() * 1000

            invalid_ts = ((df["ts"] < min_ts) | (df["ts"] > max_ts)).sum()
            quality_results["invalid_timestamps"] = int(invalid_ts)
            if invalid_ts > 0:
                self.warnings.append(
                    f"Found {invalid_ts} records with invalid timestamps"
                )

        # Calculate quality score (0-100)
        quality_issues = len(self.errors) + len(self.warnings) * 0.5
        max_possible_issues = len(df.columns) * 3  # Rough estimate
        quality_score = max(0, 100 - (quality_issues / max_possible_issues) * 100)
        quality_results["quality_score"] = float(quality_score)

        return quality_results

    def _validate_temporal_consistency(self, df: pd.DataFrame) -> dict[str, Any]:
        """Validate temporal consistency in the data."""
        temporal_results = {
            "valid_time_ordering": True,
            "reasonable_date_range": True,
            "session_consistency": True,
            "time_range_days": 0,
            "temporal_gaps": [],
        }

        if "ts" not in df.columns:
            self.warnings.append("No timestamp column found for temporal validation")
            return temporal_results

        # Convert timestamps
        df_temp = df.copy()
        df_temp["datetime"] = pd.to_datetime(df_temp["ts"], unit="ms")

        # Check date range
        date_range = df_temp["datetime"].max() - df_temp["datetime"].min()
        temporal_results["time_range_days"] = int(date_range.days)

        if date_range.days < 1:
            self.warnings.append(
                "Data spans less than 1 day - may be insufficient for analysis"
            )
        elif date_range.days > 365 * 5:  # More than 5 years
            self.warnings.append(
                f"Data spans {date_range.days} days - unusually long period"
            )

        # Check for reasonable date range
        now = datetime.now()
        if df_temp["datetime"].min() < now - timedelta(days=365 * 10):  # 10 years ago
            temporal_results["reasonable_date_range"] = False
            self.warnings.append("Data contains very old timestamps (>10 years ago)")

        if df_temp["datetime"].max() > now + timedelta(days=1):  # Future dates
            temporal_results["reasonable_date_range"] = False
            self.warnings.append("Data contains future timestamps")

        # Check session consistency
        if "sessionId" in df.columns and "userId" in df.columns:
            # Check if sessions span reasonable time periods
            session_stats = df_temp.groupby(["userId", "sessionId"]).agg(
                {"datetime": ["min", "max", "count"]}
            )

            session_stats.columns = ["session_start", "session_end", "event_count"]
            session_stats["session_duration_hours"] = (
                session_stats["session_end"] - session_stats["session_start"]
            ).dt.total_seconds() / 3600

            # Flag unusually long sessions (>24 hours)
            long_sessions = (session_stats["session_duration_hours"] > 24).sum()
            if long_sessions > 0:
                temporal_results["session_consistency"] = False
                self.warnings.append(
                    f"Found {long_sessions} sessions longer than 24 hours"
                )

        return temporal_results

    def _validate_business_logic(self, df: pd.DataFrame) -> dict[str, Any]:
        """Validate business logic and domain constraints."""
        business_results = {
            "valid_subscription_levels": True,
            "valid_event_types": True,
            "valid_http_methods": True,
            "user_behavior_reasonable": True,
            "anomalous_patterns": [],
        }

        # Validate subscription levels
        if "level" in df.columns:
            valid_levels = {"free", "paid"}
            actual_levels = set(df["level"].dropna().unique())
            invalid_levels = actual_levels - valid_levels
            if invalid_levels:
                business_results["valid_subscription_levels"] = False
                self.warnings.append(
                    f"Invalid subscription levels found: {invalid_levels}"
                )

        # Validate HTTP methods
        if "method" in df.columns:
            valid_methods = {"GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS", "PATCH"}
            actual_methods = set(df["method"].dropna().unique())
            invalid_methods = actual_methods - valid_methods
            if invalid_methods:
                business_results["valid_http_methods"] = False
                self.warnings.append(f"Invalid HTTP methods found: {invalid_methods}")

        # Check for reasonable user behavior patterns
        if "userId" in df.columns and "page" in df.columns:
            user_stats = (
                df.groupby("userId")
                .agg(
                    {
                        "page": "count",
                        "ts": lambda x: (x.max() - x.min())
                        / (1000 * 60 * 60 * 24),  # Days
                    }
                )
                .rename(columns={"page": "total_events", "ts": "activity_days"})
            )

            # Flag users with excessive activity (>1000 events per day)
            user_stats["events_per_day"] = user_stats["total_events"] / (
                user_stats["activity_days"] + 1
            )
            excessive_activity = (user_stats["events_per_day"] > 1000).sum()

            if excessive_activity > 0:
                business_results["user_behavior_reasonable"] = False
                business_results["anomalous_patterns"].append(
                    f"{excessive_activity} users with >1000 events/day"
                )
                self.warnings.append(
                    f"Found {excessive_activity} users with excessive activity (>1000 events/day)"
                )

            # Flag users with no activity variation (single event type)
            if len(df["page"].unique()) > 1:
                single_event_users = df.groupby("userId")["page"].nunique()
                monotonous_users = (single_event_users == 1).sum()

                if monotonous_users / len(single_event_users) > 0.1:  # >10% of users
                    business_results["anomalous_patterns"].append(
                        f"{monotonous_users} users with single event type"
                    )
                    self.warnings.append(
                        f"Found {monotonous_users} users with only one event type"
                    )

        return business_results

    def _check_data_leakage(self, df: pd.DataFrame) -> dict[str, Any]:
        """Check for potential data leakage scenarios."""
        leakage_results = {
            "temporal_leakage_risk": False,
            "future_information_risk": False,
            "target_leakage_risk": False,
            "recommendations": [],
        }

        # Check for temporal leakage risks
        if "ts" in df.columns:
            # Look for events that might represent churn outcomes
            # churn_events = [
            #     "Cancellation Confirmation",
            #     "Cancel",
            #     "Downgrade",
            #     "Account Closed",
            # ]
            actual_events = df["page"].unique() if "page" in df.columns else []

            found_churn_events = [
                event
                for event in actual_events
                if any(
                    churn_keyword in str(event)
                    for churn_keyword in ["cancel", "downgrade", "close", "end", "stop"]
                )
            ]

            if found_churn_events:
                leakage_results["target_leakage_risk"] = True
                leakage_results["recommendations"].append(
                    "Remove explicit churn events from features to prevent target leakage"
                )
                self.warnings.append(
                    f"Found potential target leakage events: {found_churn_events}"
                )

        # Check for future information
        if "registration" in df.columns and "ts" in df.columns:
            future_reg = (df["registration"] > df["ts"]).sum()
            if future_reg > 0:
                leakage_results["future_information_risk"] = True
                leakage_results["recommendations"].append(
                    "Registration dates after event timestamps detected - check data integrity"
                )
                self.warnings.append(
                    f"Found {future_reg} events with registration after timestamp"
                )

        # General recommendations
        leakage_results["recommendations"].extend(
            [
                "Use strict temporal splitting for train/validation/test sets",
                "Ensure features only use information available at prediction time",
                "Validate feature engineering doesn't introduce future information",
            ]
        )

        return leakage_results

    def _compile_summary(self) -> dict[str, Any]:
        """Compile validation summary."""
        return {
            "validation_passed": len(self.errors) == 0,
            "total_errors": len(self.errors),
            "total_warnings": len(self.warnings),
            "errors": self.errors,
            "warnings": self.warnings,
            "recommendations": self._generate_recommendations(),
        }

    def _generate_recommendations(self) -> list[str]:
        """Generate data cleaning and preparation recommendations."""
        recommendations = []

        # Based on validation results
        if (
            self.validation_results.get("data_quality", {}).get("duplicate_records", 0)
            > 0
        ):
            recommendations.append("Remove duplicate records before modeling")

        missing_data = self.validation_results.get("data_quality", {}).get(
            "missing_values", {}
        )
        high_missing_cols = [
            col
            for col, stats in missing_data.items()
            if stats.get("percentage", 0) > 20
        ]
        if high_missing_cols:
            recommendations.append(
                f"Handle missing values in columns: {', '.join(high_missing_cols)}"
            )

        if not self.validation_results.get("temporal_consistency", {}).get(
            "session_consistency", True
        ):
            recommendations.append("Review and clean session duration anomalies")

        if self.validation_results.get("data_leakage", {}).get(
            "target_leakage_risk", False
        ):
            recommendations.append("Remove potential target leakage features")

        # General recommendations
        recommendations.extend(
            [
                "Implement automated data quality monitoring",
                "Create data cleaning pipeline with configurable rules",
                "Set up alerts for data quality degradation",
                "Document data preprocessing steps for reproducibility",
            ]
        )

        return recommendations

    def generate_report(self, output_path: str | None = None) -> str:
        """Generate a formatted validation report."""
        if not self.validation_results:
            return "No validation results available. Run validate_dataset() first."

        report_lines = [
            "=" * 80,
            "DATA VALIDATION REPORT",
            "=" * 80,
            f"Validation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "SUMMARY",
            "-" * 40,
            f"Overall Status: {'✅ PASSED' if self.validation_results['summary']['validation_passed'] else '❌ FAILED'}",
            f"Total Errors: {self.validation_results['summary']['total_errors']}",
            f"Total Warnings: {self.validation_results['summary']['total_warnings']}",
            f"Data Quality Score: {self.validation_results.get('data_quality', {}).get('quality_score', 0):.1f}/100",
            "",
        ]

        # Add errors
        if self.errors:
            report_lines.extend(["ERRORS", "-" * 40])
            for i, error in enumerate(self.errors, 1):
                report_lines.append(f"{i}. {error}")
            report_lines.append("")

        # Add warnings
        if self.warnings:
            report_lines.extend(["WARNINGS", "-" * 40])
            for i, warning in enumerate(self.warnings, 1):
                report_lines.append(f"{i}. {warning}")
            report_lines.append("")

        # Add recommendations
        recommendations = self.validation_results["summary"]["recommendations"]
        if recommendations:
            report_lines.extend(["RECOMMENDATIONS", "-" * 40])
            for i, rec in enumerate(recommendations, 1):
                report_lines.append(f"{i}. {rec}")
            report_lines.append("")

        report_lines.append("=" * 80)

        report = "\n".join(report_lines)

        # Save to file if path provided
        if output_path:
            with open(output_path, "w") as f:
                f.write(report)
            self.logger.info(f"Validation report saved to {output_path}")

        return report


def validate_churn_data(
    data_path: str | Path, output_report: str | None = None
) -> dict[str, Any]:
    """
    Convenience function to validate customer churn data.

    Args:
        data_path: Path to data file
        output_report: Optional path to save validation report

    Returns:
        Validation results dictionary
    """
    validator = DataValidator()
    results = validator.validate_dataset(data_path)

    if output_report:
        validator.generate_report(output_report)

    return results


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        data_file = sys.argv[1]
        report_file = sys.argv[2] if len(sys.argv) > 2 else None

        print(f"Validating data file: {data_file}")
        results = validate_churn_data(data_file, report_file)

        print("\nValidation Summary:")
        print(
            f"Status: {'PASSED' if results['summary']['validation_passed'] else 'FAILED'}"
        )
        print(f"Errors: {results['summary']['total_errors']}")
        print(f"Warnings: {results['summary']['total_warnings']}")
    else:
        print("Usage: python data_validator.py <data_file> [report_file]")
