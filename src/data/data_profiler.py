"""
Data profiling module for customer churn prediction pipeline.

This module provides comprehensive data profiling including:
- Statistical summaries
- Distribution analysis
- Correlation analysis
- Data quality metrics
- Automated insights generation
"""

import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
plt.style.use("default")
sns.set_palette("husl")


class DataProfiler:
    """Comprehensive data profiler for customer churn event logs."""

    def __init__(self, output_dir: str | None = None):
        """
        Initialize data profiler.

        Args:
            output_dir: Directory to save profiling outputs
        """
        self.output_dir = Path(output_dir) if output_dir else Path(".")
        self.output_dir.mkdir(exist_ok=True)

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def profile_dataset(
        self, data: pd.DataFrame | str | Path, dataset_name: str = "Dataset"
    ) -> dict[str, Any]:
        """
        Generate comprehensive data profile.

        Args:
            data: DataFrame or path to data file
            dataset_name: Name for the dataset in reports

        Returns:
            Dictionary containing profile results
        """
        self.logger.info(f"Starting data profiling for {dataset_name}...")

        # Load data if path provided
        df = self._load_data(data) if isinstance(data, str | Path) else data.copy()

        if df.empty:
            self.logger.error("Empty dataset provided")
            return {}

        # Generate profile components
        profile_results = {
            "dataset_name": dataset_name,
            "generation_time": datetime.now().isoformat(),
            "basic_info": self._get_basic_info(df),
            "column_profiles": self._profile_columns(df),
            "missing_data_analysis": self._analyze_missing_data(df),
            "duplicate_analysis": self._analyze_duplicates(df),
            "correlation_analysis": self._analyze_correlations(df),
            "temporal_analysis": self._analyze_temporal_patterns(df),
            "user_behavior_analysis": self._analyze_user_behavior(df),
            "data_quality_metrics": self._calculate_quality_metrics(df),
            "automated_insights": self._generate_insights(df),
        }

        # Generate visualizations
        self._create_visualizations(df, dataset_name)

        # Generate HTML report
        self._generate_html_report(profile_results, dataset_name)

        self.logger.info(f"Data profiling completed for {dataset_name}")

        return profile_results

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
            self.logger.error(f"Failed to load data from {file_path}: {e}")
            return pd.DataFrame()

        return pd.DataFrame(data)

    def _get_basic_info(self, df: pd.DataFrame) -> dict[str, Any]:
        """Get basic dataset information."""
        return {
            "total_records": len(df),
            "total_columns": len(df.columns),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "numeric_columns": len(df.select_dtypes(include=[np.number]).columns),
            "categorical_columns": len(df.select_dtypes(include=["object"]).columns),
            "datetime_columns": len(df.select_dtypes(include=["datetime64"]).columns),
        }

    def _profile_columns(self, df: pd.DataFrame) -> dict[str, dict]:
        """Generate detailed profile for each column."""
        column_profiles = {}

        for col in df.columns:
            profile = {
                "dtype": str(df[col].dtype),
                "non_null_count": int(df[col].count()),
                "null_count": int(df[col].isnull().sum()),
                "null_percentage": float((df[col].isnull().sum() / len(df)) * 100),
                "unique_count": int(df[col].nunique()),
                "unique_percentage": float(
                    (df[col].nunique() / df[col].count() * 100)
                    if df[col].count() > 0
                    else 0
                ),
            }

            # Add type-specific statistics
            if df[col].dtype in ["int64", "float64"]:
                profile.update(self._get_numeric_stats(df[col]))
            elif df[col].dtype == "object":
                profile.update(self._get_categorical_stats(df[col]))
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                profile.update(self._get_datetime_stats(df[col]))

            column_profiles[col] = profile

        return column_profiles

    def _get_numeric_stats(self, series: pd.Series) -> dict[str, Any]:
        """Get statistics for numeric columns."""
        stats = series.describe()
        return {
            "mean": float(stats["mean"]) if not pd.isna(stats["mean"]) else None,
            "std": float(stats["std"]) if not pd.isna(stats["std"]) else None,
            "min": float(stats["min"]) if not pd.isna(stats["min"]) else None,
            "max": float(stats["max"]) if not pd.isna(stats["max"]) else None,
            "q25": float(stats["25%"]) if not pd.isna(stats["25%"]) else None,
            "median": float(stats["50%"]) if not pd.isna(stats["50%"]) else None,
            "q75": float(stats["75%"]) if not pd.isna(stats["75%"]) else None,
            "skewness": float(series.skew()) if not pd.isna(series.skew()) else None,
            "kurtosis": (
                float(series.kurtosis()) if not pd.isna(series.kurtosis()) else None
            ),
            "zeros_count": int((series == 0).sum()),
            "negative_count": int((series < 0).sum()),
        }

    def _get_categorical_stats(self, series: pd.Series) -> dict[str, Any]:
        """Get statistics for categorical columns."""
        value_counts = series.value_counts()
        return {
            "most_frequent_value": (
                str(value_counts.index[0]) if len(value_counts) > 0 else None
            ),
            "most_frequent_count": (
                int(value_counts.iloc[0]) if len(value_counts) > 0 else 0
            ),
            "least_frequent_value": (
                str(value_counts.index[-1]) if len(value_counts) > 0 else None
            ),
            "least_frequent_count": (
                int(value_counts.iloc[-1]) if len(value_counts) > 0 else 0
            ),
            "top_5_values": {str(k): int(v) for k, v in value_counts.head(5).items()},
            "avg_length": (
                float(series.astype(str).str.len().mean()) if series.count() > 0 else 0
            ),
            "max_length": (
                int(series.astype(str).str.len().max()) if series.count() > 0 else 0
            ),
            "min_length": (
                int(series.astype(str).str.len().min()) if series.count() > 0 else 0
            ),
        }

    def _get_datetime_stats(self, series: pd.Series) -> dict[str, Any]:
        """Get statistics for datetime columns."""
        return {
            "min_date": series.min().isoformat() if series.count() > 0 else None,
            "max_date": series.max().isoformat() if series.count() > 0 else None,
            "date_range_days": (
                int((series.max() - series.min()).days) if series.count() > 0 else 0
            ),
        }

    def _analyze_missing_data(self, df: pd.DataFrame) -> dict[str, Any]:
        """Analyze missing data patterns."""
        missing_data = {}
        total_records = len(df)

        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_pct = (missing_count / total_records) * 100

            missing_data[col] = {
                "count": int(missing_count),
                "percentage": float(missing_pct),
                "severity": (
                    "high"
                    if missing_pct > 50
                    else "medium" if missing_pct > 20 else "low"
                ),
            }

        # Analyze missing data patterns
        missing_patterns = {}
        if total_records > 0:
            # Find columns that are missing together
            missing_matrix = df.isnull()
            pattern_counts = missing_matrix.value_counts()

            # Get top 5 missing patterns
            top_patterns = pattern_counts.head(5)
            for pattern, count in top_patterns.items():
                pattern_key = str(
                    [
                        col
                        for col, is_missing in zip(df.columns, pattern, strict=False)
                        if is_missing
                    ]
                )
                missing_patterns[pattern_key] = {
                    "count": int(count),
                    "percentage": float((count / total_records) * 100),
                }

        return {
            "by_column": missing_data,
            "patterns": missing_patterns,
            "total_complete_records": int((~df.isnull().any(axis=1)).sum()),
            "complete_records_percentage": float(
                ((~df.isnull().any(axis=1)).sum() / total_records) * 100
            ),
        }

    def _analyze_duplicates(self, df: pd.DataFrame) -> dict[str, Any]:
        """Analyze duplicate records."""
        total_records = len(df)
        duplicate_records = df.duplicated().sum()

        return {
            "total_duplicates": int(duplicate_records),
            "duplicate_percentage": (
                float((duplicate_records / total_records) * 100)
                if total_records > 0
                else 0
            ),
            "unique_records": int(total_records - duplicate_records),
        }

    def _analyze_correlations(self, df: pd.DataFrame) -> dict[str, Any]:
        """Analyze correlations between numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) < 2:
            return {"message": "Insufficient numeric columns for correlation analysis"}

        corr_matrix = df[numeric_cols].corr()

        # Find high correlations (excluding diagonal)
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:  # High correlation threshold
                    high_corr_pairs.append(
                        {
                            "column1": corr_matrix.columns[i],
                            "column2": corr_matrix.columns[j],
                            "correlation": float(corr_val),
                        }
                    )

        return {
            "correlation_matrix": corr_matrix.round(3).to_dict(),
            "high_correlations": high_corr_pairs,
            "numeric_columns_analyzed": list(numeric_cols),
        }

    def _analyze_temporal_patterns(self, df: pd.DataFrame) -> dict[str, Any]:
        """Analyze temporal patterns in the data."""
        temporal_analysis = {}

        if "ts" in df.columns:
            # Convert timestamp to datetime
            df_temp = df.copy()
            df_temp["datetime"] = pd.to_datetime(df_temp["ts"], unit="ms")

            # Basic temporal info
            temporal_analysis["date_range"] = {
                "start_date": df_temp["datetime"].min().isoformat(),
                "end_date": df_temp["datetime"].max().isoformat(),
                "total_days": int(
                    (df_temp["datetime"].max() - df_temp["datetime"].min()).days + 1
                ),
            }

            # Activity by hour
            df_temp["hour"] = df_temp["datetime"].dt.hour
            hourly_activity = df_temp["hour"].value_counts().sort_index()
            temporal_analysis["hourly_activity"] = hourly_activity.to_dict()

            # Activity by day of week
            df_temp["day_of_week"] = df_temp["datetime"].dt.day_name()
            daily_activity = df_temp["day_of_week"].value_counts()
            temporal_analysis["daily_activity"] = daily_activity.to_dict()

            # Peak activity periods
            peak_hour = hourly_activity.idxmax()
            peak_day = daily_activity.idxmax()

            temporal_analysis["peak_periods"] = {
                "peak_hour": int(peak_hour),
                "peak_hour_events": int(hourly_activity[peak_hour]),
                "peak_day": str(peak_day),
                "peak_day_events": int(daily_activity[peak_day]),
            }

        return temporal_analysis

    def _analyze_user_behavior(self, df: pd.DataFrame) -> dict[str, Any]:
        """Analyze user behavior patterns."""
        behavior_analysis = {}

        if "userId" in df.columns:
            # User activity statistics
            user_stats = df.groupby("userId").agg(
                {
                    "ts": "count",  # Total events
                    "sessionId": (
                        "nunique" if "sessionId" in df.columns else lambda _: 1
                    ),
                    "page": "nunique" if "page" in df.columns else lambda _: 1,
                }
            )

            user_stats.columns = ["total_events", "unique_sessions", "unique_pages"]

            behavior_analysis["user_statistics"] = {
                "total_users": len(user_stats),
                "avg_events_per_user": float(user_stats["total_events"].mean()),
                "median_events_per_user": float(user_stats["total_events"].median()),
                "most_active_user_events": int(user_stats["total_events"].max()),
                "least_active_user_events": int(user_stats["total_events"].min()),
                "avg_sessions_per_user": float(user_stats["unique_sessions"].mean()),
                "avg_unique_pages_per_user": float(user_stats["unique_pages"].mean()),
            }

            # User engagement segments
            user_stats["engagement_level"] = pd.cut(
                user_stats["total_events"],
                bins=[0, 50, 200, 1000, float("inf")],
                labels=["Low", "Medium", "High", "Very High"],
            )

            engagement_dist = user_stats["engagement_level"].value_counts()
            behavior_analysis["engagement_distribution"] = engagement_dist.to_dict()

        # Event type analysis
        if "page" in df.columns:
            event_stats = df["page"].value_counts()
            behavior_analysis["event_types"] = {
                "total_event_types": len(event_stats),
                "most_common_event": str(event_stats.index[0]),
                "most_common_event_count": int(event_stats.iloc[0]),
                "event_distribution": event_stats.head(10).to_dict(),
            }

        return behavior_analysis

    def _calculate_quality_metrics(self, df: pd.DataFrame) -> dict[str, Any]:
        """Calculate overall data quality metrics."""
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isnull().sum().sum()
        duplicate_rows = df.duplicated().sum()

        # Quality score calculation
        completeness_score = (
            ((total_cells - missing_cells) / total_cells) * 100
            if total_cells > 0
            else 0
        )
        uniqueness_score = (
            ((len(df) - duplicate_rows) / len(df)) * 100 if len(df) > 0 else 0
        )

        # Consistency checks
        consistency_issues = 0
        if "userId" in df.columns:
            null_user_ids = df["userId"].isnull().sum()
            consistency_issues += null_user_ids

        consistency_score = max(0, 100 - (consistency_issues / len(df)) * 100)

        # Overall quality score (weighted average)
        overall_quality = (
            completeness_score * 0.4 + uniqueness_score * 0.3 + consistency_score * 0.3
        )

        return {
            "completeness_score": float(completeness_score),
            "uniqueness_score": float(uniqueness_score),
            "consistency_score": float(consistency_score),
            "overall_quality_score": float(overall_quality),
            "total_cells": int(total_cells),
            "missing_cells": int(missing_cells),
            "duplicate_rows": int(duplicate_rows),
            "quality_grade": self._get_quality_grade(overall_quality),
        }

    def _get_quality_grade(self, score: float) -> str:
        """Convert quality score to letter grade."""
        if score >= 90:
            return "A (Excellent)"
        elif score >= 80:
            return "B (Good)"
        elif score >= 70:
            return "C (Fair)"
        elif score >= 60:
            return "D (Poor)"
        else:
            return "F (Very Poor)"

    def _generate_insights(self, df: pd.DataFrame) -> list[str]:
        """Generate automated insights about the dataset."""
        insights = []

        # Dataset size insights
        if len(df) < 1000:
            insights.append(
                "📊 Small dataset: Consider collecting more data for robust analysis"
            )
        elif len(df) > 1000000:
            insights.append(
                "📊 Large dataset: Consider sampling strategies for faster processing"
            )

        # Missing data insights
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        if missing_pct > 20:
            insights.append(
                f"⚠️ High missing data: {missing_pct:.1f}% of values are missing"
            )
        elif missing_pct > 5:
            insights.append(
                f"📝 Moderate missing data: {missing_pct:.1f}% of values are missing"
            )

        # User behavior insights
        if "userId" in df.columns and "page" in df.columns:
            user_stats = df.groupby("userId").size()
            if (
                user_stats.std() / user_stats.mean() > 2
            ):  # High coefficient of variation
                insights.append(
                    "📈 Highly variable user activity: Consider user segmentation"
                )

            # Event distribution insights
            event_stats = df["page"].value_counts()
            if len(event_stats) > 0 and event_stats.iloc[0] / event_stats.sum() > 0.8:
                insights.append(
                    "🎯 Dominant event type: One event type represents >80% of data"
                )

        # Temporal insights
        if "ts" in df.columns:
            df_temp = df.copy()
            df_temp["datetime"] = pd.to_datetime(df_temp["ts"], unit="ms")
            date_range = (df_temp["datetime"].max() - df_temp["datetime"].min()).days

            if date_range < 7:
                insights.append("📅 Short time period: Data spans less than a week")
            elif date_range > 365:
                insights.append(
                    "📅 Long time period: Consider seasonal patterns in analysis"
                )

        # Data quality insights
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            dup_pct = (duplicates / len(df)) * 100
            if dup_pct > 5:
                insights.append(
                    f"🔄 High duplication: {dup_pct:.1f}% of records are duplicates"
                )

        return insights

    def _create_visualizations(self, df: pd.DataFrame, dataset_name: str):
        """Create visualization plots."""
        # Set up the plotting style
        plt.style.use("default")

        # 1. Missing data heatmap
        if df.isnull().sum().sum() > 0:
            plt.figure(figsize=(12, 6))
            missing_data = df.isnull()
            sns.heatmap(
                missing_data.transpose(), cbar=True, yticklabels=True, xticklabels=False
            )
            plt.title(f"{dataset_name}: Missing Data Pattern")
            plt.tight_layout()
            plt.savefig(
                self.output_dir
                / f'{dataset_name.lower().replace(" ", "_")}_missing_data.png',
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

        # 2. Correlation heatmap for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            plt.figure(figsize=(10, 8))
            corr_matrix = df[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, square=True)
            plt.title(f"{dataset_name}: Numeric Columns Correlation")
            plt.tight_layout()
            plt.savefig(
                self.output_dir
                / f'{dataset_name.lower().replace(" ", "_")}_correlation.png',
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

        # 3. Event type distribution
        if "page" in df.columns:
            plt.figure(figsize=(12, 6))
            event_counts = df["page"].value_counts().head(10)
            event_counts.plot(kind="bar")
            plt.title(f"{dataset_name}: Top 10 Event Types")
            plt.xlabel("Event Type")
            plt.ylabel("Count")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(
                self.output_dir
                / f'{dataset_name.lower().replace(" ", "_")}_events.png',
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

        # 4. Temporal activity patterns
        if "ts" in df.columns:
            df_temp = df.copy()
            df_temp["datetime"] = pd.to_datetime(df_temp["ts"], unit="ms")
            df_temp["hour"] = df_temp["datetime"].dt.hour

            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            hourly_activity = df_temp["hour"].value_counts().sort_index()
            hourly_activity.plot(kind="line", marker="o")
            plt.title("Activity by Hour of Day")
            plt.xlabel("Hour")
            plt.ylabel("Number of Events")
            plt.grid(True, alpha=0.3)

            plt.subplot(1, 2, 2)
            daily_activity = df_temp["datetime"].dt.day_name().value_counts()
            daily_activity.plot(kind="bar")
            plt.title("Activity by Day of Week")
            plt.xlabel("Day")
            plt.ylabel("Number of Events")
            plt.xticks(rotation=45)

            plt.tight_layout()
            plt.savefig(
                self.output_dir
                / f'{dataset_name.lower().replace(" ", "_")}_temporal.png',
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

    def _format_user_behavior_html(self, behavior_data: dict[str, Any]) -> str:
        """Format user behavior data for HTML report."""
        if not behavior_data:
            return "<p>No user behavior data available</p>"

        user_stats = behavior_data.get("user_statistics", {})
        html = f"""
        <div class="metric">Total Users: <strong>{user_stats.get('total_users', 0):,}</strong></div>
        <div class="metric">Avg Events/User: <strong>{user_stats.get('avg_events_per_user', 0):.1f}</strong></div>
        <div class="metric">Most Active User: <strong>{user_stats.get('most_active_user_events', 0):,} events</strong></div>
        """

        engagement_dist = behavior_data.get("engagement_distribution", {})
        if engagement_dist:
            html += "<h3>User Engagement Distribution:</h3><ul>"
            for level, count in engagement_dist.items():
                html += f"<li>{level}: {count} users</li>"
            html += "</ul>"

        return html

    def _format_temporal_analysis_html(self, temporal_data: dict[str, Any]) -> str:
        """Format temporal analysis data for HTML report."""
        if not temporal_data:
            return "<p>No temporal data available</p>"

        date_range = temporal_data.get("date_range", {})
        peak_periods = temporal_data.get("peak_periods", {})

        html = f"""
        <div class="metric">Date Range: <strong>{date_range.get('start_date', 'N/A')[:10]} to {date_range.get('end_date', 'N/A')[:10]}</strong></div>
        <div class="metric">Total Days: <strong>{date_range.get('total_days', 0)}</strong></div>
        <div class="metric">Peak Hour: <strong>{peak_periods.get('peak_hour', 'N/A')}:00 ({peak_periods.get('peak_hour_events', 0):,} events)</strong></div>
        <div class="metric">Peak Day: <strong>{peak_periods.get('peak_day', 'N/A')} ({peak_periods.get('peak_day_events', 0):,} events)</strong></div>
        """

        return html


def profile_churn_data(
    data_path: str | Path,
    dataset_name: str = "Customer Churn Dataset",
    output_dir: str | None = None,
) -> dict[str, Any]:
    """
    Convenience function to profile customer churn data.

    Args:
        data_path: Path to data file
        dataset_name: Name for the dataset
        output_dir: Directory to save outputs

    Returns:
        Profile results dictionary
    """
    profiler = DataProfiler(output_dir=output_dir)
    results = profiler.profile_dataset(data_path, dataset_name)

    return results


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        data_file = sys.argv[1]
        dataset_name = sys.argv[2] if len(sys.argv) > 2 else "Dataset"
        output_dir = sys.argv[3] if len(sys.argv) > 3 else "data_profile_output"

        print(f"Profiling data file: {data_file}")
        results = profile_churn_data(data_file, dataset_name, output_dir)

        print("\nProfiling Summary:")
        print(f"Records: {results['basic_info']['total_records']:,}")
        print(
            f"Quality Score: {results.get('data_quality_metrics', {}).get('overall_quality_score', 0):.1f}/100"
        )
        print(
            f"Quality Grade: {results.get('data_quality_metrics', {}).get('quality_grade', 'N/A')}"
        )
    else:
        print("Usage: python data_profiler.py <data_file> [dataset_name] [output_dir]")
