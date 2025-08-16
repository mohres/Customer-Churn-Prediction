#!/usr/bin/env python3
"""
Run MLflow Experiment Script

This script demonstrates the MLflow integration by running a complete model experiment
with automatic tracking, model registration, and comparison.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from sklearn.model_selection import train_test_split

from features.feature_store import FeatureStore
from models.mlflow_ensemble_models import run_model_experiment


def load_and_prepare_data():
    """Load and prepare feature data for modeling."""
    print("🔄 Loading feature data...")

    # Initialize feature store
    feature_store = FeatureStore()

    # Load cached features if available
    try:
        # Try loading from processed features
        features_path = "data/processed/features_selected.csv"
        features_df = feature_store.load_features(features_path)
        print(f"✅ Loaded {len(features_df)} feature records from {features_path}")
    except FileNotFoundError:
        print("❌ No processed features found. Please run feature engineering first.")
        print("Run: python -m features.feature_store")
        sys.exit(1)

    # Check for target variable
    if "is_churned" not in features_df.columns:
        print("❌ Target variable 'is_churned' not found in features")
        sys.exit(1)

    # Prepare features and target
    x = features_df.drop(["is_churned"], axis=1)
    y = features_df["is_churned"]

    print(f"📊 Dataset shape: {x.shape}")
    print(f"📊 Churn rate: {y.mean():.2%}")

    return x, y


def main():
    """Run complete MLflow experiment."""
    print("🚀 Starting MLflow Model Experiment")
    print("=" * 50)

    # Load data
    x, y = load_and_prepare_data()

    # Split data with temporal ordering
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"📊 Training set: {x_train.shape[0]} samples")
    print(f"📊 Test set: {x_test.shape[0]} samples")

    # Configure models for experiment
    models_config = {
        "xgboost": {
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 200,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
        "lightgbm": {
            "num_leaves": 31,
            "learning_rate": 0.05,
            "n_estimators": 200,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
        },
    }

    print("\n🔧 Model Configuration:")
    for model_name, config in models_config.items():
        print(f"  {model_name}: {config}")

    print("\n🚀 Running Model Experiment...")
    print("=" * 50)

    # Run experiment
    experiment_results = run_model_experiment(
        x_train, y_train, x_test, y_test, models_config
    )

    print("\n✅ Experiment Complete!")
    print("=" * 50)

    # Print summary
    best_model = experiment_results["best_model"]
    comparison = experiment_results["comparison"]

    if best_model:
        best_result = comparison.iloc[0]
        print(f"🏆 Best Model: {best_result['model']}")
        print(f"📈 Best AUC: {best_result['roc_auc']:.4f}")
        print(f"📝 Run ID: {best_result['run_id']}")
        print(f"🏷️  Registered: {'Yes' if best_result['registered'] else 'No'}")

    print("\n📊 View results in MLflow UI:")
    print(f"   Run: mlflow ui --backend-store-uri file://{Path.cwd()}/mlruns")
    print("   URL: http://localhost:5000")

    return experiment_results


if __name__ == "__main__":
    results = main()
