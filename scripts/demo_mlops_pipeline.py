#!/usr/bin/env python3
"""
MLOps Pipeline Demo Script

This script demonstrates the complete MLOps pipeline including:
- Feature engineering (if needed)
- Automated model training
- MLflow experiment tracking
- Model registry and comparison
- Artifact generation

Run this script to see the full pipeline in action!
"""

import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))


def print_banner(text: str, char: str = "="):
    """Print a banner with text."""
    print(f"\n{char * 60}")
    print(f"{text:^60}")
    print(f"{char * 60}")


def print_step(step: int, description: str):
    """Print step information."""
    print(f"\n🔹 Step {step}: {description}")
    print("-" * 40)


def check_features():
    """Check if features are available."""
    features_path = "data/processed/features_selected.csv"
    if not os.path.exists(features_path):
        print(f"❌ Features not found at {features_path}")
        print("📝 Run feature engineering first:")
        print("   python -m src.features.feature_store")
        return False
    return True


def run_training_pipeline():
    """Run the training pipeline."""
    from pipelines.training_pipeline import TrainingPipeline

    print("🚀 Starting automated training pipeline...")
    pipeline = TrainingPipeline()
    results = pipeline.run()

    if results["success"]:
        print("✅ Training pipeline completed successfully!")
        return results
    else:
        print(f"❌ Training pipeline failed: {results.get('error', 'Unknown error')}")
        return None


def print_results_summary(results):
    """Print a summary of results."""
    if not results or not results["success"]:
        return

    print_banner("RESULTS SUMMARY", "🎯")

    comparison = results["results"].get("comparison")
    if comparison is not None and not comparison.empty:
        print("\n📊 Model Performance Comparison:")
        print(
            comparison[
                ["model", "roc_auc", "precision", "recall", "f1_score", "registered"]
            ]
            .round(4)
            .to_string(index=False)
        )

        best_model = comparison.iloc[0]
        print(f"\n🏆 Best Model: {best_model['model']}")
        print(f"📈 AUC Score: {best_model['roc_auc']:.4f}")
        print(f"🎯 Precision: {best_model['precision']:.4f}")
        print(f"🎯 Recall: {best_model['recall']:.4f}")
        print(f"🏷️  Registered: {'Yes' if best_model['registered'] else 'No'}")

    print(
        f"\n📝 Performance Validation: {'✅ PASSED' if results['performance_validation_passed'] else '❌ FAILED'}"
    )
    print(f"📅 Completed: {results['timestamp']}")


def show_artifacts():
    """Show generated artifacts."""
    print_banner("GENERATED ARTIFACTS", "📁")

    # Show reports
    reports_dir = Path("reports")
    if reports_dir.exists():
        print("\n📊 Reports:")
        for file in sorted(reports_dir.glob("*.csv")):
            print(f"  📄 {file.name}")
        for file in sorted(reports_dir.glob("*.txt")):
            print(f"  📄 {file.name}")
        for file in sorted(reports_dir.glob("*.yaml")):
            print(f"  📄 {file.name}")

    # Show plots
    plots_dir = Path("plots")
    if plots_dir.exists():
        print("\n📈 Evaluation Plots:")
        for file in sorted(plots_dir.glob("*.png")):
            print(f"  🖼️  {file.name}")


def show_mlflow_info():
    """Show MLflow information."""
    print_banner("MLFLOW TRACKING", "🔬")

    mlruns_path = Path("mlruns")
    if mlruns_path.exists():
        print(f"\n📊 MLflow Data: {mlruns_path.absolute()}")
        print("🌐 Start MLflow UI with:")
        print("   python scripts/start_mlflow_ui.py")
        print("   # or")
        print("   make mlflow-ui")
        print("\n🔗 Then visit: http://localhost:5000")

        # Count experiments and runs
        experiment_dirs = [
            d for d in mlruns_path.iterdir() if d.is_dir() and d.name.isdigit()
        ]
        if experiment_dirs:
            total_runs = 0
            for exp_dir in experiment_dirs:
                runs = [
                    d for d in exp_dir.iterdir() if d.is_dir() and d.name != "meta.yaml"
                ]
                total_runs += len(runs)

            print(
                f"\n📈 Found {len(experiment_dirs)} experiments with {total_runs} runs"
            )


def main():
    """Run the complete MLOps demo."""
    print_banner("🤖 MLOPS PIPELINE DEMO", "🚀")
    print("Welcome to the Customer Churn Prediction MLOps Pipeline Demo!")
    print("This will demonstrate the complete automated ML workflow.")

    # Step 1: Check prerequisites
    print_step(1, "Checking Prerequisites")
    if not check_features():
        print("\n❌ Demo cannot continue without features.")
        print("💡 Please run feature engineering first and then rerun this demo.")
        sys.exit(1)

    print("✅ Features found - ready to proceed!")

    # Step 2: Run training pipeline
    print_step(2, "Running Automated Training Pipeline")
    start_time = time.time()
    results = run_training_pipeline()
    end_time = time.time()

    if not results:
        print("\n❌ Demo failed during training pipeline.")
        sys.exit(1)

    duration = end_time - start_time
    print(f"\n⏱️  Pipeline completed in {duration:.1f} seconds")

    # Step 3: Show results
    print_step(3, "Results Summary")
    print_results_summary(results)

    # Step 4: Show artifacts
    print_step(4, "Generated Artifacts")
    show_artifacts()

    # Step 5: MLflow info
    print_step(5, "MLflow Experiment Tracking")
    show_mlflow_info()

    # Final message
    print_banner("🎉 DEMO COMPLETE!", "✨")
    print("The MLOps pipeline has successfully:")
    print("✅ Trained multiple models with automatic parameter tuning")
    print("✅ Tracked all experiments in MLflow")
    print("✅ Registered high-performing models")
    print("✅ Generated comprehensive evaluation reports")
    print("✅ Created visualization plots")
    print("✅ Provided model comparison and selection")

    print("\n🔗 Next Steps:")
    print(
        "1. Start MLflow UI to explore experiments: python scripts/start_mlflow_ui.py"
    )
    print("2. Review generated reports in the reports/ directory")
    print("3. Examine evaluation plots in the plots/ directory")
    print("4. Customize the pipeline by editing config/training_config.yaml")
    print("5. Deploy the best model using the FastAPI application")

    print("\n🚀 Happy MLOps! 🎯")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
