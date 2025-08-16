#!/usr/bin/env python3
"""
Startup script for Churn Prediction API

This script helps set up and start the API service with proper configuration
and model loading. It provides both development and production startup modes.

Usage:
    python scripts/start_api.py --mode development
    python scripts/start_api.py --mode production --host 0.0.0.0 --port 8000
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_requirements():
    """Check if required packages are installed."""
    logger.info("Checking requirements...")

    try:
        import importlib.util

        # Check if packages are available without importing them
        required_packages = [
            "fastapi",
            "numpy",
            "pandas",
            "shap",
            "sklearn",
            "uvicorn",
            "mlflow",
        ]

        for package in required_packages:
            if importlib.util.find_spec(package) is None:
                raise ImportError(f"Package '{package}' not found")

        logger.info("✓ All required packages are installed")
        return True
    except ImportError as e:
        logger.error(f"✗ Missing required package: {e}")
        logger.info("Install requirements with: pip install -r requirements.txt")
        return False


def check_models():
    """Check if models are available."""
    logger.info("Checking for available models...")

    models_dir = project_root / "models"
    mlruns_dir = project_root / "mlruns"

    model_files = []
    if models_dir.exists():
        model_files.extend(list(models_dir.glob("*.joblib")))

    mlflow_models = []
    if mlruns_dir.exists():
        # Check for MLflow models
        for run_dir in mlruns_dir.rglob("*/artifacts"):
            if (run_dir / "MLmodel").exists():
                mlflow_models.append(run_dir.parent)

    if model_files:
        logger.info(f"✓ Found {len(model_files)} .joblib model(s):")
        for model_file in model_files:
            logger.info(f"  - {model_file.name}")

    if mlflow_models:
        logger.info(f"✓ Found {len(mlflow_models)} MLflow model(s)")

    if not model_files and not mlflow_models:
        logger.warning("⚠ No models found. Train models first using:")
        logger.warning("  python scripts/run_mlflow_experiment.py")
        return False

    return True


def check_data():
    """Check if data files are available."""
    logger.info("Checking for data files...")

    data_dir = project_root / "data"
    required_files = ["customer_churn.json", "customer_churn_mini.json"]

    missing_files = []
    for file_name in required_files:
        file_path = data_dir / file_name
        if not file_path.exists():
            missing_files.append(file_name)

    if missing_files:
        logger.warning(f"⚠ Missing data files: {missing_files}")
        logger.warning(
            "The API will still start but may not work properly without data"
        )
        return False

    logger.info("✓ All required data files found")
    return True


def setup_environment(mode: str):
    """Set up environment variables based on mode."""
    logger.info(f"Setting up environment for {mode} mode...")

    # Base environment variables
    env_vars = {
        "PYTHONPATH": str(project_root),
        "LOG_LEVEL": "INFO" if mode == "production" else "DEBUG",
    }

    # MLflow configuration
    mlflow_uri = project_root / "mlruns"
    if mlflow_uri.exists():
        env_vars["MLFLOW_TRACKING_URI"] = f"file://{mlflow_uri.absolute()}"

    # Set environment variables
    for key, value in env_vars.items():
        os.environ[key] = value
        logger.info(f"  {key}={value}")


def start_mlflow_ui():
    """Start MLflow UI in background if available."""
    logger.info("Starting MLflow UI...")

    mlruns_dir = project_root / "mlruns"
    if not mlruns_dir.exists():
        logger.warning("MLflow tracking directory not found, skipping MLflow UI")
        return

    try:
        # Start MLflow UI in background
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "mlflow",
                "ui",
                "--backend-store-uri",
                f"file://{mlruns_dir.absolute()}",
                "--host",
                "127.0.0.1",
                "--port",
                "5000",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        logger.info("✓ MLflow UI started at http://localhost:5000")
        logger.info(f"  Process PID: {process.pid}")

        return process

    except Exception as e:
        logger.warning(f"Could not start MLflow UI: {e}")
        return None


def start_api(mode: str, host: str, port: int, workers: int | None = None):
    """Start the FastAPI application."""
    logger.info(f"Starting Churn Prediction API in {mode} mode...")
    logger.info(f"Server will be available at http://{host}:{port}")

    # Build uvicorn command
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "src.api.main:app",
        "--host",
        host,
        "--port",
        str(port),
    ]

    if mode == "development":
        cmd.extend(["--reload", "--log-level", "debug"])
    else:
        if workers:
            cmd.extend(["--workers", str(workers)])
        cmd.extend(["--log-level", "info"])

    logger.info(f"Running command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, cwd=project_root, check=True)
    except KeyboardInterrupt:
        logger.info("API server stopped by user")
    except subprocess.CalledProcessError as e:
        logger.error(f"API server failed with exit code {e.returncode}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Start Churn Prediction API")
    parser.add_argument(
        "--mode",
        choices=["development", "production"],
        default="development",
        help="Startup mode",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument(
        "--workers", type=int, help="Number of worker processes (production mode only)"
    )
    parser.add_argument(
        "--skip-checks", action="store_true", help="Skip pre-startup checks"
    )
    parser.add_argument(
        "--start-mlflow", action="store_true", help="Start MLflow UI alongside the API"
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Starting Churn Prediction API")
    logger.info("=" * 60)

    # Pre-startup checks
    if not args.skip_checks:
        if not check_requirements():
            sys.exit(1)

        check_models()  # Warning only
        check_data()  # Warning only

    # Setup environment
    setup_environment(args.mode)

    # Start MLflow UI if requested
    mlflow_process = None
    if args.start_mlflow:
        mlflow_process = start_mlflow_ui()

    try:
        # Start API
        start_api(args.mode, args.host, args.port, args.workers)
    finally:
        # Clean up MLflow process
        if mlflow_process:
            logger.info("Stopping MLflow UI...")
            mlflow_process.terminate()
            mlflow_process.wait()


if __name__ == "__main__":
    main()
