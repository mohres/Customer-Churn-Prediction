#!/usr/bin/env python3
"""
Start MLflow UI Script

This script starts the MLflow UI server to view experiment results,
model registry, and artifacts.
"""

import subprocess
import sys
from pathlib import Path


def start_mlflow_ui(host: str = "127.0.0.1", port: int = 5000):
    """Start MLflow UI server."""
    mlruns_path = Path.cwd() / "mlruns"

    cmd = [
        sys.executable,
        "-m",
        "mlflow",
        "ui",
        "--host",
        host,
        "--port",
        str(port),
        "--backend-store-uri",
        f"file://{mlruns_path}",
    ]

    print("🚀 Starting MLflow UI...")
    print(f"📊 MLflow tracking URI: file://{mlruns_path}")
    print(f"🌐 UI URL: http://{host}:{port}")
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 50)

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n🛑 MLflow UI stopped")


if __name__ == "__main__":
    start_mlflow_ui()
