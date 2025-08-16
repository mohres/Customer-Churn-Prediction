#!/usr/bin/env python3
"""
Production Demo Script

This script demonstrates how to use the production churn prediction pipeline:
1. Load the trained model
2. Process user events
3. Make predictions
4. Use the API

Make sure you have trained a model first using:
    python scripts/train_production_model.py
"""

import json
from datetime import datetime

import pandas as pd

from src.pipelines.inference_pipeline import PredictionPipeline


def main():
    print("Churn Prediction Production Demo")
    print("=" * 40)

    # 1. Load trained model
    model_path = "models/production_churn_model.joblib"

    try:
        pipeline = PredictionPipeline.from_model_path(model_path)
        print(f"✓ Loaded model from {model_path}")
    except FileNotFoundError:
        print(f"✗ Model not found at {model_path}")
        print("Please train a model first using:")
        print(
            "    python scripts/train_production_model.py --data-path data/customer_churn.json"
        )
        return

    # 2. Create sample user events
    user_events = pd.DataFrame(
        [
            {
                "ts": int(datetime.now().timestamp() * 1000),
                "userId": "demo_user_001",
                "sessionId": "session_1",
                "page": "NextSong",
                "level": "paid",
                "artist": "Artist Name",
                "song": "Song Title",
            },
            {
                "ts": int(datetime.now().timestamp() * 1000) + 60000,
                "userId": "demo_user_001",
                "sessionId": "session_1",
                "page": "Playlist",
                "level": "paid",
            },
        ]
    )

    print(f"✓ Created {len(user_events)} sample events")

    # 3. Make prediction
    try:
        result = pipeline.predict_user_churn(user_events, "demo_user_001")
        print("✓ Prediction result:")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"✗ Prediction failed: {e}")

    # 4. Show pipeline stats
    stats = pipeline.get_pipeline_stats()
    print("✓ Pipeline statistics:")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
