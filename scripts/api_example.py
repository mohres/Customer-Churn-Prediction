#!/usr/bin/env python3
"""
Example script demonstrating Churn Prediction API usage

This script shows how to interact with the API endpoints for:
- Health checks
- Single predictions
- Batch predictions
- Prediction explanations
- Model information

Usage:
    python scripts/api_example.py --api-url http://localhost:8000 --api-key dev-api-key-123
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta

import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChurnPredictionClient:
    """Client for interacting with the Churn Prediction API."""

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update(
            {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        )

    def health_check(self) -> dict:
        """Check API health status."""
        logger.info("Checking API health...")
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def list_models(self) -> list[dict]:
        """List available models."""
        logger.info("Listing available models...")
        response = self.session.get(f"{self.base_url}/models")
        response.raise_for_status()
        return response.json()

    def predict_single(
        self, user_id: str, events: list[dict], model_name: str | None = None
    ) -> dict:
        """Make a single user prediction."""
        logger.info(f"Making prediction for user {user_id}...")

        payload = {"user_id": user_id, "events": events}

        if model_name:
            payload["model_name"] = model_name

        response = self.session.post(f"{self.base_url}/predict", json=payload)
        response.raise_for_status()
        return response.json()

    def predict_batch(self, users: list[dict], model_name: str | None = None) -> dict:
        """Make batch predictions."""
        logger.info(f"Making batch predictions for {len(users)} users...")

        payload = {"users": users}

        if model_name:
            payload["model_name"] = model_name

        response = self.session.post(f"{self.base_url}/predict/batch", json=payload)
        response.raise_for_status()
        return response.json()

    def explain_prediction(
        self, user_id: str, events: list[dict], model_name: str | None = None
    ) -> dict:
        """Get prediction explanation."""
        logger.info(f"Getting prediction explanation for user {user_id}...")

        payload = {"user_id": user_id, "events": events}

        if model_name:
            payload["model_name"] = model_name

        response = self.session.post(f"{self.base_url}/explain", json=payload)
        response.raise_for_status()
        return response.json()


def generate_sample_events(user_id: str, num_events: int = 50) -> list[dict]:
    """Generate sample user events for testing."""
    events = []
    base_time = datetime.now() - timedelta(days=30)

    event_types = [
        "play",
        "thumbs_up",
        "thumbs_down",
        "add_to_playlist",
        "skip",
        "logout",
        "advertisement_seen",
    ]

    artists = ["Artist A", "Artist B", "Artist C", "Artist D", "Artist E"]

    for i in range(num_events):
        event_time = base_time + timedelta(
            days=i // 2, hours=(i % 24), minutes=(i * 7) % 60
        )

        events.append(
            {
                "user_id": user_id,
                "timestamp": event_time.isoformat(),
                "event": event_types[i % len(event_types)],
                "song_id": f"song_{i % 100}",
                "artist": artists[i % len(artists)],
                "session_id": f"session_{i // 10}",
            }
        )

    return events


def main():
    parser = argparse.ArgumentParser(description="Churn Prediction API Example")
    parser.add_argument(
        "--api-url", default="http://localhost:8000", help="API base URL"
    )
    parser.add_argument(
        "--api-key", default="dev-api-key-123", help="API key for authentication"
    )
    parser.add_argument(
        "--user-id",
        default="example_user_123",
        help="User ID for single prediction example",
    )
    parser.add_argument("--model-name", help="Specific model to use for predictions")

    args = parser.parse_args()

    # Initialize client
    client = ChurnPredictionClient(args.api_url, args.api_key)

    try:
        # 1. Health Check
        print("=" * 60)
        print("1. HEALTH CHECK")
        print("=" * 60)

        health = client.health_check()
        print(f"Status: {health['status']}")
        print(f"Version: {health['version']}")
        print(f"Models loaded: {len(health['models_loaded'])}")
        print()

        # 2. List Models
        print("=" * 60)
        print("2. AVAILABLE MODELS")
        print("=" * 60)

        models = client.list_models()
        for model in models:
            print(f"- {model['model_name']} ({model['model_type']})")
            print(f"  Loaded: {model['loaded_at']}")
            print(f"  Has explainer: {model['has_explainer']}")
            print()

        if not models:
            print("No models available. Train models first.")
            return

        # Use first available model if none specified
        model_name = args.model_name or models[0]["model_name"]
        print(f"Using model: {model_name}")
        print()

        # 3. Single Prediction
        print("=" * 60)
        print("3. SINGLE PREDICTION")
        print("=" * 60)

        sample_events = generate_sample_events(args.user_id, 30)
        print(f"Generated {len(sample_events)} sample events for user {args.user_id}")

        prediction = client.predict_single(
            user_id=args.user_id, events=sample_events, model_name=model_name
        )

        print(f"User ID: {prediction['user_id']}")
        print(f"Churn Probability: {prediction['churn_probability']:.4f}")
        print(f"Churn Prediction: {prediction['churn_prediction']}")
        print(f"Confidence Score: {prediction['confidence_score']:.4f}")
        print(f"Model Used: {prediction['model_name']}")
        print()

        # 4. Batch Prediction
        print("=" * 60)
        print("4. BATCH PREDICTION")
        print("=" * 60)

        # Create batch of users
        batch_users = []
        for i in range(3):
            user_id = f"batch_user_{i}"
            events = generate_sample_events(user_id, 20 + i * 10)
            batch_users.append({"user_id": user_id, "events": events})

        batch_result = client.predict_batch(batch_users, model_name)

        print(f"Batch ID: {batch_result['batch_id']}")
        print(f"Total Users: {batch_result['total_users']}")
        print(f"Successful: {batch_result['successful_predictions']}")
        print(f"Failed: {batch_result['failed_predictions']}")
        print("\nPredictions:")

        for pred in batch_result["predictions"]:
            print(
                f"- {pred['user_id']}: {pred['churn_probability']:.4f} "
                f"({'CHURN' if pred['churn_prediction'] else 'NO CHURN'})"
            )
        print()

        # 5. Prediction Explanation (if explainer available)
        print("=" * 60)
        print("5. PREDICTION EXPLANATION")
        print("=" * 60)

        # Check if the model has an explainer
        model_info = next((m for m in models if m["model_name"] == model_name), None)

        if model_info and model_info["has_explainer"]:
            explanation = client.explain_prediction(
                user_id=args.user_id, events=sample_events, model_name=model_name
            )

            print(f"User ID: {explanation['user_id']}")
            print(f"Churn Probability: {explanation['churn_probability']:.4f}")
            print(f"Churn Prediction: {explanation['churn_prediction']}")
            print()

            print("Top features contributing to CHURN:")
            for feature in explanation["top_positive_features"][:5]:
                print(f"+ {feature}")
            print()

            print("Top features preventing CHURN:")
            for feature in explanation["top_negative_features"][:5]:
                print(f"- {feature}")
            print()

            print("Feature Details (top 10):")
            for feat in explanation["feature_explanations"][:10]:
                direction = "→ CHURN" if feat["shap_value"] > 0 else "→ RETAIN"
                print(
                    f"{feat['importance_rank']:2d}. {feat['feature_name']}: "
                    f"{feat['feature_value']:.3f} (SHAP: {feat['shap_value']:.4f}) {direction}"
                )
        else:
            print(f"SHAP explainer not available for model '{model_name}'")

        print()
        print("=" * 60)
        print("API EXAMPLE COMPLETED SUCCESSFULLY!")
        print("=" * 60)

    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}")
        if hasattr(e, "response") and e.response is not None:
            logger.error(f"Response: {e.response.text}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
