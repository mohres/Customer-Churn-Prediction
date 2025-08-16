"""
Production FastAPI Application for Churn Prediction

This API accepts raw user event data and performs real-time feature engineering
and churn prediction. Designed for production use with the full customer_churn.json dataset.

Features:
- Accepts raw event logs as input
- Real-time feature engineering using FeatureStore
- Single user and batch prediction endpoints
- Model health monitoring and metrics
- Input validation and error handling
- Comprehensive logging and monitoring

Endpoints:
- GET /health: Health check and system status
- POST /predict/user: Single user churn prediction from raw events
- POST /predict/batch: Batch churn predictions from raw events
- GET /models: List available models and their status
- GET /metrics: Prometheus-compatible metrics
"""

import json
import time
import uuid
from datetime import datetime, timezone

import joblib
import pandas as pd
import structlog
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field, field_validator
from starlette.responses import Response

from ..features.feature_store import FeatureConfig, FeatureStore

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Prometheus metrics
PREDICTION_COUNTER = Counter(
    "predictions_total", "Total predictions made", ["model_name", "prediction_type"]
)
PREDICTION_LATENCY = Histogram(
    "prediction_duration_seconds", "Time spent on predictions", ["model_name"]
)
ERROR_COUNTER = Counter(
    "prediction_errors_total", "Total prediction errors", ["error_type"]
)
FEATURE_ENGINEERING_LATENCY = Histogram(
    "feature_engineering_duration_seconds", "Time spent on feature engineering"
)

# FastAPI app initialization
app = FastAPI(
    title="Production Churn Prediction API",
    description="Real-time churn prediction API that processes raw user event data",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model cache
model_cache = {}
feature_store_cache = {}


# Pydantic models for request/response
class UserEvent(BaseModel):
    """Single user event data."""

    ts: int = Field(..., description="Timestamp in milliseconds")
    userId: str = Field(..., description="User ID")
    sessionId: str | None = Field(None, description="Session ID")
    page: str | None = Field(None, description="Page/action name")
    auth: str | None = Field(None, description="Authentication status")
    method: str | None = Field(None, description="HTTP method")
    status: int | None = Field(None, description="HTTP status code")
    level: str | None = Field(None, description="User subscription level")
    itemInSession: int | None = Field(None, description="Item number in session")
    location: str | None = Field(None, description="User location")
    userAgent: str | None = Field(None, description="User agent string")
    lastName: str | None = Field(None, description="User last name")
    firstName: str | None = Field(None, description="User first name")
    registration: int | None = Field(None, description="Registration timestamp")
    gender: str | None = Field(None, description="User gender")
    artist: str | None = Field(None, description="Artist name")
    song: str | None = Field(None, description="Song title")
    length: float | None = Field(None, description="Song length in seconds")


class UserPredictionRequest(BaseModel):
    """Request for single user churn prediction."""

    user_id: str = Field(..., description="User ID for prediction")
    events: list[UserEvent] = Field(..., description="User's event history")
    reference_date: str | None = Field(
        None, description="Reference date for prediction (ISO format)"
    )

    @field_validator("events")
    @classmethod
    def validate_events(cls, v):
        if not v:
            raise ValueError("At least one event is required")
        if len(v) > 10000:  # Reasonable limit
            raise ValueError("Too many events (max 10,000)")
        return v


class BatchPredictionRequest(BaseModel):
    """Request for batch churn predictions."""

    users: list[UserPredictionRequest] = Field(
        ..., description="List of users for prediction"
    )

    @field_validator("users")
    @classmethod
    def validate_users(cls, v):
        if not v:
            raise ValueError("At least one user is required")
        if len(v) > 100:  # Batch size limit
            raise ValueError("Too many users in batch (max 100)")
        return v


class PredictionResponse(BaseModel):
    """Response for churn prediction."""

    user_id: str
    churn_probability: float = Field(..., ge=0.0, le=1.0)
    risk_level: str = Field(..., description="LOW, MEDIUM, or HIGH")
    prediction_id: str
    timestamp: str
    model_version: str
    processing_time_ms: float


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions."""

    predictions: list[PredictionResponse]
    batch_id: str
    total_users: int
    successful_predictions: int
    failed_predictions: int
    total_processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    timestamp: str
    version: str
    models_loaded: int
    uptime_seconds: float


class ModelInfo(BaseModel):
    """Model information response."""

    name: str
    version: str
    loaded: bool
    last_prediction: str | None
    total_predictions: int


# Global state for health monitoring
app_start_time = time.time()


def load_model(model_path: str = "models/production_churn_model.joblib") -> dict:
    """Load the trained model and metadata."""
    if model_path in model_cache:
        return model_cache[model_path]

    try:
        # Load model
        model = joblib.load(model_path)

        # Load metadata
        metadata_path = model_path.replace(".joblib", "_metadata.json")
        with open(metadata_path) as f:
            metadata = json.load(f)

        model_info = {
            "model": model,
            "metadata": metadata,
            "feature_columns": metadata["feature_columns"],
            "feature_config": FeatureConfig(**metadata["feature_config"]),
            "preprocessing": metadata.get("preprocessing", {}),
            "loaded_at": datetime.now(timezone.utc),
            "prediction_count": 0,
        }

        model_cache[model_path] = model_info
        logger.info("Model loaded successfully", model_path=model_path)
        return model_info

    except Exception as e:
        logger.error("Failed to load model", model_path=model_path, error=str(e))
        raise HTTPException(
            status_code=500, detail=f"Failed to load model: {e!s}"
        ) from e


def get_feature_store(feature_config: FeatureConfig) -> FeatureStore:
    """Get or create feature store instance."""
    config_key = str(feature_config.__dict__)

    if config_key not in feature_store_cache:
        feature_store_cache[config_key] = FeatureStore(feature_config)
        logger.info("Created new feature store", config=feature_config.feature_version)

    return feature_store_cache[config_key]


def risk_level_from_probability(probability: float) -> str:
    """Convert probability to risk level."""
    if probability >= 0.7:
        return "HIGH"
    elif probability >= 0.3:
        return "MEDIUM"
    else:
        return "LOW"


def process_user_events(events: list[UserEvent], user_id: str) -> pd.DataFrame:
    """Convert user events to DataFrame for feature engineering."""
    events_data = []
    for event in events:
        event_dict = event.model_dump()
        events_data.append(event_dict)

    df = pd.DataFrame(events_data)

    # Ensure required columns exist
    if "userId" not in df.columns:
        df["userId"] = user_id

    # Basic validation
    if len(df) == 0:
        raise ValueError("No events provided")

    if df["userId"].nunique() > 1:
        raise ValueError("Multiple user IDs in single user request")

    return df


async def predict_user_churn(
    user_request: UserPredictionRequest, model_info: dict
) -> PredictionResponse:
    """Predict churn for a single user."""
    start_time = time.time()
    prediction_id = str(uuid.uuid4())

    try:
        # Convert events to DataFrame
        events_df = process_user_events(user_request.events, user_request.user_id)

        # Feature engineering
        feature_start = time.time()
        feature_store = get_feature_store(model_info["feature_config"])

        # Compute features for this user
        features_df, validation = feature_store.compute_features(events_df)

        if not validation.get("passed", False):
            logger.warning(
                "Feature validation failed",
                user_id=user_request.user_id,
                warnings=validation.get("warnings", []),
            )

        feature_time = time.time() - feature_start
        FEATURE_ENGINEERING_LATENCY.observe(feature_time)

        # Prepare features for prediction
        feature_columns = model_info["feature_columns"]

        # Ensure all required features are present
        missing_features = set(feature_columns) - set(features_df.columns)
        if missing_features:
            logger.warning(
                "Missing features, filling with 0",
                missing_features=list(missing_features),
            )
            for feature in missing_features:
                features_df[feature] = 0

        # Apply preprocessing before feature selection
        preprocessing = model_info["metadata"].get("preprocessing", {})
        if "label_encoders" in preprocessing:
            for column, categories in preprocessing["label_encoders"].items():
                if column in features_df.columns:
                    logger.info(
                        f"Encoding {column}: {features_df[column].unique()} -> {categories}"
                    )
                    # Map categorical values to numeric
                    category_map = {cat: idx for idx, cat in enumerate(categories)}
                    original_values = features_df[column].copy()
                    features_df[column] = (
                        features_df[column].map(category_map).fillna(0).astype(int)
                    )
                    logger.info(
                        f"Encoded {column}: {original_values.iloc[0]} -> {features_df[column].iloc[0]}, dtype: {features_df[column].dtype}"
                    )

        # Select and order features (after preprocessing)
        x = features_df[feature_columns].copy()

        # Specific fix for current_subscription_level
        if (
            "current_subscription_level" in x.columns
            and x["current_subscription_level"].dtype == "object"
        ):
            logger.warning(
                f"Manual fix for current_subscription_level: {x['current_subscription_level'].unique()}"
            )
            # Force encode: paid=1, free=0, everything else=0
            x["current_subscription_level"] = (
                x["current_subscription_level"]
                .map({"paid": 1, "free": 0})
                .fillna(0)
                .astype(int)
            )
            logger.info(
                f"Fixed current_subscription_level dtype: {x['current_subscription_level'].dtype}"
            )

        # Force convert any remaining object columns to numeric
        for col in x.columns:
            if x[col].dtype == "object":
                logger.warning(
                    f"Converting object column {col} to numeric: {x[col].unique()}"
                )
                # Try to force conversion to numeric
                x[col] = pd.to_numeric(x[col], errors="coerce")

        # Fill NaN values
        x = x.fillna(0)

        # Final verification - convert any remaining object types
        for col in x.columns:
            if x[col].dtype == "object":
                logger.error(f"Still object after conversion: {col}")
                x[col] = 0  # Set to 0 as fallback

        # Debug: check data types before prediction
        object_columns = x.select_dtypes(include=["object"]).columns.tolist()
        if object_columns:
            logger.error(f"Object columns found: {object_columns}")
            for col in object_columns:
                logger.error(f"{col} values: {x[col].unique()}")

        # Make prediction
        model = model_info["model"]

        # Debug: print exact data types and values before prediction
        logger.info(f"Final x shape: {x.shape}")
        logger.info(f"Final x dtypes: {x.dtypes.to_dict()}")

        # Check for any remaining object columns
        remaining_objects = x.select_dtypes(include=["object"]).columns.tolist()
        if remaining_objects:
            logger.error(f"STILL HAVE OBJECT COLUMNS: {remaining_objects}")
            for col in remaining_objects:
                logger.error(f"{col}: {x[col].values} (dtype: {x[col].dtype})")

        churn_probability = float(model.predict_proba(x)[0, 1])

        # Update model stats
        model_info["prediction_count"] += 1

        processing_time = (time.time() - start_time) * 1000

        # Record metrics
        PREDICTION_COUNTER.labels(
            model_name=model_info["metadata"].get("model_type", "unknown"),
            prediction_type="single",
        ).inc()

        PREDICTION_LATENCY.labels(
            model_name=model_info["metadata"].get("model_type", "unknown")
        ).observe(processing_time / 1000)

        return PredictionResponse(
            user_id=user_request.user_id,
            churn_probability=churn_probability,
            risk_level=risk_level_from_probability(churn_probability),
            prediction_id=prediction_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_version=model_info["metadata"]
            .get("feature_config", {})
            .get("feature_version", "unknown"),
            processing_time_ms=processing_time,
        )

    except Exception as e:
        ERROR_COUNTER.labels(error_type="prediction_error").inc()
        logger.error(
            "Prediction failed",
            user_id=user_request.user_id,
            prediction_id=prediction_id,
            error=str(e),
        )
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e!s}") from e


# API Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    uptime = time.time() - app_start_time

    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc).isoformat(),
        version="2.0.0",
        models_loaded=len(model_cache),
        uptime_seconds=uptime,
    )


@app.get("/models", response_model=list[ModelInfo])
async def list_models():
    """List loaded models and their status."""
    models = []
    for model_path, model_info in model_cache.items():
        models.append(
            ModelInfo(
                name=model_path,
                version=model_info["metadata"]
                .get("feature_config", {})
                .get("feature_version", "unknown"),
                loaded=True,
                last_prediction=model_info.get("last_prediction"),
                total_predictions=model_info.get("prediction_count", 0),
            )
        )

    return models


@app.post("/predict/user", response_model=PredictionResponse)
async def predict_single_user(request: UserPredictionRequest):
    """Predict churn probability for a single user based on their event history."""
    logger.info(
        "Single user prediction request",
        user_id=request.user_id,
        events_count=len(request.events),
    )

    # Load model if not already loaded
    model_info = load_model()

    # Make prediction
    result = await predict_user_churn(request, model_info)

    logger.info(
        "Single user prediction completed",
        user_id=request.user_id,
        churn_probability=result.churn_probability,
        risk_level=result.risk_level,
    )

    return result


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch_users(request: BatchPredictionRequest):
    """Predict churn probability for multiple users."""
    batch_id = str(uuid.uuid4())
    start_time = time.time()

    logger.info(
        "Batch prediction request", batch_id=batch_id, users_count=len(request.users)
    )

    # Load model if not already loaded
    model_info = load_model()

    predictions = []
    successful = 0
    failed = 0

    for user_request in request.users:
        try:
            prediction = await predict_user_churn(user_request, model_info)
            predictions.append(prediction)
            successful += 1
        except Exception as e:
            logger.error(
                "Batch prediction failed for user",
                user_id=user_request.user_id,
                batch_id=batch_id,
                error=str(e),
            )
            failed += 1

    total_time = (time.time() - start_time) * 1000

    PREDICTION_COUNTER.labels(
        model_name=model_info["metadata"].get("model_type", "unknown"),
        prediction_type="batch",
    ).inc()

    logger.info(
        "Batch prediction completed",
        batch_id=batch_id,
        successful=successful,
        failed=failed,
        total_time_ms=total_time,
    )

    return BatchPredictionResponse(
        predictions=predictions,
        batch_id=batch_id,
        total_users=len(request.users),
        successful_predictions=successful,
        failed_predictions=failed,
        total_processing_time_ms=total_time,
    )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    logger.info("Starting Production Churn Prediction API")

    # Try to pre-load the default model
    try:
        load_model()
        logger.info("Default model pre-loaded successfully")
    except Exception as e:
        logger.warning("Could not pre-load default model", error=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
