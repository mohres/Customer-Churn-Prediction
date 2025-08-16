"""
FastAPI Application for Churn Prediction Model Serving

This module provides a production-ready API for serving churn prediction models
with comprehensive monitoring, validation, and explanation capabilities.

Features:
- Single user and batch prediction endpoints
- Model health monitoring and metrics
- Input validation and sanitization
- SHAP-based prediction explanations
- Rate limiting and authentication
- Comprehensive logging and tracing
- API versioning strategy
- Auto-generated OpenAPI documentation

Endpoints:
- GET /health: Health check and system status
- GET /models: List available models and their status
- POST /predict: Single user churn prediction
- POST /predict/batch: Batch churn predictions
- POST /explain: Get prediction explanation with SHAP values
- GET /metrics: Prometheus-compatible metrics
"""

import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import shap
import structlog
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field, field_validator
from starlette.responses import Response

import mlflow

from ..features.feature_store import FeatureConfig, FeatureStore
from .auth import (
    AuthenticatedUser,
    add_rate_limit_headers,
    require_batch_permission,
    require_explain_permission,
    require_models_permission,
    require_predict_permission,
)

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

# FastAPI app initialization
app = FastAPI(
    title="Churn Prediction API",
    description="Production-ready API for customer churn prediction with ML model serving",
    version="1.0.0",
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

# Add rate limiting middleware
app.middleware("http")(add_rate_limit_headers)


# Global model storage
class ModelStore:
    """Global model store for managing loaded models."""

    def __init__(self):
        self.models: dict[str, any] = {}
        self.feature_store = None
        self.explainers: dict[str, shap.Explainer] = {}
        self.model_metadata: dict[str, dict] = {}

    def load_model(self, model_name: str, model_path: str) -> None:
        """Load a model and its metadata."""
        try:
            # Load model
            if model_path.endswith(".joblib"):
                model = joblib.load(model_path)
            else:
                model = mlflow.sklearn.load_model(model_path)

            self.models[model_name] = model

            # Initialize SHAP explainer
            # For tree-based models, use TreeExplainer
            try:
                self.explainers[model_name] = shap.TreeExplainer(model)
                logger.info(f"SHAP TreeExplainer initialized for {model_name}")
            except Exception as e:
                logger.warning(
                    f"Could not initialize SHAP explainer for {model_name}: {e}"
                )

            # Store metadata
            self.model_metadata[model_name] = {
                "loaded_at": datetime.now(timezone.utc).isoformat(),
                "model_path": model_path,
                "model_type": type(model).__name__,
            }

            logger.info(f"Model {model_name} loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to load model: {e}"
            ) from e


model_store = ModelStore()


# Pydantic models for request/response validation
class UserEvent(BaseModel):
    """Individual user event for feature computation."""

    user_id: str = Field(..., description="Unique user identifier")
    timestamp: str = Field(..., description="Event timestamp in ISO format")
    event: str = Field(
        ..., description="Event type (e.g., thumbs_up, thumbs_down, play)"
    )
    song_id: str | None = Field(None, description="Song identifier if applicable")
    artist: str | None = Field(None, description="Artist name if applicable")
    session_id: str | None = Field(None, description="Session identifier")

    @field_validator("timestamp")
    def validate_timestamp(cls, v):
        try:
            datetime.fromisoformat(v.replace("Z", "+00:00"))
            return v
        except ValueError as e:
            raise ValueError("Invalid timestamp format. Use ISO format.") from e


class PredictionRequest(BaseModel):
    """Single user churn prediction request."""

    user_id: str = Field(..., description="Unique user identifier")
    events: list[UserEvent] = Field(
        ..., description="List of user events for feature computation"
    )
    reference_date: str | None = Field(
        None, description="Reference date for feature computation"
    )
    model_name: str | None = Field(
        "lightgbm_churn_model", description="Model to use for prediction"
    )

    @field_validator("reference_date")
    def validate_reference_date(cls, v):
        if v is not None:
            try:
                datetime.fromisoformat(v.replace("Z", "+00:00"))
                return v
            except ValueError as e:
                raise ValueError(
                    "Invalid reference_date format. Use ISO format."
                ) from e
        return v


class BatchPredictionRequest(BaseModel):
    """Batch churn prediction request."""

    users: list[PredictionRequest] = Field(
        ..., description="List of user prediction requests"
    )
    model_name: str | None = Field(
        "lightgbm_churn_model", description="Model to use for predictions"
    )


class PredictionResponse(BaseModel):
    """Churn prediction response."""

    user_id: str
    churn_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Probability of churn (0-1)"
    )
    churn_prediction: bool = Field(..., description="Binary churn prediction")
    prediction_id: str = Field(..., description="Unique prediction identifier")
    timestamp: str = Field(..., description="Prediction timestamp")
    model_name: str = Field(..., description="Model used for prediction")
    confidence_score: float | None = Field(
        None, description="Prediction confidence score"
    )


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""

    predictions: list[PredictionResponse]
    batch_id: str = Field(..., description="Unique batch identifier")
    total_users: int = Field(..., description="Total number of users processed")
    successful_predictions: int = Field(
        ..., description="Number of successful predictions"
    )
    failed_predictions: int = Field(..., description="Number of failed predictions")


class ExplanationRequest(BaseModel):
    """Prediction explanation request."""

    user_id: str = Field(..., description="Unique user identifier")
    events: list[UserEvent] = Field(
        ..., description="List of user events for feature computation"
    )
    reference_date: str | None = Field(
        None, description="Reference date for feature computation"
    )
    model_name: str | None = Field(
        "lightgbm_churn_model", description="Model to use for explanation"
    )


class FeatureExplanation(BaseModel):
    """Individual feature explanation."""

    feature_name: str
    feature_value: float
    shap_value: float
    importance_rank: int


class ExplanationResponse(BaseModel):
    """Prediction explanation response."""

    user_id: str
    churn_probability: float
    churn_prediction: bool
    feature_explanations: list[FeatureExplanation]
    top_positive_features: list[str] = Field(
        ..., description="Features most contributing to churn"
    )
    top_negative_features: list[str] = Field(
        ..., description="Features most preventing churn"
    )
    explanation_id: str
    timestamp: str
    model_name: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Health check timestamp")
    version: str = Field(..., description="API version")
    models_loaded: dict[str, dict] = Field(..., description="Status of loaded models")
    system_info: dict[str, str | float] = Field(..., description="System information")


class ModelInfo(BaseModel):
    """Model information response."""

    model_name: str
    model_type: str
    loaded_at: str
    model_path: str
    has_explainer: bool


# Startup event to load models
@app.on_event("startup")
async def startup_event():
    """Initialize models and services on startup."""
    logger.info("Starting Churn Prediction API")

    # Initialize feature store
    config = FeatureConfig(
        activity_windows=[7, 14, 30], trend_windows=[7, 14, 21], enable_caching=True
    )
    model_store.feature_store = FeatureStore(config)

    # Load available models
    models_dir = Path("models")
    # mlflow_models_dir = Path("mlruns")

    # Load local .joblib models
    if models_dir.exists():
        for model_file in models_dir.glob("*.joblib"):
            model_name = model_file.stem
            try:
                model_store.load_model(model_name, str(model_file))
            except Exception as e:
                logger.error(f"Failed to load {model_name}: {e}")

    # Load MLflow models (production versions)
    try:
        client = mlflow.tracking.MlflowClient()
        for registered_model in client.search_registered_models():
            model_name = registered_model.name
            try:
                # Get production version
                production_version = client.get_latest_versions(
                    model_name, stages=["Production"]
                )
                if production_version:
                    model_uri = f"models:/{model_name}/Production"
                    model_store.load_model(model_name, model_uri)
                else:
                    # Fallback to latest version
                    latest_version = client.get_latest_versions(model_name)[0]
                    model_uri = f"models:/{model_name}/{latest_version.version}"
                    model_store.load_model(model_name, model_uri)
            except Exception as e:
                logger.error(f"Failed to load MLflow model {model_name}: {e}")
    except Exception as e:
        logger.warning(f"Could not connect to MLflow: {e}")

    logger.info(
        f"Loaded {len(model_store.models)} models: {list(model_store.models.keys())}"
    )


# Middleware for request logging and tracing
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with tracing information."""
    start_time = time.time()
    request_id = str(uuid.uuid4())

    # Add request ID to logs
    logger.info(
        "Request started",
        request_id=request_id,
        method=request.method,
        url=str(request.url),
        headers=dict(request.headers),
    )

    response = await call_next(request)

    process_time = time.time() - start_time
    logger.info(
        "Request completed",
        request_id=request_id,
        status_code=response.status_code,
        process_time=process_time,
    )

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = str(process_time)

    return response


# Helper functions
def get_model(model_name: str):
    """Get model from store with validation."""
    if model_name not in model_store.models:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found. Available models: {list(model_store.models.keys())}",
        )
    return model_store.models[model_name]


def compute_user_features(
    events: list[UserEvent], reference_date: str | None = None
) -> pd.DataFrame:
    """Compute features for a user based on their events."""
    # Convert events to DataFrame
    events_data = []
    for event in events:
        events_data.append(
            {
                "user_id": event.user_id,
                "timestamp": event.timestamp,
                "event": event.event,
                "song_id": event.song_id,
                "artist": event.artist,
                "session_id": event.session_id,
            }
        )

    events_df = pd.DataFrame(events_data)

    # Set reference date
    if reference_date:
        ref_date = datetime.fromisoformat(reference_date.replace("Z", "+00:00"))
    else:
        ref_date = datetime.now(timezone.utc)

    # Update feature store config
    model_store.feature_store.config.reference_date = ref_date

    # Compute features
    features_df = model_store.feature_store.compute_features(events_df)

    return features_df


# API Endpoints


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with comprehensive system status."""
    try:
        system_info = {
            "cpu_count": "N/A",  # Could use psutil for actual system info
            "memory_usage": "N/A",
            "disk_usage": "N/A",
        }

        return HealthResponse(
            status="healthy",
            timestamp=datetime.now(timezone.utc).isoformat(),
            version="1.0.0",
            models_loaded=model_store.model_metadata,
            system_info=system_info,
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed") from e


@app.get("/models", response_model=list[ModelInfo])
async def list_models(_user: AuthenticatedUser = Depends(require_models_permission)):
    """List all available models and their status."""
    models_info = []
    for model_name, metadata in model_store.model_metadata.items():
        models_info.append(
            ModelInfo(
                model_name=model_name,
                model_type=metadata["model_type"],
                loaded_at=metadata["loaded_at"],
                model_path=metadata["model_path"],
                has_explainer=model_name in model_store.explainers,
            )
        )
    return models_info


@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(
    request: PredictionRequest,
    _user: AuthenticatedUser = Depends(require_predict_permission),
):
    """Predict churn probability for a single user."""
    start_time = time.time()
    prediction_id = str(uuid.uuid4())

    try:
        # Get model
        model = get_model(request.model_name)

        # Compute features
        features_df = compute_user_features(request.events, request.reference_date)

        if features_df.empty:
            raise HTTPException(
                status_code=400,
                detail="Could not compute features from provided events",
            )

        # Make prediction
        churn_prob = model.predict_proba(features_df)[0][
            1
        ]  # Probability of churn (class 1)
        churn_prediction = churn_prob >= 0.5

        # Calculate confidence score (distance from decision boundary)
        confidence_score = abs(churn_prob - 0.5) * 2

        # Update metrics
        PREDICTION_COUNTER.labels(
            model_name=request.model_name, prediction_type="single"
        ).inc()
        PREDICTION_LATENCY.labels(model_name=request.model_name).observe(
            time.time() - start_time
        )

        response = PredictionResponse(
            user_id=request.user_id,
            churn_probability=float(churn_prob),
            churn_prediction=churn_prediction,
            prediction_id=prediction_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_name=request.model_name,
            confidence_score=float(confidence_score),
        )

        logger.info(
            "Prediction completed",
            prediction_id=prediction_id,
            user_id=request.user_id,
            churn_probability=churn_prob,
            model_name=request.model_name,
        )

        return response

    except Exception as e:
        ERROR_COUNTER.labels(error_type="prediction_error").inc()
        logger.error(f"Prediction failed: {e}", prediction_id=prediction_id)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e!s}") from e


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_churn_batch(
    request: BatchPredictionRequest,
    _user: AuthenticatedUser = Depends(require_batch_permission),
):
    """Predict churn probability for multiple users."""
    batch_id = str(uuid.uuid4())
    start_time = time.time()

    try:
        predictions = []
        successful = 0
        failed = 0

        for user_request in request.users:
            try:
                # Replicate single prediction logic without auth dependency
                model_name = request.model_name or user_request.model_name
                model = get_model(model_name)

                # Compute features
                features_df = compute_user_features(
                    user_request.events, user_request.reference_date
                )

                if features_df.empty:
                    failed += 1
                    continue

                # Make prediction
                churn_prob = model.predict_proba(features_df)[0][1]
                churn_prediction = churn_prob >= 0.5
                confidence_score = abs(churn_prob - 0.5) * 2

                prediction = PredictionResponse(
                    user_id=user_request.user_id,
                    churn_probability=float(churn_prob),
                    churn_prediction=churn_prediction,
                    prediction_id=str(uuid.uuid4()),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    model_name=model_name,
                    confidence_score=float(confidence_score),
                )

                predictions.append(prediction)
                successful += 1

            except Exception as e:
                logger.error(f"Failed prediction for user {user_request.user_id}: {e}")
                failed += 1
                continue

        # Update metrics
        PREDICTION_COUNTER.labels(
            model_name=request.model_name, prediction_type="batch"
        ).inc(len(request.users))
        PREDICTION_LATENCY.labels(model_name=request.model_name).observe(
            time.time() - start_time
        )

        response = BatchPredictionResponse(
            predictions=predictions,
            batch_id=batch_id,
            total_users=len(request.users),
            successful_predictions=successful,
            failed_predictions=failed,
        )

        logger.info(
            "Batch prediction completed",
            batch_id=batch_id,
            total_users=len(request.users),
            successful=successful,
            failed=failed,
        )

        return response

    except Exception as e:
        ERROR_COUNTER.labels(error_type="batch_prediction_error").inc()
        logger.error(f"Batch prediction failed: {e}", batch_id=batch_id)
        raise HTTPException(
            status_code=500, detail=f"Batch prediction failed: {e!s}"
        ) from e


@app.post("/explain", response_model=ExplanationResponse)
async def explain_prediction(
    request: ExplanationRequest,
    _user: AuthenticatedUser = Depends(require_explain_permission),
):
    """Get SHAP-based explanation for a prediction."""
    explanation_id = str(uuid.uuid4())

    try:
        # Get model and explainer
        model = get_model(request.model_name)

        if request.model_name not in model_store.explainers:
            raise HTTPException(
                status_code=404,
                detail=f"SHAP explainer not available for model '{request.model_name}'",
            )

        explainer = model_store.explainers[request.model_name]

        # Compute features
        features_df = compute_user_features(request.events, request.reference_date)

        if features_df.empty:
            raise HTTPException(
                status_code=400,
                detail="Could not compute features from provided events",
            )

        # Make prediction
        churn_prob = model.predict_proba(features_df)[0][1]
        churn_prediction = churn_prob >= 0.5

        # Get SHAP values
        shap_values = explainer.shap_values(features_df)

        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            # For binary classification, take the positive class
            shap_values = shap_values[1]

        # Create feature explanations
        feature_names = features_df.columns.tolist()
        feature_values = features_df.iloc[0].values
        shap_vals = shap_values[0] if shap_values.ndim > 1 else shap_values

        feature_explanations = []
        for i, (name, value, shap_val) in enumerate(
            zip(feature_names, feature_values, shap_vals, strict=False)
        ):
            feature_explanations.append(
                FeatureExplanation(
                    feature_name=name,
                    feature_value=float(value),
                    shap_value=float(shap_val),
                    importance_rank=i + 1,
                )
            )

        # Sort by absolute SHAP value to get most important features
        feature_explanations.sort(key=lambda x: abs(x.shap_value), reverse=True)

        # Update importance ranks
        for i, explanation in enumerate(feature_explanations):
            explanation.importance_rank = i + 1

        # Get top positive and negative features
        top_positive = [
            f.feature_name for f in feature_explanations if f.shap_value > 0
        ][:5]
        top_negative = [
            f.feature_name for f in feature_explanations if f.shap_value < 0
        ][:5]

        response = ExplanationResponse(
            user_id=request.user_id,
            churn_probability=float(churn_prob),
            churn_prediction=churn_prediction,
            feature_explanations=feature_explanations,
            top_positive_features=top_positive,
            top_negative_features=top_negative,
            explanation_id=explanation_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_name=request.model_name,
        )

        logger.info(
            "Explanation completed",
            explanation_id=explanation_id,
            user_id=request.user_id,
            model_name=request.model_name,
        )

        return response

    except Exception as e:
        ERROR_COUNTER.labels(error_type="explanation_error").inc()
        logger.error(f"Explanation failed: {e}", explanation_id=explanation_id)
        raise HTTPException(status_code=500, detail=f"Explanation failed: {e!s}") from e


@app.get("/metrics")
async def get_metrics():
    """Prometheus-compatible metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Custom exception handlers
@app.exception_handler(ValueError)
async def value_error_handler(_request: Request, exc: ValueError):
    ERROR_COUNTER.labels(error_type="validation_error").inc()
    return JSONResponse(
        status_code=400, content={"detail": f"Validation error: {exc!s}"}
    )


@app.exception_handler(Exception)
async def general_exception_handler(_request: Request, exc: Exception):
    ERROR_COUNTER.labels(error_type="internal_error").inc()
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)  # nosec B104
