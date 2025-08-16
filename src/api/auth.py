"""
Authentication and Rate Limiting Module

This module provides authentication mechanisms and rate limiting for the FastAPI application.

Features:
- API key-based authentication
- Rate limiting with configurable limits
- Role-based access control
- Request throttling per user/API key
- Authentication bypass for health checks
"""

from collections import defaultdict
from datetime import datetime, timedelta

import structlog
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.status import HTTP_429_TOO_MANY_REQUESTS

logger = structlog.get_logger()

# Authentication configuration
API_KEYS = {
    "dev-api-key-123": {
        "name": "Development Key",
        "role": "developer",
        "rate_limit": 1000,  # requests per hour
        "permissions": ["predict", "explain", "batch", "models"],
    },
    "prod-api-key-456": {
        "name": "Production Key",
        "role": "production",
        "rate_limit": 10000,
        "permissions": ["predict", "explain", "batch", "models"],
    },
    "monitoring-key-789": {
        "name": "Monitoring Key",
        "role": "monitoring",
        "rate_limit": 100,
        "permissions": ["health", "metrics", "models"],
    },
}


# Rate limiting storage (in production, use Redis or similar)
class RateLimiter:
    """In-memory rate limiter with sliding window."""

    def __init__(self):
        self.requests: dict[str, list] = defaultdict(list)
        self.blocked_until: dict[str, datetime] = {}

    def is_allowed(
        self, key: str, limit: int, window_seconds: int = 3600
    ) -> tuple[bool, dict]:
        """Check if request is allowed under rate limit."""
        now = datetime.now()

        # Check if still blocked
        if key in self.blocked_until and now < self.blocked_until[key]:
            remaining_time = (self.blocked_until[key] - now).total_seconds()
            return False, {
                "allowed": False,
                "limit": limit,
                "remaining": 0,
                "reset_time": remaining_time,
            }

        # Remove blocked status if expired
        if key in self.blocked_until and now >= self.blocked_until[key]:
            del self.blocked_until[key]

        # Clean old requests outside window
        cutoff = now - timedelta(seconds=window_seconds)
        self.requests[key] = [
            req_time for req_time in self.requests[key] if req_time > cutoff
        ]

        # Check current request count
        current_requests = len(self.requests[key])

        if current_requests >= limit:
            # Block for remaining window time
            oldest_request = min(self.requests[key])
            self.blocked_until[key] = oldest_request + timedelta(seconds=window_seconds)
            remaining_time = (self.blocked_until[key] - now).total_seconds()

            return False, {
                "allowed": False,
                "limit": limit,
                "remaining": 0,
                "reset_time": remaining_time,
            }

        # Allow request and record it
        self.requests[key].append(now)

        return True, {
            "allowed": True,
            "limit": limit,
            "remaining": limit - current_requests - 1,
            "reset_time": (
                window_seconds - (now - min(self.requests[key])).total_seconds()
                if self.requests[key]
                else window_seconds
            ),
        }


# Global rate limiter instance
rate_limiter = RateLimiter()

# Security scheme
security = HTTPBearer(auto_error=False)


class AuthenticatedUser:
    """Represents an authenticated user with permissions."""

    def __init__(self, api_key: str, user_info: dict):
        self.api_key = api_key
        self.name = user_info["name"]
        self.role = user_info["role"]
        self.rate_limit = user_info["rate_limit"]
        self.permissions = set(user_info["permissions"])

    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission."""
        return permission in self.permissions


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> AuthenticatedUser | None:
    """Extract and validate API key from request."""

    # Skip authentication for health check and metrics endpoints
    if request.url.path in ["/health", "/metrics", "/docs", "/redoc", "/openapi.json"]:
        return None

    # Extract API key from Authorization header or query parameter
    api_key = None

    if credentials:
        api_key = credentials.credentials
    elif "api_key" in request.query_params:
        api_key = request.query_params["api_key"]
    elif "x-api-key" in request.headers:
        api_key = request.headers["x-api-key"]

    if not api_key:
        logger.warning("No API key provided", path=request.url.path)
        raise HTTPException(
            status_code=401,
            detail="API key required. Provide via Authorization header, x-api-key header, or api_key query parameter.",
        )

    # Validate API key
    if api_key not in API_KEYS:
        logger.warning("Invalid API key provided", api_key=api_key[:10] + "...")
        raise HTTPException(status_code=401, detail="Invalid API key")

    user_info = API_KEYS[api_key]
    user = AuthenticatedUser(api_key, user_info)

    logger.info("User authenticated", user_name=user.name, role=user.role)
    return user


async def check_rate_limit(
    request: Request, user: AuthenticatedUser | None = Depends(get_current_user)
) -> AuthenticatedUser | None:
    """Check rate limiting for authenticated requests."""

    # Skip rate limiting for unauthenticated endpoints
    if user is None:
        return None

    # Check rate limit
    allowed, rate_info = rate_limiter.is_allowed(
        user.api_key, user.rate_limit, window_seconds=3600  # 1 hour window
    )

    if not allowed:
        logger.warning(
            "Rate limit exceeded",
            user_name=user.name,
            api_key=user.api_key[:10] + "...",
            limit=user.rate_limit,
            reset_time=rate_info["reset_time"],
        )

        raise HTTPException(
            status_code=HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Limit: {rate_info['limit']} requests/hour. "
            f"Try again in {rate_info['reset_time']:.0f} seconds.",
            headers={
                "X-RateLimit-Limit": str(rate_info["limit"]),
                "X-RateLimit-Remaining": str(rate_info["remaining"]),
                "X-RateLimit-Reset": str(rate_info["reset_time"]),
            },
        )

    # Add rate limit headers to response (handled in middleware)
    request.state.rate_limit_info = rate_info

    logger.info(
        "Rate limit check passed",
        user_name=user.name,
        remaining=rate_info["remaining"],
        limit=rate_info["limit"],
    )

    return user


def require_permission(permission: str):
    """Dependency factory to require specific permissions."""

    async def check_permission(user: AuthenticatedUser = Depends(check_rate_limit)):
        if user is None:
            # Skip permission check for unauthenticated endpoints
            return user

        if not user.has_permission(permission):
            logger.warning(
                "Permission denied",
                user_name=user.name,
                role=user.role,
                required_permission=permission,
                user_permissions=list(user.permissions),
            )
            raise HTTPException(
                status_code=403,
                detail=f"Permission '{permission}' required. Your role '{user.role}' does not have this permission.",
            )

        return user

    return check_permission


# Convenience dependencies for different permissions
require_predict_permission = require_permission("predict")
require_explain_permission = require_permission("explain")
require_batch_permission = require_permission("batch")
require_models_permission = require_permission("models")
require_monitoring_permission = require_permission("monitoring")


# Rate limit middleware
async def add_rate_limit_headers(request: Request, call_next):
    """Middleware to add rate limit headers to responses."""
    response = await call_next(request)

    # Add rate limit headers if available
    if hasattr(request.state, "rate_limit_info"):
        rate_info = request.state.rate_limit_info
        response.headers["X-RateLimit-Limit"] = str(rate_info["limit"])
        response.headers["X-RateLimit-Remaining"] = str(rate_info["remaining"])
        response.headers["X-RateLimit-Reset"] = str(rate_info["reset_time"])

    return response
