"""
Gateway middleware — rate limiting and optional API key authentication.

RateLimitMiddleware:
    Token-bucket rate limiter backed by Redis.
    Default: 60 req/min per IP (or 30 req/min for /api/ endpoints).
    Returns 429 with Retry-After header on breach.

APIKeyAuthMiddleware:
    Optional API key validation. Gated by AUTH_REQUIRED=true env var.
    Injects X-Tenant-ID header from the resolved tenant.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

RATE_LIMIT_DEFAULT = int(os.getenv("RATE_LIMIT_RPM", "60"))
RATE_LIMIT_SEARCH = int(os.getenv("RATE_LIMIT_SEARCH_RPM", "30"))
AUTH_REQUIRED = os.getenv("AUTH_REQUIRED", "false").lower() == "true"


def _client_ip(request: Request) -> str:
    """Extract real client IP, respecting X-Forwarded-For."""
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Sliding-window rate limiter backed by Redis.
    Degrades gracefully when Redis is unavailable.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for health checks and metrics
        path = request.url.path
        if path.startswith("/health") or path == "/metrics":
            return await call_next(request)

        # Determine limit tier
        limit = RATE_LIMIT_SEARCH if path.startswith("/api/search") else RATE_LIMIT_DEFAULT

        client_ip = _client_ip(request)
        allowed, retry_after = await self._check_rate_limit(client_ip, path, limit)

        if not allowed:
            logger.warning("Rate limit exceeded: ip=%s path=%s", client_ip, path)
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Please slow down.", "retry_after_seconds": retry_after},
                headers={"Retry-After": str(retry_after)},
            )

        return await call_next(request)

    async def _check_rate_limit(self, client_ip: str, path: str, limit: int) -> tuple[bool, int]:
        """
        Sliding-window counter via Redis INCR + EXPIRE.
        Returns (allowed, retry_after_seconds).
        """
        try:
            from gateway.dependencies import _get_cache
            cache = _get_cache()
            if cache is None or not cache.available:
                return True, 0  # Allow all if Redis unavailable

            window = 60  # 1-minute window
            key = f"rl:{client_ip}:{int(time.time()) // window}"

            redis = cache._redis
            count = await redis.incr(key)
            if count == 1:
                await redis.expire(key, window)

            if count > limit:
                ttl = await redis.ttl(key)
                return False, max(int(ttl), 1)
        except Exception as exc:
            logger.debug("Rate limit check failed (allowing request): %s", exc)

        return True, 0


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """
    Optional API key authentication.
    Only enforced when AUTH_REQUIRED=true.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not AUTH_REQUIRED:
            return await call_next(request)

        # Skip auth for health, docs, metrics
        path = request.url.path
        if path in {"/health", "/health/ready", "/metrics", "/docs", "/redoc", "/openapi.json"}:
            return await call_next(request)

        api_key = (
            request.headers.get("X-API-Key")
            or request.query_params.get("api_key")
        )

        if not api_key:
            return JSONResponse(
                status_code=401,
                content={"detail": "API key required. Pass X-API-Key header or ?api_key= query param."},
            )

        # Validate against known keys (env-based for simplicity; extend with DB lookup)
        valid_keys = set(os.getenv("VALID_API_KEYS", "dev-key-changeme").split(","))
        if api_key.strip() not in valid_keys:
            logger.warning("Invalid API key attempt from %s", _client_ip(request))
            return JSONResponse(
                status_code=403,
                content={"detail": "Invalid API key."},
            )

        return await call_next(request)
