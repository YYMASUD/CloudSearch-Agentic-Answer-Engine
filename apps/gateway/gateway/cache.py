"""
Redis query & result cache — eliminates duplicate LLM calls.

Cache key: search:{sha256(query+mode)} → serialized search result.
TTL configurable via CACHE_TTL_SECONDS (default 300s).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

CACHE_TTL = int(os.getenv("CACHE_TTL_SECONDS", "300"))


def _cache_key(query: str, mode: str) -> str:
    raw = f"{query.strip().lower()}:{mode}"
    h = hashlib.sha256(raw.encode()).hexdigest()[:24]
    return f"search:{h}"


class RedisCache:
    """Async Redis cache for search results and answers."""

    def __init__(self, redis_url: str | None = None) -> None:
        self._redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self._redis: Any = None  # redis.asyncio.Redis instance

    async def initialize(self) -> None:
        try:
            import redis.asyncio as aioredis
            self._redis = aioredis.from_url(
                self._redis_url,
                decode_responses=True,
                socket_connect_timeout=3,
            )
            await self._redis.ping()
            logger.info("RedisCache connected to %s", self._redis_url)
        except Exception as exc:
            logger.warning("RedisCache unavailable (%s) — caching disabled.", exc)
            self._redis = None

    async def close(self) -> None:
        if self._redis:
            await self._redis.aclose()
            self._redis = None

    @property
    def available(self) -> bool:
        return self._redis is not None

    async def get(self, query: str, mode: str) -> dict | None:
        """Return cached result dict or None on miss."""
        if not self._redis:
            return None
        try:
            key = _cache_key(query, mode)
            raw = await self._redis.get(key)
            if raw:
                logger.debug("Cache HIT for key=%s", key)
                return json.loads(raw)
        except Exception as exc:
            logger.warning("Cache GET error: %s", exc)
        return None

    async def set(self, query: str, mode: str, data: dict, ttl: int = CACHE_TTL) -> None:
        """Store result dict with TTL."""
        if not self._redis:
            return
        try:
            key = _cache_key(query, mode)
            await self._redis.setex(key, ttl, json.dumps(data, default=str))
            logger.debug("Cache SET key=%s ttl=%ds", key, ttl)
        except Exception as exc:
            logger.warning("Cache SET error: %s", exc)

    async def invalidate(self, query: str, mode: str) -> None:
        """Remove a specific cache entry."""
        if not self._redis:
            return
        try:
            key = _cache_key(query, mode)
            await self._redis.delete(key)
        except Exception as exc:
            logger.warning("Cache DELETE error: %s", exc)

    async def health_check(self) -> bool:
        if not self._redis:
            return False
        try:
            return await self._redis.ping()
        except Exception:
            return False
