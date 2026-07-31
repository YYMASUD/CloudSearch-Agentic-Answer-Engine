"""
Resilience primitives — circuit breaker and retry with exponential backoff.

Used by fan-out (per-provider circuit breakers) and the LLM router
(retry on transient streaming errors).
"""
from __future__ import annotations

import asyncio
import functools
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


# ─── Circuit Breaker ──────────────────────────────────────────────────────────

class CircuitState(str, Enum):
    CLOSED = "CLOSED"         # Normal — requests pass through
    OPEN = "OPEN"             # Tripped — fail-fast, no requests
    HALF_OPEN = "HALF_OPEN"   # Probing — allow one request through


@dataclass
class CircuitBreaker:
    """
    Per-provider circuit breaker.

    CLOSED → failures exceed threshold → OPEN
    OPEN   → recovery_timeout expires  → HALF_OPEN
    HALF_OPEN → next call succeeds     → CLOSED
    HALF_OPEN → next call fails        → OPEN
    """
    name: str
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    last_failure_time: float = 0.0
    _success_count: int = field(default=0, repr=False)

    def record_success(self) -> None:
        """Record a successful call."""
        self.failure_count = 0
        self._success_count += 1
        if self.state == CircuitState.HALF_OPEN:
            logger.info("CircuitBreaker[%s]: HALF_OPEN → CLOSED (probe succeeded)", self.name)
            self.state = CircuitState.CLOSED

    def record_failure(self) -> None:
        """Record a failed call."""
        self.failure_count += 1
        self.last_failure_time = time.monotonic()

        if self.state == CircuitState.HALF_OPEN:
            logger.warning("CircuitBreaker[%s]: HALF_OPEN → OPEN (probe failed)", self.name)
            self.state = CircuitState.OPEN
        elif self.failure_count >= self.failure_threshold:
            logger.warning(
                "CircuitBreaker[%s]: CLOSED → OPEN (%d consecutive failures)",
                self.name, self.failure_count,
            )
            self.state = CircuitState.OPEN

    def allow_request(self) -> bool:
        """Check if a request should be allowed through."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            elapsed = time.monotonic() - self.last_failure_time
            if elapsed >= self.recovery_timeout:
                logger.info("CircuitBreaker[%s]: OPEN → HALF_OPEN (recovery timeout elapsed)", self.name)
                self.state = CircuitState.HALF_OPEN
                return True
            return False

        # HALF_OPEN — allow one probe
        return True

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self._success_count,
        }


# ─── Retry with exponential backoff ──────────────────────────────────────────

def with_retry(
    max_retries: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    retryable_exceptions: tuple = (Exception,),
):
    """
    Async decorator: retry with exponential backoff + jitter.

    Usage:
        @with_retry(max_retries=3)
        async def flaky_call():
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc = None
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except retryable_exceptions as exc:
                    last_exc = exc
                    if attempt < max_retries:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        jitter = random.uniform(0, delay * 0.3)
                        total_delay = delay + jitter
                        logger.warning(
                            "Retry %d/%d for %s after %.2fs (error: %s)",
                            attempt + 1, max_retries, func.__name__, total_delay, exc,
                        )
                        await asyncio.sleep(total_delay)
                    else:
                        logger.error(
                            "All %d retries exhausted for %s: %s",
                            max_retries, func.__name__, exc,
                        )
            raise last_exc  # type: ignore[misc]
        return wrapper  # type: ignore[return-value]
    return decorator
