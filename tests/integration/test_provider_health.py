"""
Integration tests — provider health probes and circuit breaker behavior.

Requires:
    docker compose --profile lite up -d

Run:
    pytest tests/integration/test_provider_health.py -v
"""
from __future__ import annotations

import os
import pytest
import httpx

GATEWAY_URL = os.getenv("GATEWAY_URL", "http://localhost:8000")


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="session")
async def client():
    async with httpx.AsyncClient(base_url=GATEWAY_URL, timeout=15.0) as c:
        yield c


class TestProviderHealth:
    async def test_ready_endpoint_returns_checks(self, client: httpx.AsyncClient):
        """Readiness endpoint must always return the checks dict."""
        resp = await client.get("/health/ready")
        assert resp.status_code in {200, 503}
        body = resp.json()
        assert isinstance(body.get("checks"), dict)
        assert "elapsed_ms" in body

    async def test_model_router_check_present(self, client: httpx.AsyncClient):
        resp = await client.get("/health/ready")
        body = resp.json()
        assert "model_router" in body["checks"]
        assert "healthy" in body["checks"]["model_router"]

    async def test_redis_check_present(self, client: httpx.AsyncClient):
        resp = await client.get("/health/ready")
        body = resp.json()
        assert "redis" in body["checks"]

    async def test_postgres_check_present(self, client: httpx.AsyncClient):
        resp = await client.get("/health/ready")
        body = resp.json()
        assert "postgres" in body["checks"]

    async def test_response_time_under_5s(self, client: httpx.AsyncClient):
        """Health checks must complete within 5 seconds even if providers are slow."""
        import time
        t0 = time.monotonic()
        resp = await client.get("/health/ready")
        elapsed = time.monotonic() - t0
        assert elapsed < 5.0, f"Health check took {elapsed:.1f}s (expected < 5s)"


class TestCircuitBreaker:
    """Unit-level circuit breaker tests (no HTTP needed)."""

    def test_circuit_opens_after_threshold(self):
        from cloudsearch_shared.resilience import CircuitBreaker, CircuitState
        cb = CircuitBreaker(name="test", failure_threshold=3)
        assert cb.state == CircuitState.CLOSED
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_open_circuit_blocks_requests(self):
        from cloudsearch_shared.resilience import CircuitBreaker
        cb = CircuitBreaker(name="test", failure_threshold=1)
        cb.record_failure()
        assert not cb.allow_request()

    def test_circuit_transitions_to_half_open_after_timeout(self):
        import time
        from cloudsearch_shared.resilience import CircuitBreaker, CircuitState
        cb = CircuitBreaker(name="test", failure_threshold=1, recovery_timeout=0.01)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        time.sleep(0.02)
        assert cb.allow_request()
        assert cb.state == CircuitState.HALF_OPEN

    def test_circuit_closes_on_probe_success(self):
        import time
        from cloudsearch_shared.resilience import CircuitBreaker, CircuitState
        cb = CircuitBreaker(name="test", failure_threshold=1, recovery_timeout=0.01)
        cb.record_failure()
        time.sleep(0.02)
        cb.allow_request()  # Transition to HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_half_open_reopens_on_failure(self):
        import time
        from cloudsearch_shared.resilience import CircuitBreaker, CircuitState
        cb = CircuitBreaker(name="test", failure_threshold=1, recovery_timeout=0.01)
        cb.record_failure()
        time.sleep(0.02)
        cb.allow_request()  # Transition to HALF_OPEN
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
