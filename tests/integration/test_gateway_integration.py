"""
Integration tests — full gateway pipeline.

Requires:
    docker compose --profile lite up -d

Run:
    pytest tests/integration -v
"""
from __future__ import annotations

import asyncio
import json
import os
import time

import httpx
import pytest

GATEWAY_URL = os.getenv("GATEWAY_URL", "http://localhost:8000")


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="session")
async def client():
    async with httpx.AsyncClient(base_url=GATEWAY_URL, timeout=30.0) as c:
        yield c


# ─── Health ───────────────────────────────────────────────────────────────────

class TestHealth:
    async def test_liveness(self, client: httpx.AsyncClient):
        resp = await client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["service"] == "cloudsearch-gateway"

    async def test_readiness_shape(self, client: httpx.AsyncClient):
        resp = await client.get("/health/ready")
        # May be 200 or 503 depending on infra, but shape must match
        assert resp.status_code in {200, 503}
        body = resp.json()
        assert "ready" in body
        assert "checks" in body
        assert "elapsed_ms" in body


# ─── Search (sync) ────────────────────────────────────────────────────────────

class TestSearchSync:
    @pytest.mark.skipif(
        os.getenv("SKIP_LIVE_SEARCH", "true") == "true",
        reason="Live search disabled (SKIP_LIVE_SEARCH=true)"
    )
    async def test_search_response_schema(self, client: httpx.AsyncClient):
        resp = await client.post("/api/search", json={
            "query": "What is the Python GIL?",
            "mode": "web",
            "max_results": 5,
        })
        assert resp.status_code == 200
        body = resp.json()
        assert "session_id" in body
        assert "answer" in body
        assert isinstance(body["sources"], list)
        assert isinstance(body["citations"], list)
        assert "fusion_stats" in body

    async def test_search_validation_empty_query(self, client: httpx.AsyncClient):
        resp = await client.post("/api/search", json={"query": ""})
        assert resp.status_code == 422  # Pydantic validation error

    async def test_search_validation_query_too_long(self, client: httpx.AsyncClient):
        resp = await client.post("/api/search", json={"query": "x" * 1001})
        assert resp.status_code == 422


# ─── SSE Streaming ────────────────────────────────────────────────────────────

class TestSearchStream:
    @pytest.mark.skipif(
        os.getenv("SKIP_LIVE_SEARCH", "true") == "true",
        reason="Live search disabled"
    )
    async def test_sse_event_types(self, client: httpx.AsyncClient):
        event_types: list[str] = []
        async with client.stream("GET", "/api/search/stream?q=hello+world&mode=web") as resp:
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers.get("content-type", "")
            async for line in resp.aiter_lines():
                if line.startswith("event: "):
                    event_types.append(line[7:].strip())
                if len(event_types) >= 3:
                    break
        assert len(event_types) > 0, "No SSE events received"


# ─── Cache ────────────────────────────────────────────────────────────────────

class TestCache:
    @pytest.mark.skipif(
        os.getenv("SKIP_LIVE_SEARCH", "true") == "true",
        reason="Live search disabled"
    )
    async def test_cache_hit_on_second_request(self, client: httpx.AsyncClient):
        payload = {"query": "unique cache test query 12345", "mode": "web", "max_results": 3}
        # First request — should be a MISS
        resp1 = await client.post("/api/search", json=payload)
        assert resp1.status_code == 200
        # Second request — should be a HIT
        resp2 = await client.post("/api/search", json=payload)
        assert resp2.status_code == 200
        # Both answers should match
        assert resp1.json()["answer"] == resp2.json()["answer"]


# ─── Rate Limiting ────────────────────────────────────────────────────────────

class TestRateLimit:
    @pytest.mark.skipif(
        os.getenv("SKIP_RATE_LIMIT_TEST", "true") == "true",
        reason="Rate limit test disabled"
    )
    async def test_rate_limit_triggers_429(self, client: httpx.AsyncClient):
        """Burst 70 health requests — some must be rate-limited."""
        responses = await asyncio.gather(*[
            client.get("/api/search/stream?q=test")
            for _ in range(70)
        ], return_exceptions=True)
        status_codes = [
            r.status_code for r in responses
            if isinstance(r, httpx.Response)
        ]
        assert 429 in status_codes, f"Expected 429 in responses, got: {set(status_codes)}"
