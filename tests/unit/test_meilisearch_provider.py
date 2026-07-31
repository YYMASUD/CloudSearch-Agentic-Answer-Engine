"""
Unit tests for MeilisearchProvider.
Uses httpx mock to avoid requiring a live Meilisearch instance.

Run with: pytest tests/unit/test_meilisearch_provider.py -v
"""
from __future__ import annotations

import json
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import pytest_asyncio

from cloudsearch_shared.document import NormalizedDocument, SourceType
from services.providers.base import SearchOptions
from services.providers.meilisearch_provider import MeilisearchProvider


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def search_options() -> SearchOptions:
    return SearchOptions(max_results=5, semantic_ratio=0.5)


@pytest.fixture
def sample_hits() -> list[dict]:
    return [
        {
            "id": "abc123",
            "title": "AI Search Engines",
            "url": "https://example.com/ai-search",
            "snippet": "A comprehensive overview of AI-powered search engines.",
            "content": "AI search engines use retrieval augmented generation...",
            "score": 0.0,  # Will be overridden by _rankingScore
            "_rankingScore": 0.92,
            "source_type": "INDEXED",
            "metadata": {"category": "ai"},
            "chunk_idx": 0,
            "doc_id": "deadbeef",
        },
        {
            "id": "def456",
            "title": "Vector Databases Overview",
            "url": "https://example.com/vector-db",
            "snippet": "Comparing Qdrant, Pinecone, and Weaviate.",
            "content": "Vector databases store high-dimensional embeddings...",
            "score": 0.0,
            "_rankingScore": 0.78,
            "source_type": "INDEXED",
            "metadata": {},
            "chunk_idx": 0,
            "doc_id": "cafebabe",
        },
    ]


_DUMMY_REQUEST = httpx.Request("POST", "http://localhost:7700/indexes/test_docs/search")

def _make_response(status: int, body: dict) -> httpx.Response:
    return httpx.Response(
        status_code=status,
        content=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
        request=_DUMMY_REQUEST,
    )


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestMeilisearchProvider:
    @pytest.fixture(autouse=True)
    def provider(self):
        return MeilisearchProvider(
            base_url="http://localhost:7700",
            api_key="test-key",
            index_name="test_docs",
            embedder_source="local",
        )

    # ── health_check ──────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_health_check_true_when_ok(self, provider):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=_make_response(200, {"status": "available"}))
        provider._client = mock_client
        assert await provider.health_check() is True

    @pytest.mark.asyncio
    async def test_health_check_false_when_unreachable(self, provider):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
        provider._client = mock_client
        assert await provider.health_check() is False

    @pytest.mark.asyncio
    async def test_health_check_false_when_not_initialized(self, provider):
        # Client is None before initialize()
        assert await provider.health_check() is False

    # ── search ────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_search_returns_normalized_documents(self, provider, search_options, sample_hits):
        mock_client = AsyncMock()

        # health_check
        mock_client.get = AsyncMock(return_value=_make_response(200, {"status": "available"}))
        # search
        mock_client.post = AsyncMock(
            return_value=_make_response(200, {"hits": sample_hits, "estimatedTotalHits": 2})
        )

        provider._client = mock_client
        provider._hybrid_enabled = False

        results = []
        async for doc in provider.search("ai search", search_options):
            results.append(doc)

        assert len(results) == 2
        assert all(isinstance(d, NormalizedDocument) for d in results)

    @pytest.mark.asyncio
    async def test_search_scores_extracted_from_ranking_score(self, provider, search_options, sample_hits):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=_make_response(200, {"status": "available"}))
        mock_client.post = AsyncMock(
            return_value=_make_response(200, {"hits": sample_hits})
        )
        provider._client = mock_client
        provider._hybrid_enabled = False

        results = []
        async for doc in provider.search("query", search_options):
            results.append(doc)

        assert results[0].score == pytest.approx(0.92)
        assert results[1].score == pytest.approx(0.78)

    @pytest.mark.asyncio
    async def test_search_degrades_gracefully_when_unreachable(self, provider, search_options):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=_make_response(503, {}))
        provider._client = mock_client

        results = []
        async for doc in provider.search("query", search_options):
            results.append(doc)

        assert results == []  # Graceful degradation — no crash, no results

    @pytest.mark.asyncio
    async def test_search_degrades_on_http_error(self, provider, search_options):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=_make_response(200, {"status": "available"}))
        mock_client.post = AsyncMock(side_effect=httpx.HTTPError("500"))
        provider._client = mock_client
        provider._hybrid_enabled = False

        results = []
        async for doc in provider.search("query", search_options):
            results.append(doc)

        assert results == []  # No crash

    @pytest.mark.asyncio
    async def test_search_uses_hybrid_payload_when_enabled(self, provider, search_options, sample_hits):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=_make_response(200, {"status": "available"}))
        mock_client.post = AsyncMock(
            return_value=_make_response(200, {"hits": sample_hits})
        )
        provider._client = mock_client
        provider._hybrid_enabled = True

        async for _ in provider.search("query", search_options):
            pass

        call_args = mock_client.post.call_args
        payload = call_args.kwargs.get("json") or call_args.args[1] if len(call_args.args) > 1 else None
        if payload is None:
            payload = call_args.kwargs["json"]
        assert "hybrid" in payload
        assert payload["hybrid"]["semanticRatio"] == search_options.semantic_ratio

    # ── source_type and name ──────────────────────────────────────────

    def test_source_type(self, provider):
        assert provider.source_type == SourceType.INDEXED

    def test_name(self, provider):
        assert provider.name == "meilisearch"

    # ── _hit_to_document ──────────────────────────────────────────────

    def test_hit_to_document_mapping(self, provider, sample_hits):
        hit = sample_hits[0]
        doc = provider._hit_to_document(hit)
        assert doc.title == "AI Search Engines"
        assert doc.url == "https://example.com/ai-search"
        assert doc.score == pytest.approx(0.92)
        assert doc.source_type == SourceType.INDEXED
        assert doc.chunk_idx == 0

    def test_hit_to_document_score_clamped(self, provider):
        hit = {
            "title": "Test",
            "url": "https://example.com",
            "content": "test content",
            "snippet": "test",
            "_rankingScore": 1.5,  # Out of range
            "source_type": "INDEXED",
            "metadata": {},
            "chunk_idx": 0,
        }
        doc = provider._hit_to_document(hit)
        assert doc.score == 1.0  # Clamped

    # ── index_documents ───────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_index_documents_raises_without_initialize(self, provider):
        with pytest.raises(RuntimeError, match="not initialized"):
            await provider.index_documents([])

    @pytest.mark.asyncio
    async def test_index_documents_calls_correct_endpoint(self, provider):
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            return_value=_make_response(202, {"taskUid": 1})
        )
        mock_client.get = AsyncMock(
            side_effect=[
                _make_response(200, {"status": "available"}),  # health
                _make_response(200, {"status": "succeeded"}),  # task poll
            ]
        )
        provider._client = mock_client

        doc = NormalizedDocument.create(
            title="Test",
            url="https://example.com",
            content="content",
            score=0.5,
            source_type=SourceType.INDEXED,
        )
        await provider.index_documents([doc])

        assert mock_client.post.called
        endpoint = mock_client.post.call_args.args[0]
        assert "documents" in endpoint
