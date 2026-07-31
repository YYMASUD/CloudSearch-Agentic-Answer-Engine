"""
MeilisearchProvider — BM25 + vector hybrid search adapter.

Uses Meilisearch v1.6+ hybrid search (semanticRatio controls the blend).
Falls back to BM25-only if the embedder is not configured.
Degrades gracefully (yields nothing + logs) if Meilisearch is unreachable.

References:
    https://www.meilisearch.com/docs/learn/ai-powered-search/hybrid-search
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import AsyncIterator, Any

import httpx

from cloudsearch_shared.document import NormalizedDocument, SourceType
from .base import SearchOptions, SearchProvider

logger = logging.getLogger(__name__)

# ─── Default index settings ────────────────────────────────────────────────────

_DEFAULT_SETTINGS: dict[str, Any] = {
    "searchableAttributes": ["title", "content", "snippet", "url"],
    "displayedAttributes": ["id", "title", "url", "snippet", "content", "score", "source_type", "metadata", "chunk_idx", "doc_id"],
    "filterableAttributes": ["source_type", "doc_id", "chunk_idx"],
    "sortableAttributes": ["score"],
    "rankingRules": [
        "words",
        "typo",
        "proximity",
        "attribute",
        "sort",
        "exactness",
    ],
    "typoTolerance": {
        "enabled": True,
        "minWordSizeForTypos": {"oneTypo": 5, "twoTypos": 9},
    },
    "pagination": {"maxTotalHits": 1000},
}

_LOCAL_EMBEDDER_SETTINGS: dict[str, Any] = {
    "embedders": {
        "default": {
            "source": "huggingFace",
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "documentTemplate": "{{doc.title}} {{doc.snippet}}",
        }
    }
}


class MeilisearchProvider(SearchProvider):
    """
    Retrieval backend backed by Meilisearch hybrid search.

    Configuration via environment variables:
        MEILISEARCH_URL          (default: http://localhost:7700)
        MEILISEARCH_MASTER_KEY   (default: changeme-master-key)
        MEILISEARCH_INDEX_NAME   (default: documents)
        MEILISEARCH_EMBEDDER_SOURCE  (default: local)

    The provider uses the async httpx client internally so it fits
    naturally in asyncio fan-out without blocking the event loop.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        index_name: str | None = None,
        embedder_source: str | None = None,
        timeout_s: float = 5.0,
    ) -> None:
        self._base_url = (base_url or os.getenv("MEILISEARCH_URL", "http://localhost:7700")).rstrip("/")
        self._api_key = api_key or os.getenv("MEILISEARCH_MASTER_KEY", "changeme-master-key")
        self._index_name = index_name or os.getenv("MEILISEARCH_INDEX_NAME", "documents")
        self._embedder_source = embedder_source or os.getenv("MEILISEARCH_EMBEDDER_SOURCE", "local")
        self._timeout = timeout_s
        self._client: httpx.AsyncClient | None = None
        self._hybrid_enabled: bool = False

    # ─── Lifecycle ────────────────────────────────────────────────────

    @property
    def source_type(self) -> SourceType:
        return SourceType.INDEXED

    @property
    def name(self) -> str:
        return "meilisearch"

    async def initialize(self) -> None:
        """Create the async HTTP client and ensure index + settings exist."""
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=self._timeout,
        )

        if not await self.health_check():
            logger.warning(
                "Meilisearch not reachable at %s — provider will degrade gracefully.",
                self._base_url,
            )
            return

        await self._ensure_index()
        await self._apply_settings()
        self._hybrid_enabled = await self._check_hybrid_support()
        logger.info(
            "MeilisearchProvider initialized. Index=%r hybrid=%s",
            self._index_name,
            self._hybrid_enabled,
        )

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def health_check(self) -> bool:
        if not self._client:
            return False
        try:
            resp = await self._client.get("/health", timeout=2.0)
            return resp.status_code == 200
        except Exception:
            return False

    # ─── Search ───────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        opts: SearchOptions,
    ) -> AsyncIterator[NormalizedDocument]:
        """Hybrid BM25 + vector search with graceful degradation."""
        if not self._client:
            logger.error("MeilisearchProvider not initialized — skipping.")
            return

        if not await self.health_check():
            logger.warning("Meilisearch unreachable — skipping provider.")
            return

        try:
            hits = await self._execute_search(query, opts)
            for hit in hits:
                doc = self._hit_to_document(hit)
                yield doc
        except Exception as exc:
            logger.exception(
                "MeilisearchProvider.search failed for query %r: %s", query, exc
            )
            return  # Degrade: yield nothing, let other providers continue

    # ─── Private helpers ──────────────────────────────────────────────

    async def _execute_search(
        self, query: str, opts: SearchOptions
    ) -> list[dict[str, Any]]:
        """Build and fire the search request; return raw Meilisearch hits."""
        payload: dict[str, Any] = {
            "q": query,
            "limit": opts.max_results,
            "showRankingScore": True,
        }

        if self._hybrid_enabled:
            payload["hybrid"] = {
                "semanticRatio": opts.semantic_ratio,
                "embedder": "default",
            }

        # Apply source_type filter if mode restricts it
        if opts.mode != "web":
            payload["filter"] = f'source_type = "{opts.mode.upper()}"'

        resp = await self._client.post(
            f"/indexes/{self._index_name}/search",
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("hits", [])

    def _hit_to_document(self, hit: dict[str, Any]) -> NormalizedDocument:
        """Convert a raw Meilisearch hit dict to a NormalizedDocument."""
        # Meilisearch returns _rankingScore in [0, 1] when showRankingScore=True
        score = float(hit.get("_rankingScore", 0.5))

        return NormalizedDocument.create(
            title=hit.get("title", "Untitled"),
            url=hit.get("url", ""),
            content=hit.get("content", ""),
            snippet=hit.get("snippet", hit.get("content", "")[:300]),
            score=score,
            source_type=SourceType(hit.get("source_type", SourceType.INDEXED.value)),
            chunk_idx=int(hit.get("chunk_idx", 0)),
            metadata={
                **hit.get("metadata", {}),
                "meilisearch_id": hit.get("id"),
            },
        )

    async def _ensure_index(self) -> None:
        """Create the index if it does not exist."""
        try:
            resp = await self._client.get(f"/indexes/{self._index_name}")
            if resp.status_code == 404:
                create_resp = await self._client.post(
                    "/indexes",
                    json={"uid": self._index_name, "primaryKey": "id"},
                )
                # Wait for the task to complete
                task_uid = create_resp.json().get("taskUid")
                if task_uid:
                    await self._wait_for_task(task_uid)
        except Exception as exc:
            logger.warning("Could not ensure Meilisearch index: %s", exc)

    async def _apply_settings(self) -> None:
        """Apply index settings (searchable attributes, ranking, embedder)."""
        try:
            resp = await self._client.patch(
                f"/indexes/{self._index_name}/settings",
                json=_DEFAULT_SETTINGS,
            )
            task_uid = resp.json().get("taskUid")
            if task_uid:
                await self._wait_for_task(task_uid)

            # Apply embedder settings for hybrid search
            embedder_settings = self._build_embedder_settings()
            if embedder_settings:
                emb_resp = await self._client.patch(
                    f"/indexes/{self._index_name}/settings",
                    json={"embedders": embedder_settings},
                )
                emb_task = emb_resp.json().get("taskUid")
                if emb_task:
                    await self._wait_for_task(emb_task)
        except Exception as exc:
            logger.warning("Could not apply Meilisearch settings: %s", exc)

    def _build_embedder_settings(self) -> dict[str, Any]:
        """Build embedder config based on MEILISEARCH_EMBEDDER_SOURCE."""
        source = self._embedder_source.lower()
        if source == "openai":
            openai_key = os.getenv("OPENAI_API_KEY", "")
            if not openai_key:
                logger.warning("OPENAI_API_KEY not set, falling back to local embedder.")
                source = "local"
            else:
                return {
                    "default": {
                        "source": "openAi",
                        "apiKey": openai_key,
                        "model": "text-embedding-3-small",
                        "documentTemplate": "{{doc.title}} {{doc.snippet}}",
                    }
                }
        if source in ("local", "huggingface"):
            return {
                "default": {
                    "source": "huggingFace",
                    "model": "sentence-transformers/all-MiniLM-L6-v2",
                    "documentTemplate": "{{doc.title}} {{doc.snippet}}",
                }
            }
        # No embedder configured → BM25 only
        return {}

    async def _check_hybrid_support(self) -> bool:
        """Return True if the index has an embedder configured."""
        try:
            resp = await self._client.get(f"/indexes/{self._index_name}/settings/embedders")
            if resp.status_code == 200:
                embedders = resp.json()
                return bool(embedders)
        except Exception:
            pass
        return False

    async def _wait_for_task(self, task_uid: int, max_wait_s: float = 30.0) -> None:
        """Poll Meilisearch task until succeeded or failed."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max_wait_s
        while loop.time() < deadline:
            try:
                resp = await self._client.get(f"/tasks/{task_uid}")
                status = resp.json().get("status")
                if status == "succeeded":
                    return
                if status == "failed":
                    logger.error("Meilisearch task %d failed: %s", task_uid, resp.json())
                    return
            except Exception:
                pass
            await asyncio.sleep(0.5)

    # ─── Document ingestion (for indexing documents into Meilisearch) ─

    async def index_documents(self, documents: list[NormalizedDocument]) -> None:
        """
        Bulk-index documents into Meilisearch.

        Used by the ingestion pipeline and tests. Not part of the
        SearchProvider ABC (ingestion is separate from retrieval).
        """
        if not self._client:
            raise RuntimeError("Provider not initialized — call initialize() first.")

        docs = [doc.to_dict() for doc in documents]
        resp = await self._client.post(
            f"/indexes/{self._index_name}/documents",
            json=docs,
            params={"primaryKey": "id"},
        )
        resp.raise_for_status()
        task_uid = resp.json().get("taskUid")
        if task_uid:
            await self._wait_for_task(task_uid)
        logger.info("Indexed %d documents into Meilisearch.", len(docs))

    async def delete_document(self, doc_id: str) -> None:
        """Delete a single document by primary key."""
        if not self._client:
            raise RuntimeError("Provider not initialized.")
        await self._client.delete(f"/indexes/{self._index_name}/documents/{doc_id}")
