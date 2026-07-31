"""
MetasearchProvider — live web search via SearXNG + Brave/Serper fallback.

Priority chain:
  1. SearXNG (self-hosted aggregator)
  2. Brave Search API
  3. Serper.dev API

Falls back through the chain until one succeeds. Yields nothing
(graceful degradation) if all sources fail.
"""
from __future__ import annotations

import logging
import os
from typing import AsyncIterator

import httpx

from cloudsearch_shared.document import NormalizedDocument, SourceType
from .base import SearchOptions, SearchProvider

logger = logging.getLogger(__name__)


class MetasearchProvider(SearchProvider):
    """Live web search aggregating SearXNG, Brave, and Serper."""

    def __init__(
        self,
        searxng_url: str | None = None,
        brave_api_key: str | None = None,
        serper_api_key: str | None = None,
        timeout_s: float = 6.0,
    ) -> None:
        self._searxng_url = (searxng_url or os.getenv("SEARXNG_BASE_URL", "http://localhost:8888")).rstrip("/")
        self._brave_key = brave_api_key or os.getenv("BRAVE_API_KEY", "")
        self._serper_key = serper_api_key or os.getenv("SERPER_API_KEY", "")
        self._timeout = timeout_s
        self._client: httpx.AsyncClient | None = None

    @property
    def source_type(self) -> SourceType:
        return SourceType.WEB

    @property
    def name(self) -> str:
        return "metasearch"

    async def initialize(self) -> None:
        self._client = httpx.AsyncClient(timeout=self._timeout)

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()

    async def health_check(self) -> bool:
        if not self._client:
            return False
        try:
            resp = await self._client.get(f"{self._searxng_url}/healthz", timeout=2.0)
            return resp.status_code == 200
        except Exception:
            # SearXNG might not have /healthz; check if Brave/Serper are configured
            return bool(self._brave_key or self._serper_key)

    async def search(self, query: str, opts: SearchOptions) -> AsyncIterator[NormalizedDocument]:
        if not self._client:
            logger.error("MetasearchProvider not initialized.")
            return

        results = await self._try_searxng(query, opts)
        if not results and self._brave_key:
            results = await self._try_brave(query, opts)
        if not results and self._serper_key:
            results = await self._try_serper(query, opts)

        for doc in results:
            yield doc

    async def _try_searxng(self, query: str, opts: SearchOptions) -> list[NormalizedDocument]:
        try:
            resp = await self._client.get(
                f"{self._searxng_url}/search",
                params={
                    "q": query,
                    "format": "json",
                    "language": opts.language,
                    "pageno": 1,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            results = []
            for i, item in enumerate(data.get("results", [])[:opts.max_results]):
                score = max(0.0, 1.0 - (i * 0.08))
                results.append(NormalizedDocument.create(
                    title=item.get("title", "Untitled"),
                    url=item.get("url", ""),
                    content=item.get("content", ""),
                    snippet=item.get("content", "")[:300],
                    score=score,
                    source_type=SourceType.WEB,
                    metadata={"engine": item.get("engine", "searxng"), "score": item.get("score")},
                ))
            logger.debug("SearXNG returned %d results.", len(results))
            return results
        except Exception as exc:
            logger.debug("SearXNG failed: %s", exc)
            return []

    async def _try_brave(self, query: str, opts: SearchOptions) -> list[NormalizedDocument]:
        try:
            resp = await self._client.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": opts.max_results, "search_lang": opts.language},
                headers={"Accept": "application/json", "X-Subscription-Token": self._brave_key},
            )
            resp.raise_for_status()
            data = resp.json()
            results = []
            for i, item in enumerate(data.get("web", {}).get("results", [])[:opts.max_results]):
                score = max(0.0, 1.0 - (i * 0.08))
                results.append(NormalizedDocument.create(
                    title=item.get("title", "Untitled"),
                    url=item.get("url", ""),
                    content=item.get("description", ""),
                    snippet=item.get("description", "")[:300],
                    score=score,
                    source_type=SourceType.WEB,
                    metadata={"engine": "brave", "age": item.get("age")},
                ))
            logger.debug("Brave returned %d results.", len(results))
            return results
        except Exception as exc:
            logger.debug("Brave search failed: %s", exc)
            return []

    async def _try_serper(self, query: str, opts: SearchOptions) -> list[NormalizedDocument]:
        try:
            resp = await self._client.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": self._serper_key, "Content-Type": "application/json"},
                json={"q": query, "num": opts.max_results, "gl": "us", "hl": opts.language},
            )
            resp.raise_for_status()
            data = resp.json()
            results = []
            for i, item in enumerate(data.get("organic", [])[:opts.max_results]):
                score = max(0.0, 1.0 - (i * 0.08))
                results.append(NormalizedDocument.create(
                    title=item.get("title", "Untitled"),
                    url=item.get("link", ""),
                    content=item.get("snippet", ""),
                    snippet=item.get("snippet", "")[:300],
                    score=score,
                    source_type=SourceType.WEB,
                    metadata={"engine": "serper", "position": item.get("position")},
                ))
            logger.debug("Serper returned %d results.", len(results))
            return results
        except Exception as exc:
            logger.debug("Serper search failed: %s", exc)
            return []
