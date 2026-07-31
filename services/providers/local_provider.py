"""
LocalProvider — fully offline retrieval using BM25 + local Qdrant.

No external HTTP calls. Suitable for air-gapped deployments.
Documents must be pre-ingested via the RAG pipeline.
"""
from __future__ import annotations

import logging
import os
from typing import AsyncIterator

from cloudsearch_shared.document import NormalizedDocument, SourceType
from .base import SearchOptions, SearchProvider

logger = logging.getLogger(__name__)


class LocalProvider(SearchProvider):
    """
    Offline-capable local retrieval using rank-bm25 + Qdrant local.
    Falls back to BM25-only if Qdrant is unreachable.
    """

    def __init__(self) -> None:
        self._bm25 = None
        self._corpus: list[NormalizedDocument] = []
        self._vector_store = None

    @property
    def source_type(self) -> SourceType:
        return SourceType.LOCAL

    @property
    def name(self) -> str:
        return "local"

    async def initialize(self) -> None:
        # Try to load local vector store
        try:
            from services.rag.vector_store import VectorStore
            self._vector_store = VectorStore()
            await self._vector_store.initialize()
            logger.info("LocalProvider: Qdrant vector store initialized.")
        except Exception as exc:
            logger.warning("LocalProvider: vector store unavailable: %s", exc)
            self._vector_store = None

    async def close(self) -> None:
        if self._vector_store:
            await self._vector_store.close()

    async def health_check(self) -> bool:
        # Always healthy — BM25 is always available even without vector store
        return True

    async def search(self, query: str, opts: SearchOptions) -> AsyncIterator[NormalizedDocument]:
        # Try vector search first
        if self._vector_store and await self._vector_store.health_check():
            try:
                from services.rag.embedder import embedder_router
                qvec = await embedder_router.embed_one(query)
                docs = await self._vector_store.search(
                    query_vector=qvec,
                    top_k=opts.max_results,
                    source_type_filter=SourceType.LOCAL,
                    tenant_id=opts.tenant_id,
                )
                for doc in docs:
                    yield doc
                return
            except Exception as exc:
                logger.warning("LocalProvider vector search failed: %s — falling back to BM25", exc)

        # BM25 fallback using in-memory corpus
        if self._corpus:
            results = self._bm25_search(query, opts.max_results)
            for doc in results:
                yield doc

    def _bm25_search(self, query: str, top_k: int) -> list[NormalizedDocument]:
        """Simple BM25 search over in-memory corpus."""
        try:
            from rank_bm25 import BM25Okapi  # Optional dep: pip install rank-bm25
        except ImportError:
            logger.warning("rank-bm25 not installed — BM25 search unavailable. Run: pip install rank-bm25")
            return []

        try:
            import re

            def tokenize(t: str) -> list[str]:
                return re.findall(r"\b\w+\b", t.lower())

            corpus_tokens = [tokenize(f"{d.title} {d.content}") for d in self._corpus]
            bm25 = BM25Okapi(corpus_tokens)
            scores = bm25.get_scores(tokenize(query))

            pairs = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
            max_score = pairs[0][1] if pairs else 1.0

            results = []
            for idx, score in pairs:
                if score <= 0:
                    continue
                doc = self._corpus[idx]
                doc.score = float(score) / max(float(max_score), 1.0)
                results.append(doc)
            return results
        except Exception as exc:
            logger.warning("BM25 search failed: %s", exc)
            return []

    def add_documents(self, docs: list[NormalizedDocument]) -> None:
        """Add documents to the in-memory BM25 corpus."""
        self._corpus.extend(docs)
