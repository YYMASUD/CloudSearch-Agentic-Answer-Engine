"""
Re-ranker — cross-encoder and LLM-based re-ranking of retrieved documents.

Two strategies:
1. CrossEncoderReranker  — fast, open-source model (ms-marco-MiniLM-L-6-v2).
   Scores (query, document) pairs; O(n) inference per document.

2. LLMReranker — sends a batch relevance scoring prompt to the LLM.
   Higher quality but slower; used as optional second-pass.

The Reranker class wraps both strategies behind a unified interface.
"""
from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from cloudsearch_shared.document import NormalizedDocument

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    """A document with its re-ranked score."""
    document: NormalizedDocument
    rerank_score: float


class BaseReranker(ABC):
    """Abstract re-ranker interface."""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: list[NormalizedDocument],
        top_k: int | None = None,
    ) -> list[RerankResult]:
        """
        Re-rank documents for a given query.

        Args:
            query:     The user query.
            documents: Candidate documents to re-rank.
            top_k:     Return only the top N results. None = return all.

        Returns:
            List of RerankResult sorted by rerank_score descending.
        """
        ...


class CrossEncoderReranker(BaseReranker):
    """
    Re-ranker using sentence-transformers CrossEncoder.
    Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (fast, open-source)

    Scores each (query, passage) pair; higher score = more relevant.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        self._model_name = model_name
        self._model: Any = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)
            logger.info("Loaded CrossEncoder model %r", self._model_name)
        except ImportError:
            logger.error("sentence-transformers not installed for CrossEncoder.")
            raise

    async def rerank(
        self,
        query: str,
        documents: list[NormalizedDocument],
        top_k: int | None = None,
    ) -> list[RerankResult]:
        if not documents:
            return []

        import asyncio
        import concurrent.futures

        try:
            self._load_model()
        except Exception:
            # Degrade: return original order with original scores
            logger.warning("CrossEncoder unavailable — returning original order.")
            results = [RerankResult(doc, doc.score) for doc in documents]
            return results[:top_k] if top_k else results

        pairs = [(query, doc.content[:512]) for doc in documents]  # truncate for speed

        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            raw_scores = await loop.run_in_executor(
                pool,
                lambda: self._model.predict(pairs),
            )

        # Normalize scores to [0, 1] using sigmoid
        import math
        def sigmoid(x: float) -> float:
            return 1.0 / (1.0 + math.exp(-x))

        results = [
            RerankResult(document=doc, rerank_score=sigmoid(float(score)))
            for doc, score in zip(documents, raw_scores)
        ]
        results.sort(key=lambda r: r.rerank_score, reverse=True)
        return results[:top_k] if top_k else results


class LLMReranker(BaseReranker):
    """
    LLM-based re-ranker using a structured scoring prompt.
    More expensive but can reason about relevance holistically.

    The LLM is asked to score each snippet 1-5 for relevance to the query.
    Scores are parsed from the structured response and normalized.
    """

    def __init__(self, llm_url: str | None = None) -> None:
        self._llm_url = llm_url or os.getenv("LLM_URL", "http://localhost:8003")

    async def rerank(
        self,
        query: str,
        documents: list[NormalizedDocument],
        top_k: int | None = None,
    ) -> list[RerankResult]:
        if not documents:
            return []

        import httpx

        snippets = [
            f"[{i+1}] {doc.title}: {doc.snippet[:200]}"
            for i, doc in enumerate(documents)
        ]
        prompt = (
            f"Query: {query}\n\n"
            "Rate each snippet's relevance to the query on a scale of 1-5.\n"
            "Respond ONLY with a JSON array of integers: [score1, score2, ...]\n\n"
            + "\n".join(snippets)
        )

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    f"{self._llm_url}/rerank",
                    json={"prompt": prompt, "count": len(documents)},
                )
                resp.raise_for_status()
                scores_raw = resp.json().get("scores", [])
        except Exception as exc:
            logger.warning("LLMReranker failed: %s — returning original order.", exc)
            results = [RerankResult(doc, doc.score) for doc in documents]
            return results[:top_k] if top_k else results

        # Normalize 1-5 → 0-1
        results = []
        for i, doc in enumerate(documents):
            raw = scores_raw[i] if i < len(scores_raw) else 3
            normalized = (float(raw) - 1.0) / 4.0
            results.append(RerankResult(document=doc, rerank_score=normalized))

        results.sort(key=lambda r: r.rerank_score, reverse=True)
        return results[:top_k] if top_k else results


class Reranker:
    """
    Unified re-ranker that selects strategy based on environment config.

    RERANKER_STRATEGY: "cross_encoder" (default) | "llm" | "none"
    """

    def __init__(self) -> None:
        strategy = os.getenv("RERANKER_STRATEGY", "cross_encoder").lower()
        if strategy == "llm":
            self._impl: BaseReranker = LLMReranker()
        elif strategy == "none":
            self._impl = _PassthroughReranker()
        else:
            self._impl = CrossEncoderReranker()

    async def rerank(
        self,
        query: str,
        documents: list[NormalizedDocument],
        top_k: int | None = None,
    ) -> list[NormalizedDocument]:
        """Re-rank and return documents (scores updated in-place)."""
        results = await self._impl.rerank(query, documents, top_k)
        # Update document scores with re-rank scores
        output = []
        for r in results:
            r.document.score = r.rerank_score
            output.append(r.document)
        return output


class _PassthroughReranker(BaseReranker):
    """No-op re-ranker — returns documents in original order."""

    async def rerank(
        self,
        query: str,
        documents: list[NormalizedDocument],
        top_k: int | None = None,
    ) -> list[RerankResult]:
        results = [RerankResult(doc, doc.score) for doc in documents]
        return results[:top_k] if top_k else results
