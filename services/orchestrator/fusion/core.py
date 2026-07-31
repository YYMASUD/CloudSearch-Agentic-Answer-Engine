"""
Fusion Core ★ — the key differentiator.

Merges results from all SearchProvider backends into a single
normalized, ranked stream via a composable ranker pipeline:

  raw docs → Dedup → RRF → Diversity → [optional cross-encoder] → top-K

The pipeline is configurable at construction time. Default pipeline:
  1. DeduplicatingRanker  — remove URL-duplicate results
  2. RRFFusion            — merge per-source ranked lists via RRF
  3. DiversityRanker      — MMR-style domain diversity

An optional CrossEncoder re-rank pass (Phase 2 reranker) can be wired in
as a fourth stage. When enabled it's applied only to the top-N candidates
to keep latency manageable.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

from cloudsearch_shared.document import NormalizedDocument
from services.orchestrator.agent.fan_out import ProviderResult, merge_provider_results
from .rankers import (
    DeduplicatingRanker,
    DiversityRanker,
    Ranker,
    RRFFusion,
    ScoreNormFusion,
)

logger = logging.getLogger(__name__)

# Default top-K returned by the Fusion Core to the RAG / Generation layers
DEFAULT_FUSION_TOP_K = int(os.getenv("FUSION_TOP_K", "15"))

# If > 0, run cross-encoder re-rank on the top-N candidates after RRF
CROSS_ENCODER_TOP_N = int(os.getenv("FUSION_CROSS_ENCODER_TOP_N", "0"))


@dataclass
class FusionStats:
    """Diagnostic statistics from a single fusion run."""
    total_raw_docs: int
    after_dedup: int
    after_rrf: int
    after_diversity: int
    final_count: int
    provider_counts: dict[str, int] = field(default_factory=dict)
    elapsed_ms: int = 0


@dataclass
class FusionResult:
    """Output of the Fusion Core."""
    documents: list[NormalizedDocument]
    stats: FusionStats


class FusionCore:
    """
    Composable fusion pipeline.

    The pipeline is expressed as a list of Ranker objects applied in
    sequence. Each ranker receives the full list and returns a re-ordered
    (and possibly truncated) list.

    Default pipeline: Dedup → RRF → Diversity
    Alternative:      Dedup → ScoreNorm → Diversity  (set FUSION_STRATEGY=score_norm)
    """

    def __init__(
        self,
        pipeline: list[Ranker] | None = None,
        top_k: int = DEFAULT_FUSION_TOP_K,
    ) -> None:
        self._top_k = top_k
        self._pipeline = pipeline or self._default_pipeline()

    # ─── Public API ───────────────────────────────────────────────────

    def fuse(
        self,
        query: str,
        provider_results: list[ProviderResult],
    ) -> FusionResult:
        """
        Run the fusion pipeline over raw provider results.

        Args:
            query:            User query (passed to rankers for context).
            provider_results: Output of fan_out().

        Returns:
            FusionResult with ranked, deduplicated, diverse documents.
        """
        import time
        start = time.monotonic()

        # Collect stats
        provider_counts = {
            pr.source_type.value: len(pr.documents)
            for pr in provider_results
        }
        raw_docs = merge_provider_results(provider_results)
        total_raw = len(raw_docs)

        if not raw_docs:
            logger.warning("FusionCore: no documents received from any provider.")
            return FusionResult(
                documents=[],
                stats=FusionStats(
                    total_raw_docs=0,
                    after_dedup=0,
                    after_rrf=0,
                    after_diversity=0,
                    final_count=0,
                    provider_counts=provider_counts,
                ),
            )

        # Run pipeline stages
        docs = raw_docs
        stage_counts: list[int] = []

        for ranker in self._pipeline:
            docs = ranker.rank(query, docs, top_k=None)
            stage_counts.append(len(docs))
            logger.debug(
                "FusionCore after %s: %d docs",
                ranker.__class__.__name__,
                len(docs),
            )

        # Final top-K truncation
        final = docs[: self._top_k]

        elapsed_ms = int((time.monotonic() - start) * 1000)
        logger.info(
            "FusionCore: %d raw → %d final in %dms",
            total_raw, len(final), elapsed_ms,
        )

        # Map stage counts to stat fields (default 0 if pipeline is custom)
        after_dedup   = stage_counts[0] if len(stage_counts) > 0 else total_raw
        after_rrf     = stage_counts[1] if len(stage_counts) > 1 else after_dedup
        after_diversity = stage_counts[2] if len(stage_counts) > 2 else after_rrf

        return FusionResult(
            documents=final,
            stats=FusionStats(
                total_raw_docs=total_raw,
                after_dedup=after_dedup,
                after_rrf=after_rrf,
                after_diversity=after_diversity,
                final_count=len(final),
                provider_counts=provider_counts,
                elapsed_ms=elapsed_ms,
            ),
        )

    # ─── Pipeline factory ─────────────────────────────────────────────

    @staticmethod
    def _default_pipeline() -> list[Ranker]:
        strategy = os.getenv("FUSION_STRATEGY", "rrf").lower()
        if strategy == "score_norm":
            merge_ranker: Ranker = ScoreNormFusion()
        else:
            merge_ranker = RRFFusion(k=60)

        lambda_ = float(os.getenv("FUSION_DIVERSITY_LAMBDA", "0.7"))
        max_per_domain = int(os.getenv("FUSION_MAX_PER_DOMAIN", "3"))

        return [
            DeduplicatingRanker(),
            merge_ranker,
            DiversityRanker(lambda_=lambda_, max_per_domain=max_per_domain),
        ]

    # ─── Convenience builder ──────────────────────────────────────────

    @classmethod
    def with_cross_encoder(cls, top_k: int = DEFAULT_FUSION_TOP_K) -> "FusionCore":
        """
        Build a FusionCore with cross-encoder re-rank as the final stage.
        Requires sentence-transformers. Falls back to default if unavailable.
        """
        from services.rag.reranker import CrossEncoderReranker

        class _CrossEncoderRanker(Ranker):
            """Adapter: wraps async CrossEncoderReranker for sync Ranker ABC."""

            def __init__(self) -> None:
                self._reranker = CrossEncoderReranker()

            def rank(self, query: str, documents: list[NormalizedDocument], top_k=None):
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # Can't nest event loops — return original order
                        logger.warning("CrossEncoder unavailable inside running loop — using original order.")
                        return documents[:top_k] if top_k else documents
                    result = loop.run_until_complete(
                        self._reranker.rerank(query, documents, top_k)
                    )
                    return result
                except Exception as exc:
                    logger.warning("CrossEncoderRanker failed: %s", exc)
                    return documents[:top_k] if top_k else documents

        pipeline = cls._default_pipeline() + [_CrossEncoderRanker()]
        return cls(pipeline=pipeline, top_k=top_k)
