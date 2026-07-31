"""
Fusion rankers — pluggable re-ranking strategies for the Fusion Core.

Each ranker implements the Ranker ABC:
    rank(query, documents) → list[NormalizedDocument] (sorted desc by score)

Available rankers:
    RRFFusion          — Reciprocal Rank Fusion (fast, parameter-free default)
    ScoreNormFusion    — Score normalization + weighted merge
    DiversityRanker    — MMR-style diversity re-ranking (dedup by domain)
"""
from __future__ import annotations

import abc
import logging
import math
from collections import defaultdict
from urllib.parse import urlparse

from cloudsearch_shared.document import NormalizedDocument

logger = logging.getLogger(__name__)


class Ranker(abc.ABC):
    """Abstract ranker interface."""

    @abc.abstractmethod
    def rank(
        self,
        query: str,
        documents: list[NormalizedDocument],
        top_k: int | None = None,
    ) -> list[NormalizedDocument]:
        """
        Re-rank documents for the given query.

        Args:
            query:     Original (or rewritten) user query.
            documents: Flat merged list from all providers.
            top_k:     Truncate to top_k results. None = return all.

        Returns:
            Documents sorted by descending relevance score.
        """
        ...


class RRFFusion(Ranker):
    """
    Reciprocal Rank Fusion (RRF).

    RRF merges ranked lists from multiple sources by summing reciprocal
    rank scores: score(d) = Σ 1/(k + rank_i(d)) across all lists.

    The ``k`` parameter (default 60) controls how much top-rank documents
    are penalized. Lower k → more aggressive top-rank boosting.

    Reference: Cormack, Clarke & Buettcher (2009) — SIGIR.
    """

    def __init__(self, k: int = 60) -> None:
        self.k = k

    def rank(
        self,
        query: str,
        documents: list[NormalizedDocument],
        top_k: int | None = None,
    ) -> list[NormalizedDocument]:
        if not documents:
            return []

        # Group documents by source_type to form per-source ranked lists
        by_source: dict[str, list[NormalizedDocument]] = defaultdict(list)
        for doc in documents:
            by_source[doc.source_type.value].append(doc)

        # Sort each source list by its own score (desc)
        for source_docs in by_source.values():
            source_docs.sort(key=lambda d: d.score, reverse=True)

        # Accumulate RRF scores keyed by doc.id
        rrf_scores: dict[str, float] = defaultdict(float)
        doc_map: dict[str, NormalizedDocument] = {}

        for source_docs in by_source.values():
            for rank, doc in enumerate(source_docs, start=1):
                rrf_scores[doc.id] += 1.0 / (self.k + rank)
                doc_map[doc.id] = doc

        # Normalize RRF scores to [0, 1]
        max_rrf = max(rrf_scores.values()) if rrf_scores else 1.0
        for doc_id in rrf_scores:
            doc_map[doc_id].score = rrf_scores[doc_id] / max_rrf

        ranked = sorted(doc_map.values(), key=lambda d: d.score, reverse=True)
        return ranked[:top_k] if top_k else ranked


class ScoreNormFusion(Ranker):
    """
    Weighted score normalization fusion.

    Normalizes each source's scores to [0, 1] then combines with
    source-specific weights. Simpler than RRF but requires tuned weights.
    """

    def __init__(self, source_weights: dict[str, float] | None = None) -> None:
        # Default: equal weights for all sources
        self._weights = source_weights or {}

    def rank(
        self,
        query: str,
        documents: list[NormalizedDocument],
        top_k: int | None = None,
    ) -> list[NormalizedDocument]:
        if not documents:
            return []

        # Group + normalize per source
        by_source: dict[str, list[NormalizedDocument]] = defaultdict(list)
        for doc in documents:
            by_source[doc.source_type.value].append(doc)

        doc_map: dict[str, NormalizedDocument] = {}
        combined_scores: dict[str, float] = defaultdict(float)

        for source_name, source_docs in by_source.items():
            if not source_docs:
                continue
            max_s = max(d.score for d in source_docs) or 1.0
            min_s = min(d.score for d in source_docs)
            span = max_s - min_s or 1.0
            weight = self._weights.get(source_name, 1.0)

            for doc in source_docs:
                normalized = (doc.score - min_s) / span
                combined_scores[doc.id] += normalized * weight
                doc_map[doc.id] = doc

        # Write back combined scores
        max_combined = max(combined_scores.values()) if combined_scores else 1.0
        for doc_id, score in combined_scores.items():
            doc_map[doc_id].score = score / max_combined

        ranked = sorted(doc_map.values(), key=lambda d: d.score, reverse=True)
        return ranked[:top_k] if top_k else ranked


class DiversityRanker(Ranker):
    """
    Maximal Marginal Relevance (MMR) -style diversity ranker.

    Selects the top result, then iteratively selects the next result
    that maximises: λ * relevance - (1-λ) * max_similarity_to_selected.

    Similarity is approximated by domain matching (same domain = high sim).
    This prevents 10 results from the same website dominating the feed.

    Args:
        lambda_: Trade-off between relevance (1.0) and diversity (0.0).
        max_per_domain: Hard cap on results per domain.
    """

    def __init__(self, lambda_: float = 0.7, max_per_domain: int = 3) -> None:
        self.lambda_ = lambda_
        self.max_per_domain = max_per_domain

    def rank(
        self,
        query: str,
        documents: list[NormalizedDocument],
        top_k: int | None = None,
    ) -> list[NormalizedDocument]:
        if not documents:
            return []

        limit = top_k or len(documents)
        sorted_docs = sorted(documents, key=lambda d: d.score, reverse=True)

        selected: list[NormalizedDocument] = []
        selected_domains: dict[str, int] = defaultdict(int)
        remaining = list(sorted_docs)

        while remaining and len(selected) < limit:
            best: NormalizedDocument | None = None
            best_score = -math.inf

            for doc in remaining:
                domain = self._domain(doc.url)

                # Hard cap per domain
                if selected_domains[domain] >= self.max_per_domain:
                    continue

                # MMR score
                domain_penalty = selected_domains[domain] * 0.15
                mmr_score = self.lambda_ * doc.score - (1 - self.lambda_) * domain_penalty

                if mmr_score > best_score:
                    best_score = mmr_score
                    best = doc

            if best is None:
                break  # All remaining docs hit domain cap

            selected.append(best)
            selected_domains[self._domain(best.url)] += 1
            remaining.remove(best)

        return selected

    @staticmethod
    def _domain(url: str) -> str:
        try:
            return urlparse(url).netloc.lower()
        except Exception:
            return url


class DeduplicatingRanker(Ranker):
    """
    URL-normalized deduplication pass.
    Removes duplicate documents (same canonical URL) keeping the highest-scored copy.
    Should be applied before RRF to avoid inflating scores for duplicates.
    """

    def rank(
        self,
        query: str,
        documents: list[NormalizedDocument],
        top_k: int | None = None,
    ) -> list[NormalizedDocument]:
        seen_urls: dict[str, NormalizedDocument] = {}
        for doc in sorted(documents, key=lambda d: d.score, reverse=True):
            canonical = self._canonical_url(doc.url)
            if canonical not in seen_urls:
                seen_urls[canonical] = doc

        deduped = list(seen_urls.values())
        deduped.sort(key=lambda d: d.score, reverse=True)
        return deduped[:top_k] if top_k else deduped

    @staticmethod
    def _canonical_url(url: str) -> str:
        """Normalize URL for dedup: lowercase, strip trailing slash + fragment."""
        try:
            p = urlparse(url.lower().rstrip("/"))
            return f"{p.scheme}://{p.netloc}{p.path}"
        except Exception:
            return url.lower()
