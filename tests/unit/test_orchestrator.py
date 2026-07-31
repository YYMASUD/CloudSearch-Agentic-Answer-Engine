"""
Unit tests for Orchestrator: Planner, SourceRouter, FanOut, FusionCore.
Run with: pytest tests/unit/test_orchestrator.py -v
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from cloudsearch_shared.document import NormalizedDocument, SourceType
from services.orchestrator.agent.planner import Planner, QueryIntent, classify_intent, rewrite_query
from services.orchestrator.agent.router import SourceRoute, SourceRouter
from services.orchestrator.agent.fan_out import fan_out, merge_provider_results, ProviderResult
from services.orchestrator.fusion.core import FusionCore
from services.orchestrator.fusion.rankers import (
    DeduplicatingRanker, DiversityRanker, RRFFusion, ScoreNormFusion,
)
from services.providers.base import SearchOptions


# ─── Planner Tests ────────────────────────────────────────────────────────────

class TestPlanner:
    def test_code_intent_from_def_keyword(self):
        assert classify_intent("def fibonacci(n):") == QueryIntent.CODE

    def test_code_intent_from_error(self):
        assert classify_intent("TypeError: unsupported operand type") == QueryIntent.CODE

    def test_code_intent_from_how_to_write(self):
        assert classify_intent("how to write a REST API in Python") == QueryIntent.CODE

    def test_github_intent_from_url(self):
        assert classify_intent("github.com/microsoft/vscode issues") == QueryIntent.GITHUB

    def test_github_intent_from_pr(self):
        assert classify_intent("How do I review PR #42?") == QueryIntent.GITHUB

    def test_research_intent_long_query(self):
        assert classify_intent("how does retrieval augmented generation work in detail") == QueryIntent.RESEARCH

    def test_factual_intent(self):
        assert classify_intent("What is the capital of France?") == QueryIntent.FACTUAL

    def test_mode_override_wins(self):
        # Even a code-looking query with mode_override=local → LOCAL
        assert classify_intent("def foo():", mode_override="local") == QueryIntent.LOCAL

    def test_mode_override_github(self):
        assert classify_intent("search something", mode_override="github") == QueryIntent.GITHUB

    def test_rewrite_code_adds_context(self):
        rewritten = rewrite_query("binary search", QueryIntent.CODE)
        assert "code" in rewritten.lower() or "example" in rewritten.lower()

    def test_rewrite_strips_trailing_punctuation(self):
        rewritten = rewrite_query("What is Python?", QueryIntent.FACTUAL)
        assert not rewritten.endswith("?")

    def test_planner_returns_result(self):
        planner = Planner()
        result = planner.plan("What is RAG?")
        assert result.intent == QueryIntent.FACTUAL
        assert result.original_query == "What is RAG?"
        assert result.rewritten_query  # non-empty

    def test_planner_mode_override_high_confidence(self):
        planner = Planner()
        result = planner.plan("search docs", mode_override="local")
        assert result.intent == QueryIntent.LOCAL
        assert result.confidence == pytest.approx(0.9)


# ─── SourceRouter Tests ───────────────────────────────────────────────────────

class TestSourceRouter:
    def test_code_intent_routes_to_code_and_indexed(self):
        router = SourceRouter()
        route = router.route(QueryIntent.CODE)
        assert SourceType.CODE in route.sources
        assert SourceType.INDEXED in route.sources

    def test_local_intent_routes_to_local_only(self):
        router = SourceRouter()
        route = router.route(QueryIntent.LOCAL)
        assert route.sources == {SourceType.LOCAL}

    def test_code_weight_higher_than_indexed(self):
        router = SourceRouter()
        route = router.route(QueryIntent.CODE)
        assert route.weight_for(SourceType.CODE) > route.weight_for(SourceType.INDEXED)

    def test_disabled_providers_filtered(self, monkeypatch):
        monkeypatch.setenv("DISABLED_PROVIDERS", "WEB")
        router = SourceRouter()
        route = router.route(QueryIntent.FACTUAL)
        assert SourceType.WEB not in route.sources

    def test_unknown_intent_returns_web_and_indexed(self):
        router = SourceRouter()
        route = router.route(QueryIntent.UNKNOWN)
        assert SourceType.WEB in route.sources
        assert SourceType.INDEXED in route.sources


# ─── FanOut Tests ─────────────────────────────────────────────────────────────

def _make_doc(url: str, score: float, source_type: SourceType) -> NormalizedDocument:
    return NormalizedDocument.create(
        title="Test",
        url=url,
        content="test content",
        score=score,
        source_type=source_type,
    )


def _mock_provider(source_type: SourceType, docs: list[NormalizedDocument]):
    """Create an async mock SearchProvider yielding the given docs."""
    async def _search(query, opts):
        for doc in docs:
            yield doc

    provider = MagicMock()
    provider.source_type = source_type
    provider.name = source_type.value.lower()
    provider.search = _search
    return provider


@pytest.mark.asyncio
async def test_fan_out_collects_from_all_providers():
    web_docs = [_make_doc("https://web.com/1", 0.9, SourceType.WEB)]
    idx_docs = [_make_doc("https://indexed.com/1", 0.8, SourceType.INDEXED)]

    providers = {
        SourceType.WEB: _mock_provider(SourceType.WEB, web_docs),
        SourceType.INDEXED: _mock_provider(SourceType.INDEXED, idx_docs),
    }
    route = SourceRoute(sources={SourceType.WEB, SourceType.INDEXED})
    results = await fan_out("test query", providers, route)

    all_docs = merge_provider_results(results)
    assert len(all_docs) == 2
    urls = {d.url for d in all_docs}
    assert "https://web.com/1" in urls
    assert "https://indexed.com/1" in urls


@pytest.mark.asyncio
async def test_fan_out_missing_provider_skipped():
    """Provider registered in route but not in providers dict → skipped."""
    providers = {SourceType.INDEXED: _mock_provider(SourceType.INDEXED, [])}
    route = SourceRoute(sources={SourceType.WEB, SourceType.INDEXED})
    results = await fan_out("test", providers, route)
    # WEB skipped, INDEXED ran (0 docs)
    assert len(results) == 1


@pytest.mark.asyncio
async def test_fan_out_timeout_degrades_gracefully(monkeypatch):
    """Provider that hangs should be cancelled after timeout."""
    # monkeypatch.setenv won't update an already-evaluated module constant —
    # patch the constant in the module namespace directly.
    import sys
    import importlib
    fan_out_mod = sys.modules.get("services.orchestrator.agent.fan_out") or \
        importlib.import_module("services.orchestrator.agent.fan_out")
    monkeypatch.setattr(fan_out_mod, "PROVIDER_TIMEOUT_S", 0.1)

    async def _slow_search(query, opts):
        import asyncio
        await asyncio.sleep(10)  # Will be cancelled
        yield _make_doc("https://slow.com", 0.5, SourceType.WEB)

    slow = MagicMock()
    slow.source_type = SourceType.WEB
    slow.name = "slow"
    slow.search = _slow_search

    providers = {SourceType.WEB: slow}
    route = SourceRoute(sources={SourceType.WEB})
    results = await fan_out("test", providers, route)
    assert results[0].timed_out is True
    assert results[0].documents == []


# ─── FusionCore Tests ─────────────────────────────────────────────────────────

class TestRRFFusion:
    def _make_docs(self, n: int, source: SourceType, base_score: float = 1.0) -> list[NormalizedDocument]:
        return [
            _make_doc(f"https://{source.value.lower()}.com/{i}", base_score - i * 0.05, source)
            for i in range(n)
        ]

    def test_rrf_merges_two_sources(self):
        web_docs = self._make_docs(5, SourceType.WEB)
        idx_docs = self._make_docs(5, SourceType.INDEXED, base_score=0.9)
        all_docs = web_docs + idx_docs

        ranker = RRFFusion(k=60)
        result = ranker.rank("query", all_docs)
        assert len(result) == 10
        # Scores should be normalized to [0, 1]
        assert all(0.0 <= d.score <= 1.0 for d in result)
        # Should be sorted descending
        scores = [d.score for d in result]
        assert scores == sorted(scores, reverse=True)

    def test_rrf_empty_input(self):
        assert RRFFusion().rank("q", []) == []


class TestDeduplicatingRanker:
    def test_removes_url_duplicates(self):
        doc1 = _make_doc("https://example.com/page", 0.9, SourceType.WEB)
        doc2 = _make_doc("https://example.com/page", 0.7, SourceType.INDEXED)  # Same URL
        doc3 = _make_doc("https://other.com/page", 0.8, SourceType.WEB)

        result = DeduplicatingRanker().rank("q", [doc1, doc2, doc3])
        assert len(result) == 2
        urls = {d.url for d in result}
        assert "https://example.com/page" in urls

    def test_keeps_highest_scored_duplicate(self):
        doc1 = _make_doc("https://example.com/", 0.9, SourceType.WEB)
        doc2 = _make_doc("https://example.com/", 0.5, SourceType.INDEXED)
        result = DeduplicatingRanker().rank("q", [doc1, doc2])
        assert result[0].score == pytest.approx(0.9)


class TestDiversityRanker:
    def test_limits_results_per_domain(self):
        docs = [
            _make_doc(f"https://same-domain.com/{i}", 1.0 - i * 0.01, SourceType.WEB)
            for i in range(10)
        ] + [_make_doc("https://other.com/1", 0.5, SourceType.WEB)]

        result = DiversityRanker(max_per_domain=3).rank("q", docs, top_k=5)
        domain_counts: dict[str, int] = {}
        from urllib.parse import urlparse
        for d in result:
            dom = urlparse(d.url).netloc
            domain_counts[dom] = domain_counts.get(dom, 0) + 1

        assert domain_counts.get("same-domain.com", 0) <= 3


class TestFusionCore:
    def test_fuse_returns_fusion_result(self):
        docs = [_make_doc(f"https://src.com/{i}", 0.9 - i * 0.1, SourceType.WEB) for i in range(5)]
        pr = ProviderResult(
            source_type=SourceType.WEB,
            documents=docs,
            elapsed_ms=50,
        )
        core = FusionCore(top_k=3)
        result = core.fuse("query", [pr])
        assert len(result.documents) <= 3
        assert result.stats.total_raw_docs == 5
        assert result.stats.final_count <= 3

    def test_fuse_empty_input(self):
        core = FusionCore()
        result = core.fuse("query", [])
        assert result.documents == []
        assert result.stats.final_count == 0

    def test_fuse_deduplicates_cross_provider(self):
        """Same URL from two providers should be deduplicated."""
        url = "https://example.com/shared"
        pr_web = ProviderResult(
            source_type=SourceType.WEB,
            documents=[_make_doc(url, 0.9, SourceType.WEB)],
            elapsed_ms=10,
        )
        pr_idx = ProviderResult(
            source_type=SourceType.INDEXED,
            documents=[_make_doc(url, 0.7, SourceType.INDEXED)],
            elapsed_ms=5,
        )
        core = FusionCore()
        result = core.fuse("query", [pr_web, pr_idx])
        urls = [d.url for d in result.documents]
        assert urls.count(url) == 1
