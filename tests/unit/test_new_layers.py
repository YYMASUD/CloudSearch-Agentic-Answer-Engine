"""
Unit tests for newly completed architecture layer components:
GitHubProvider, OnyxProvider, GraphRAGBuilder, RAPTORTreeBuilder, and KafkaIngestionWorker.
"""
import pytest
from cloudsearch_shared.document import NormalizedDocument, SourceType
from services.providers.github_provider import GitHubProvider
from services.providers.onyx_provider import OnyxProvider
from services.rag.graph_rag import GraphRAGBuilder
from services.rag.raptor import RAPTORTreeBuilder
from services.rag.kafka_worker import KafkaIngestionWorker
from services.providers.base import SearchOptions


@pytest.mark.asyncio
async def test_github_provider_fallback():
    provider = GitHubProvider()
    await provider.initialize()
    opts = SearchOptions(max_results=5)
    results = [doc async for doc in provider.search("react hooks", opts)]
    await provider.close()

    assert len(results) > 0
    assert results[0].source_type == SourceType.CODE
    assert "react" in results[0].content.lower() or "cloudsearch" in results[0].content.lower()


@pytest.mark.asyncio
async def test_onyx_provider_fallback():
    provider = OnyxProvider()
    await provider.initialize()
    opts = SearchOptions(max_results=5, tenant_id="acme_corp")
    results = [doc async for doc in provider.search("security policy", opts)]
    await provider.close()

    assert len(results) > 0
    assert results[0].source_type == SourceType.PRIVATE
    assert results[0].metadata["tenant_id"] == "acme_corp"


def test_graph_rag_builder():
    builder = GraphRAGBuilder()
    doc = NormalizedDocument.create(
        title="Architecture Overview",
        url="https://cloudsearch.internal/docs",
        content="CloudSearch uses Meilisearch for BM25 and Qdrant for Vector Store retrieval. GraphRAG extracts entities.",
        score=1.0,
        source_type=SourceType.INDEXED
    )
    graph_stats = builder.build_knowledge_graph([doc])

    assert graph_stats["node_count"] > 0
    context = builder.query_graph_context("Meilisearch")
    assert len(context) > 0


def test_raptor_tree_builder():
    builder = RAPTORTreeBuilder(max_levels=2, cluster_size=2)
    docs = [
        NormalizedDocument.create(
            title=f"Doc {i}",
            url=f"https://example.com/doc/{i}",
            content=f"Content chunk number {i} describing search features.",
            score=0.9,
            source_type=SourceType.INDEXED
        )
        for i in range(4)
    ]
    tree_docs = builder.build_tree(docs)

    assert len(tree_docs) > len(docs)
    summary_docs = [d for d in tree_docs if d.metadata.get("is_raptor_summary")]
    assert len(summary_docs) > 0


@pytest.mark.asyncio
async def test_kafka_ingestion_worker():
    worker = KafkaIngestionWorker()
    event_data = {
        "title": "CloudSearch Ingest Test",
        "url": "https://example.com/test",
        "content": "Kafka ingestion stream test for RAGFlow deep document parsing and GraphRAG indexing.",
        "tenant_id": "test_tenant"
    }
    result = await worker.process_document_event(event_data)

    assert result["status"] == "PROCESSED"
    assert result["chunks_count"] > 0
