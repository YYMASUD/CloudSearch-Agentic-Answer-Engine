#!/usr/bin/env python3
"""
Meilisearch index initialization script.
Run this once after `docker compose up` to create and configure the index.

Usage:
    python infra/meilisearch/init-index.py
"""
import asyncio
import os
import sys

# Add repo root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from cloudsearch_shared.document import NormalizedDocument, SourceType
from services.providers.meilisearch_provider import MeilisearchProvider


SAMPLE_DOCUMENTS = [
    NormalizedDocument.create(
        title="CloudSearch Architecture Overview",
        url="https://example.com/cloudsearch-architecture",
        content=(
            "CloudSearch is a 6-layer agentic answer engine that combines multiple "
            "retrieval backends with a Fusion Core for result blending and cross-source "
            "re-ranking. The system supports streaming answers with inline citations."
        ),
        snippet="CloudSearch is a 6-layer agentic answer engine...",
        score=1.0,
        source_type=SourceType.INDEXED,
        metadata={"category": "docs", "version": "0.1.0"},
    ),
    NormalizedDocument.create(
        title="Retrieval Augmented Generation (RAG) Explained",
        url="https://example.com/rag-explained",
        content=(
            "RAG combines document retrieval with language model generation. "
            "Documents are chunked, embedded, and stored in a vector database. "
            "At query time, relevant chunks are retrieved and injected into the LLM prompt."
        ),
        snippet="RAG combines document retrieval with language model generation.",
        score=0.95,
        source_type=SourceType.INDEXED,
        metadata={"category": "ml"},
    ),
    NormalizedDocument.create(
        title="Meilisearch Hybrid Search Documentation",
        url="https://www.meilisearch.com/docs/learn/ai-powered-search/hybrid-search",
        content=(
            "Meilisearch hybrid search combines BM25 keyword matching with vector "
            "semantic search. The semanticRatio parameter controls the blend from "
            "0.0 (pure BM25) to 1.0 (pure vector)."
        ),
        snippet="Meilisearch hybrid search combines BM25 with vector search.",
        score=0.9,
        source_type=SourceType.INDEXED,
        metadata={"category": "docs"},
    ),
]


async def main() -> None:
    provider = MeilisearchProvider()
    await provider.initialize()

    if not await provider.health_check():
        print("❌ Meilisearch is not reachable. Start it with: docker compose up meilisearch")
        sys.exit(1)

    print(f"✓ Meilisearch reachable. Indexing {len(SAMPLE_DOCUMENTS)} sample documents...")
    await provider.index_documents(SAMPLE_DOCUMENTS)
    print("✓ Sample documents indexed successfully.")

    # Test a search
    from services.providers.base import SearchOptions
    opts = SearchOptions(max_results=5)
    results = []
    async for doc in provider.search("agentic search engine", opts):
        results.append(doc)

    print(f"✓ Test search returned {len(results)} results.")
    for doc in results:
        print(f"  [{doc.score:.3f}] {doc.title} — {doc.url}")

    await provider.close()
    print("✓ Initialization complete.")


if __name__ == "__main__":
    asyncio.run(main())
