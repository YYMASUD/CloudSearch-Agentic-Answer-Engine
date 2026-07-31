"""
Strawberry GraphQL schema for CloudSearch.

Provides a non-streaming query/mutation API for programmatic clients.
Streaming is handled separately via SSE at /api/search/stream.
"""
from __future__ import annotations

from typing import Optional

import strawberry
from strawberry.fastapi import GraphQLRouter


@strawberry.type
class SourceCard:
    index: int
    title: str
    url: str
    snippet: str
    source_type: str
    score: float
    favicon_url: str


@strawberry.type
class CitationItem:
    number: int
    url: str
    title: str
    snippet: str
    confidence: float


@strawberry.type
class SearchResult:
    session_id: str
    query: str
    answer: str
    sources: list[SourceCard]
    citations: list[CitationItem]


@strawberry.type
class Query:
    @strawberry.field(description="Execute a synchronous search query.")
    async def search(
        self,
        query: str,
        mode: Optional[str] = "web",
        max_results: Optional[int] = 10,
        tenant_id: Optional[str] = "default",
    ) -> SearchResult:
        """GraphQL search — delegates to the REST pipeline."""
        import httpx
        import os

        gateway_url = os.getenv("GATEWAY_INTERNAL_URL", "http://localhost:8000")
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{gateway_url}/api/search",
                json={
                    "query": query,
                    "mode": mode,
                    "max_results": max_results,
                    "tenant_id": tenant_id,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        return SearchResult(
            session_id=data["session_id"],
            query=data["query"],
            answer=data["answer"],
            sources=[
                SourceCard(
                    index=s["index"],
                    title=s["title"],
                    url=s["url"],
                    snippet=s["snippet"],
                    source_type=s["source_type"],
                    score=s["score"],
                    favicon_url=s["favicon_url"],
                )
                for s in data.get("sources", [])
            ],
            citations=[
                CitationItem(
                    number=c["number"],
                    url=c["url"],
                    title=c["title"],
                    snippet=c["snippet"],
                    confidence=c["confidence"],
                )
                for c in data.get("citations", [])
            ],
        )


schema = strawberry.Schema(query=Query)
graphql_app = GraphQLRouter(schema, graphiql=True)
