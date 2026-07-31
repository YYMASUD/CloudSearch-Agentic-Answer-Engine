"""
Search router — the core API endpoints.

POST /api/search          — JSON response (non-streaming, for GraphQL / quick clients)
GET  /api/search/stream   — SSE streaming response (primary UI endpoint)
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Annotated, AsyncIterator

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from cloudsearch_shared.document import SourceType
from services.orchestrator.agent.fan_out import fan_out
from services.orchestrator.agent.planner import Planner
from services.orchestrator.agent.router import SourceRouter
from services.orchestrator.fusion.core import FusionCore
from services.rag.reranker import Reranker
from services.rag.citation_grounder import CitationGrounder
from services.llm.router import ModelRouter
from services.llm.synthesizer import stream_answer
from services.llm.streaming import stream_search_response, stream_error
from services.providers.base import SearchOptions
from gateway.dependencies import (
    get_fusion_core,
    get_grounder,
    get_model_router,
    get_planner,
    get_provider_registry,
    get_reranker,
    get_source_router,
    get_cache,
    get_session_store,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ─── Request / Response schemas ───────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="User search query")
    mode: str = Field("web", description="Search mode: web | code | github | local | private")
    max_results: int = Field(10, ge=1, le=50)
    semantic_ratio: float = Field(0.5, ge=0.0, le=1.0)
    tenant_id: str = Field("default")
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class SourceCardResponse(BaseModel):
    index: int
    id: str
    title: str
    url: str
    snippet: str
    source_type: str
    score: float
    favicon_url: str


class SearchResponse(BaseModel):
    session_id: str
    query: str
    answer: str
    citations: list[dict]
    sources: list[SourceCardResponse]
    fusion_stats: dict


# ─── Core pipeline helper ─────────────────────────────────────────────────────

async def _run_pipeline(
    request: SearchRequest,
    planner: Planner,
    source_router: SourceRouter,
    providers: dict,
    fusion_core: FusionCore,
    reranker: Reranker,
):
    """Shared pipeline: plan → route → fan-out → fuse → rerank."""
    # 1. Plan
    plan = planner.plan(request.query, mode_override=request.mode if request.mode != "web" else None)

    # 2. Route
    route = source_router.route(plan.intent)

    # 3. Fan-out
    opts = SearchOptions(
        max_results=request.max_results,
        semantic_ratio=request.semantic_ratio,
        tenant_id=request.tenant_id,
        mode=request.mode,
    )
    provider_results = await fan_out(plan.rewritten_query, providers, route, opts)

    # 4. Fuse
    fusion_result = fusion_core.fuse(plan.rewritten_query, provider_results)

    # 5. Re-rank
    reranked = await reranker.rerank(plan.rewritten_query, fusion_result.documents, top_k=10)

    return plan, fusion_result, reranked


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/search", response_model=SearchResponse, summary="Synchronous search")
async def search(
    body: SearchRequest,
    planner: Annotated[Planner, Depends(get_planner)],
    source_router: Annotated[SourceRouter, Depends(get_source_router)],
    providers: Annotated[dict, Depends(get_provider_registry)],
    fusion_core: Annotated[FusionCore, Depends(get_fusion_core)],
    reranker: Annotated[Reranker, Depends(get_reranker)],
    model_router: Annotated[ModelRouter, Depends(get_model_router)],
    grounder: Annotated[CitationGrounder, Depends(get_grounder)],
):
    """
    Full synchronous search — returns complete answer with citations.
    Suitable for programmatic clients. For the UI, use /search/stream.
    """
    import time as _time
    t_start = _time.monotonic()

    # ── Cache check ────────────────────────────────────────────────────────
    try:
        cache = get_cache()
        cached = await cache.get(body.query, body.mode)
        if cached:
            from fastapi.responses import JSONResponse
            return JSONResponse(content=cached, headers={"X-Cache": "HIT"})
    except Exception:
        pass  # Cache unavailable — continue to pipeline

    plan, fusion_result, reranked = await _run_pipeline(
        body, planner, source_router, providers, fusion_core, reranker
    )

    # Generate answer (collect full stream)
    answer_tokens = stream_answer(plan.rewritten_query, reranked, model_router)
    full_answer = "".join([token async for token in answer_tokens])

    # Ground citations
    grounding = grounder.ground(full_answer, reranked)

    sources = [
        SourceCardResponse(
            index=i + 1,
            id=doc.id,
            title=doc.title,
            url=doc.url,
            snippet=doc.snippet,
            source_type=doc.source_type.value,
            score=round(doc.score, 4),
            favicon_url=f"https://www.google.com/s2/favicons?domain={doc.url}&sz=32",
        )
        for i, doc in enumerate(reranked)
    ]

    response = SearchResponse(
        session_id=body.session_id,
        query=body.query,
        answer=grounding.answer_with_citations,
        citations=grounding.to_dict()["citations"],
        sources=sources,
        fusion_stats={"total_raw": fusion_result.stats.total_raw_docs,
                      "final_count": fusion_result.stats.final_count,
                      "elapsed_ms": fusion_result.stats.elapsed_ms},
    )

    duration_ms = int((_time.monotonic() - t_start) * 1000)
    response_dict = response.model_dump()

    # ── Cache set (fire-and-forget) ───────────────────────────────────────────
    try:
        await get_cache().set(body.query, body.mode, response_dict)
    except Exception:
        pass

    # ── Session persistence (fire-and-forget) ─────────────────────────────
    try:
        await get_session_store().save_session(
            session_id=body.session_id,
            query=body.query,
            rewritten_query=plan.rewritten_query,
            intent=plan.intent,
            sources_used=[s.source_type for s in sources],
            answer_text=full_answer,
            citations=grounding.to_dict()["citations"],
            duration_ms=duration_ms,
            tenant_id=body.tenant_id,
        )
    except Exception:
        pass

    from fastapi.responses import JSONResponse
    return JSONResponse(content=response_dict, headers={"X-Cache": "MISS"})


@router.get("/search/stream", summary="SSE streaming search")
async def search_stream(
    q: str = Query(..., min_length=1, max_length=1000, description="Search query"),
    mode: str = Query("web", description="Search mode"),
    max_results: int = Query(10, ge=1, le=50),
    semantic_ratio: float = Query(0.5, ge=0.0, le=1.0),
    tenant_id: str = Query("default"),
    session_id: str = Query(default="", description="Client session ID (auto-generated if empty)"),
    planner: Planner = Depends(get_planner),
    source_router: SourceRouter = Depends(get_source_router),
    providers: dict = Depends(get_provider_registry),
    fusion_core: FusionCore = Depends(get_fusion_core),
    reranker: Reranker = Depends(get_reranker),
    model_router: ModelRouter = Depends(get_model_router),
    grounder: CitationGrounder = Depends(get_grounder),
):
    """
    SSE streaming search endpoint.

    Emits events in order:
      source_card  (N events, one per source)
      answer_chunk (M events, one per LLM token)
      answer_done  (1 event, final answer + citations)

    Connect with EventSource in the browser:
        const es = new EventSource('/api/search/stream?q=...')
    """
    body = SearchRequest(
        query=q,
        mode=mode,
        max_results=max_results,
        semantic_ratio=semantic_ratio,
        tenant_id=tenant_id,
        session_id=session_id or str(uuid.uuid4()),
    )

    async def _event_generator() -> AsyncIterator[str]:
        try:
            plan, fusion_result, reranked = await _run_pipeline(
                body, planner, source_router, providers, fusion_core, reranker
            )

            if not reranked:
                async for chunk in stream_error("No sources found for your query.", recoverable=True):
                    yield chunk
                return

            # Stream answer tokens lazily (generator not yet consumed)
            answer_tokens = stream_answer(plan.rewritten_query, reranked, model_router)

            # Collect tokens + build citations simultaneously
            collected_tokens: list[str] = []

            async def _collecting_tokens():
                async for tok in answer_tokens:
                    collected_tokens.append(tok)
                    yield tok

            async for chunk in stream_search_response(
                query=body.query,
                sources=reranked,
                answer_tokens=_collecting_tokens(),
            ):
                yield chunk

            full_answer = "".join(collected_tokens)
            grounding = grounder.ground(full_answer, reranked)
            # Emit a separate citations event for progressive UI update
            yield f"event: citations\ndata: {json.dumps(grounding.to_dict()['citations'])}\n\n"

        except Exception as exc:
            logger.exception("SSE stream error: %s", exc)
            async for chunk in stream_error(str(exc)):
                yield chunk

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
