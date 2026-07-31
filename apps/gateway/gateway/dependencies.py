"""
Gateway dependency injection — provider registry and shared clients.

All SearchProvider instances are created once at startup (initialize())
and stored in a module-level registry. FastAPI route handlers access them
via get_provider_registry() dependency.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from cloudsearch_shared.document import SourceType
from services.providers.base import SearchProvider
from services.providers.meilisearch_provider import MeilisearchProvider
from services.llm.router import ModelRouter
from services.orchestrator.agent.planner import Planner
from services.orchestrator.agent.router import SourceRouter
from services.orchestrator.fusion.core import FusionCore
from services.rag.citation_grounder import CitationGrounder
from services.rag.reranker import Reranker

logger = logging.getLogger(__name__)

# ─── Shared singletons ────────────────────────────────────────────────────────
_provider_registry: dict[SourceType, SearchProvider] = {}
_model_router: ModelRouter | None = None
_planner: Planner | None = None
_source_router: SourceRouter | None = None
_fusion_core: FusionCore | None = None
_reranker: Reranker | None = None
_grounder: CitationGrounder | None = None


async def init_providers() -> None:
    """Called once at startup — initialize all providers."""
    global _provider_registry, _model_router, _planner, _source_router
    global _fusion_core, _reranker, _grounder

    # ── Retrieval providers ────────────────────────────────────────────
    meili = MeilisearchProvider()
    await meili.initialize()
    _provider_registry[SourceType.INDEXED] = meili

    # Conditionally add other providers based on feature flags
    if os.getenv("ENABLE_LOCAL_PROVIDER", "true").lower() == "true":
        try:
            from services.providers.local_provider import LocalProvider
            local = LocalProvider()
            await local.initialize()
            _provider_registry[SourceType.LOCAL] = local
            logger.info("LocalProvider initialized.")
        except Exception as exc:
            logger.warning("LocalProvider not available: %s", exc)

    if os.getenv("BRAVE_API_KEY") or os.getenv("SERPER_API_KEY"):
        try:
            from services.providers.metasearch_provider import MetasearchProvider
            web = MetasearchProvider()
            await web.initialize()
            _provider_registry[SourceType.WEB] = web
            logger.info("MetasearchProvider (WEB) initialized.")
        except Exception as exc:
            logger.warning("MetasearchProvider not available: %s", exc)

    logger.info(
        "Provider registry: %s",
        [st.value for st in _provider_registry],
    )

    # ── LLM + Orchestration ───────────────────────────────────────────
    _model_router = ModelRouter()
    _planner = Planner()
    _source_router = SourceRouter()
    _fusion_core = FusionCore()
    _reranker = Reranker()
    _grounder = CitationGrounder()

    logger.info("Gateway dependencies initialized.")


async def close_providers() -> None:
    """Called at shutdown — close provider connections."""
    for provider in _provider_registry.values():
        try:
            await provider.close()
        except Exception as exc:
            logger.warning("Error closing provider %r: %s", provider.name, exc)


# ─── FastAPI dependency functions ─────────────────────────────────────────────

def get_provider_registry() -> dict[SourceType, SearchProvider]:
    return _provider_registry


def get_model_router() -> ModelRouter:
    assert _model_router is not None, "ModelRouter not initialized"
    return _model_router


def get_planner() -> Planner:
    assert _planner is not None, "Planner not initialized"
    return _planner


def get_source_router() -> SourceRouter:
    assert _source_router is not None, "SourceRouter not initialized"
    return _source_router


def get_fusion_core() -> FusionCore:
    assert _fusion_core is not None, "FusionCore not initialized"
    return _fusion_core


def get_reranker() -> Reranker:
    assert _reranker is not None, "Reranker not initialized"
    return _reranker


def get_grounder() -> CitationGrounder:
    assert _grounder is not None, "CitationGrounder not initialized"
    return _grounder
