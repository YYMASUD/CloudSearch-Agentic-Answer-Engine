"""
Parallel fan-out — queries multiple SearchProviders concurrently.

Uses asyncio.gather with per-provider timeouts. A failing or slow
provider never blocks results from healthy ones. All results are
collected into a single flat list tagged with source weights for
the Fusion Core.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass

from cloudsearch_shared.document import NormalizedDocument, SourceType
from cloudsearch_shared.resilience import CircuitBreaker
from services.providers.base import SearchOptions, SearchProvider
from .router import SourceRoute

logger = logging.getLogger(__name__)

# Per-provider timeout — providers exceeding this are cancelled gracefully
PROVIDER_TIMEOUT_S: float = float(os.getenv("PROVIDER_TIMEOUT_S", "8.0"))

# Per-provider circuit breakers (module-level, persist across requests)
_circuit_breakers: dict[str, CircuitBreaker] = {}


def _get_circuit_breaker(provider_name: str) -> CircuitBreaker:
    if provider_name not in _circuit_breakers:
        _circuit_breakers[provider_name] = CircuitBreaker(name=provider_name)
    return _circuit_breakers[provider_name]


@dataclass
class ProviderResult:
    """Raw results from a single provider, pre-fusion."""
    source_type: SourceType
    documents: list[NormalizedDocument]
    elapsed_ms: int
    timed_out: bool = False
    error: str | None = None


async def _collect_provider(
    provider: SearchProvider,
    query: str,
    opts: SearchOptions,
    weight: float,
) -> ProviderResult:
    """
    Collect all results from a single provider with timeout protection
    and circuit breaker integration.
    """
    cb = _get_circuit_breaker(provider.name)

    # Circuit breaker: fail-fast if circuit is open
    if not cb.allow_request():
        logger.info("Provider %r circuit OPEN — skipping.", provider.name)
        return ProviderResult(
            source_type=provider.source_type,
            documents=[],
            elapsed_ms=0,
            timed_out=False,
            error=f"Circuit breaker OPEN for {provider.name}",
        )

    start = time.monotonic()
    docs: list[NormalizedDocument] = []
    timed_out = False
    error = None

    try:
        async def _run():
            async for doc in provider.search(query, opts):
                doc.score = min(1.0, doc.score * weight)
                docs.append(doc)
        await asyncio.wait_for(_run(), timeout=PROVIDER_TIMEOUT_S)
        cb.record_success()
    except asyncio.TimeoutError:
        timed_out = True
        cb.record_failure()
        logger.warning(
            "Provider %r timed out after %.1fs for query %r",
            provider.name,
            PROVIDER_TIMEOUT_S,
            query[:60],
        )
    except Exception as exc:
        error = str(exc)
        cb.record_failure()
        logger.exception("Provider %r raised: %s", provider.name, exc)

    elapsed_ms = int((time.monotonic() - start) * 1000)
    logger.debug(
        "Provider %r: %d docs in %dms (timeout=%s, error=%s)",
        provider.name, len(docs), elapsed_ms, timed_out, error,
    )
    return ProviderResult(
        source_type=provider.source_type,
        documents=docs,
        elapsed_ms=elapsed_ms,
        timed_out=timed_out,
        error=error,
    )


async def fan_out(
    query: str,
    providers: dict[SourceType, SearchProvider],
    route: SourceRoute,
    base_opts: SearchOptions | None = None,
) -> list[ProviderResult]:
    """
    Query all providers in the route concurrently.

    Args:
        query:     Rewritten user query.
        providers: Map of SourceType → SearchProvider (all available providers).
        route:     Routing decision from SourceRouter (which to activate + weights).
        base_opts: Base SearchOptions; per-provider max_results is set from route.

    Returns:
        List of ProviderResult (one per active, healthy provider).
        Providers that are missing from the registry are silently skipped.
    """
    opts_base = base_opts or SearchOptions()

    tasks = []
    active_providers = []

    for source_type in route.sources:
        provider = providers.get(source_type)
        if provider is None:
            logger.debug("No provider registered for %s — skipping.", source_type)
            continue

        # Per-provider options
        opts = SearchOptions(
            max_results=route.max_results_per_source,
            semantic_ratio=opts_base.semantic_ratio,
            language=opts_base.language,
            tenant_id=opts_base.tenant_id,
            filters=opts_base.filters,
            mode=opts_base.mode,
            include_content=opts_base.include_content,
        )
        weight = route.weight_for(source_type)
        tasks.append(_collect_provider(provider, query, opts, weight))
        active_providers.append(source_type)

    if not tasks:
        logger.warning("fan_out: no providers available for route %s", route.sources)
        return []

    logger.info(
        "fan_out: querying %d providers concurrently: %s",
        len(tasks),
        [st.value for st in active_providers],
    )

    results = await asyncio.gather(*tasks, return_exceptions=False)
    return list(results)


def merge_provider_results(provider_results: list[ProviderResult]) -> list[NormalizedDocument]:
    """
    Flatten all ProviderResult.documents into a single list.
    Used by the Fusion Core as its input.
    """
    merged: list[NormalizedDocument] = []
    for pr in provider_results:
        merged.extend(pr.documents)
    return merged
