"""Health check router — liveness + deep readiness with per-provider probing."""
import asyncio
import time
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


@router.get("", response_model=HealthResponse, summary="Liveness probe")
async def health():
    return HealthResponse(status="ok", service="cloudsearch-gateway", version="0.1.0")


@router.get("/ready", summary="Deep readiness probe")
async def ready():
    """
    Concurrently probe all registered providers and critical infrastructure.
    Returns 503 if any critical dependency is unhealthy.
    """
    from gateway.dependencies import (
        get_provider_registry, get_model_router, _get_cache, _session_store
    )

    checks: dict = {}
    start = time.monotonic()

    # ── Provider health checks (concurrent) ──────────────────────────
    try:
        providers = get_provider_registry()
        async def _probe(name: str, provider) -> tuple[str, dict]:
            t0 = time.monotonic()
            try:
                ok = await asyncio.wait_for(provider.health_check(), timeout=3.0)
            except Exception as exc:
                ok = False
            elapsed = int((time.monotonic() - t0) * 1000)
            return name, {"healthy": ok, "latency_ms": elapsed}

        tasks = [_probe(st.value, p) for st, p in providers.items()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, tuple):
                n, stat = r
                checks[f"provider.{n}"] = stat
    except Exception as exc:
        checks["providers"] = {"healthy": False, "error": str(exc)}

    # ── Model router ─────────────────────────────────────────────────
    try:
        checks["model_router"] = {"healthy": get_model_router() is not None}
    except AssertionError:
        checks["model_router"] = {"healthy": False}

    # ── Redis cache ──────────────────────────────────────────────────
    try:
        cache = _get_cache()
        if cache:
            ok = await asyncio.wait_for(cache.health_check(), timeout=2.0)
            checks["redis"] = {"healthy": ok}
        else:
            checks["redis"] = {"healthy": False, "note": "not initialized"}
    except Exception as exc:
        checks["redis"] = {"healthy": False, "error": str(exc)}

    # ── Postgres session store ───────────────────────────────────────
    try:
        from gateway.dependencies import _session_store as ss
        if ss:
            ok = await asyncio.wait_for(ss.health_check(), timeout=2.0)
            checks["postgres"] = {"healthy": ok}
        else:
            checks["postgres"] = {"healthy": False, "note": "not initialized"}
    except Exception as exc:
        checks["postgres"] = {"healthy": False, "error": str(exc)}

    elapsed_ms = int((time.monotonic() - start) * 1000)

    # Critical: Meilisearch (INDEXED) and model_router must be healthy
    critical_ok = checks.get("model_router", {}).get("healthy", False)
    indexed = checks.get("provider.INDEXED", {})
    if indexed:
        critical_ok = critical_ok and indexed.get("healthy", False)

    status_code = 200 if critical_ok else 503
    return JSONResponse(
        status_code=status_code,
        content={
            "ready": critical_ok,
            "elapsed_ms": elapsed_ms,
            "checks": checks,
        },
    )
