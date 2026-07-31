"""
CloudSearch API Gateway — FastAPI application entry point.

Routes:
    GET  /health                  — liveness probe
    POST /api/search              — synchronous search (JSON response)
    GET  /api/search/stream       — SSE streaming search
    POST /graphql                 — Strawberry GraphQL endpoint
    GET  /metrics                 — Prometheus metrics

The gateway is the single entry point for the web client.
It coordinates with the orchestrator, RAG, and LLM services.
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from cloudsearch_shared.telemetry import setup_telemetry

# Import routers
from gateway.routers.health import router as health_router
from gateway.routers.search import router as search_router
from gateway.routers.graphql import graphql_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize + teardown app-level resources."""
    logger.info("CloudSearch Gateway starting up…")

    # Bootstrap OpenTelemetry
    try:
        setup_telemetry("cloudsearch-gateway")
    except Exception as exc:
        logger.warning("OTel setup failed (continuing without tracing): %s", exc)

    # Initialize provider registry (shared across requests)
    from gateway.dependencies import init_providers
    await init_providers()

    yield

    logger.info("CloudSearch Gateway shutting down…")
    from gateway.dependencies import close_providers
    await close_providers()


app = FastAPI(
    title="CloudSearch API Gateway",
    description="Agentic answer engine — Perplexity-style multi-source search",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ─── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Prometheus metrics ───────────────────────────────────────────────────────
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# ─── Routers ──────────────────────────────────────────────────────────────────
app.include_router(health_router, prefix="/health", tags=["health"])
app.include_router(search_router, prefix="/api", tags=["search"])
app.mount("/graphql", graphql_app)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "gateway.main:app",
        host="0.0.0.0",
        port=int(os.getenv("GATEWAY_PORT", "8000")),
        reload=os.getenv("ENV", "development") == "development",
        log_level="info",
    )
