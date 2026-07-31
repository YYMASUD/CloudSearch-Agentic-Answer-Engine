"""Health check router."""
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


@router.get("", response_model=HealthResponse, summary="Liveness probe")
async def health():
    return HealthResponse(status="ok", service="cloudsearch-gateway", version="0.1.0")


@router.get("/ready", summary="Readiness probe")
async def ready():
    """Check that all required dependencies are initialized."""
    from gateway.dependencies import get_provider_registry, get_model_router
    try:
        providers = get_provider_registry()
        router_ok = get_model_router() is not None
        checks = {
            "providers": len(providers) > 0,
            "model_router": router_ok,
        }
    except AssertionError:
        checks = {"providers": False, "model_router": False}

    all_ready = all(checks.values())
    return {
        "ready": all_ready,
        "checks": checks,
    }
