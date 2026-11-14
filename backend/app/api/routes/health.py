from fastapi import APIRouter
from app.core.config import settings
from app.schemas.health import HealthResponse

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health and Readiness Check",
    description="Returns current backend status, version, device, and model readiness.",
)
async def get_health() -> HealthResponse:
    """Return backend operational status."""
    return HealthResponse(
        status="healthy",
        version=settings.VERSION,
        model_loaded=False,  # Unloaded until ML pipeline is integrated in Step 3
        model_architecture="Pending Integration (Phase 2 Baseline)",
        device=settings.DEVICE,
        supported_domains=7,
    )

