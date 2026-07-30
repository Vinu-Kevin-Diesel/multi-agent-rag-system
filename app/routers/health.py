from datetime import datetime, timezone

from fastapi import APIRouter

from app.config import settings
from app.schemas import AblationFlags, HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc),
        # Reported so the eval harness can record the configuration a run was actually collected
        # under, rather than trusting the --config label it was given.
        flags=AblationFlags(
            router_mode=settings.router_mode,
            router_model=settings.effective_router_model,
            decompose_enabled=settings.decompose_enabled,
            critic_mode=settings.critic_mode,
        ),
    )
