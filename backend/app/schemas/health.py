from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """System health and readiness status schema"""

    status: str = Field(default="healthy", description="Application service status")
    version: str = Field(description="Backend application version")
    model_loaded: bool = Field(default=False, description="Whether ML weights are loaded in memory")
    model_architecture: Optional[str] = Field(
        default=None, description="Active temporal recognition model architecture"
    )
    device: str = Field(description="Compute execution device (cpu / cuda)")
    supported_domains: int = Field(default=7, description="Number of supported UI context domains")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of the health check",
    )

