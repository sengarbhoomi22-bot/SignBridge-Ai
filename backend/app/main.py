from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.routes import health


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context for startup and shutdown initialization."""
    # Startup: logging and resource check
    print(f"[{settings.PROJECT_NAME}] Starting service v{settings.VERSION} on {settings.DEVICE}")
    yield
    # Shutdown: clean up resources if needed
    print(f"[{settings.PROJECT_NAME}] Shutting down service")


def create_application() -> FastAPI:
    """Factory creating and configuring the FastAPI application."""
    application = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description="SignBridge AI - Indian Sign Language to Text & Speech Backend API",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS Middleware Configuration
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount Health Check Endpoints (both root /health and /api/v1/health)
    application.include_router(health.router)
    application.include_router(health.router, prefix=settings.API_V1_STR)

    @application.get("/", tags=["Root"])
    async def root():
        """Root status endpoint providing API metadata."""
        return {
            "name": settings.PROJECT_NAME,
            "version": settings.VERSION,
            "docs_url": "/docs",
            "health_url": "/health",
            "api_v1_health": f"{settings.API_V1_STR}/health",
        }

    return application


app = create_application()

