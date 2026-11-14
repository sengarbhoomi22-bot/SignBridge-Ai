import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.core.config import settings

client = TestClient(app)


def test_root_endpoint():
    """Verify root status endpoint returns API metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == settings.PROJECT_NAME
    assert data["version"] == settings.VERSION
    assert "/docs" in data["docs_url"]
    assert "/health" in data["health_url"]


def test_health_check_endpoint():
    """Verify /health returns 200 with complete HealthResponse schema."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == settings.VERSION
    assert data["model_loaded"] is False  # Explicitly false until Phase 2 integration
    assert data["device"] == settings.DEVICE
    assert data["supported_domains"] == 7
    assert "timestamp" in data


def test_api_v1_health_check_endpoint():
    """Verify /api/v1/health returns 200 matching root health check."""
    response = client.get(f"{settings.API_V1_STR}/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == settings.VERSION
    assert data["supported_domains"] == 7


def test_cors_preflight_headers():
    """Verify CORS preflight returns allowed origins for local frontend."""
    headers = {
        "Origin": "http://localhost:3000",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "Content-Type",
    }
    response = client.options("/api/v1/health", headers=headers)
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"
