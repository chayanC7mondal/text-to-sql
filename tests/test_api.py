import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "services" in data


def test_readiness_check(client):
    """Test readiness check endpoint."""
    response = client.get("/api/v1/health/ready")
    assert response.status_code == 200
    data = response.json()
    assert "ready" in data


def test_query_endpoint_missing_body(client):
    """Test query endpoint with missing body."""
    response = client.post("/api/v1/query")
    assert response.status_code == 422  # Validation error


def test_query_endpoint_invalid_question(client):
    """Test query endpoint with empty question."""
    response = client.post("/api/v1/query", json={"question": ""})
    assert response.status_code == 422  # Validation error


def test_metrics_endpoint(client):
    """Test metrics endpoint."""
    response = client.get("/api/v1/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_queries" in data
    assert "model_metrics" in data
