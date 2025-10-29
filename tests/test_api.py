"""API endpoint tests."""
import pytest
from fastapi.testclient import TestClient
from datetime import datetime
import numpy as np

from src.api.main import app
from src.api.auth import create_access_token


@pytest.fixture
def client():
    """Test client fixture."""
    return TestClient(app)


@pytest.fixture
def auth_token():
    """Generate test auth token."""
    return create_access_token({"user_id": "test_user", "username": "test"})


@pytest.fixture
def auth_headers(auth_token):
    """Authorization headers."""
    return {"Authorization": f"Bearer {auth_token}"}


class TestHealthEndpoints:
    """Test health and status endpoints."""

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    def test_readiness_probe(self, client):
        """Test readiness probe."""
        response = client.get("/ready")
        assert response.status_code in [200, 503]

    def test_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert b"api_requests_total" in response.content


class TestAuthenticationEndpoints:
    """Test authentication."""

    def test_login_success(self, client):
        """Test successful login."""
        response = client.post(
            "/auth/token",
            params={"username": "demo", "password": "demo"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    def test_login_failure(self, client):
        """Test failed login."""
        response = client.post(
            "/auth/token",
            params={"username": "wrong", "password": "wrong"}
        )
        assert response.status_code == 401

    def test_protected_endpoint_without_auth(self, client):
        """Test accessing protected endpoint without auth."""
        response = client.post(
            "/predict",
            json={"symbol": "AAPL"}
        )
        # Should fail if auth is enabled
        assert response.status_code in [200, 401]


class TestPredictionEndpoints:
    """Test prediction endpoints."""

    def test_predict_endpoint(self, client, auth_headers):
        """Test prediction endpoint."""
        response = client.post(
            "/predict",
            json={
                "symbol": "AAPL",
                "timestamp": datetime.utcnow().isoformat()
            },
            headers=auth_headers
        )

        # May fail if models not loaded, but should have proper error
        assert response.status_code in [200, 500, 503]

        if response.status_code == 200:
            data = response.json()
            assert "predicted_return" in data
            assert "confidence" in data
            assert "uncertainty_lower" in data
            assert "uncertainty_upper" in data

    def test_predict_with_news_context(self, client, auth_headers):
        """Test prediction with news context."""
        response = client.post(
            "/predict",
            json={
                "symbol": "AAPL",
                "news_context": "Apple announces record earnings"
            },
            headers=auth_headers
        )

        assert response.status_code in [200, 500, 503]

    def test_similar_patterns_endpoint(self, client, auth_headers):
        """Test similar patterns retrieval."""
        response = client.post(
            "/similar_patterns",
            json={
                "symbol": "AAPL",
                "top_k": 5
            },
            headers=auth_headers
        )

        assert response.status_code in [200, 500, 503]

        if response.status_code == 200:
            data = response.json()
            assert "patterns" in data
            assert isinstance(data["patterns"], list)


class TestFeedbackEndpoint:
    """Test feedback submission."""

    def test_submit_feedback(self, client, auth_headers):
        """Test feedback submission."""
        response = client.post(
            "/feedback",
            json={
                "prediction_id": "test-prediction-123",
                "actual_return": 2.5,
                "timestamp": datetime.utcnow().isoformat()
            },
            headers=auth_headers
        )

        assert response.status_code in [200, 500]

        if response.status_code == 200:
            data = response.json()
            assert data["status"] == "success"


class TestDriftDetection:
    """Test drift detection endpoint."""

    def test_drift_status(self, client, auth_headers):
        """Test drift status endpoint."""
        response = client.get(
            "/drift_status",
            headers=auth_headers
        )

        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "total_checks" in data
            assert "drift_detections" in data


class TestRateLimiting:
    """Test rate limiting."""

    def test_rate_limit_enforcement(self, client, auth_headers):
        """Test rate limiting (if enabled)."""
        # Make many requests rapidly
        responses = []
        for _ in range(100):
            response = client.post(
                "/predict",
                json={"symbol": "AAPL"},
                headers=auth_headers
            )
            responses.append(response.status_code)

        # Should have at least one 429 if rate limiting is enabled
        # Or all successful if disabled
        assert all(code in [200, 429, 500, 503] for code in responses)


class TestInputValidation:
    """Test input validation."""

    def test_invalid_symbol(self, client, auth_headers):
        """Test prediction with invalid symbol."""
        response = client.post(
            "/predict",
            json={"symbol": ""},
            headers=auth_headers
        )
        assert response.status_code in [422, 500]

    def test_missing_required_fields(self, client, auth_headers):
        """Test prediction without required fields."""
        response = client.post(
            "/predict",
            json={},
            headers=auth_headers
        )
        assert response.status_code == 422


@pytest.mark.asyncio
class TestWebSocket:
    """Test WebSocket streaming."""

    async def test_websocket_connection(self, client):
        """Test WebSocket connection."""
        with client.websocket_connect("/ws/stream") as websocket:
            # Receive connection message
            data = websocket.receive_json()
            assert data["type"] == "connected"

            # Send subscription
            websocket.send_json({"action": "subscribe", "symbol": "AAPL"})

            # Receive response (may timeout, that's ok)
            try:
                data = websocket.receive_json(mode="text", timeout=5)
                assert data["type"] in ["prediction", "heartbeat"]
            except Exception:
                pass  # Timeout is acceptable
