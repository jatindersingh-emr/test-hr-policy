
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Assume the FastAPI app is defined in app.main as 'app'
# and AgentResponseModel is in app.models
from app.main import app

@pytest.fixture(scope="module")
def client():
    """
    Fixture to provide a FastAPI test client.
    """
    with TestClient(app) as c:
        yield c

@pytest.fixture
def valid_hr_policy_response():
    """
    Fixture to provide a mock HR policy response from the agent.
    """
    return {
        "success": True,
        "response": "The company's leave policy allows employees 20 days of paid leave per year.",
        "error_type": None
    }

def mock_verify_token(token: str):
    """
    Mock function to simulate token verification.
    Returns True for '<valid_token>', False otherwise.
    """
    return token == "<valid_token>"

def mock_agent_ask(user_input: str, user_context: dict):
    """
    Mock function to simulate the agent's response to a policy question.
    """
    return {
        "success": True,
        "response": "The company's leave policy allows employees 20 days of paid leave per year.",
        "error_type": None
    }

def test_functional_successful_hr_policy_query_via_api_ask(client, valid_hr_policy_response):
    """
    Functional test:
    Validates that a user can successfully submit a well-formed HR policy question to the /api/ask endpoint
    and receive a relevant response.
    """
    # Patch the authentication and agent logic to avoid real dependencies
    with patch("app.api.dependencies.verify_token", side_effect=mock_verify_token) as mock_auth, \
         patch("app.api.routes.agent.ask_agent", side_effect=mock_agent_ask) as mock_ask:

        payload = {
            "user_input": "What is the company's leave policy?",
            "user_context": {"token": "<valid_token>"}
        }
        response = client.post("/api/ask", json=payload)
        assert response.status_code == 200, "Expected HTTP 200 for valid HR policy query"

        data = response.json()
        assert data["success"] is True, "Expected success=True in response"
        assert isinstance(data["response"], str) and data["response"], "Expected non-empty response string"
        assert data["error_type"] is None, "Expected error_type to be None"

        # Ensure mocks were called as expected
        mock_auth.assert_called_once_with("<valid_token>")
        mock_ask.assert_called_once_with(payload["user_input"], payload["user_context"])
