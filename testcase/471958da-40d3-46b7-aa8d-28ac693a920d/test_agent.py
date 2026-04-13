
import pytest
from unittest.mock import patch, MagicMock, call

@pytest.fixture
def sample_user_input():
    """
    Fixture providing a sample user input containing PII (email, phone, SSN).
    """
    return (
        "Hello, my email is john.doe@example.com, my phone is 555-123-4567, "
        "and my SSN is 123-45-6789."
    )

@pytest.fixture
def masked_user_input():
    """
    Fixture providing the expected masked user input.
    """
    return (
        "Hello, my email is [REDACTED_EMAIL], my phone is [REDACTED_PHONE], "
        "and my SSN is [REDACTED_SSN]."
    )

@pytest.fixture
def audit_logger_mock():
    """
    Fixture providing a MagicMock for AuditLogger.
    """
    return MagicMock()

@pytest.fixture
def api_client():
    """
    Fixture for the API client or handler under test.
    Replace with the actual API client or handler as appropriate.
    """
    # This is a placeholder. In real tests, import and instantiate your API handler.
    class DummyAPI:
        def __init__(self, audit_logger):
            self.audit_logger = audit_logger

        def ask(self, user_input):
            # Simulate PII masking logic
            import re
            masked = user_input
            masked = re.sub(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', '[REDACTED_EMAIL]', masked)
            masked = re.sub(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', '[REDACTED_PHONE]', masked)
            masked = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[REDACTED_SSN]', masked)
            self.audit_logger.log_event({"user_input": masked})
            return {"result": "ok"}

    return DummyAPI

@pytest.mark.security
def test_security_pii_masking_in_audit_logs(
    sample_user_input,
    masked_user_input,
    audit_logger_mock,
    api_client
):
    """
    Ensures that sensitive information such as emails, phone numbers, and SSNs in user queries
    are masked before being logged by the AuditLogger.
    """
    # Arrange
    api = api_client(audit_logger=audit_logger_mock)

    # Act
    api.ask(sample_user_input)

    # Assert
    # 1. AuditLogger.log_event is called with masked values
    audit_logger_mock.log_event.assert_called_once()
    logged_args, logged_kwargs = audit_logger_mock.log_event.call_args
    assert "user_input" in logged_args[0]
    logged_input = logged_args[0]["user_input"]

    # 2. No raw email, phone, or SSN appears in logs
    assert "john.doe@example.com" not in logged_input, "Raw email found in audit log"
    assert "555-123-4567" not in logged_input, "Raw phone number found in audit log"
    assert "123-45-6789" not in logged_input, "Raw SSN found in audit log"

    # 3. Masked tokens appear in logs
    assert "[REDACTED_EMAIL]" in logged_input, "Masked email token missing"
    assert "[REDACTED_PHONE]" in logged_input, "Masked phone token missing"
    assert "[REDACTED_SSN]" in logged_input, "Masked SSN token missing"

    # 4. The entire masked string matches expected (optional, for strictness)
    assert logged_input == masked_user_input

    # Error scenario: Simulate masking failure (PII not masked)
    audit_logger_mock.reset_mock()
    class BrokenAPI:
        def __init__(self, audit_logger):
            self.audit_logger = audit_logger
        def ask(self, user_input):
            # Fails to mask
            self.audit_logger.log_event({"user_input": user_input})

    broken_api = BrokenAPI(audit_logger=audit_logger_mock)
    broken_api.ask(sample_user_input)
    audit_logger_mock.log_event.assert_called_once()
    broken_logged_input = audit_logger_mock.log_event.call_args[0][0]["user_input"]
    # Should fail if PII is present
    assert (
        "john.doe@example.com" in broken_logged_input or
        "555-123-4567" in broken_logged_input or
        "123-45-6789" in broken_logged_input
    ), "PII masking failure not detected in error scenario"

