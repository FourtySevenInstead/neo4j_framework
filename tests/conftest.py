"""
Pytest Configuration and Test Running Guide
tests/conftest.py
"""

import os
import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from neo4j import Driver

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ============================================================================
# Environment Fixtures
# ============================================================================


@pytest.fixture
def test_env_file(tmp_path):
    """Create a temporary .env file for testing."""
    env_file = tmp_path / ".env.test"
    env_file.write_text(
        """
NEO4J_URI=neo4j://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=test_password
NEO4J_DATABASE=test_db
NEO4J_ENCRYPTED=true
NEO4J_MAX_CONNECTION_POOL_SIZE=100
NEO4J_CONNECTION_TIMEOUT=30.0
NEO4J_MAX_TRANSACTION_RETRY_TIME=30.0
"""
    )
    return str(env_file)


@pytest.fixture
def test_env_file_invalid(tmp_path):
    """Create a temporary .env file with invalid values for testing validation."""
    env_file = tmp_path / ".env.invalid"
    env_file.write_text(
        """
NEO4J_URI=neo4j://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=test_password
NEO4J_MAX_CONNECTION_POOL_SIZE=9999
NEO4J_CONNECTION_TIMEOUT=-5.0
"""
    )
    return str(env_file)


@pytest.fixture
def clean_env():
    """Clean environment before and after test."""
    # Save original environment
    original_env = os.environ.copy()

    # Clear Neo4j-related variables
    for key in list(os.environ.keys()):
        if key.startswith("NEO4J_") or key.startswith("TEST_"):
            del os.environ[key]

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


# ============================================================================
# Mock Neo4j Driver Fixtures
# ============================================================================


@pytest.fixture
def mock_neo4j_driver():
    """Create a mock Neo4j driver with proper session support."""
    mock_driver = MagicMock(spec=Driver)
    mock_session = MagicMock()
    mock_result = MagicMock()
    mock_record = MagicMock()

    # Setup record data
    mock_record.data.return_value = {"n": {"name": "TestNode", "id": 1}}
    mock_result.__iter__.return_value = [mock_record]

    # Setup session behavior
    mock_session.run.return_value = mock_result
    mock_session.execute_read.return_value = [mock_record]
    mock_session.execute_write.return_value = mock_result
    mock_session.__enter__.return_value = mock_session
    mock_session.__exit__.return_value = None

    # Setup driver behavior
    mock_driver.session.return_value = mock_session
    mock_driver.verify_connectivity.return_value = None
    mock_driver.close.return_value = None

    return mock_driver


@pytest.fixture
def mock_neo4j_connection(mock_neo4j_driver):
    """Create a mock Neo4jConnection instance."""
    from neo4j_framework.db.connection import Neo4jConnection

    conn = Neo4jConnection(
        uri="neo4j://localhost:7687",
        username="neo4j",
        password="password",
        database="test_db",
        encrypted=True,
    )

    # Inject mock driver
    conn._driver = mock_neo4j_driver

    return conn


# ============================================================================
# File Path Fixtures
# ============================================================================


@pytest.fixture
def test_csv_file(tmp_path):
    """Create a temporary CSV file for testing."""
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("name,age\nAlice,30\nBob,25\n")
    return csv_file


@pytest.fixture
def test_cert_files(tmp_path):
    """Create temporary certificate files for mTLS testing."""
    cert_file = tmp_path / "client.crt"
    key_file = tmp_path / "client.key"

    cert_file.write_text(
        "-----BEGIN CERTIFICATE-----\nFAKE_CERT\n-----END CERTIFICATE-----\n"
    )
    key_file.write_text(
        "-----BEGIN PRIVATE KEY-----\nFAKE_KEY\n-----END PRIVATE KEY-----\n"
    )

    return {"cert": str(cert_file), "key": str(key_file)}


# ============================================================================
# Pytest Configuration
# ============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "security: mark test as security-related")
    config.addinivalue_line("markers", "validation: mark test as validation-related")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add markers based on test file name
        if "test_security" in str(item.fspath):
            item.add_marker(pytest.mark.security)
        if "test_validators" in str(item.fspath):
            item.add_marker(pytest.mark.validation)
        if "test_integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)


# ============================================================================
# Logging Setup for Tests
# ============================================================================


@pytest.fixture(autouse=True)
def setup_test_logging():
    """Setup logging for all tests."""
    import logging

    # Configure root logger
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Reduce noise from neo4j driver
    logging.getLogger("neo4j").setLevel(logging.WARNING)

    yield

    # Cleanup handlers
    logging.getLogger().handlers.clear()
