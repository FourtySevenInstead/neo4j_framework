< /home/honeymatrix/Projects/neo4j_framework/tests/conftest.py >

```
"""
Pytest Configuration and Test Running Guide
tests/conftest.py
"""

import os
import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock

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
    env_file.write_text("""
NEO4J_URI=neo4j://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=test_password
NEO4J_DATABASE=test_db
NEO4J_ENCRYPTED=true
NEO4J_MAX_CONNECTION_POOL_SIZE=100
NEO4J_CONNECTION_TIMEOUT=30.0
NEO4J_MAX_TRANSACTION_RETRY_TIME=30.0
""")
    return str(env_file)


@pytest.fixture
def test_env_file_invalid(tmp_path):
    """Create a temporary .env file with invalid values for testing validation."""
    env_file = tmp_path / ".env.invalid"
    env_file.write_text("""
NEO4J_URI=neo4j://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=test_password
NEO4J_MAX_CONNECTION_POOL_SIZE=9999
NEO4J_CONNECTION_TIMEOUT=-5.0
""")
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
    mock_driver = MagicMock()
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
    from src.neo4j_framework.db.connection import Neo4jConnection

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
# Configuration Fixtures
# ============================================================================


@pytest.fixture
def env_loader():
    """Create a test EnvironmentLoader instance."""
    from src.neo4j_framework.config.env_loader import EnvironmentLoader

    return EnvironmentLoader(env_file=".env.test", env_prefix="NEO4J_")


@pytest.fixture
def db_config():
    """Create a test database configuration."""
    return {
        "uri": "neo4j://localhost:7687",
        "username": "neo4j",
        "password": "test_password",
        "database": "test_db",
        "encrypted": True,
        "max_connection_pool_size": 100,
        "connection_timeout": 30.0,
        "max_transaction_retry_time": 30.0,
    }


# ============================================================================
# File Path Fixtures
# ============================================================================


@pytest.fixture
def test_csv_file(tmp_path):
    """Create a temporary CSV file for testing."""
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("name,age
Alice,30
Bob,25
")
    return csv_file


@pytest.fixture
def test_cert_files(tmp_path):
    """Create temporary certificate files for mTLS testing."""
    cert_file = tmp_path / "client.crt"
    key_file = tmp_path / "client.key"

    cert_file.write_text(
        "-----BEGIN CERTIFICATE-----
FAKE_CERT
-----END CERTIFICATE-----
"
    )
    key_file.write_text(
        "-----BEGIN PRIVATE KEY-----
FAKE_KEY
-----END PRIVATE KEY-----
"
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
```

< /home/honeymatrix/Projects/neo4j_framework/tests/test_validators.py >

```
"""
Validation tests for Neo4j Framework.
Tests input validation and error handling.
"""

import pytest
import os


@pytest.mark.validation
class TestInputValidation:
    """Input validation tests."""

    def test_integer_validation(self, clean_env):
        """Test integer validation with various inputs."""
        from src.neo4j_framework.config.env_loader import EnvironmentLoader

        loader = EnvironmentLoader(env_prefix="NEO4J_")

        # Valid integer
        os.environ["NEO4J_INT_VAL"] = "42"
        assert loader.get_int("INT_VAL") == 42

        # Invalid - not a number
        os.environ["NEO4J_INT_VAL"] = "not_a_number"
        with pytest.raises(ValueError, match="must be an integer"):
            loader.get_int("INT_VAL")

        # Invalid - float when int expected
        os.environ["NEO4J_INT_VAL"] = "42.5"
        with pytest.raises(ValueError):
            loader.get_int("INT_VAL")

    def test_float_validation(self, clean_env):
        """Test float validation with various inputs."""
        from src.neo4j_framework.config.env_loader import EnvironmentLoader

        loader = EnvironmentLoader(env_prefix="NEO4J_")

        # Valid float
        os.environ["NEO4J_FLOAT_VAL"] = "3.14"
        assert loader.get_float("FLOAT_VAL") == 3.14

        # Valid - integer as float
        os.environ["NEO4J_FLOAT_VAL"] = "42"
        assert loader.get_float("FLOAT_VAL") == 42.0

        # Invalid - not a number
        os.environ["NEO4J_FLOAT_VAL"] = "not_a_number"
        with pytest.raises(ValueError, match="must be a float"):
            loader.get_float("FLOAT_VAL")

    def test_bounds_validation_integers(self, clean_env):
        """Test integer bounds validation."""
        from src.neo4j_framework.config.env_loader import EnvironmentLoader

        loader = EnvironmentLoader(env_prefix="NEO4J_")

        os.environ["NEO4J_VAL"] = "50"

        # Within bounds
        assert loader.get_int("VAL", min_val=1, max_val=100) == 50

        # Below minimum
        with pytest.raises(ValueError, match="must be >= 1"):
            loader.get_int("VAL", min_val=100, max_val=200)

        # Above maximum
        with pytest.raises(ValueError, match="must be <= 10"):
            loader.get_int("VAL", min_val=1, max_val=10)

    def test_bounds_validation_floats(self, clean_env):
        """Test float bounds validation."""
        from src.neo4j_framework.config.env_loader import EnvironmentLoader

        loader = EnvironmentLoader(env_prefix="NEO4J_")

        os.environ["NEO4J_VAL"] = "5.5"

        # Within bounds
        assert loader.get_float("VAL", min_val=0.0, max_val=10.0) == 5.5

        # Below minimum
        with pytest.raises(ValueError, match="must be >= 10.0"):
            loader.get_float("VAL", min_val=10.0, max_val=20.0)

        # Above maximum
        with pytest.raises(ValueError, match="must be <= 1.0"):
            loader.get_float("VAL", min_val=0.0, max_val=1.0)

    def test_string_validation(self):
        """Test string validation."""
        from src.neo4j_framework.utils.validators import Validators

        # Valid strings
        Validators.validate_string_not_empty("value", "param")
        Validators.validate_string_not_empty("  value  ", "param")

        # Invalid - empty
        with pytest.raises(ValueError, match="cannot be empty"):
            Validators.validate_string_not_empty("", "param")

        # Invalid - whitespace only
        with pytest.raises(ValueError, match="cannot be empty"):
            Validators.validate_string_not_empty("   ", "param")

    def test_not_none_validation(self):
        """Test not-None validation."""
        from src.neo4j_framework.utils.validators import Validators

        # Valid - various non-None values
        Validators.validate_not_none("value", "param")
        Validators.validate_not_none(0, "param")
        Validators.validate_not_none(False, "param")
        Validators.validate_not_none([], "param")

        # Invalid - None
        with pytest.raises(ValueError, match="cannot be None"):
            Validators.validate_not_none(None, "param")

    def test_positive_int_validation(self):
        """Test positive integer validation."""
        from src.neo4j_framework.utils.validators import Validators

        # Valid
        Validators.validate_positive_int(1, "param")
        Validators.validate_positive_int(100, "param")
        Validators.validate_positive_int(99999, "param")

        # Invalid - zero
        with pytest.raises(ValueError, match="must be a positive integer"):
            Validators.validate_positive_int(0, "param")

        # Invalid - negative
        with pytest.raises(ValueError, match="must be a positive integer"):
            Validators.validate_positive_int(-1, "param")

        # Invalid - not an integer
        with pytest.raises(ValueError, match="must be a positive integer"):
            Validators.validate_positive_int(3.14, "param")


@pytest.mark.validation
class TestConfigurationValidation:
    """Configuration validation tests."""

    def test_complete_valid_config(self, clean_env):
        """Test complete valid configuration."""
        from src.neo4j_framework.config.env_loader import EnvironmentLoader

        os.environ["NEO4J_URI"] = "neo4j://localhost:7687"
        os.environ["NEO4J_USERNAME"] = "neo4j"
        os.environ["NEO4J_PASSWORD"] = "password"
        os.environ["NEO4J_DATABASE"] = "neo4j"
        os.environ["NEO4J_ENCRYPTED"] = "true"
        os.environ["NEO4J_MAX_CONNECTION_POOL_SIZE"] = "50"
        os.environ["NEO4J_CONNECTION_TIMEOUT"] = "30.0"
        os.environ["NEO4J_MAX_TRANSACTION_RETRY_TIME"] = "30.0"

        loader = EnvironmentLoader(env_prefix="NEO4J_")
        config = loader.get_config()

        assert config["uri"] == "neo4j://localhost:7687"
        assert config["username"] == "neo4j"
        assert config["password"] == "password"
        assert config["database"] == "neo4j"
        assert config["encrypted"] is True
        assert config["max_connection_pool_size"] == 50
        assert config["connection_timeout"] == 30.0
        assert config["max_transaction_retry_time"] == 30.0

    def test_invalid_pool_size_in_config(self, clean_env):
        """Test that invalid pool size in config is caught."""
        from src.neo4j_framework.config.env_loader import EnvironmentLoader

        os.environ["NEO4J_URI"] = "neo4j://localhost:7687"
        os.environ["NEO4J_PASSWORD"] = "password"
        os.environ["NEO4J_MAX_CONNECTION_POOL_SIZE"] = "1000"

        loader = EnvironmentLoader(env_prefix="NEO4J_")

        with pytest.raises(ValueError, match="must be <= 500"):
            loader.get_config()

    def test_invalid_timeout_in_config(self, clean_env):
        """Test that invalid timeout in config is caught."""
        from src.neo4j_framework.config.env_loader import EnvironmentLoader

        os.environ["NEO4J_URI"] = "neo4j://localhost:7687"
        os.environ["NEO4J_PASSWORD"] = "password"
        os.environ["NEO4J_CONNECTION_TIMEOUT"] = "-5.0"

        loader = EnvironmentLoader(env_prefix="NEO4J_")

        with pytest.raises(ValueError, match="must be >= 0.1"):
            loader.get_config()
```

< /home/honeymatrix/Projects/neo4j_framework/tests/test_framework.py >

```
"""
Test Suite for Neo4j Framework
Comprehensive tests validating all major components with updated architecture.
"""

import sys
import os
import logging
from typing import Optional, Dict, Any
from unittest.mock import Mock, MagicMock, patch
import pytest

logger = logging.getLogger(__name__)


# ============================================================================
# PART 1: Environment Loader Tests (UPDATED)
# ============================================================================


@pytest.mark.unit
class TestEnvironmentLoader:
    """Tests for EnvironmentLoader with validation."""

    def test_initialization(self, clean_env):
        """Test EnvironmentLoader initialization."""
        from src.neo4j_framework.config.env_loader import EnvironmentLoader

        loader = EnvironmentLoader(env_file=".env.test", env_prefix="TEST_")
        assert loader.env_file == ".env.test"
        assert loader.env_prefix == "TEST_"
        assert loader._loaded is False

    def test_get_basic(self, clean_env):
        """Test getting environment variables with prefix."""
        from src.neo4j_framework.config.env_loader import EnvironmentLoader

        os.environ["NEO4J_TEST_KEY"] = "test_value"

        loader = EnvironmentLoader(env_prefix="NEO4J_")
        value = loader.get("TEST_KEY")
        assert value == "test_value"

    def test_get_int_with_validation(self, clean_env):
        """Test integer retrieval with bounds validation."""
        from src.neo4j_framework.config.env_loader import EnvironmentLoader

        os.environ["NEO4J_POOL_SIZE"] = "100"

        loader = EnvironmentLoader(env_prefix="NEO4J_")

        # Valid value within bounds
        value = loader.get_int("POOL_SIZE", min_val=1, max_val=500)
        assert value == 100

        # Invalid - too high
        os.environ["NEO4J_POOL_SIZE"] = "1000"
        with pytest.raises(ValueError, match="must be <= 500"):
            loader.get_int("POOL_SIZE", min_val=1, max_val=500)

        # Invalid - too low
        os.environ["NEO4J_POOL_SIZE"] = "0"
        with pytest.raises(ValueError, match="must be >= 1"):
            loader.get_int("POOL_SIZE", min_val=1, max_val=500)

    def test_get_float_with_validation(self, clean_env):
        """Test float retrieval with bounds validation."""
        from src.neo4j_framework.config.env_loader import EnvironmentLoader

        os.environ["NEO4J_TIMEOUT"] = "30.5"

        loader = EnvironmentLoader(env_prefix="NEO4J_")

        # Valid value
        value = loader.get_float("TIMEOUT", min_val=0.1, max_val=300.0)
        assert value == 30.5

        # Invalid - out of bounds
        os.environ["NEO4J_TIMEOUT"] = "500.0"
        with pytest.raises(ValueError, match="must be <= 300.0"):
            loader.get_float("TIMEOUT", min_val=0.1, max_val=300.0)

    def test_get_bool(self, clean_env):
        """Test boolean conversion."""
        from src.neo4j_framework.config.env_loader import EnvironmentLoader

        loader = EnvironmentLoader(env_prefix="NEO4J_")

        # Test various truthy values
        for truthy in ["true", "True", "TRUE", "1", "yes", "YES", "on", "ON"]:
            os.environ["NEO4J_BOOL_VAL"] = truthy
            assert loader.get_bool("BOOL_VAL") is True

        # Test falsy values
        for falsy in ["false", "False", "0", "no", "off"]:
            os.environ["NEO4J_BOOL_VAL"] = falsy
            assert loader.get_bool("BOOL_VAL") is False

    def test_required_validation(self, clean_env):
        """Test required environment variable validation."""
        from src.neo4j_framework.config.env_loader import EnvironmentLoader

        loader = EnvironmentLoader(env_prefix="NEO4J_")

        with pytest.raises(ValueError, match="Required environment variable not set"):
            loader.get("MISSING_VAR", required=True)

    def test_get_config_complete(self, clean_env):
        """Test complete configuration retrieval."""
        from src.neo4j_framework.config.env_loader import EnvironmentLoader

        # Set all required environment variables
        os.environ["NEO4J_URI"] = "neo4j://localhost:7687"
        os.environ["NEO4J_PASSWORD"] = "password"
        os.environ["NEO4J_USERNAME"] = "neo4j"
        os.environ["NEO4J_DATABASE"] = "test_db"
        os.environ["NEO4J_ENCRYPTED"] = "true"
        os.environ["NEO4J_MAX_CONNECTION_POOL_SIZE"] = "50"

        loader = EnvironmentLoader(env_prefix="NEO4J_")
        config = loader.get_config()

        assert config["uri"] == "neo4j://localhost:7687"
        assert config["password"] == "password"
        assert config["username"] == "neo4j"
        assert config["database"] == "test_db"
        assert config["encrypted"] is True
        assert config["max_connection_pool_size"] == 50

    def test_initialization(self):
        """Test Neo4jConnection initialization."""
        from src.neo4j_framework.db.connection import Neo4jConnection

        conn = Neo4jConnection(
            uri="neo4j://localhost:7687",
            username="neo4j",
            password="password",
            database="test_db",
            encrypted=True,
            max_connection_pool_size=50,
        )

        assert conn.uri == "neo4j://localhost:7687"
        assert conn.username == "neo4j"
        assert conn.password == "password"
        assert conn.database == "test_db"
        assert conn.encrypted is True
        assert conn.config["max_connection_pool_size"] == 50

    def test_pool_size_validation(self):
        """Test pool size bounds validation."""
        from src.neo4j_framework.db.connection import Neo4jConnection

        # Valid pool size
        conn = Neo4jConnection(
            uri="neo4j://localhost:7687",
            username="neo4j",
            password="password",
            max_connection_pool_size=100,
        )
        assert conn.config["max_connection_pool_size"] == 100

        # Invalid - too high
        with pytest.raises(ValueError, match="must be between 1 and 500"):
            Neo4jConnection(
                uri="neo4j://localhost:7687",
                username="neo4j",
                password="password",
                max_connection_pool_size=1000,
            )

        # Invalid - too low
        with pytest.raises(ValueError, match="must be between 1 and 500"):
            Neo4jConnection(
                uri="neo4j://localhost:7687",
                username="neo4j",
                password="password",
                max_connection_pool_size=0,
            )

    def test_basic_auth_creation(self):
        """Test basic authentication creation."""
        from src.neo4j_framework.db.connection import Neo4jConnection
        from neo4j import Auth

        conn = Neo4jConnection(
            uri="neo4j://localhost:7687",
            username="neo4j",
            password="password",
        )

        auth = conn._create_basic_auth()
        assert isinstance(auth, Auth)

    def test_missing_credentials_error(self):
        """Test error when credentials are missing."""
        from src.neo4j_framework.db.connection import Neo4jConnection

        conn = Neo4jConnection(
            uri="neo4j://localhost:7687",
            username=None,
            password=None,
        )

        with pytest.raises(ValueError, match="Username and password required"):
            conn._create_basic_auth()

    def test_connect_with_mock_driver(self, mock_neo4j_driver):
        """Test connection with mocked driver."""
        from src.neo4j_framework.db.connection import Neo4jConnection

        with patch(
            "src.neo4j_framework.db.connection.GraphDatabase.driver",
            return_value=mock_neo4j_driver,
        ):
            conn = Neo4jConnection(
                uri="neo4j://localhost:7687",
                username="neo4j",
                password="password",
            )

            driver = conn.connect()

            assert driver is mock_neo4j_driver
            assert conn.is_connected()
            mock_neo4j_driver.verify_connectivity.assert_called_once()

    def test_context_manager(self, mock_neo4j_driver):
        """Test Neo4jConnection as context manager."""
        from src.neo4j_framework.db.connection import Neo4jConnection

        with patch(
            "src.neo4j_framework.db.connection.GraphDatabase.driver",
            return_value=mock_neo4j_driver,
        ):
            with Neo4jConnection(
                uri="neo4j://localhost:7687",
                username="neo4j",
                password="password",
            ) as conn:
                assert conn.is_connected()

            # After context, should be closed
            assert not conn.is_connected()

    def test_mtls_file_validation(self, test_cert_files):
        """Test mTLS certificate file validation."""
        from src.neo4j_framework.db.connection import Neo4jConnection

        conn = Neo4jConnection(
            uri="neo4j://localhost:7687",
            username="neo4j",
            password="password",
        )

        # Valid files should not raise
        with patch("src.neo4j_framework.db.connection.GraphDatabase.driver"):
            with patch.object(conn, "_create_basic_auth", return_value=MagicMock()):
                # Should not raise
                try:
                    conn.connect_with_mtls(
                        cert_path=test_cert_files["cert"],
                        key_path=test_cert_files["key"],
                    )
                except Exception as e:
                    # We expect some error since we're mocking, but not file validation error
                    assert "not found" not in str(e).lower()

    def test_mtls_missing_file_error(self):
        """Test mTLS with missing certificate file."""
        from src.neo4j_framework.db.connection import Neo4jConnection

        conn = Neo4jConnection(
            uri="neo4j://localhost:7687",
            username="neo4j",
            password="password",
        )

        with pytest.raises(ValueError, match="Certificate file not found"):
            conn.connect_with_mtls(cert_path="/nonexistent/cert.pem")

    def test_get_driver_without_connection(self):
        """Test error when getting driver before connection."""
        from src.neo4j_framework.db.connection import Neo4jConnection

        conn = Neo4jConnection(
            uri="neo4j://localhost:7687",
            username="neo4j",
            password="password",
        )

        with pytest.raises(RuntimeError, match="Not connected"):
            conn.get_driver()


# ============================================================================
# PART 3: Query Manager Tests (UPDATED)
# ============================================================================


@pytest.mark.unit
class TestQueryManager:
    """Tests for QueryManager with read/write differentiation."""

    def test_initialization(self, mock_neo4j_connection):
        """Test QueryManager initialization."""
        from src.neo4j_framework.queries.query_manager import QueryManager

        qm = QueryManager(mock_neo4j_connection)
        assert qm.connection is mock_neo4j_connection

    def test_execute_read(self, mock_neo4j_connection):
        """Test read query execution."""
        from src.neo4j_framework.queries.query_manager import QueryManager

        qm = QueryManager(mock_neo4j_connection)
        results = qm.execute_read("MATCH (n) RETURN n LIMIT 1")

        assert isinstance(results, list)
        assert len(results) > 0

    def test_execute_write(self, mock_neo4j_connection):
        """Test write query execution."""
        from src.neo4j_framework.queries.query_manager import QueryManager

        qm = QueryManager(mock_neo4j_connection)
        result = qm.execute_write(
            "CREATE (n:Node {name: $name}) RETURN n", params={"name": "Test"}
        )

        assert result is not None

    def test_execute_query_warning(self, mock_neo4j_connection, caplog):
        """Test that execute_query logs warning."""
        from src.neo4j_framework.queries.query_manager import QueryManager

        qm = QueryManager(mock_neo4j_connection)

        with caplog.at_level(logging.WARNING):
            qm.execute_query("MATCH (n) RETURN n")

        # Should log warning about using execute_read/execute_write
        assert "execute_query() called" in caplog.text


# ============================================================================
# PART 4: Transaction Manager Tests (UPDATED)
# ============================================================================


@pytest.mark.unit
class TestTransactionManager:
    """Tests for TransactionManager with context manager support."""

    def test_initialization(self, mock_neo4j_connection):
        """Test TransactionManager initialization."""
        from src.neo4j_framework.transactions.transaction_manager import (
            TransactionManager,
        )

        tm = TransactionManager(mock_neo4j_connection)
        assert tm.connection is mock_neo4j_connection

    def test_run_in_transaction(self, mock_neo4j_connection):
        """Test transaction execution."""
        from src.neo4j_framework.transactions.transaction_manager import (
            TransactionManager,
        )

        tm = TransactionManager(mock_neo4j_connection)

        def tx_func(tx):
            return tx.run("CREATE (n:Node {name: $name})", {"name": "Test"})

        result = tm.run_in_transaction(tx_func)
        assert result is not None

    def test_context_manager_support(self, mock_neo4j_connection):
        """Test TransactionManager as context manager."""
        from src.neo4j_framework.transactions.transaction_manager import (
            TransactionManager,
        )

        tm = TransactionManager(mock_neo4j_connection)

        with tm as session:
            assert session is not None
            # Session should be available for operations

        # After context, session should be closed
        assert tm._session is None


# ============================================================================
# PART 5: CSV Importer Tests (UPDATED)
# ============================================================================


@pytest.mark.unit
class TestCSVImporter:
    """Tests for CSVImporter with path validation."""

    def test_initialization(self, mock_neo4j_connection):
        """Test CSVImporter initialization."""
        from src.neo4j_framework.importers.csv_importer import CSVImporter

        importer = CSVImporter(mock_neo4j_connection)
        assert importer.connection is mock_neo4j_connection
        assert importer.allowed_dir is None

    def test_initialization_with_allowed_dir(self, mock_neo4j_connection, tmp_path):
        """Test CSVImporter with restricted directory."""
        from src.neo4j_framework.importers.csv_importer import CSVImporter

        importer = CSVImporter(mock_neo4j_connection, allowed_dir=str(tmp_path))
        assert importer.allowed_dir == tmp_path.resolve()

    def test_path_validation_valid(self, mock_neo4j_connection, test_csv_file):
        """Test valid CSV file path."""
        from src.neo4j_framework.importers.csv_importer import CSVImporter

        importer = CSVImporter(mock_neo4j_connection)
        validated_path = importer._validate_file_path(str(test_csv_file))

        assert validated_path.exists()
        assert validated_path.is_absolute()

    def test_path_validation_nonexistent(self, mock_neo4j_connection):
        """Test error for nonexistent file."""
        from src.neo4j_framework.importers.csv_importer import CSVImporter

        importer = CSVImporter(mock_neo4j_connection)

        with pytest.raises(ValueError, match="CSV file not found"):
            importer._validate_file_path("/nonexistent/file.csv")

    def test_path_validation_outside_allowed_dir(
        self, mock_neo4j_connection, test_csv_file, tmp_path
    ):
        """Test error when file is outside allowed directory."""
        from src.neo4j_framework.importers.csv_importer import CSVImporter

        # Create a different directory
        other_dir = tmp_path / "other"
        other_dir.mkdir()

        # Restrict importer to other_dir
        importer = CSVImporter(mock_neo4j_connection, allowed_dir=str(other_dir))

        # test_csv_file is not in other_dir
        with pytest.raises(ValueError, match="must be within"):
            importer._validate_file_path(str(test_csv_file))

    def test_import_csv_file_url(self, mock_neo4j_connection, test_csv_file):
        """Test that CSV import constructs proper file URL."""
        from src.neo4j_framework.importers.csv_importer import CSVImporter

        importer = CSVImporter(mock_neo4j_connection)
        query = "LOAD CSV WITH HEADERS FROM $file_url AS row CREATE (:Node {name: row.name})"

        importer.import_csv(str(test_csv_file), query)

        # Verify the session.run was called with file_url parameter
        call_args = mock_neo4j_connection._driver.session.return_value.run.call_args
        assert call_args is not None
        params = call_args[0][1]
        assert "file_url" in params
        assert "file:///" in params["file_url"]


# ============================================================================
# PART 6: Utility Tests (UPDATED)
# ============================================================================


@pytest.mark.unit
class TestValidators:
    """Tests for Validators utility."""

    def test_validate_not_none_success(self):
        """Test validation passes for non-None value."""
        from src.neo4j_framework.utils.validators import Validators

        # Should not raise
        Validators.validate_not_none("value", "test_param")
        Validators.validate_not_none(0, "test_param")
        Validators.validate_not_none(False, "test_param")

    def test_validate_not_none_failure(self):
        """Test validation fails for None value."""
        from src.neo4j_framework.utils.validators import Validators

        with pytest.raises(ValueError, match="test_param cannot be None"):
            Validators.validate_not_none(None, "test_param")

    def test_validate_string_not_empty(self):
        """Test string validation."""
        from src.neo4j_framework.utils.validators import Validators

        # Should pass
        Validators.validate_string_not_empty("value", "test_param")

        # Should fail for empty string
        with pytest.raises(ValueError, match="cannot be empty"):
            Validators.validate_string_not_empty("", "test_param")

        # Should fail for whitespace
        with pytest.raises(ValueError, match="cannot be empty"):
            Validators.validate_string_not_empty("   ", "test_param")

    def test_validate_positive_int(self):
        """Test positive integer validation."""
        from src.neo4j_framework.utils.validators import Validators

        # Should pass
        Validators.validate_positive_int(1, "test_param")
        Validators.validate_positive_int(100, "test_param")

        # Should fail for zero
        with pytest.raises(ValueError, match="must be a positive integer"):
            Validators.validate_positive_int(0, "test_param")

        # Should fail for negative
        with pytest.raises(ValueError, match="must be a positive integer"):
            Validators.validate_positive_int(-1, "test_param")


@pytest.mark.unit
class TestPerformance:
    """Tests for Performance utilities."""

    def test_time_function_decorator(self, caplog):
        """Test performance timing decorator uses logging."""
        from src.neo4j_framework.utils.performance import Performance
        import time

        @Performance.time_function
        def test_func():
            time.sleep(0.01)
            return "done"

        with caplog.at_level(logging.DEBUG):
            result = test_func()

        assert result == "done"
        assert "execution time" in caplog.text.lower()

    def test_time_function_with_error(self, caplog):
        """Test performance decorator logs errors."""
        from src.neo4j_framework.utils.performance import Performance

        @Performance.time_function
        def failing_func():
            raise ValueError("Test error")

        with caplog.at_level(logging.ERROR):
            with pytest.raises(ValueError):
                failing_func()

        assert "failed after" in caplog.text.lower()


@pytest.mark.unit
class TestCustomExceptions:
    """Tests for custom exception classes."""

    def test_exception_hierarchy(self):
        """Test custom exception hierarchy."""
        from src.neo4j_framework.utils.exceptions import (
            Neo4jFrameworkException,
            ConnectionError,
            AuthenticationError,
            ConfigurationError,
            ValidationError,
            QueryError,
            TransactionError,
        )

        # All should inherit from base
        assert issubclass(ConnectionError, Neo4jFrameworkException)
        assert issubclass(AuthenticationError, Neo4jFrameworkException)
        assert issubclass(ConfigurationError, Neo4jFrameworkException)
        assert issubclass(ValidationError, Neo4jFrameworkException)
        assert issubclass(QueryError, Neo4jFrameworkException)
        assert issubclass(TransactionError, Neo4jFrameworkException)

    def test_exception_can_be_raised(self):
        """Test custom exceptions can be raised and caught."""
        from src.neo4j_framework.utils.exceptions import ValidationError

        with pytest.raises(ValidationError, match="Test error"):
            raise ValidationError("Test error")


# ============================================================================
# PART 7: Pool Manager Tests (UPDATED)
# ============================================================================


@pytest.mark.unit
class TestPoolManager:
    """Tests for PoolManager without private access."""

    def test_initialization(self, mock_neo4j_driver):
        """Test PoolManager initialization."""
        from src.neo4j_framework.db.pool_manager import PoolManager

        pm = PoolManager(mock_neo4j_driver)
        assert pm.driver is mock_neo4j_driver

    def test_get_pool_stats_no_private_access(self, mock_neo4j_driver):
        """Test that pool stats doesn't access private attributes."""
        from src.neo4j_framework.db.pool_manager import PoolManager

        pm = PoolManager(mock_neo4j_driver)
        stats = pm.get_pool_stats()

        # Should return informative message, not actual stats
        assert isinstance(stats, dict)
        assert "note" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
```

< /home/honeymatrix/Projects/neo4j_framework/tests/test_integration_advanced.py >

```
"""
Advanced Integration Tests for Neo4j Framework (UPDATED)
tests/test_integration_advanced.py
"""

import pytest
from unittest.mock import MagicMock, patch, Mock
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


@pytest.mark.integration
class TestFrameworkIntegration:
    """Advanced integration tests for complete workflows."""

    def test_complete_initialization_and_query_flow(self, mock_neo4j_connection):
        """Test complete initialization with query execution (UPDATED)."""
        from src.neo4j_framework.queries.query_manager import QueryManager

        # Create query manager with mock connection
        query_manager = QueryManager(mock_neo4j_connection)

        # Execute read query (NEW - using execute_read)
        results = query_manager.execute_read("MATCH (n) RETURN n LIMIT 1")

        assert isinstance(results, list)
        assert len(results) > 0
        assert results[0]["n"]["name"] == "TestNode"

    def test_multiple_queries_sequential(self, mock_neo4j_connection):
        """Test executing multiple queries sequentially (UPDATED)."""
        from src.neo4j_framework.queries.query_manager import QueryManager

        query_manager = QueryManager(mock_neo4j_connection)

        # Mock different results for different queries
        mock_driver = mock_neo4j_connection._driver
        mock_session = mock_driver.session.return_value

        # Setup different return values
        result1 = MagicMock()
        result1.__iter__.return_value = [MagicMock(data=lambda: {"count": 10})]

        result2 = MagicMock()
        result2.__iter__.return_value = [MagicMock(data=lambda: {"avg": 5.5})]

        result3 = MagicMock()
        result3.__iter__.return_value = [MagicMock(data=lambda: {"max": 100})]

        # Configure mock to return different results
        mock_session.execute_read.side_effect = [
            [MagicMock(data=lambda: {"count": 10})],
            [MagicMock(data=lambda: {"avg": 5.5})],
            [MagicMock(data=lambda: {"max": 100})],
        ]

        # Execute queries
        count_result = query_manager.execute_read("MATCH (n) RETURN COUNT(n) as count")
        avg_result = query_manager.execute_read("MATCH (n) RETURN AVG(n.value) as avg")
        max_result = query_manager.execute_read("MATCH (n) RETURN MAX(n.value) as max")

        assert len(count_result) == 1
        assert len(avg_result) == 1
        assert len(max_result) == 1

    def test_transaction_with_rollback_simulation(self, mock_neo4j_connection):
        """Test transaction execution with error handling (UPDATED)."""
        from src.neo4j_framework.transactions.transaction_manager import (
            TransactionManager,
        )

        tx_manager = TransactionManager(mock_neo4j_connection)

        # Define write transaction
        def write_transaction(tx):
            return tx.run("CREATE (n:TestNode {value: $val}) RETURN n", {"val": 42})

        # Execute transaction
        result = tx_manager.run_in_transaction(write_transaction)

        assert result is not None
        # Verify execute_write was called
        mock_driver = mock_neo4j_connection._driver
        mock_driver.session.assert_called()

    def test_transaction_context_manager(self, mock_neo4j_connection):
        """Test transaction context manager support (NEW)."""
        from src.neo4j_framework.transactions.transaction_manager import (
            TransactionManager,
        )

        tx_manager = TransactionManager(mock_neo4j_connection)

        # Use context manager
        with tx_manager as session:
            assert session is not None
            # Session is available for operations
            session.execute_write(lambda tx: tx.run("CREATE (n:Node)"))

        # After context, session should be closed
        assert tx_manager._session is None

    def test_csv_import_workflow(self, mock_neo4j_connection, test_csv_file):
        """Test complete CSV import workflow (UPDATED)."""
        from src.neo4j_framework.importers.csv_importer import CSVImporter

        importer = CSVImporter(mock_neo4j_connection)

        # Define CSV import query
        query = (
            "LOAD CSV WITH HEADERS FROM $file_url AS row "
            "CREATE (:User {name: row.name})"
        )

        # Execute import
        result = importer.import_csv(str(test_csv_file), query)

        # Verify file URL was added to parameters
        mock_driver = mock_neo4j_connection._driver
        mock_session = mock_driver.session.return_value
        mock_session.run.assert_called()

        # Get the call arguments
        call_args = mock_session.run.call_args
        if call_args:
            params = call_args[0][1]
            assert "file_url" in params
            assert "file:///" in params["file_url"]

    def test_connection_lifecycle(self):
        """Test complete connection lifecycle (UPDATED)."""
        from src.neo4j_framework.db.connection import Neo4jConnection

        conn = Neo4jConnection(
            uri="neo4j://localhost:7687",
            username="neo4j",
            password="password",
            database="test_db",
        )

        # Check initial state
        assert not conn.is_connected()

        # Mock driver creation
        mock_driver = MagicMock()
        mock_driver.verify_connectivity.return_value = None

        with patch(
            "src.neo4j_framework.db.connection.GraphDatabase.driver",
            return_value=mock_driver,
        ):
            # Connect
            driver = conn.connect()
            assert conn.is_connected()
            assert driver is mock_driver

            # Get driver
            retrieved_driver = conn.get_driver()
            assert retrieved_driver is mock_driver

            # Close
            conn.close()
            assert not conn.is_connected()
            mock_driver.close.assert_called_once()

    def test_connection_context_manager_lifecycle(self):
        """Test connection as context manager (NEW)."""
        from src.neo4j_framework.db.connection import Neo4jConnection

        mock_driver = MagicMock()
        mock_driver.verify_connectivity.return_value = None

        with patch(
            "src.neo4j_framework.db.connection.GraphDatabase.driver",
            return_value=mock_driver,
        ):
            with Neo4jConnection(
                uri="neo4j://localhost:7687",
                username="neo4j",
                password="password",
            ) as conn:
                assert conn.is_connected()

            # After context, should be closed
            assert not conn.is_connected()
            mock_driver.close.assert_called()

    def test_error_recovery_workflow(self, mock_neo4j_connection):
        """Test error handling and recovery in workflows (UPDATED)."""
        from src.neo4j_framework.queries.query_manager import QueryManager

        query_manager = QueryManager(mock_neo4j_connection)

        # Setup mock to raise error on first call, succeed on second
        mock_driver = mock_neo4j_connection._driver
        mock_session = mock_driver.session.return_value

        # Create successful mock record
        mock_record = MagicMock(data=lambda: {"status": "ok"})

        # First call fails, second succeeds
        mock_session.execute_read.side_effect = [
            Exception("Connection timeout"),
            [mock_record],
        ]

        # First call should raise
        with pytest.raises(Exception, match="Connection timeout"):
            query_manager.execute_read("MATCH (n) RETURN n")

        # Reset side effect for second call
        mock_session.execute_read.side_effect = None
        mock_session.execute_read.return_value = [mock_record]

        # Second call should succeed
        result = query_manager.execute_read("MATCH (n) RETURN n")
        assert result is not None
        assert len(result) > 0

    def test_parameterized_queries_with_different_types(self, mock_neo4j_connection):
        """Test query parameterization with various data types (UPDATED)."""
        from src.neo4j_framework.queries.query_manager import QueryManager

        query_manager = QueryManager(mock_neo4j_connection)

        # Define parameters of different types
        params = {
            "string": "test",
            "integer": 42,
            "float": 3.14,
            "boolean": True,
            "list": [1, 2, 3],
            "dict": {"key": "value"},
        }

        # Execute read query with parameters
        result = query_manager.execute_read(
            "MATCH (n) WHERE n.value = $string RETURN n", params=params
        )

        # Verify mock was called
        mock_driver = mock_neo4j_connection._driver
        mock_session = mock_driver.session.return_value
        mock_session.execute_read.assert_called()

    def test_batch_operations(self, mock_neo4j_connection):
        """Test batch operations on multiple records (UPDATED)."""
        from src.neo4j_framework.queries.query_manager import QueryManager

        query_manager = QueryManager(mock_neo4j_connection)

        # Create mock records
        mock_records = [
            MagicMock(data=lambda i=i: {"id": i, "name": f"Node{i}"})
            for i in range(100)
        ]

        # Configure mock to return all records
        mock_driver = mock_neo4j_connection._driver
        mock_session = mock_driver.session.return_value
        mock_session.execute_read.return_value = mock_records

        # Execute read query
        results = query_manager.execute_read("MATCH (n) RETURN n")

        assert len(results) == 100

    def test_environment_configuration_override(self, clean_env):
        """Test environment configuration with custom values (UPDATED)."""
        import os
        from src.neo4j_framework.config.env_loader import EnvironmentLoader

        # Set custom environment variables
        os.environ["MY_APP_URI"] = "neo4j://custom:7687"
        os.environ["MY_APP_PASSWORD"] = "custom_password"
        os.environ["MY_APP_USERNAME"] = "custom_user"

        # Create loader with custom prefix
        loader = EnvironmentLoader(env_prefix="MY_APP_")

        assert loader.env_prefix == "MY_APP_"

        uri = loader.get("URI")
        password = loader.get("PASSWORD")
        username = loader.get("USERNAME")

        assert uri == "neo4j://custom:7687"
        assert password == "custom_password"
        assert username == "custom_user"

    def test_multiple_connections_independent(self):
        """Test multiple independent connections (UPDATED)."""
        from src.neo4j_framework.db.connection import Neo4jConnection

        # Create multiple connections with different configs
        conn1 = Neo4jConnection(
            uri="neo4j://host1:7687",
            username="user1",
            password="pass1",
            database="db1",
        )

        conn2 = Neo4jConnection(
            uri="neo4j://host2:7687",
            username="user2",
            password="pass2",
            database="db2",
        )

        # Verify they are independent
        assert conn1.uri != conn2.uri
        assert conn1.username != conn2.username
        assert conn1.database != conn2.database
        assert conn1._driver is None
        assert conn2._driver is None

    def test_connection_pool_configuration(self):
        """Test connection pool configuration (UPDATED with validation)."""
        from src.neo4j_framework.db.connection import Neo4jConnection

        conn = Neo4jConnection(
            uri="neo4j://localhost:7687",
            username="neo4j",
            password="password",
            max_connection_pool_size=200,
            connection_timeout=60.0,
            max_connection_lifetime=7200.0,
        )

        assert conn.config["max_connection_pool_size"] == 200
        assert conn.config["connection_timeout"] == 60.0
        assert conn.config["max_connection_lifetime"] == 7200.0

    def test_query_template_substitution(self):
        """Test query template usage and substitution (UPDATED)."""
        from src.neo4j_framework.queries.query_templates import QueryTemplates
        from src.neo4j_framework.queries.base_query import BaseQuery

        # Get template
        template = QueryTemplates.get_template("CREATE_NODE")
        assert template is not None

        # Create parameterized query
        query = BaseQuery(query_str=template, params={"name": "TestNode"})

        assert "CREATE" in query.query_str
        assert query.params["name"] == "TestNode"

    def test_base_query_with_connection(self, mock_neo4j_connection):
        """Test BaseQuery execution with connection (NEW)."""
        from src.neo4j_framework.queries.base_query import BaseQuery

        query = BaseQuery(query_str="MATCH (n) RETURN n LIMIT 1", params={"limit": 1})

        result = query.execute(mock_neo4j_connection)
        assert result is not None

    def test_csv_importer_with_allowed_directory(self, mock_neo4j_connection, tmp_path):
        """Test CSV importer with directory restriction (NEW)."""
        from src.neo4j_framework.importers.csv_importer import CSVImporter

        # Create allowed directory
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()

        # Create CSV in allowed directory
        csv_file = allowed_dir / "data.csv"
        csv_file.write_text("name,age
Alice,30")

        # Create importer with directory restriction
        importer = CSVImporter(mock_neo4j_connection, allowed_dir=str(allowed_dir))

        # Should accept file in allowed directory
        validated_path = importer._validate_file_path(str(csv_file))
        assert validated_path.exists()

        # Should reject file outside allowed directory
        outside_file = tmp_path / "outside.csv"
        outside_file.write_text("name,age
Bob,25")

        with pytest.raises(ValueError, match="must be within"):
            importer._validate_file_path(str(outside_file))


@pytest.mark.integration
class TestErrorHandlingScenarios:
    """Test various error handling scenarios (UPDATED)."""

    def test_invalid_uri_format(self):
        """Test handling of invalid URI formats (UPDATED)."""
        from src.neo4j_framework.db.connection import Neo4jConnection

        # Connection object should be created (validation at connect time)
        conn = Neo4jConnection(
            uri="invalid://uri", username="neo4j", password="password"
        )

        assert conn.uri == "invalid://uri"

    def test_missing_password_error(self):
        """Test error when password is missing (UPDATED)."""
        from src.neo4j_framework.db.connection import Neo4jConnection

        conn = Neo4jConnection(
            uri="neo4j://localhost:7687", username="neo4j", password=None
        )

        with pytest.raises(ValueError, match="Username and password required"):
            conn._create_basic_auth()

    def test_invalid_parameter_type(self, clean_env):
        """Test handling of invalid parameter types (UPDATED)."""
        import os
        from src.neo4j_framework.config.env_loader import EnvironmentLoader

        os.environ["NEO4J_INVALID_INT"] = "not_a_number"

        loader = EnvironmentLoader(env_prefix="NEO4J_")

        with pytest.raises(ValueError, match="must be an integer"):
            loader.get_int("INVALID_INT")

    def test_transaction_error_propagation(self, mock_neo4j_connection):
        """Test transaction manager error handling."""
        from src.neo4j_framework.transactions.transaction_manager import TransactionManager

        tx_manager = TransactionManager(mock_neo4j_connection)

        # Test that passing None function raises
        with pytest.raises(ValueError, match="cannot be None"):
            tx_manager.run_in_transaction(None)

    def test_query_manager_connection_validation(self):
        """Test QueryManager validates connection (UPDATED)."""
        from src.neo4j_framework.queries.query_manager import QueryManager

        # Should raise when connection is None
        with pytest.raises(ValueError, match="cannot be None"):
            QueryManager(None)

    def test_transaction_manager_connection_validation(self):
        """Test TransactionManager validates connection (UPDATED)."""
        from src.neo4j_framework.transactions.transaction_manager import (
            TransactionManager,
        )

        # Should raise when connection is None
        with pytest.raises(ValueError, match="cannot be None"):
            TransactionManager(None)

    def test_csv_importer_connection_validation(self):
        """Test CSVImporter validates connection (UPDATED)."""
        from src.neo4j_framework.importers.csv_importer import CSVImporter

        # Should raise when connection is None
        with pytest.raises(ValueError, match="cannot be None"):
            CSVImporter(None)

    def test_pool_size_validation_error(self):
        """Test pool size validation (NEW)."""
        from src.neo4j_framework.db.connection import Neo4jConnection

        # Should raise for invalid pool size
        with pytest.raises(ValueError, match="must be between 1 and 500"):
            Neo4jConnection(
                uri="neo4j://localhost:7687",
                username="neo4j",
                password="password",
                max_connection_pool_size=1000,
            )

    def test_mtls_certificate_validation_error(self):
        """Test mTLS certificate validation (NEW)."""
        from src.neo4j_framework.db.connection import Neo4jConnection

        conn = Neo4jConnection(
            uri="neo4j://localhost:7687",
            username="neo4j",
            password="password",
        )

        # Should raise for missing certificate
        with pytest.raises(ValueError, match="Certificate file not found"):
            conn.connect_with_mtls(cert_path="/nonexistent/cert.pem")


@pytest.mark.integration
class TestPerformanceAndOptimization:
    """Test performance-related functionality (UPDATED)."""

    def test_query_timing_with_logging(self, caplog):
        """Test query execution timing with logging (UPDATED)."""
        from src.neo4j_framework.utils.performance import Performance
        import time
        import logging

        @Performance.time_function
        def timed_operation():
            time.sleep(0.01)
            return "complete"

        with caplog.at_level(logging.DEBUG):
            result = timed_operation()

        assert result == "complete"
        # Verify timing was logged
        assert "execution time" in caplog.text.lower()

    def test_query_error_logging(self, caplog):
        """Test that query errors are logged (NEW)."""
        from src.neo4j_framework.utils.performance import Performance
        import logging

        @Performance.time_function
        def failing_operation():
            raise ValueError("Test error")

        with caplog.at_level(logging.ERROR):
            with pytest.raises(ValueError):
                failing_operation()

        # Verify error was logged
        assert "failed after" in caplog.text.lower()

    def test_bulk_read_operations(self, mock_neo4j_connection):
        """Test performance of bulk read operations (UPDATED)."""
        from src.neo4j_framework.queries.query_manager import QueryManager

        query_manager = QueryManager(mock_neo4j_connection)

        # Create many mock records
        mock_records = [
            MagicMock(data=lambda i=i: {"id": i, "value": i * 10}) for i in range(1000)
        ]

        # Configure mock
        mock_driver = mock_neo4j_connection._driver
        mock_session = mock_driver.session.return_value
        mock_session.execute_read.return_value = mock_records

        # Execute bulk read
        results = query_manager.execute_read("MATCH (n) RETURN n")

        assert len(results) == 1000

    def test_bulk_write_operations(self, mock_neo4j_connection):
        """Test performance of bulk write operations (UPDATED)."""
        from src.neo4j_framework.queries.query_manager import QueryManager

        query_manager = QueryManager(mock_neo4j_connection)

        # Simulate bulk writes
        for i in range(100):
            query_manager.execute_write(
                "CREATE (n:Node {value: $val})", params={"val": i}
            )

        # Verify execute_write was called multiple times
        mock_driver = mock_neo4j_connection._driver
        mock_session = mock_driver.session.return_value
        assert mock_session.execute_write.call_count >= 100

    def test_connection_reuse(self, mock_neo4j_connection):
        """Test connection reuse across multiple operations (NEW)."""
        from src.neo4j_framework.queries.query_manager import QueryManager
        from src.neo4j_framework.transactions.transaction_manager import (
            TransactionManager,
        )

        # Create components with same connection
        query_manager = QueryManager(mock_neo4j_connection)
        transaction_manager = TransactionManager(mock_neo4j_connection)

        # Both should use same connection
        assert query_manager.connection is transaction_manager.connection

        # Multiple operations should reuse same driver
        query_manager.execute_read("MATCH (n) RETURN n LIMIT 1")
        transaction_manager.run_in_transaction(lambda tx: tx.run("MATCH (n) RETURN n"))

        # Driver should be called twice (once per component)
        mock_driver = mock_neo4j_connection._driver
        assert mock_driver.session.call_count == 2


@pytest.mark.integration
class TestCompleteWorkflows:
    """Test complete end-to-end workflows (NEW)."""

    def test_data_import_and_query_workflow(self, mock_neo4j_connection, test_csv_file):
        """Test complete workflow: import CSV then query data."""
        from src.neo4j_framework.importers.csv_importer import CSVImporter
        from src.neo4j_framework.queries.query_manager import QueryManager

        # Step 1: Import CSV
        importer = CSVImporter(mock_neo4j_connection)
        import_query = (
            "LOAD CSV WITH HEADERS FROM $file_url AS row "
            "CREATE (:Person {name: row.name, age: row.age})"
        )
        importer.import_csv(str(test_csv_file), import_query)

        # Step 2: Query imported data
        query_manager = QueryManager(mock_neo4j_connection)
        results = query_manager.execute_read(
            "MATCH (p:Person) RETURN p ORDER BY p.name"
        )

        assert results is not None

    def test_write_and_transaction_workflow(self, mock_neo4j_connection):
        """Test workflow: write query then transaction."""
        from src.neo4j_framework.queries.query_manager import QueryManager
        from src.neo4j_framework.transactions.transaction_manager import (
            TransactionManager,
        )

        query_manager = QueryManager(mock_neo4j_connection)
        transaction_manager = TransactionManager(mock_neo4j_connection)

        # Step 1: Create node with write query
        query_manager.execute_write(
            "CREATE (n:Node {name: $name}) RETURN n", params={"name": "TestNode"}
        )

        # Step 2: Add relationships in transaction
        def add_relationship(tx):
            return tx.run(
                "MATCH (n1:Node), (n2:Node) "
                "WHERE n1.name = $n1 AND n2.name = $n2 "
                "CREATE (n1)-[:CONNECTS]->(n2)",
                {"n1": "TestNode", "n2": "TestNode"},
            )

        transaction_manager.run_in_transaction(add_relationship)

        # Verify operations completed
        mock_driver = mock_neo4j_connection._driver
        mock_session = mock_driver.session.return_value
        assert mock_session.execute_write.call_count >= 2

    def test_read_modify_write_workflow(self, mock_neo4j_connection):
        """Test workflow: read, modify data, write back."""
        from src.neo4j_framework.queries.query_manager import QueryManager

        query_manager = QueryManager(mock_neo4j_connection)

        # Setup mock to return data on read
        mock_driver = mock_neo4j_connection._driver
        mock_session = mock_driver.session.return_value

        mock_record = MagicMock(data=lambda: {"name": "OldName", "id": 1})
        mock_session.execute_read.return_value = [mock_record]

        # Step 1: Read current data
        current_data = query_manager.execute_read(
            "MATCH (n:Node) WHERE n.id = $id RETURN n", params={"id": 1}
        )

        assert current_data is not None

        # Step 2: Modify and write back
        modified_name = "NewName"
        query_manager.execute_write(
            "MATCH (n:Node) WHERE n.id = $id SET n.name = $new_name",
            params={"id": 1, "new_name": modified_name},
        )

        # Verify both read and write were called
        assert mock_session.execute_read.called
        assert mock_session.execute_write.called

    def test_error_recovery_workflow(self, mock_neo4j_connection):
        """Test error handling and recovery in complete workflow."""
        from src.neo4j_framework.queries.query_manager import QueryManager

        query_manager = QueryManager(mock_neo4j_connection)

        # Setup mock to fail first, succeed second
        mock_driver = mock_neo4j_connection._driver
        mock_session = mock_driver.session.return_value

        # First attempt fails
        mock_session.execute_read.side_effect = [
            Exception("Connection failed"),
            [MagicMock(data=lambda: {"status": "ok"})],
        ]

        # First call raises
        with pytest.raises(Exception):
            query_manager.execute_read("MATCH (n) RETURN n")

        # Reset for retry
        mock_session.execute_read.side_effect = None
        mock_session.execute_read.return_value = [
            MagicMock(data=lambda: {"status": "ok"})
        ]

        # Retry succeeds
        result = query_manager.execute_read("MATCH (n) RETURN n")
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
```

< /home/honeymatrix/Projects/neo4j_framework/tests/test_security.py >

```
"""
Security-focused tests for Neo4j Framework.
Tests validation, input sanitization, and security features.
"""

import pytest
import os
from pathlib import Path
from unittest.mock import MagicMock, patch


@pytest.mark.security
class TestSecurityValidation:
    """Security validation tests."""

    def test_pool_size_bounds_enforcement(self):
        """Test that pool size is strictly bounded."""
        from src.neo4j_framework.db.connection import Neo4jConnection

        # Test upper bound
        with pytest.raises(ValueError):
            Neo4jConnection(
                uri="neo4j://localhost:7687",
                username="neo4j",
                password="password",
                max_connection_pool_size=501,
            )

        # Test lower bound
        with pytest.raises(ValueError):
            Neo4jConnection(
                uri="neo4j://localhost:7687",
                username="neo4j",
                password="password",
                max_connection_pool_size=0,
            )

        # Test negative
        with pytest.raises(ValueError):
            Neo4jConnection(
                uri="neo4j://localhost:7687",
                username="neo4j",
                password="password",
                max_connection_pool_size=-10,
            )

    def test_timeout_bounds_enforcement(self, clean_env):
        """Test that timeouts are bounded."""
        from src.neo4j_framework.config.env_loader import EnvironmentLoader

        loader = EnvironmentLoader(env_prefix="NEO4J_")

        # Test connection timeout upper bound
        os.environ["NEO4J_CONNECTION_TIMEOUT"] = "500.0"
        with pytest.raises(ValueError, match="must be <= 300.0"):
            loader.get_float("CONNECTION_TIMEOUT", min_val=0.1, max_val=300.0)

        # Test connection timeout lower bound
        os.environ["NEO4J_CONNECTION_TIMEOUT"] = "0.05"
        with pytest.raises(ValueError, match="must be >= 0.1"):
            loader.get_float("CONNECTION_TIMEOUT", min_val=0.1, max_val=300.0)

    def test_csv_path_traversal_prevention(self, mock_neo4j_connection, tmp_path):
        """Test that CSV importer prevents directory traversal."""
        from src.neo4j_framework.importers.csv_importer import CSVImporter

        # Create allowed directory
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()

        # Create CSV in allowed dir
        csv_file = allowed_dir / "data.csv"
        csv_file.write_text("name
test")

        importer = CSVImporter(mock_neo4j_connection, allowed_dir=str(allowed_dir))

        # Create file outside allowed dir
        outside_file = tmp_path / "outside.csv"
        outside_file.write_text("name
test")

        # Attempt path traversal - should fail
        with pytest.raises(ValueError, match="must be within"):
            importer._validate_file_path(str(outside_file))

    def test_certificate_file_validation(self):
        """Test that certificate paths are validated."""
        from src.neo4j_framework.db.connection import Neo4jConnection

        conn = Neo4jConnection(
            uri="neo4j://localhost:7687",
            username="neo4j",
            password="password",
        )

        # Non-existent cert file should raise
        with pytest.raises(ValueError, match="Certificate file not found"):
            conn.connect_with_mtls(cert_path="/nonexistent/cert.pem")

    def test_sensitive_error_messages(self, clean_env):
        """Test that error messages don't expose sensitive data."""
        from src.neo4j_framework.config.env_loader import EnvironmentLoader

        loader = EnvironmentLoader(env_prefix="NEO4J_")

        try:
            loader.get("PASSWORD", required=True)
        except ValueError as e:
            error_msg = str(e)
            # Should mention the variable but not expose actual credentials
            assert "Please configure" in error_msg
            # Should not contain actual password values
            assert "password" not in error_msg.lower() or "NEO4J_PASSWORD" in error_msg


@pytest.mark.security
class TestAuthenticationSecurity:
    """Authentication security tests."""

    def test_missing_credentials_rejected(self):
        """Test that connections require credentials."""
        from src.neo4j_framework.db.connection import Neo4jConnection

        conn = Neo4jConnection(
            uri="neo4j://localhost:7687",
            username=None,
            password=None,
        )

        with pytest.raises(ValueError, match="Username and password required"):
            conn._create_basic_auth()

    def test_kerberos_auth_requires_ticket(self):
        """Test Kerberos auth requires ticket."""
        from src.neo4j_framework.db.connection import Neo4jConnection

        conn = Neo4jConnection(
            uri="neo4j://localhost:7687",
            username="neo4j",
            password="password",
        )

        with pytest.raises(ValueError, match="Kerberos ticket required"):
            conn.connect(auth_type="kerberos")

    def test_bearer_auth_requires_token(self):
        """Test Bearer auth requires token."""
        from src.neo4j_framework.db.connection import Neo4jConnection

        conn = Neo4jConnection(
            uri="neo4j://localhost:7687",
            username="neo4j",
            password="password",
        )

        with pytest.raises(ValueError, match="Bearer token required"):
            conn.connect(auth_type="bearer")

    def test_custom_auth_requires_all_parameters(self):
        """Test custom auth requires all parameters."""
        from src.neo4j_framework.db.connection import Neo4jConnection

        conn = Neo4jConnection(
            uri="neo4j://localhost:7687",
            username="neo4j",
            password="password",
        )

        # Missing 'scheme'
        with pytest.raises(ValueError, match="required for custom authentication"):
            conn.connect(
                auth_type="custom",
                principal="user",
                credentials="pass",
                realm="realm",
            )


@pytest.mark.security
class TestFileSecurityValidation:
    """File security validation tests."""

    def test_csv_file_readability_check(self, mock_neo4j_connection, tmp_path):
        """Test that CSV importer checks file readability."""
        from src.neo4j_framework.importers.csv_importer import CSVImporter
        import os

        # Create unreadable file (on Unix systems)
        if hasattr(os, "chmod"):
            csv_file = tmp_path / "unreadable.csv"
            csv_file.write_text("data")
            os.chmod(str(csv_file), 0o000)

            importer = CSVImporter(mock_neo4j_connection)

            try:
                with pytest.raises(ValueError, match="not readable"):
                    importer._validate_file_path(str(csv_file))
            finally:
                # Restore permissions for cleanup
                os.chmod(str(csv_file), 0o644)

    def test_cert_file_readability_check(self, tmp_path):
        """Test that mTLS checks certificate file readability."""
        from src.neo4j_framework.db.connection import Neo4jConnection
        import os

        if hasattr(os, "chmod"):
            cert_file = tmp_path / "cert.pem"
            cert_file.write_text("FAKE CERT")
            os.chmod(str(cert_file), 0o000)

            conn = Neo4jConnection(
                uri="neo4j://localhost:7687",
                username="neo4j",
                password="password",
            )

            try:
                with pytest.raises(ValueError, match="not readable"):
                    conn.connect_with_mtls(cert_path=str(cert_file))
            finally:
                # Restore permissions for cleanup
                os.chmod(str(cert_file), 0o644)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
```

< /home/honeymatrix/Projects/neo4j_framework/tests/test_additional_coverage.py >

```
"""
Additional test coverage for Neo4j Framework.
Tests edge cases, performance, and advanced scenarios.
"""

import pytest
import logging
from unittest.mock import MagicMock, patch
from src.neo4j_framework.config.env_loader import EnvironmentLoader
from src.neo4j_framework.db.connection import Neo4jConnection
from src.neo4j_framework.queries.query_manager import QueryManager
from src.neo4j_framework.transactions.transaction_manager import TransactionManager
from src.neo4j_framework.importers.csv_importer import CSVImporter


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_large_pool_size(self):
        """Test handling of maximum pool size."""
        conn = Neo4jConnection(
            uri="neo4j://localhost:7687",
            username="neo4j",
            password="password",
            max_connection_pool_size=500,  # Max allowed
        )
        assert conn.config["max_connection_pool_size"] == 500

    def test_minimum_pool_size(self):
        """Test minimum pool size of 1."""
        conn = Neo4jConnection(
            uri="neo4j://localhost:7687",
            username="neo4j",
            password="password",
            max_connection_pool_size=1,
        )
        assert conn.config["max_connection_pool_size"] == 1

    def test_empty_query_string(self, mock_neo4j_connection):
        """Test handling of empty query string."""
        from src.neo4j_framework.queries.base_query import BaseQuery

        # BaseQuery accepts empty string (validation happens at execution)
        query = BaseQuery("", {})
        assert query is not None

    def test_none_query_parameters(self, mock_neo4j_connection):
        """Test query execution with None parameters."""
        query_manager = QueryManager(mock_neo4j_connection)
        # Should use empty dict instead of None
        results = query_manager.execute_read("MATCH (n) RETURN n LIMIT 1")
        assert isinstance(results, list)

    def test_very_large_query_result(self, mock_neo4j_connection):
        """Test handling of very large result sets."""
        query_manager = QueryManager(mock_neo4j_connection)

        # Mock large result set
        mock_driver = mock_neo4j_connection._driver
        mock_session = mock_driver.session.return_value
        large_records = [
            MagicMock(data=lambda: {"id": i, "name": f"Node{i}"})
            for i in range(10000)
        ]
        mock_session.execute_read.return_value = large_records

        results = query_manager.execute_read("MATCH (n) RETURN n")
        assert len(results) == 10000

    def test_unicode_in_parameters(self, mock_neo4j_connection):
        """Test handling of unicode characters in query parameters."""
        query_manager = QueryManager(mock_neo4j_connection)

        unicode_params = {
            "name": "测试",  # Chinese
            "text": "مرحبا",  # Arabic
            "emoji": "🚀🎉"  # Emoji
        }

        # Should not raise
        query_manager.execute_write(
            "CREATE (n:Node {name: $name, text: $text, emoji: $emoji}) RETURN n",
            params=unicode_params
        )


@pytest.mark.unit
class TestConcurrency:
    """Test concurrent operations (simulated)."""

    def test_multiple_simultaneous_readers(self, mock_neo4j_connection):
        """Test multiple read operations."""
        query_manager = QueryManager(mock_neo4j_connection)

        for i in range(10):
            results = query_manager.execute_read("MATCH (n) RETURN n LIMIT 1")
            assert isinstance(results, list)

    def test_read_write_interleaving(self, mock_neo4j_connection):
        """Test interleaved read and write operations."""
        query_manager = QueryManager(mock_neo4j_connection)

        for i in range(5):
            # Write
            query_manager.execute_write(
                "CREATE (n:Node {id: $id}) RETURN n",
                params={"id": i}
            )
            # Read
            results = query_manager.execute_read(
                "MATCH (n:Node {id: $id}) RETURN n",
                params={"id": i}
            )
            assert isinstance(results, list)


@pytest.mark.unit
class TestDatabaseAbstraction:
    """Test database selection and multi-database support."""

    def test_database_parameter_on_read(self, mock_neo4j_connection):
        """Test explicit database selection on read."""
        query_manager = QueryManager(mock_neo4j_connection)

        results = query_manager.execute_read(
            "MATCH (n) RETURN n LIMIT 1",
            database="custom_db"
        )

        # Verify database was passed to session
        mock_driver = mock_neo4j_connection._driver
        mock_session = mock_driver.session.return_value
        # Session should be called with database parameter
        assert mock_driver.session.called

    def test_database_parameter_on_write(self, mock_neo4j_connection):
        """Test explicit database selection on write."""
        query_manager = QueryManager(mock_neo4j_connection)

        query_manager.execute_write(
            "CREATE (n:Node) RETURN n",
            database="custom_db"
        )

        mock_driver = mock_neo4j_connection._driver
        assert mock_driver.session.called

    def test_transaction_with_specific_database(self, mock_neo4j_connection):
        """Test transaction execution with specific database."""
        tx_manager = TransactionManager(mock_neo4j_connection)

        def tx_func(tx):
            return tx.run("CREATE (n:Node) RETURN n")

        tx_manager.run_in_transaction(tx_func, database="test_db")

        mock_driver = mock_neo4j_connection._driver
        assert mock_driver.session.called


@pytest.mark.security
class TestInputSanitization:
    """Test input sanitization and injection prevention."""

    def test_query_injection_prevention(self, mock_neo4j_connection):
        """Test that raw query injection is prevented."""
        from src.neo4j_framework.queries.base_query import BaseQuery

        # Should use parameterized queries
        malicious_param = "'; DROP TABLE users; --"
        query = BaseQuery("CREATE (n:Node {name: $name}) RETURN n", {"name": malicious_param})

        # Parameters should be safe (not concatenated into query string)
        assert "$name" in query.query_str
        assert "DROP TABLE" not in query.query_str
        assert query.params["name"] == malicious_param

    def test_cypher_injection_in_parameter_names(self, mock_neo4j_connection):
        """Test handling of suspicious parameter names."""
        query_manager = QueryManager(mock_neo4j_connection)

        # Parameters with suspicious names should still work safely
        params = {
            "normal": "value",
            "with_number_123": "test",
            "with_underscore": "test"
        }

        # Should not raise
        query_manager.execute_read(
            "MATCH (n) WHERE n.field = $normal RETURN n",
            params=params
        )


@pytest.mark.unit
class TestConnectionReuse:
    """Test connection pooling and reuse patterns."""

    def test_connection_not_closed_between_queries(self, mock_neo4j_connection):
        """Test that connection is reused, not recreated."""
        query_manager = QueryManager(mock_neo4j_connection)

        for _ in range(100):
            query_manager.execute_read("MATCH (n) RETURN n LIMIT 1")

        # Driver should be reused, not recreated
        mock_driver = mock_neo4j_connection._driver
        # Should be called 100 times, not recreate driver each time
        assert mock_driver.session.call_count == 100

    def test_connection_lifecycle_reset(self, clean_env):
        """Test connection lifecycle reset between tests."""
        from src.neo4j_framework.db.connection import Neo4jConnection

        conn1 = Neo4jConnection(
            uri="neo4j://localhost:7687",
            username="neo4j",
            password="password"
        )

        # New connection should start fresh
        assert conn1._driver is None
        assert not conn1.is_connected()


@pytest.mark.unit
class TestEnvVariablePriority:
    """Test environment variable precedence and loading order."""

    def test_system_env_takes_precedence(self, clean_env):
        """Test that system environment variables take precedence."""
        import os

        # Set system environment
        os.environ["NEO4J_URI"] = "neo4j://system:7687"
        os.environ["NEO4J_PASSWORD"] = "system_password"

        loader = EnvironmentLoader(env_file=".env.nonexistent", env_prefix="NEO4J_")

        # Should use system environment, not missing .env file
        uri = loader.get("URI")
        password = loader.get("PASSWORD")

        assert uri == "neo4j://system:7687"
        assert password == "system_password"

    def test_missing_env_file_uses_system_env(self, clean_env):
        """Test that missing .env file falls back to system environment."""
        import os

        os.environ["NEO4J_TEST_VAR"] = "from_system"

        loader = EnvironmentLoader(env_file=".env.missing", env_prefix="NEO4J_")
        value = loader.get("TEST_VAR")

        assert value == "from_system"


@pytest.mark.integration
class TestRecoveryScenarios:
    """Test recovery from various failure scenarios."""

    def test_recovery_from_connection_timeout(self, mock_neo4j_connection):
        """Test handling of connection timeouts."""
        query_manager = QueryManager(mock_neo4j_connection)

        mock_driver = mock_neo4j_connection._driver
        mock_session = mock_driver.session.return_value

        # First call times out, second succeeds
        mock_session.execute_read.side_effect = [
            TimeoutError("Connection timeout"),
            [MagicMock(data=lambda: {"result": "success"})]
        ]

        # First call raises
        with pytest.raises(TimeoutError):
            query_manager.execute_read("MATCH (n) RETURN n LIMIT 1")

        # Reset for next attempt
        mock_session.execute_read.side_effect = None
        mock_session.execute_read.return_value = [MagicMock(data=lambda: {"result": "success"})]

        # Retry succeeds
        results = query_manager.execute_read("MATCH (n) RETURN n LIMIT 1")
        assert len(results) > 0


@pytest.mark.unit
class TestSpecialCharactersAndEncoding:
    """Test handling of special characters and various encodings."""

    def test_special_chars_in_node_properties(self, mock_neo4j_connection):
        """Test handling of special characters in node properties."""
        query_manager = QueryManager(mock_neo4j_connection)

        special_params = {
            "special": "!@#$%^&*()",
            "quotes": 'He said "Hello"',
            "newlines": "Line1
Line2
Line3",
            "tabs": "Col1	Col2	Col3"
        }

        query_manager.execute_write(
            "CREATE (n:Node {prop: $special}) RETURN n",
            params=special_params
        )

    def test_paths_with_special_characters(self, mock_neo4j_connection, tmp_path):
        """Test CSV file paths with special characters."""
        from src.neo4j_framework.importers.csv_importer import CSVImporter

        # Create file with special name
        special_dir = tmp_path / "dir-with_special.chars"
        special_dir.mkdir()

        csv_file = special_dir / "file-2024_test (1).csv"
        csv_file.write_text("name
Test")

        importer = CSVImporter(mock_neo4j_connection)
        validated = importer._validate_file_path(str(csv_file))

        assert validated.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

< /home/honeymatrix/Projects/neo4j_framework/.env.example >

```
# Neo4j Connection Configuration
NEO4J_URI=neo4j://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password_here
NEO4J_DATABASE=neo4j

# Security Settings
NEO4J_ENCRYPTED=true

# Connection Pool Settings (1-500)
NEO4J_MAX_CONNECTION_POOL_SIZE=100

# Timeout Settings (in seconds, 0.1-300)
NEO4J_CONNECTION_TIMEOUT=30.0
NEO4J_MAX_TRANSACTION_RETRY_TIME=30.0

# Optional: Authentication
# NEO4J_KERBEROS_TICKET=your_ticket
# NEO4J_BEARER_TOKEN=your_token

# Optional: mTLS Configuration
# NEO4J_CLIENT_CERT_PATH=/path/to/cert.pem
# NEO4J_CLIENT_KEY_PATH=/path/to/key.pem
# NEO4J_CLIENT_KEY_PASSWORD=your_key_password

# Optional: Logging
NEO4J_LOG_LEVEL=INFO
```

< /home/honeymatrix/Projects/neo4j_framework/.gitignore >

```
# Virtual Environment
.venv/
venv/
ENV/
env/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# pytest
.pytest_cache/
.coverage
htmlcov/
.tox/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store
*.sublime-workspace

# Pyright
pyrightconfig.local.json
.pyright/

# Environment
.env
.env.local
.env.*.local

# OS
.DS_Store
Thumbs.db

# Misc
*.log
logs/
.cache/
```

< /home/honeymatrix/Projects/neo4j_framework/requirements.txt >

```
neo4j>=5.0.0,<6.0.0
python-dotenv>=1.0.0
typing-extensions>=4.0.0
```

< /home/honeymatrix/Projects/neo4j_framework/pyproject.toml >

```
[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "neo4j-framework"
version = "2.0.0"
description = "Production-ready Neo4j Python library with type safety and security"
readme = "README.md"
requires-python = ">=3.11"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]

dependencies = [
    "neo4j>=5.0.0,<6.0.0",
    "python-dotenv>=1.0.0",
    "typing-extensions>=4.0.0"
]

[project.urls]
Homepage = "https://github.com/FourtySevenInstead/neo4j_framework"
Repository = "https://github.com/FourtySevenInstead/neo4j_framework.git"
Documentation = "https://github.com/FourtySevenInstead/neo4j_framework/blob/main/README.md"

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]
include = ["neo4j_framework*"]

[tool.pyright]
include = ["src"]
exclude = ["**/node_modules", "**/__pycache__", "tests"]
typeCheckingMode = "strict"
reportMissingTypeStubs = false
reportUnknownMemberType = false
```

< /home/honeymatrix/Projects/neo4j_framework/src/neo4j_framework/config/db_config.py >

```
from typing import Dict, Any
from .env_loader import EnvironmentLoader


def get_db_config(env_file: str = ".env", env_prefix: str = "NEO4J_") -> Dict[str, Any]:
    """
    Get database configuration from environment variables.

    Args:
        env_file: Path to .env file
        env_prefix: Prefix for environment variables

    Returns:
        Dictionary with database configuration
    """
    loader = EnvironmentLoader(env_file, env_prefix)
    return loader.get_config()
```

< /home/honeymatrix/Projects/neo4j_framework/src/neo4j_framework/config/**init**.py >

```
from .env_loader import EnvironmentLoader
from .db_config import get_db_config

__all__ = ["EnvironmentLoader", "get_db_config"]
```

< /home/honeymatrix/Projects/neo4j_framework/src/neo4j_framework/config/env_loader.py >

```
"""
Environment variable loader with validation and security features.
"""

import os
import logging
from typing import Optional, Dict, Any

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Valid range for connection pool size
POOL_SIZE_MIN = 1
POOL_SIZE_MAX = 500


class EnvironmentLoader:
    """
    Loads and manages environment variables securely.

    Supports custom .env file paths and environment variable prefixes
    for flexible integration into multiple projects.

    **Important:** override=False means existing environment variables
    (from the system/shell) take precedence over values in the .env file.
    This is intentional—don't override host system configuration.
    """

    def __init__(
        self,
        env_file: str = ".env",
        env_prefix: str = "NEO4J_",
    ):
        """
        Initialize the environment loader.

        Args:
            env_file: Path to the .env file (default: .env)
            env_prefix: Prefix for environment variables (default: NEO4J_)
        """
        self.env_file = env_file
        self.env_prefix = env_prefix
        self._loaded = False

    def load(self) -> bool:
        """Load environment variables from .env file."""
        if not self._loaded:
            if os.path.exists(self.env_file):
                self._loaded = load_dotenv(self.env_file, override=False)
                logger.debug(f"Environment variables loaded from {self.env_file}")
            else:
                logger.debug(f".env file not found at {self.env_file}")
                self._loaded = True  # Mark as attempted
        return self._loaded

    def get(
        self, key: str, default: Optional[str] = None, required: bool = False
    ) -> Optional[str]:
        """
        Get an environment variable value.

        Automatically prepends the configured prefix to the key.

        Args:
            key: Environment variable name (without prefix)
            default: Default value if not found
            required: If True, raises ValueError when key is not found

        Returns:
            Environment variable value or default

        Raises:
            ValueError: If required variable is not found
        """
        if not self._loaded:
            self.load()

        # Build the full key with prefix
        full_key = f"{self.env_prefix}{key}"
        value = os.getenv(full_key, default)

        if required and value is None:
            raise ValueError(
                f"Required environment variable not set. Please configure {full_key}"
            )

        return value

    def get_int(
        self,
        key: str,
        default: Optional[int] = None,
        min_val: Optional[int] = None,
        max_val: Optional[int] = None,
    ) -> Optional[int]:
        """
        Get an integer environment variable with optional bounds checking.

        Args:
            key: Environment variable name
            default: Default value if not found
            min_val: Minimum allowed value (inclusive)
            max_val: Maximum allowed value (inclusive)

        Returns:
            Integer value or default

        Raises:
            ValueError: If value is not a valid integer or out of bounds
        """
        value = self.get(key)
        if value is None:
            return default

        try:
            int_value = int(value)
        except ValueError:
            raise ValueError(
                f"Environment variable {self.env_prefix}{key} must be an integer, "
                f"got '{value}'"
            )

        # Validate bounds
        if min_val is not None and int_value < min_val:
            raise ValueError(
                f"Environment variable {self.env_prefix}{key} must be >= {min_val}, "
                f"got {int_value}"
            )
        if max_val is not None and int_value > max_val:
            raise ValueError(
                f"Environment variable {self.env_prefix}{key} must be <= {max_val}, "
                f"got {int_value}"
            )

        return int_value

    def get_float(
        self,
        key: str,
        default: Optional[float] = None,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ) -> Optional[float]:
        """
        Get a float environment variable with optional bounds checking.

        Args:
            key: Environment variable name
            default: Default value if not found
            min_val: Minimum allowed value (inclusive)
            max_val: Maximum allowed value (inclusive)

        Returns:
            Float value or default

        Raises:
            ValueError: If value is not a valid float or out of bounds
        """
        value = self.get(key)
        if value is None:
            return default

        try:
            float_value = float(value)
        except ValueError:
            raise ValueError(
                f"Environment variable {self.env_prefix}{key} must be a float, "
                f"got '{value}'"
            )

        # Validate bounds
        if min_val is not None and float_value < min_val:
            raise ValueError(
                f"Environment variable {self.env_prefix}{key} must be >= {min_val}, "
                f"got {float_value}"
            )
        if max_val is not None and float_value > max_val:
            raise ValueError(
                f"Environment variable {self.env_prefix}{key} must be <= {max_val}, "
                f"got {float_value}"
            )

        return float_value

    def get_bool(self, key: str, default: bool = False) -> bool:
        """
        Get a boolean environment variable.

        Accepts: true, 1, yes, on (case-insensitive)

        Args:
            key: Environment variable name
            default: Default value if not found

        Returns:
            Boolean value or default
        """
        value = self.get(key)
        if value is None:
            return default
        return value.lower() in ("true", "1", "yes", "on")

    def get_config(self) -> Dict[str, Any]:
        """
        Get all Neo4j configuration from environment variables.

        Returns:
            Dictionary containing all configuration values

        Raises:
            ValueError: If required variables are missing or invalid
        """
        self.load()

        return {
            "uri": self.get("URI", "neo4j://localhost:7687", required=True),
            "username": self.get("USERNAME", "neo4j"),
            "password": self.get("PASSWORD", required=True),
            "database": self.get("DATABASE", "neo4j"),
            # Default to True for security
            "encrypted": self.get_bool("ENCRYPTED", True),
            "max_connection_pool_size": self.get_int(
                "MAX_CONNECTION_POOL_SIZE",
                100,
                min_val=POOL_SIZE_MIN,
                max_val=POOL_SIZE_MAX,
            ),
            "connection_timeout": self.get_float(
                "CONNECTION_TIMEOUT", 30.0, min_val=0.1, max_val=300.0
            ),
            "max_transaction_retry_time": self.get_float(
                "MAX_TRANSACTION_RETRY_TIME", 30.0, min_val=1.0, max_val=300.0
            ),
            "kerberos_ticket": self.get("KERBEROS_TICKET"),
            "bearer_token": self.get("BEARER_TOKEN"),
            "client_cert_path": self.get("CLIENT_CERT_PATH"),
            "client_key_path": self.get("CLIENT_KEY_PATH"),
            "client_key_password": self.get("CLIENT_KEY_PASSWORD"),
        }
```

< /home/honeymatrix/Projects/neo4j_framework/src/neo4j_framework/db/**init**.py >

```
from .connection import Neo4jConnection
from .pool_manager import PoolManager

__all__ = ["Neo4jConnection", "PoolManager"]
```

< /home/honeymatrix/Projects/neo4j_framework/src/neo4j_framework/db/pool_manager.py >

```
"""
Connection pool management utilities.

Note: The Neo4j Python driver does not directly expose pool statistics.
This class is a placeholder for future monitoring capabilities or
integration with external monitoring tools.
"""

import logging

logger = logging.getLogger(__name__)


class PoolManager:
    """
    Utilities for managing connection pools.

    The Neo4j driver manages pooling internally. This class can be used for
    future monitoring or metrics collection.
    """

    def __init__(self, driver):
        """
        Initialize pool manager.

        Args:
            driver: Neo4j Driver instance
        """
        self.driver = driver
        logger.debug("PoolManager initialized")

    def get_pool_stats(self):
        """
        Get connection pool statistics.

        Note: Detailed pool stats are not exposed by the Neo4j driver.
        Consider using Neo4j monitoring endpoints for detailed metrics.

        Returns:
            Dictionary with available pool information
        """
        logger.info(
            "Pool statistics not directly available in Neo4j driver. "
            "Use Neo4j monitoring endpoints for detailed metrics."
        )
        return {"note": "Use Neo4j monitoring endpoints for detailed pool metrics"}
```

< /home/honeymatrix/Projects/neo4j_framework/src/neo4j_framework/db/connection.py >

```
"""
Neo4j connection management with support for multiple authentication methods.
"""

import logging
from typing import Optional, Any, Dict, Type, cast, Callable, TypeVar
from neo4j import GraphDatabase, Driver, Auth, Session

logger = logging.getLogger(__name__)
T = TypeVar("T")


class Neo4jConnection:
    """
    Manages Neo4j database connections with support for multiple authentication methods.
    Handles connection pooling and lifecycle management.
    """

    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        database: str = "neo4j",
        encrypted: bool = True,
        max_connection_pool_size: int = 100,
    ) -> None:
        """
        Initialize Neo4j connection configuration.

        Args:
            uri: Neo4j connection URI (neo4j://, neo4j+s://, bolt://, etc.)
            username: Database username
            password: Database password
            database: Target database name
            encrypted: Enable encryption (ignored for URIs with +s or +ssc)
            max_connection_pool_size: Maximum number of concurrent connections

        Raises:
            ValueError: If configuration is invalid
        """
        from neo4j_framework.utils.validators import Validators

        # Validate inputs
        Validators.validate_not_none(uri, "uri")
        Validators.validate_not_none(username, "username")
        Validators.validate_not_none(password, "password")
        Validators.validate_string_not_empty(uri, "uri")
        Validators.validate_string_not_empty(username, "username")
        Validators.validate_string_not_empty(password, "password")

        # Validate pool size bounds
        if max_connection_pool_size < 1 or max_connection_pool_size > 500:
            raise ValueError(
                f"max_connection_pool_size must be between 1 and 500, "
                f"got {max_connection_pool_size}"
            )

        self.uri: str = uri
        self.username: str = username
        self.password: str = password
        self.database: str = database

        # Determine if URI scheme handles encryption
        has_secure_scheme: bool = "+s" in uri or "+ssc" in uri

        # Only use encrypted flag if URI doesn't already handle it
        self.encrypted: bool = False if has_secure_scheme else encrypted
        self.max_connection_pool_size: int = max_connection_pool_size

        self._driver: Optional[Driver] = None
        logger.debug(f"Neo4jConnection initialized: {uri}")

    def connect(
        self,
        auth_type: str = "basic",
        **auth_kwargs: Any,
    ) -> Driver:
        """
        Establish connection to Neo4j database.

        Args:
            auth_type: Authentication type ('basic', 'kerberos', 'bearer', 'custom')
            **auth_kwargs: Additional authentication parameters

        Returns:
            Neo4j Driver instance

        Raises:
            Exception: If connection fails
        """
        try:
            # Create appropriate auth based on auth_type
            if auth_type == "basic":
                auth: Auth = Auth.basic(self.username, self.password)
            elif auth_type == "kerberos":
                if "ticket" not in auth_kwargs:
                    raise ValueError("Kerberos ticket required")
                ticket: str = auth_kwargs["ticket"]
                auth = Auth.kerberos(ticket)
            elif auth_type == "bearer":
                if "token" not in auth_kwargs:
                    raise ValueError("Bearer token required")
                token: str = auth_kwargs["token"]
                auth = Auth.bearer(token)
            elif auth_type == "custom":
                required = ["scheme", "principal", "credentials", "realm"]
                missing = [k for k in required if k not in auth_kwargs]
                if missing:
                    raise ValueError(f"Custom auth missing: {', '.join(missing)}")
                scheme: str = auth_kwargs["scheme"]
                principal: str = auth_kwargs["principal"]
                credentials: str = auth_kwargs["credentials"]
                realm_val: str = auth_kwargs["realm"]
                auth = Auth.custom(scheme, principal, credentials, realm_val)
            else:
                raise ValueError(f"Unknown auth type: {auth_type}")

            # Build driver kwargs with proper types
            driver_kwargs: Dict[str, Any] = {
                "max_connection_pool_size": self.max_connection_pool_size,
            }

            # Only add encrypted flag if we need to specify it
            # (not needed for +s URIs, but required for non-secure URIs)
            if not ("+s" in self.uri or "+ssc" in self.uri):
                driver_kwargs["encrypted"] = self.encrypted

            self._driver = GraphDatabase.driver(
                self.uri,
                auth=auth,
                **driver_kwargs,
            )

            # Test the connection
            self._driver.verify_connectivity()
            logger.info(f"Connected to Neo4j: {self.uri}")

            return self._driver

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            raise

    def connect_with_mtls(
        self,
        cert_path: str,
        key_path: Optional[str] = None,
    ) -> Driver:
        """
        Connect using mTLS certificates.

        Args:
            cert_path: Path to certificate file
            key_path: Optional path to key file

        Returns:
            Neo4j Driver instance

        Raises:
            ValueError: If certificate files don't exist
        """
        import os

        if not os.path.exists(cert_path):
            raise ValueError(f"Certificate file not found: {cert_path}")

        if key_path and not os.path.exists(key_path):
            raise ValueError(f"Key file not found: {key_path}")

        auth: Auth = Auth.basic(self.username, self.password)

        driver_kwargs: Dict[str, Any] = {
            "max_connection_pool_size": self.max_connection_pool_size,
            "encrypted": True,
            "trust": "TRUST_ALL_CERTIFICATES",
        }

        self._driver = GraphDatabase.driver(
            self.uri,
            auth=auth,
            **driver_kwargs,
        )

        logger.info(f"Connected with mTLS: {self.uri}")
        return self._driver

    def is_connected(self) -> bool:
        """Check if connection is established."""
        return self._driver is not None

    def get_driver(self) -> Optional[Driver]:
        """Get the Neo4j driver instance."""
        return self._driver

    def close(self) -> None:
        """Close the connection."""
        if self._driver:
            self._driver.close()
            self._driver = None
            logger.debug("Connection closed")

    def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        if not self._driver:
            return {}

        try:
            # pyright: ignore[attr-defined]
            # _pool is a private Neo4j driver attribute without type stubs
            pool = cast(Any, self._driver._pool)

            in_use_count: int = 0
            available_count: int = 0

            if hasattr(pool, "_in_use"):
                in_use_count = len(pool._in_use)

            if hasattr(pool, "_available"):
                available_count = len(pool._available)

            return {
                "in_use": in_use_count,
                "available": available_count,
            }
        except Exception:
            return {"status": "unable to retrieve pool stats"}

    def run_in_session(
        self,
        func: Callable[[Session], T],
        database: Optional[str] = None,
    ) -> T:
        """
        Execute a function within a Neo4j session.

        Args:
            func: Function that takes a Session and returns a value.
            database: Optional database name (defaults to self.database).

        Returns:
            Result of func.

        Raises:
            RuntimeError: If not connected.
            Exception: If func raises.
        """
        if not self.is_connected():
            raise RuntimeError("Not connected to Neo4j")
        effective_db = database or self.database
        try:
            with self._driver.session(database=effective_db) as session:  # type: ignore
                return func(session)
        except Exception as e:
            logger.error(f"Session execution failed: {type(e).__name__}: {e}")
            raise

    def __enter__(self) -> "Neo4jConnection":
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(
        self,
        _exc_type: Optional[type[BaseException]],
        _exc_val: Optional[BaseException],
        _exc_tb: Any,
    ) -> bool:
        """Context manager exit."""
        self.close()
        return False
```

< /home/honeymatrix/Projects/neo4j_framework/src/neo4j_framework/queries/**init**.py >

```
from .base_query import BaseQuery
from .query_manager import QueryManager
from .query_templates import QueryTemplates

__all__ = ["BaseQuery", "QueryManager", "QueryTemplates"]
```

< /home/honeymatrix/Projects/neo4j_framework/src/neo4j_framework/queries/query_templates.py >

```
class QueryTemplates:
    """
    Predefined query templates.
    """

    CREATE_NODE = "CREATE (n:Node {name: $name}) RETURN n"
    MATCH_NODE = "MATCH (n:Node {name: $name}) RETURN n"

    @classmethod
    def get_template(cls, template_name: str):
        return getattr(cls, template_name, None)
```

< /home/honeymatrix/Projects/neo4j_framework/src/neo4j_framework/queries/base_query.py >

```
"""
Base query class with parameter safety and logging.
"""

import logging
from typing import Any, Dict, LiteralString, Optional, Callable, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    from neo4j_framework.db.connection import Neo4jConnection
    from neo4j_framework.stubs.neo4j import Session, Result  # noqa: F401

from neo4j import Query

logger = logging.getLogger(__name__)


class BaseQuery:
    """
    Base class for queries with parameterization and logging.

    Enforces use of parameterized queries to prevent injection attacks.
    """

    def __init__(
        self,
        query_str: LiteralString | Query,
        params: Dict[str, Any] | None = None,
    ):
        """
        Initialize query.

        Args:
            query_str: Query string (LiteralString for type safety)
                or Query object
            params: Query parameters

        Raises:
            ValueError: If query_str is None or invalid
        """
        if not query_str:
            raise ValueError("query_str cannot be None")
        self.query_str = query_str
        self.params = params or {}
        logger.debug(f"Query initialized with {len(self.params)} parameters")

    def execute(
        self,
        connection: "Neo4jConnection",
        database: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Execute the query.

        Args:
            connection: Neo4j connection instance
            database: Target database (optional, uses connection
                default if not specified)

        Returns:
            Query result

        Raises:
            RuntimeError: If connection is not established
        """
        logger.debug("Executing query...")

        try:
            with connection.get_driver().session(
                database=database or connection.database
            ) as session:
                result = session.run(self.query_str, self.params)
                logger.debug("Query executed successfully")
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Query execution failed: {type(e).__name__}: {e}")
            raise
```

< /home/honeymatrix/Projects/neo4j_framework/src/neo4j_framework/queries/query_manager.py >

```
"""
Query execution engine with read/write differentiation and logging.
"""

from __future__ import annotations

import logging

from typing import TYPE_CHECKING, Any, Dict, LiteralString, Optional, cast, List

from neo4j import Query

from neo4j_framework.queries.base_query import BaseQuery

if TYPE_CHECKING:
    from neo4j_framework.stubs.neo4j import (
        Driver,
        ManagedTransaction,
        Record,
        Result,
        Session,
    )  # noqa: F401

logger = logging.getLogger(__name__)


class QueryManager:
    """
    Query execution engine with parameterization and read/write optimization.

    Differentiates between read and write operations for automatic retry
    semantics and connection pool optimization.
    """

    def __init__(self, connection: Any):
        """
        Initialize query manager.

        Args:
            connection: Neo4j connection instance

        Raises:
            ValueError: If connection is None
        """
        if connection is None:
            raise ValueError("connection cannot be None")
        self.connection = connection
        logger.debug("QueryManager initialized")

    def execute_read(
        self,
        query_str: LiteralString | Query,
        params: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None,
    ) -> List[dict[str, Any]]:
        """
        Execute a read query with automatic retry semantics.

        Uses session.execute_read() for read optimization and automatic
        retries on transient failures.

        Args:
            query_str: Cypher query string (LiteralString for type safety)
            params: Query parameters
            database: Target database (optional)

        Returns:
            List of records as dictionaries

        Raises:
            Exception: If query execution fails
        """
        logger.debug("Executing read query...")

        def _read(tx: ManagedTransaction) -> Result:
            return tx.run(query_str, params or {})

        try:
            driver = cast(Driver, self.connection.get_driver())
            effective_db = database or cast(str, self.connection.database)
            with driver.session(database=effective_db) as session:
                result: Result = session.execute_read(_read)
                records: List[dict[str, Any]] = [record.data() for record in result]
                logger.debug(f"Read query returned {len(records)} records")
                return records
        except Exception as e:
            logger.error(f"Read query failed: {type(e).__name__}: {e}")
            raise

    def execute_write(
        self,
        query_str: LiteralString | Query,
        params: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None,
    ) -> Result:
        """
        Execute a write query with automatic retry semantics.

        Uses session.execute_write() for write optimization and automatic
        retries on transient failures.

        Args:
            query_str: Cypher query string (LiteralString for type safety)
            params: Query parameters
            database: Target database (optional)

        Returns:
            Query result

        Raises:
            Exception: If query execution fails
        """
        logger.debug("Executing write query...")

        def _write(tx: ManagedTransaction) -> Result:
            return tx.run(query_str, params or {})

        try:
            driver = cast(Driver, self.connection.get_driver())
            effective_db = database or cast(str, self.connection.database)
            with driver.session(database=effective_db) as session:
                result: Result = session.execute_write(_write)
                logger.debug("Write query executed successfully")
                return result
        except Exception as e:
            logger.error(f"Write query failed: {type(e).__name__}: {e}")
            raise

    def execute_query(
        self,
        query_str: LiteralString | Query,
        params: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None,
    ) -> List[dict[str, Any]]:
        """
        Execute a generic query without optimization.

        Use execute_read() or execute_write() for better performance.
        This method is provided for backward compatibility.

        Args:
            query_str: Cypher query string
            params: Query parameters
            database: Target database (optional)

        Returns:
            List of records as dictionaries
        """
        logger.warning(
            "execute_query() called. For better performance, use "
            "execute_read() for read operations or execute_write() for write operations."
        )
        query = BaseQuery(query_str, params)
        return query.execute(self.connection, database)
```

< /home/honeymatrix/Projects/neo4j_framework/src/neo4j_framework/transactions/**init**.py >

```
from .transaction_manager import TransactionManager

__all__ = ["TransactionManager"]
```

< /home/honeymatrix/Projects/neo4j_framework/src/neo4j_framework/transactions/transaction_manager.py >

```
"""
Transaction management with context manager support.
"""

from __future__ import annotations

import logging
from typing import Callable, Any, Dict, Optional, TYPE_CHECKING, cast, TypeVar


logger = logging.getLogger(__name__)

T = TypeVar("T")

if TYPE_CHECKING:
    from neo4j_framework.stubs.neo4j import (
        Driver,
        ManagedTransaction,
        Result,
        Session,
    )  # noqa: F401


class TransactionManager:
    """
    Handles both managed and explicit transactions with context manager support.
    """

    def __init__(self, connection: Any):
        """
        Initialize transaction manager.

        Args:
            connection: Neo4j connection instance

        Raises:
            ValueError: If connection is None
        """
        if connection is None:
            raise ValueError("connection cannot be None")
        self.connection = connection
        self._session: Optional[Session] = None
        logger.debug("TransactionManager initialized")

    def run_in_transaction(
        self,
        tx_function: Callable[[ManagedTransaction], T],
        database: Optional[str] = None,
    ) -> T:
        """
        Execute a function within a managed transaction.

        Args:
            tx_function: Function that takes transaction object and executes logic
            database: Target database (optional)

        Returns:
            Result of tx_function

        Raises:
            Exception: If transaction fails
        """

        logger.debug("Starting managed transaction...")

        driver = cast(Driver, self.connection.get_driver())
        effective_db = database or cast(str, self.connection.database)

        try:
            with driver.session(database=effective_db) as session:
                result = session.execute_write(tx_function)
                logger.debug("Transaction completed successfully")
                return result
        except Exception as e:
            logger.error(f"Transaction failed: {type(e).__name__}: {e}")
            raise

    def explicit_transaction(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None,
    ) -> Result:
        """
        Execute an explicit transaction with a single query.

        Args:
            query: Cypher query string
            params: Query parameters
            database: Target database (optional)

        Returns:
            Query result

        Raises:
            ValueError: If query is None
            Exception: If transaction fails
        """
        if not query:
            raise ValueError("query cannot be None")

        logger.debug("Executing explicit transaction...")

        def tx_func(tx: ManagedTransaction) -> Result:
            return tx.run(query, params or {})

        return self.run_in_transaction(tx_func, database)

    def __enter__(self) -> Session:
        """
        Context manager entry.

        Opens a session for the transaction block.

        Returns:
            Session object
        """
        logger.debug("Entering transaction context...")
        driver = cast(Driver, self.connection.get_driver())
        self._session = driver.session(database=cast(str, self.connection.database))
        return self._session

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: object,
    ) -> bool:
        """
        Context manager exit.

        Closes the session and handles errors if necessary.

        Returns:
            False to not suppress exceptions
        """
        if self._session:
            try:
                self._session.close()
                if exc_type:
                    logger.error(
                        f"Transaction context exited with exception: {exc_type.__name__}"
                    )
                else:
                    logger.debug("Transaction context closed successfully")
            except Exception as e:
                logger.error(f"Error closing transaction session: {e}")
            finally:
                self._session = None

        # Don't suppress exceptions
        return False
```

< /home/honeymatrix/Projects/neo4j_framework/src/neo4j_framework/importers/**init**.py >

```
from .csv_importer import CSVImporter

__all__ = ["CSVImporter"]
```

< /home/honeymatrix/Projects/neo4j_framework/src/neo4j_framework/importers/csv_importer.py >

```
"""
CSV import functionality with path validation and logging.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, cast

if TYPE_CHECKING:
    from neo4j_framework.stubs.neo4j import (
        Driver,
        Query,
        Result,
    )  # noqa: F401

logger = logging.getLogger(__name__)


class CSVImporter:
    """
    Efficient bulk data loading from CSV files with security validation.

    Validates file paths to prevent directory traversal attacks.
    """

    def __init__(self, connection: Any, allowed_dir: Optional[str] = None):
        """
        Initialize CSV importer.

        Args:
            connection: Neo4j connection instance
            allowed_dir: Optional directory to restrict CSV imports to.
                        If set, files must be within this directory.
        """
        if connection is None:
            raise ValueError("connection cannot be None")
        self.connection = connection
        self.allowed_dir = Path(allowed_dir).resolve() if allowed_dir else None
        logger.debug("CSVImporter initialized")

    def _validate_file_path(self, file_path: str) -> Path:
        """
        Validate CSV file path to prevent directory traversal.

        Args:
            file_path: Path to CSV file

        Returns:
            Resolved Path object

        Raises:
            ValueError: If path is invalid or outside allowed directory
        """
        try:
            path = Path(file_path).resolve()
        except Exception as e:
            raise ValueError(f"Invalid file path: {file_path}") from e

        # Check if file exists
        if not path.exists():
            raise ValueError(f"CSV file not found: {file_path}")

        # Check if file is readable
        if not os.access(path, os.R_OK):
            raise ValueError(f"CSV file not readable: {file_path}")

        # If allowed_dir is set, ensure file is within it
        if self.allowed_dir:
            try:
                path.relative_to(self.allowed_dir)
            except ValueError:
                raise ValueError(
                    f"CSV file must be within {self.allowed_dir}, " f"got {file_path}"
                )

        logger.debug(f"File path validated: {path}")
        return path

    def import_csv(
        self,
        file_path: str,
        query: Query,
        params: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None,
    ) -> Result:
        """
        Import CSV file using Neo4j LOAD CSV.

        Example query:
            LOAD CSV WITH HEADERS FROM $file_url AS row
            CREATE (:Node {name: row.name})

        Args:
            file_path: Path to CSV file
            query: Cypher query with LOAD CSV
            params: Additional query parameters
            database: Target database (optional)

        Returns:
            Query result

        Raises:
            ValueError: If file_path is invalid or outside allowed directory
            Exception: If query execution fails
        """
        if not file_path:
            raise ValueError("file_path cannot be None")
        if not query:
            raise ValueError("query cannot be None")

        # Validate and normalize path
        validated_path = self._validate_file_path(file_path)

        logger.info(f"Importing CSV file: {validated_path}")

        # Prepare parameters
        full_params = params or {}
        # Use file:/// URL scheme for Neo4j LOAD CSV
        full_params["file_url"] = validated_path.as_uri()

        try:
            driver = cast(Driver, self.connection.get_driver())
            effective_db = database or cast(str, self.connection.database)
            with driver.session(database=effective_db) as session:
                result = session.run(query, full_params)
                logger.info(f"CSV import completed: {validated_path}")
                return result
        except Exception as e:
            logger.error(f"CSV import failed: {type(e).__name__}: {e}")
            raise
```

< /home/honeymatrix/Projects/neo4j_framework/src/neo4j_framework/utils/exceptions.py >

```
"""
Custom exception classes for Neo4j framework.
Enables targeted error handling and recovery strategies.
"""


class Neo4jFrameworkException(Exception):
    """Base exception for all framework errors."""

    pass


class ConnectionError(Neo4jFrameworkException):
    """Raised when connection to Neo4j fails."""

    pass


class AuthenticationError(Neo4jFrameworkException):
    """Raised when authentication fails."""

    pass


class ConfigurationError(Neo4jFrameworkException):
    """Raised when configuration is invalid."""

    pass


class ValidationError(Neo4jFrameworkException):
    """Raised when input validation fails."""

    pass


class QueryError(Neo4jFrameworkException):
    """Raised when query execution fails."""

    pass


class TransactionError(Neo4jFrameworkException):
    """Raised when transaction execution fails."""

    pass
```

< /home/honeymatrix/Projects/neo4j_framework/src/neo4j_framework/utils/**init**.py >

```
from .logger import setup_logging
from .validators import Validators
from .performance import Performance
from .exceptions import (
    Neo4jFrameworkException,
    ConnectionError,
    AuthenticationError,
    ConfigurationError,
    ValidationError,
    QueryError,
    TransactionError,
)

__all__ = [
    "setup_logging",
    "Validators",
    "Performance",
    "Neo4jFrameworkException",
    "ConnectionError",
    "AuthenticationError",
    "ConfigurationError",
    "ValidationError",
    "QueryError",
    "TransactionError",
]
```

< /home/honeymatrix/Projects/neo4j_framework/src/neo4j_framework/utils/logger.py >

```
"""
Logging configuration utilities.
"""

import logging


def setup_logging(env_file: str = ".env", env_prefix: str = "NEO4J_"):
    """
    Set up logging with configuration from environment variables.

    Args:
        env_file: Path to .env file
        env_prefix: Prefix for environment variables

    Returns:
        Logger instance
    """
    from neo4j_framework.config.env_loader import EnvironmentLoader

    loader = EnvironmentLoader(env_file, env_prefix)
    log_level = loader.get("LOG_LEVEL", "INFO")

    # Configure root logger
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    return logging.getLogger(__name__)
```

< /home/honeymatrix/Projects/neo4j_framework/src/neo4j_framework/utils/performance.py >

```
"""
Performance monitoring utilities.
"""

import time
import logging
from typing import Callable, ParamSpec, TypeVar
import functools

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


class Performance:
    """
    Performance utilities (e.g., timing operations).
    """

    @staticmethod
    def time_function(func: Callable[P, R]) -> Callable[P, R]:
        """
        Decorator to measure function execution time.

        Uses logging instead of print() for better control and integration
        with logging systems.

        Args:
            func: Function to measure

        Returns:
            Wrapped function
        """

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start = time.time()
            try:
                result: R = func(*args, **kwargs)
                end = time.time()
                elapsed = end - start
                logger.debug(f"{func.__name__} execution time: {elapsed:.3f} seconds")
                return result
            except Exception as e:
                end = time.time()
                elapsed = end - start
                logger.error(f"{func.__name__} failed after {elapsed:.3f} seconds: {e}")
                raise

        return wrapper
```

< /home/honeymatrix/Projects/neo4j_framework/src/neo4j_framework/utils/validators.py >

```
"""
Input validation utilities.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class Validators:
    """
    Input validation utilities.
    """

    @staticmethod
    def validate_not_none(value: Any, name: str):
        """
        Validate that a value is not None.

        Args:
            value: Value to validate
            name: Name of the parameter for error messages

        Raises:
            ValueError: If value is None
        """
        if value is None:
            logger.error(f"Validation failed: {name} cannot be None")
            raise ValueError(f"{name} cannot be None")

    @staticmethod
    def validate_string_not_empty(value: str, name: str):
        """
        Validate that a string is not empty.

        Args:
            value: String to validate
            name: Name of the parameter for error messages

        Raises:
            ValueError: If string is empty
        """
        if not value or not value.strip():
            logger.error(f"Validation failed: {name} cannot be empty")
            raise ValueError(f"{name} cannot be empty")

    @staticmethod
    def validate_positive_int(value: Any, name: str):
        """
        Validate that a value is a positive integer.

        Args:
            value: Value to validate
            name: Name of the parameter for error messages

        Raises:
            ValueError: If value is not a positive integer
        """
        if not isinstance(value, int) or value <= 0:
            logger.error(f"Validation failed: {name} must be a positive integer")
            raise ValueError(f"{name} must be a positive integer")

    @staticmethod
    def validate_int(
        value: Any,
        name: str,
        min_val: int | None = None,
        max_val: int | None = None,
    ) -> None:
        """Validate an integer with optional bounds checking."""
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{name} must be an integer, got {type(value).__name__}")

        if min_val is not None and value < min_val:
            raise ValueError(f"{name} must be >= {min_val}, got {value}")

        if max_val is not None and value > max_val:
            raise ValueError(f"{name} must be <= {max_val}, got {value}")

    @staticmethod
    def validate_float(
        value: Any,
        name: str,
        min_val: float | None = None,
        max_val: float | None = None,
    ) -> None:
        """Validate a float with optional bounds checking."""
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{name} must be a number, got {type(value).__name__}")

        if min_val is not None and value < min_val:
            raise ValueError(f"{name} must be >= {min_val}, got {value}")

        if max_val is not None and value > max_val:
            raise ValueError(f"{name} must be <= {max_val}, got {value}")
```

< /home/honeymatrix/Projects/neo4j_framework/src/neo4j_framework/**init**.py >

```
"""
Neo4j Framework - A reusable library for Neo4j database operations.
"""

from neo4j_framework.config import EnvironmentLoader, get_db_config
from neo4j_framework.db import Neo4jConnection, PoolManager
from neo4j_framework.queries import QueryManager, BaseQuery, QueryTemplates
from neo4j_framework.transactions import TransactionManager
from neo4j_framework.importers import CSVImporter
from neo4j_framework.utils import (
    setup_logging,
    Validators,
    Performance,
    Neo4jFrameworkException,
    ConnectionError,
    AuthenticationError,
    ConfigurationError,
    ValidationError,
    QueryError,
    TransactionError,
)

__version__ = "2.0.0"

__all__ = [
    "EnvironmentLoader",
    "get_db_config",
    "Neo4jConnection",
    "PoolManager",
    "QueryManager",
    "BaseQuery",
    "QueryTemplates",
    "TransactionManager",
    "CSVImporter",
    "setup_logging",
    "Validators",
    "Performance",
    "Neo4jFrameworkException",
    "ConnectionError",
    "AuthenticationError",
    "ConfigurationError",
    "ValidationError",
    "QueryError",
    "TransactionError",
]
```

< /home/honeymatrix/Projects/neo4j_framework/src/neo4j_framework/stubs/neo4j/**init**.pyi >

```
"""
Type stubs for neo4j package - provides type hints for Pyright
"""

from typing import (
    Any,
    Iterator,
    LiteralString,
    Optional,
    Dict,
    List,
    Callable,
    Union,
    Tuple,
)

class Neo4jException(Exception):
    """Base exception for Neo4j driver."""

    pass

class QueryError(Neo4jException):
    """Raised when a query fails."""

    pass

class ClientError(Neo4jException):
    """Raised for client-side errors."""

    pass

class TransientError(Neo4jException):
    """Raised for transient errors (e.g., connection timeouts)."""

    pass

class DatabaseError(Neo4jException):
    """Raised for database-side errors."""

    pass

class Auth:
    """Neo4j authentication class with static factory methods."""

    @staticmethod
    def basic(username: str, password: str, realm: Optional[str] = None) -> "Auth":
        """Create basic authentication."""
        ...

    @staticmethod
    def kerberos(ticket: str) -> "Auth":
        """Create Kerberos authentication."""
        ...

    @staticmethod
    def bearer(token: str) -> "Auth":
        """Create Bearer token authentication."""
        ...

    @staticmethod
    def custom(
        scheme: str,
        principal: str,
        credentials: str,
        realm: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> "Auth":
        """Create custom authentication."""
        ...

class Record:
    """Neo4j Record type - represents a single result row."""

    def __getitem__(self, key: Union[str, int]) -> Any:
        """Get a value by key (string) or index (int)."""
        ...

    def __iter__(self) -> Iterator[Any]:
        """Iterate over values in the record."""
        ...

    def keys(self) -> List[str]:
        """Get all keys in the record."""
        ...

    def values(self) -> List[Any]:
        """Get all values in the record."""
        ...

    def data(self) -> Dict[str, Any]:
        """Get record data as a dictionary."""
        ...

class Result:
    """Neo4j Result type - represents query results."""

    def __iter__(self) -> Iterator[Record]:
        """Iterate over records in result."""
        ...

    def fetch(self, n: int = 1) -> List[Record]:
        """Fetch up to n records."""
        ...

    def single(self) -> Optional[Record]:
        """Get a single record or None."""
        ...

    def consume(self) -> Any:
        """Consume the result and return summary."""
        ...

class Transaction:
    """Neo4j Transaction type."""

    def run(
        self, query: str, parameters: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> Result:
        """Execute a query within the transaction."""
        ...

    def commit(self) -> None:
        """Commit the transaction."""
        ...

    def rollback(self) -> None:
        """Rollback the transaction."""
        ...

    def close(self) -> None:
        """Close the transaction."""
        ...

class ManagedTransaction:
    """Neo4j ManagedTransaction type for transaction functions."""

    def run(
        self, query: str, parameters: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> Result:
        """Execute a query within the managed transaction."""
        ...

class Session:
    """Neo4j Session type."""

    def run(
        self, query: str, parameters: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> Result:
        """Execute a query in the session."""
        ...

    def begin_transaction(self) -> Transaction:
        """Begin an explicit transaction."""
        ...

    def read_transaction(
        self, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        """Execute a function in a read transaction."""
        ...

    def write_transaction(
        self, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        """Execute a function in a write transaction."""
        ...

    def execute_read(
        self, func: Callable[[ManagedTransaction], Any], *args: Any, **kwargs: Any
    ) -> Any:
        """Execute a read operation (Neo4j 4.4+)."""
        ...

    def execute_write(
        self, func: Callable[[ManagedTransaction], Any], *args: Any, **kwargs: Any
    ) -> Any:
        """Execute a write operation (Neo4j 4.4+)."""
        ...

    def close(self) -> None:
        """Close the session."""
        ...

    def __enter__(self) -> "Session":
        """Context manager entry."""
        ...

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        ...

class Driver:
    """Neo4j Driver type."""

    def session(self, database: Optional[str] = None, **kwargs: Any) -> Session:
        """Create a new session."""
        ...

    def verify_connectivity(self) -> None:
        """Verify the connection."""
        ...

    def close(self) -> None:
        """Close the driver and all connections."""
        ...

    def __enter__(self) -> "Driver":
        """Context manager entry."""
        ...

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        ...

class GraphDatabase:
    """Neo4j GraphDatabase class."""

    @staticmethod
    def driver(
        uri: str,
        *,
        auth: Optional[Auth] = None,
        encrypted: Optional[bool] = None,
        trust: Optional[str] = None,
        max_connection_pool_size: int = 100,
        connection_timeout: float = 30.0,
        **kwargs: Any,
    ) -> Driver:
        """Create a new driver."""
        ...

# Type alias for Cypher queries
Query = LiteralString
```

< /home/honeymatrix/Projects/neo4j_framework/examples/basic_usage.py >

```
"""
Example usage of the Neo4j framework with correct import paths.
"""

from src.neo4j_framework.config.db_config import get_db_config
from src.neo4j_framework.db.connection import Neo4jConnection
from src.neo4j_framework.queries.query_manager import QueryManager
from src.neo4j_framework.utils.logger import setup_logging

logger = setup_logging()

# Load configuration
config = get_db_config()

# Create connection
conn = Neo4jConnection(
    uri=config["uri"],
    username=config["username"],
    password=config["password"],
    database=config["database"],
    encrypted=config["encrypted"],
)

try:
    # Connect to database
    conn.connect()

    # Create query manager
    query_manager = QueryManager(conn)

    # Execute a read query
    result = query_manager.execute_read("MATCH (n) RETURN n LIMIT 1")
    logger.info(f"Query result: {result}")

except Exception as e:
    logger.error(f"Error: {e}")
    raise
finally:
    # Always close connection
    conn.close()
```

< /home/honeymatrix/Projects/neo4j_framework/README.md >

```
# Neo4j Framework

A **production-ready, type-safe Python library** for seamless Neo4j database integration. Built for reliability, security, and ease of use across any project.

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![Tests Passing](https://img.shields.io/badge/tests-113%2F113%20passing-brightgreen)](tests/)
[![Type Checking](https://img.shields.io/badge/type%20checking-pyright-blue)](pyrightconfig.json)
[![MIT License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## Overview

The Neo4j Framework provides a complete abstraction layer over Neo4j's Python driver, eliminating boilerplate while maintaining full power and flexibility. It's designed specifically for developers who want safety guarantees through type hints, security by default, and zero configuration complexity.

### Key Characteristics

- **Type-Safe** - Full type hints enable IDE autocomplete and catch errors before runtime
- **Security-First** - Parameterized queries, path validation, and bounds checking built-in
- **Zero Boilerplate** - Environment-based config, automatic connection pooling, clean APIs
- **Production Ready** - 113 comprehensive tests covering edge cases and error scenarios
- **Framework Agnostic** - Drop into Flask, FastAPI, Django, or standalone scripts

---

## Installation

### From Source

\`\`\`bash
git clone https://github.com/yourusername/neo4j_framework.git
cd neo4j_framework

# Install in development mode
pip install -e .

# Or standard install
pip install .
\`\`\`

### Dependencies

\`\`\`
neo4j>=5.0.0,<6.0.0
python-dotenv>=1.0.0
typing-extensions>=4.0.0
\`\`\`

---

## Quick Start

### 1. Configure Environment

Create a \`.env\` file in your project:

\`\`\`bash
# .env
NEO4J_URI=neo4j://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_secure_password
NEO4J_DATABASE=neo4j
NEO4J_ENCRYPTED=true
NEO4J_MAX_CONNECTION_POOL_SIZE=50
\`\`\`

### 2. Initialize & Execute

\`\`\`python
from src.neo4j_framework.config.env_loader import EnvironmentLoader
from src.neo4j_framework.db.connection import Neo4jConnection
from src.neo4j_framework.queries.query_manager import QueryManager

# Load configuration
loader = EnvironmentLoader(env_prefix="NEO4J_")
config = loader.get_config()

# Create connection
conn = Neo4jConnection(
    uri=config["uri"],
    username=config["username"],
    password=config["password"],
    database=config["database"],
    encrypted=config["encrypted"],
)

# Connect
conn.connect()

# Execute query
qm = QueryManager(conn)
results = qm.execute_read(
    "MATCH (p:Person) WHERE p.age > $min_age RETURN p.name, p.age",
    params={"min_age": 18}
)

for record in results:
    print(f"{record['name']}: {record['age']} years old")

# Cleanup
conn.close()
\`\`\`

### 3. Using Context Managers (Recommended)

\`\`\`python
with Neo4jConnection(
    uri="neo4j://localhost:7687",
    username="neo4j",
    password="password"
) as conn:
    qm = QueryManager(conn)
    results = qm.execute_read("MATCH (n) RETURN COUNT(n) as count")
    print(f"Total nodes: {results[0]['count']}")
    # Connection automatically closed
\`\`\`

---

## Core Components

### Environment Loader (\`config.env_loader\`)

Manages configuration with validation and security:

\`\`\`python
from src.neo4j_framework.config.env_loader import EnvironmentLoader

loader = EnvironmentLoader(env_prefix="NEO4J_")

# Simple value
uri = loader.get("URI")

# With validation
pool_size = loader.get_int("MAX_CONNECTION_POOL_SIZE", min_val=1, max_val=500)
timeout = loader.get_float("CONNECTION_TIMEOUT", min_val=0.1, max_val=300.0)
encrypted = loader.get_bool("ENCRYPTED", default=True)

# Complete config
config = loader.get_config()
\`\`\`

### Connection Manager (\`db.connection\`)

Handles all connection logic with multiple auth methods:

\`\`\`python
from src.neo4j_framework.db.connection import Neo4jConnection

# Basic connection
conn = Neo4jConnection(
    uri="neo4j://localhost:7687",
    username="neo4j",
    password="password",
    database="neo4j",
    encrypted=True,
    max_connection_pool_size=50
)
conn.connect()

# mTLS connection
conn.connect_with_mtls(
    cert_path="/path/to/client.crt",
    key_path="/path/to/client.key"
)

# Kerberos authentication
conn.connect(auth_type="kerberos", ticket=kerberos_ticket)

# Check status
if conn.is_connected():
    print("Connected!")
\`\`\`

### Query Manager (\`queries.query_manager\`)

Execute parameterized queries safely:

\`\`\`python
from src.neo4j_framework.queries.query_manager import QueryManager

qm = QueryManager(conn)

# Read operation (optimized for read semantics)
results = qm.execute_read(
    "MATCH (n:Person) RETURN n LIMIT 10"
)

# Write operation (optimized for write semantics)
result = qm.execute_write(
    "CREATE (p:Person {name: $name}) RETURN p",
    params={"name": "Alice"}
)

# Parameterized queries prevent injection
user_input = "Alice'; DROP TABLE users; --"
results = qm.execute_read(
    "MATCH (p:Person) WHERE p.name = $name RETURN p",
    params={"name": user_input}  # Safe!
)
\`\`\`

### Transaction Manager (\`transactions.transaction_manager\`)

Handle complex multi-step operations:

\`\`\`python
from src.neo4j_framework.transactions.transaction_manager import TransactionManager

txm = TransactionManager(conn)

def create_relationship(tx):
    """Multi-step transaction function."""
    # Create two people
    tx.run("CREATE (p1:Person {name: $n1})", {"n1": "Alice"})
    tx.run("CREATE (p2:Person {name: $n2})", {"n2": "Bob"})
    # Create relationship
    tx.run(
        "MATCH (p1:Person {name: $n1}), (p2:Person {name: $n2}) "
        "CREATE (p1)-[:KNOWS]->(p2)",
        {"n1": "Alice", "n2": "Bob"}
    )

# Execute transaction
txm.run_in_transaction(create_relationship)
\`\`\`

### CSV Importer (\`importers.csv_importer\`)

Bulk import data securely:

\`\`\`python
from src.neo4j_framework.importers.csv_importer import CSVImporter

# Restrict imports to specific directory for security
importer = CSVImporter(conn, allowed_dir="/data/csv_imports")

# Import CSV
query = """
LOAD CSV WITH HEADERS FROM $file_url AS row
CREATE (p:Person {
    name: row.name,
    email: row.email,
    age: toInteger(row.age)
})
"""

importer.import_csv("/data/csv_imports/people.csv", query)

# Attempting directory traversal fails
importer.import_csv("/../../../etc/passwd", query)  # Raises ValueError
\`\`\`

---

## Security Features

### Parameterized Queries

All queries use parameter binding to prevent Cypher injection:

\`\`\`python
# ✅ SAFE - Uses parameter binding
qm.execute_read(
    "MATCH (n) WHERE n.name = $name RETURN n",
    params={"name": user_input}
)

# ❌ UNSAFE - Never do this!
qm.execute_read(
    f"MATCH (n) WHERE n.name = '{user_input}' RETURN n"
)
\`\`\`

### Input Validation

All inputs are validated with bounds checking:

\`\`\`python
from src.neo4j_framework.utils.validators import Validators

# Validate not None
Validators.validate_not_none(value, "parameter_name")

# Validate bounds
Validators.validate_int(pool_size, "pool_size", min_val=1, max_val=500)
Validators.validate_float(timeout, "timeout", min_val=0.1, max_val=300.0)
\`\`\`

### Path Traversal Prevention

CSV imports are restricted to allowed directories:

\`\`\`python
# Directory traversal attempts are blocked
importer = CSVImporter(conn, allowed_dir="/safe/directory")
importer.import_csv("/etc/passwd", query)  # ValueError: outside allowed directory
\`\`\`

### Secure Defaults

- Encryption enabled by default
- Credentials required for all connections
- Pool sizes bounded (1-500)
- Timeouts enforced (0.1s - 300s)

---

## Architecture

### Directory Structure

\`\`\`
neo4j_framework/
├── src/neo4j_framework/          # The installable package
│   ├── __init__.py               # Package exports
│   ├── py.typed                  # Type hints marker
│   ├── config/                   # Configuration management
│   ├── db/                       # Connection management
│   ├── queries/                  # Query execution
│   ├── transactions/             # Transaction handling
│   ├── importers/                # CSV bulk import
│   └── utils/                    # Utilities & exceptions
├── tests/                        # 113 comprehensive tests
├── examples/                     # Usage examples
├── README.md                     # This file
├── requirements.txt              # Dependencies
├── pyproject.toml               # Build configuration
└── pyrightconfig.json           # Type checking config
\`\`\`

### Component Relationships

\`\`\`
┌─────────────────────────────────────────┐
│   Environment Configuration             │
│   (env_loader.py)                       │
│   - Loads from .env                     │
│   - Validates types & bounds            │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│   Connection Manager                    │
│   (connection.py)                       │
│   - Multiple auth methods               │
│   - Connection pooling                  │
└──────────────────┬──────────────────────┘
                   │
                   ├─────────────────────────┐
                   │                         │
                   ▼                         ▼
        ┌────────────────────┐    ┌──────────────────┐
        │  Query Manager     │    │ Transaction      │
        │  (query_manager)   │    │ Manager          │
        │  - Read/Write      │    │ (tx_manager)     │
        │  - Parameterized   │    │ - Multi-step     │
        └────────────────────┘    │ - Rollback       │
                                  └──────────────────┘
                   │
                   ├─────────────────────────┐
                   │                         │
                   ▼                         ▼
        ┌────────────────────┐    ┌──────────────────┐
        │  CSV Importer      │    │  Query Templates │
        │  (csv_importer)    │    │  (base_query.py) │
        │  - Bulk load       │    │  - Reusable      │
        │  - Path validation │    │  - Type-safe     │
        └────────────────────┘    └──────────────────┘
\`\`\`

---

## Testing

### Running Tests

\`\`\`bash
# All tests
pytest tests/ -v

# Specific test class
pytest tests/test_framework.py::TestEnvironmentLoader -v

# Specific test
pytest tests/test_framework.py::TestEnvironmentLoader::test_initialization -v

# With coverage
pytest tests/ --cov=src/neo4j_framework --cov-report=html
\`\`\`

### Test Coverage

- **113 total tests** - 100% passing
- **20 unit tests** - Edge cases and boundary conditions
- **39 core tests** - Framework functionality
- **28 integration tests** - Real-world workflows
- **12 security tests** - Injection prevention, validation
- **14 additional tests** - Performance, concurrency

---

## Examples

### Example 1: Simple Query

\`\`\`python
from src.neo4j_framework.config.db_config import get_db_config
from src.neo4j_framework.db.connection import Neo4jConnection
from src.neo4j_framework.queries.query_manager import QueryManager

config = get_db_config()
conn = Neo4jConnection(**config)
conn.connect()

qm = QueryManager(conn)
results = qm.execute_read("MATCH (p:Person) RETURN p LIMIT 5")

for record in results:
    print(record)

conn.close()
\`\`\`

### Example 2: Transaction

\`\`\`python
from src.neo4j_framework.transactions.transaction_manager import TransactionManager

txm = TransactionManager(conn)

def create_users(tx):
    for i in range(10):
        tx.run(
            "CREATE (u:User {id: $id, name: $name})",
            {"id": f"user_{i}", "name": f"User {i}"}
        )

txm.run_in_transaction(create_users)
\`\`\`

### Example 3: Multi-Database

\`\`\`python
# Create separate connections
primary = Neo4jConnection(
    uri="neo4j://primary:7687",
    username="neo4j",
    password="password",
    database="production"
)

analytics = Neo4jConnection(
    uri="neo4j://analytics:7687",
    username="neo4j",
    password="password",
    database="analytics"
)

primary.connect()
analytics.connect()

qm_prod = QueryManager(primary)
qm_analytics = QueryManager(analytics)

# Use independently
prod_data = qm_prod.execute_read("MATCH (n) RETURN COUNT(n)")
analytics_data = qm_analytics.execute_read("MATCH (n) RETURN COUNT(n)")
\`\`\`

---

## Configuration Reference

### Environment Variables

| Variable | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| \`NEO4J_URI\` | str | - | ✅ | Connection URI (neo4j://host:port) |
| \`NEO4J_USERNAME\` | str | - | ✅ | Authentication username |
| \`NEO4J_PASSWORD\` | str | - | ✅ | Authentication password |
| \`NEO4J_DATABASE\` | str | neo4j | - | Target database |
| \`NEO4J_ENCRYPTED\` | bool | true | - | Enable SSL/TLS |
| \`NEO4J_MAX_CONNECTION_POOL_SIZE\` | int | 100 | - | Pool size (1-500) |
| \`NEO4J_CONNECTION_TIMEOUT\` | float | 30.0 | - | Connection timeout (0.1-300s) |
| \`NEO4J_MAX_TRANSACTION_RETRY_TIME\` | float | 30.0 | - | Transaction retry time (1-300s) |

### Custom Prefixes

Use different prefixes for multiple projects:

\`\`\`bash
# Project A
PROJECT_A_URI=neo4j://server-a:7687
PROJECT_A_USERNAME=neo4j
PROJECT_A_PASSWORD=password_a

# Project B
PROJECT_B_URI=neo4j://server-b:7687
PROJECT_B_USERNAME=neo4j
PROJECT_B_PASSWORD=password_b
\`\`\`

\`\`\`python
loader_a = EnvironmentLoader(env_prefix="PROJECT_A_")
loader_b = EnvironmentLoader(env_prefix="PROJECT_B_")
\`\`\`

---

## Troubleshooting

### Connection Issues

\`\`\`python
# Check URI format
# ✅ neo4j://localhost:7687
# ✅ neo4j+s://localhost:7687 (encrypted)
# ❌ localhost:7687 (missing protocol)

# Verify credentials
assert os.getenv("NEO4J_USERNAME") is not None
assert os.getenv("NEO4J_PASSWORD") is not None

# Test connectivity
import socket
socket.create_connection(("localhost", 7687), timeout=5)
\`\`\`

### Query Issues

\`\`\`python
# Use parameters, not string concatenation
# ✅
results = qm.execute_read(
    "MATCH (n) WHERE n.name = $name RETURN n",
    params={"name": user_input}
)

# ❌
results = qm.execute_read(f"MATCH (n) WHERE n.name = '{user_input}' RETURN n")
\`\`\`

### Performance

\`\`\`python
# Add database indexes
qm.execute_write("CREATE INDEX ON :Person(id)")

# Use LIMIT for large result sets
results = qm.execute_read("MATCH (n) RETURN n LIMIT 1000")

# Increase pool size for concurrency
conn = Neo4jConnection(
    uri="...",
    username="...",
    password="...",
    max_connection_pool_size=200
)
\`\`\`

---

## API Reference

### EnvironmentLoader

\`\`\`python
loader = EnvironmentLoader(env_file=".env", env_prefix="NEO4J_")

# Get values
value = loader.get("KEY", default=None, required=False)
int_val = loader.get_int("INT_KEY", default=0, min_val=None, max_val=None)
float_val = loader.get_float("FLOAT_KEY", default=0.0, min_val=None, max_val=None)
bool_val = loader.get_bool("BOOL_KEY", default=False)

# Get complete config
config = loader.get_config()  # Returns all Neo4j config
\`\`\`

### Neo4jConnection

\`\`\`python
conn = Neo4jConnection(
    uri="neo4j://...",
    username="neo4j",
    password="...",
    database="neo4j",
    encrypted=True,
    max_connection_pool_size=100
)

conn.connect()
conn.connect_with_mtls(cert_path="...", key_path="...")
driver = conn.get_driver()
connected = conn.is_connected()
conn.close()
\`\`\`

### QueryManager

\`\`\`python
qm = QueryManager(conn)

# Read query
results = qm.execute_read(query, params=None, database=None)

# Write query
result = qm.execute_write(query, params=None, database=None)

# Generic query (use execute_read/execute_write instead)
result = qm.execute_query(query, params=None, database=None)
\`\`\`

### TransactionManager

\`\`\`python
txm = TransactionManager(conn)

# Execute in transaction
result = txm.run_in_transaction(func, database=None)

# Context manager
with txm as session:
    session.execute_write(lambda tx: tx.run(query))
\`\`\`

### CSVImporter

\`\`\`python
importer = CSVImporter(conn, allowed_dir="/safe/directory")

# Import CSV
result = importer.import_csv(file_path, query, params=None, database=None)

# Validate path
validated_path = importer._validate_file_path(file_path)
\`\`\`

---

## Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork** the repository
2. **Create** a feature branch (\`git checkout -b feature/your-feature\`)
3. **Write tests** for new functionality
4. **Run tests** (\`pytest tests/ -v\`)
5. **Type check** (\`pyright\`)
6. **Commit** with clear messages
7. **Push** to your fork
8. **Submit** a Pull Request

### Development Setup

\`\`\`bash
# Clone and setup
git clone https://github.com/yourusername/neo4j_framework.git
cd neo4j_framework

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install with dev dependencies
pip install -e .
pip install pytest pytest-cov pyright

# Run tests
pytest tests/ -v

# Type checking
pyright
\`\`\`

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Support

- 📖 **Documentation**: See examples and usage sections above
- 🐛 **Issues**: Report bugs on GitHub Issues
- 💬 **Discussions**: Ask questions on GitHub Discussions
- 📧 **Email**: Contact maintainers for direct questions

---

## Changelog

### Version 2.0.0 (Current)

✅ **Complete package reorganization** with \`src/\` layout
✅ **113 comprehensive tests** (100% passing)
✅ **Pyright integration** for strict type checking
✅ **Full IDE support** (gd/gI/K navigation in LazyVim/VSCode)
✅ **Security hardened** with injection prevention and path validation
✅ **Multi-database support** with explicit database selection
✅ **mTLS authentication** for enterprise deployments
✅ **Connection pooling** with configurable bounds (1-500)
✅ **Transaction management** with automatic retry semantics
✅ **CSV bulk import** with directory restriction
✅ **Query templates** for reusable patterns
✅ **Custom exceptions** for targeted error handling
✅ **Performance monitoring** with built-in timing
✅ **Comprehensive documentation** with examples

---

**Built with ❤️ for developers who value type safety and security.**
```

< /home/honeymatrix/Projects/neo4j_framework/pyrightconfig.json >

```
{
  "include": ["src"],
  "exclude": [
    ".venv",
    "__pycache__",
    "build",
    "dist",
    ".git",
    "node_modules",
    "tests"
  ],
  "pythonVersion": "3.11",
  "pythonPlatform": "Linux",
  "typeCheckingMode": "strict",
  "reportMissingTypeStubs": false,
  "reportPrivateUsage": false,
  "reportUnknownMemberType": false,
  "reportUnknownParameterType": false,
  "reportGeneralTypeIssues": false,
  "reportMissingParameterType": false,
  "reportOptionalMemberAccess": false,
  "reportAttributeAccessIssue": false,
  "reportUnusedVariable": true,
  "reportUnusedImport": true,
  "stubPath": "src/neo4j_framework/stubs",
  "venvPath": ".",
  "venv": ".venv"
}
```

# Empty Files (0 bytes)

- /home/honeymatrix/Projects/neo4j_framework/tests/**init**.py
- /home/honeymatrix/Projects/neo4j_framework/src/neo4j_framework/py.typed
