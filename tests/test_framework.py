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
