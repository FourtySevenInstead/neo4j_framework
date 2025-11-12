"""
Advanced Integration Tests for Neo4j Framework (UPDATED)
tests/test_integration_advanced.py
"""

import pytest
from unittest.mock import MagicMock, patch

from neo4j_framework.queries.query_manager import QueryManager
from neo4j_framework.transactions.transaction_manager import TransactionManager
from neo4j_framework.importers.csv_importer import CSVImporter


@pytest.mark.integration
class TestFrameworkIntegration:
    """Advanced integration tests for complete workflows."""

    def test_complete_initialization_and_query_flow(self, mock_neo4j_connection):
        """Test complete initialization with query execution (UPDATED)."""
        # Create query manager with mock connection
        query_manager = QueryManager(mock_neo4j_connection)

        # Execute read query (NEW - using execute_read)
        results = query_manager.execute_read("MATCH (n) RETURN n LIMIT 1")

        assert isinstance(results, list)
        assert len(results) > 0
        assert results[0]["n"]["name"] == "TestNode"

    def test_multiple_queries_sequential(self, mock_neo4j_connection):
        """Test executing multiple queries sequentially (UPDATED)."""
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
        tm = TransactionManager(mock_neo4j_connection)

        # Define write transaction
        def write_transaction(tx):
            return tx.run("CREATE (n:TestNode {value: $val}) RETURN n", {"val": 42})

        # Execute transaction
        result = tm.run_in_transaction(write_transaction)

        assert result is not None
        # Verify execute_write was called
        mock_driver = mock_neo4j_connection._driver
        mock_driver.session.assert_called()

    def test_transaction_context_manager(self, mock_neo4j_connection):
        """Test transaction context manager support (NEW)."""
        tm = TransactionManager(mock_neo4j_connection)

        # Use context manager
        with tm as session:
            assert session is not None
            # Session is available for operations
            session.execute_write(lambda tx: tx.run("CREATE (n:Node)"))

        # After context, session should be closed
        assert tm._session is None

    def test_csv_import_workflow(self, mock_neo4j_connection, test_csv_file):
        """Test complete CSV import workflow (UPDATED)."""
        importer = CSVImporter(mock_neo4j_connection)

        # Define CSV import query
        query = (
            "LOAD CSV WITH HEADERS FROM $file_url AS row "
            "CREATE (:User {name: row.name, age: row.age})"
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
        from neo4j_framework.db.connection import Neo4jConnection

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
            "neo4j_framework.db.connection.GraphDatabase.driver",
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
        from neo4j_framework.db.connection import Neo4jConnection

        mock_driver = MagicMock()
        mock_driver.verify_connectivity.return_value = None

        with patch(
            "neo4j_framework.db.connection.GraphDatabase.driver",
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
        query_manager = QueryManager(mock_neo4j_connection)

        # Setup mock to raise error on first call, succeed on second
        mock_driver = mock_neo4j_connection._driver
        mock_session = mock_driver.session.return_value

        # First call fails, second succeeds
        mock_session.execute_read.side_effect = [
            Exception("Connection timeout"),
            [MagicMock(data=lambda: {"result": "success"})],
        ]

        # First call should raise
        with pytest.raises(Exception, match="Connection timeout"):
            query_manager.execute_read("MATCH (n) RETURN n")

        # Reset side effect for second call
        mock_session.execute_read.side_effect = None
        mock_session.execute_read.return_value = [
            MagicMock(data=lambda: {"result": "success"})
        ]

        # Second call should succeed
        result = query_manager.execute_read("MATCH (n) RETURN n")
        assert len(result) > 0

    def test_parameterized_queries_with_different_types(self, mock_neo4j_connection):
        """Test query parameterization with various data types (UPDATED)."""
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
        query_manager = QueryManager(mock_neo4j_connection)

        # Create mock records
        mock_records = [
            MagicMock(data=lambda: {"id": i, "name": f"Node{i}"}) for i in range(100)
        ]

        # Configure mock to return all records
        mock_driver = mock_neo4j_connection._driver
        mock_session = mock_driver.session.return_value
        mock_session.execute_read.return_value = mock_records

        # Execute bulk read
        results = query_manager.execute_read("MATCH (n) RETURN n")

        assert len(results) == 100

    def test_environment_configuration_override(self, clean_env):
        """Test environment configuration with custom values (UPDATED)."""
        import os
        from neo4j_framework.config.env_loader import EnvironmentLoader

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
        from neo4j_framework.db.connection import Neo4jConnection

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
        from neo4j_framework.db.connection import Neo4jConnection

        conn = Neo4jConnection(
            uri="neo4j://localhost:7687",
            username="neo4j",
            password="password",
            max_connection_pool_size=200,
            connection_timeout=60.0,
            max_connection_lifetime=7200.0,
        )

        assert conn.max_connection_pool_size == 200

    def test_query_template_substitution(self):
        """Test query template usage and substitution (UPDATED)."""
        from neo4j_framework.queries.query_templates import QueryTemplates
        from neo4j_framework.queries.base_query import BaseQuery

        # Get template
        template = QueryTemplates.get_template("CREATE_NODE")
        assert template is not None

        # Create parameterized query
        query = BaseQuery(query_str=template, params={"name": "TestNode"})

        assert "CREATE" in query.query_str
        assert query.params["name"] == "TestNode"

    def test_base_query_with_connection(self, mock_neo4j_connection):
        """Test BaseQuery execution with connection (NEW)."""
        from neo4j_framework.queries.base_query import BaseQuery

        query = BaseQuery(query_str="MATCH (n) RETURN n LIMIT 1", params={"limit": 1})

        result = query.execute(mock_neo4j_connection)
        assert result is not None

    def test_csv_importer_with_allowed_directory(self, mock_neo4j_connection, tmp_path):
        """Test CSV importer with directory restriction (NEW)."""
        # Create allowed directory
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()

        # Create CSV in allowed directory
        csv_file = allowed_dir / "data.csv"
        csv_file.write_text("name,age\nAlice,30")

        # Create importer with directory restriction
        importer = CSVImporter(mock_neo4j_connection, allowed_dir=str(allowed_dir))

        # Should accept file in allowed directory
        validated_path = importer._validate_file_path(str(csv_file))
        assert validated_path.exists()

        # Should reject file outside allowed directory
        outside_file = tmp_path / "outside.csv"
        outside_file.write_text("name,age\nBob,25")

        with pytest.raises(ValueError, match="must be within"):
            importer._validate_file_path(str(outside_file))


@pytest.mark.integration
class TestErrorHandlingScenarios:
    """Test various error handling scenarios (UPDATED)."""

    def test_invalid_uri_format(self):
        """Test handling of invalid URI formats (UPDATED)."""
        from neo4j_framework.db.connection import Neo4jConnection

        # Connection object should be created (validation at connect time)
        conn = Neo4jConnection(
            uri="invalid://uri", username="neo4j", password="password"
        )

        assert conn.uri == "invalid://uri"

    def test_missing_password_error(self):
        """Test error when password is missing (UPDATED)."""
        from neo4j_framework.db.connection import Neo4jConnection

        with pytest.raises(ValueError, match="Username and password required"):
            Neo4jConnection(
                uri="neo4j://localhost:7687", username="neo4j", password=None
            )

    def test_invalid_parameter_type(self, clean_env):
        """Test handling of invalid parameter types (UPDATED)."""
        import os
        from neo4j_framework.config.env_loader import EnvironmentLoader

        os.environ["NEO4J_INVALID_INT"] = "not_a_number"

        loader = EnvironmentLoader(env_prefix="NEO4J_")

        with pytest.raises(ValueError, match="must be an integer"):
            loader.get_int("INVALID_INT")

    def test_transaction_error_propagation(self, mock_neo4j_connection):
        """Test transaction manager error handling."""
        from neo4j_framework.transactions.transaction_manager import TransactionManager

        tm = TransactionManager(mock_neo4j_connection)

        # Test that passing None function raises
        with pytest.raises(TypeError, match="object is not callable"):
            tm.run_in_transaction(None)

    def test_query_manager_connection_validation(self):
        """Test QueryManager validates connection (UPDATED)."""
        from neo4j_framework.queries.query_manager import QueryManager

        # Should raise when connection is None
        with pytest.raises(ValueError, match="cannot be None"):
            QueryManager(None)

    def test_transaction_manager_connection_validation(self):
        """Test TransactionManager validates connection (UPDATED)."""
        from neo4j_framework.transactions.transaction_manager import (
            TransactionManager,
        )

        # Should raise when connection is None
        with pytest.raises(ValueError, match="cannot be None"):
            TransactionManager(None)

    def test_csv_importer_connection_validation(self):
        """Test CSVImporter validates connection (UPDATED)."""
        from neo4j_framework.importers.csv_importer import CSVImporter

        # Should raise when connection is None
        with pytest.raises(ValueError, match="cannot be None"):
            CSVImporter(None)

    def test_pool_size_validation_error(self):
        """Test pool size validation (NEW)."""
        from neo4j_framework.db.connection import Neo4jConnection

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
        from neo4j_framework.db.connection import Neo4jConnection

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
        from neo4j_framework.utils.performance import Performance
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
        from neo4j_framework.utils.performance import Performance
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
        query_manager = QueryManager(mock_neo4j_connection)

        # Create many mock records
        mock_records = [
            MagicMock(data=lambda i=i: {"id": i, "value": i * 10}) for i in range(1000)
        ]

        # Configure mock to return all records
        mock_driver = mock_neo4j_connection._driver
        mock_session = mock_driver.session.return_value
        mock_session.execute_read.return_value = mock_records

        # Execute bulk read
        results = query_manager.execute_read("MATCH (n) RETURN n")

        assert len(results) == 1000

    def test_bulk_write_operations(self, mock_neo4j_connection):
        """Test performance of bulk write operations (UPDATED)."""
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
        from neo4j_framework.queries.query_manager import QueryManager
        from neo4j_framework.transactions.transaction_manager import (
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
        with pytest.raises(Exception, match="Connection failed"):
            query_manager.execute_read("MATCH (n) RETURN n")

        # Reset for retry
        mock_session.execute_read.side_effect = None
        mock_session.execute_read.return_value = [
            MagicMock(data=lambda: {"status": "ok"})
        ]

        # Retry succeeds
        result = query_manager.execute_read("MATCH (n) RETURN n")
        assert len(result) > 0
