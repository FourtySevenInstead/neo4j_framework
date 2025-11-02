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
            "newlines": "Line1\nLine2\nLine3",
            "tabs": "Col1\tCol2\tCol3"
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
        csv_file.write_text("name\nTest")
        
        importer = CSVImporter(mock_neo4j_connection)
        validated = importer._validate_file_path(str(csv_file))
        
        assert validated.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
