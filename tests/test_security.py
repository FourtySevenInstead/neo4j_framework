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
        csv_file.write_text("name\ntest")

        importer = CSVImporter(mock_neo4j_connection, allowed_dir=str(allowed_dir))

        # Create file outside allowed dir
        outside_file = tmp_path / "outside.csv"
        outside_file.write_text("name\ntest")

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
