"""
Validation tests for Neo4j Framework.
Tests input validation and error handling.
"""

import pytest
import os

from neo4j_framework.config.env_loader import EnvironmentLoader
from neo4j_framework.utils.validators import Validators


@pytest.mark.validation
class TestInputValidation:
    """Input validation tests."""

    def test_integer_validation(self, clean_env):
        """Test integer validation with various inputs."""
        # Valid integer
        os.environ["NEO4J_INT_VAL"] = "42"
        assert EnvironmentLoader(env_prefix="NEO4J_").get_int("INT_VAL") == 42

        # Invalid - not a number
        os.environ["NEO4J_INT_VAL"] = "not_a_number"
        with pytest.raises(ValueError, match="must be an integer"):
            EnvironmentLoader(env_prefix="NEO4J_").get_int("INT_VAL")

        # Invalid - float when int expected
        os.environ["NEO4J_INT_VAL"] = "42.5"
        with pytest.raises(ValueError):
            EnvironmentLoader(env_prefix="NEO4J_").get_int("INT_VAL")

    def test_float_validation(self, clean_env):
        """Test float validation with various inputs."""
        # Valid float
        os.environ["NEO4J_FLOAT_VAL"] = "3.14"
        assert EnvironmentLoader(env_prefix="NEO4J_").get_float("FLOAT_VAL") == 3.14

        # Valid - integer as float
        os.environ["NEO4J_FLOAT_VAL"] = "42"
        assert EnvironmentLoader(env_prefix="NEO4J_").get_float("FLOAT_VAL") == 42.0

        # Invalid - not a number
        os.environ["NEO4J_FLOAT_VAL"] = "not_a_number"
        with pytest.raises(ValueError, match="must be a float"):
            EnvironmentLoader(env_prefix="NEO4J_").get_float("FLOAT_VAL")

    def test_bounds_validation_integers(self, clean_env):
        """Test integer bounds validation."""
        os.environ["NEO4J_VAL"] = "50"

        # Within bounds
        assert (
            EnvironmentLoader(env_prefix="NEO4J_").get_int(
                "VAL", min_val=1, max_val=100
            )
            == 50
        )

        # Below minimum
        with pytest.raises(ValueError, match="must be >= 1"):
            EnvironmentLoader(env_prefix="NEO4J_").get_int(
                "VAL", min_val=100, max_val=200
            )

        # Above maximum
        with pytest.raises(ValueError, match="must be <= 10"):
            EnvironmentLoader(env_prefix="NEO4J_").get_int("VAL", min_val=1, max_val=10)

    def test_bounds_validation_floats(self, clean_env):
        """Test float bounds validation."""
        os.environ["NEO4J_VAL"] = "5.5"

        # Within bounds
        assert (
            EnvironmentLoader(env_prefix="NEO4J_").get_float(
                "VAL", min_val=0.0, max_val=10.0
            )
            == 5.5
        )

        # Below minimum
        with pytest.raises(ValueError, match="must be >= 10.0"):
            EnvironmentLoader(env_prefix="NEO4J_").get_float(
                "VAL", min_val=10.0, max_val=20.0
            )

        # Above maximum
        with pytest.raises(ValueError, match="must be <= 1.0"):
            EnvironmentLoader(env_prefix="NEO4J_").get_float(
                "VAL", min_val=0.0, max_val=1.0
            )

    def test_string_validation(self):
        """Test string validation."""
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
        # Valid
        Validators.validate_positive_int(1, "param")
        Validators.validate_positive_int(100, "param")

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
        os.environ["NEO4J_URI"] = "neo4j://localhost:7687"
        os.environ["NEO4J_PASSWORD"] = "password"
        os.environ["NEO4J_MAX_CONNECTION_POOL_SIZE"] = "1000"

        loader = EnvironmentLoader(env_prefix="NEO4J_")

        with pytest.raises(ValueError, match="must be <= 500"):
            loader.get_config()

    def test_invalid_timeout_in_config(self, clean_env):
        """Test that invalid timeout in config is caught."""
        os.environ["NEO4J_URI"] = "neo4j://localhost:7687"
        os.environ["NEO4J_PASSWORD"] = "password"
        os.environ["NEO4J_CONNECTION_TIMEOUT"] = "-5.0"

        loader = EnvironmentLoader(env_prefix="NEO4J_")

        with pytest.raises(ValueError, match="must be >= 0.1"):
            loader.get_config()
