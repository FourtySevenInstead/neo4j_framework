import time
import pytest
import logging
from neo4j_framework.utils.exceptions import (
    Neo4jFrameworkException,
    ConnectionError,
    AuthenticationError,
    ConfigurationError,
    ValidationError,
    QueryError,
    TransactionError,
)
from neo4j_framework.utils.logger import setup_logging
from neo4j_framework.utils.performance import Performance
from neo4j_framework.utils.validators import Validators


def test_exceptions():
    assert issubclass(ConnectionError, Neo4jFrameworkException)
    # ... similarly for others
    with pytest.raises(ConnectionError):
        raise ConnectionError("test")


def test_setup_logging(mocker):
    mocker.patch(
        "neo4j_framework.config.env_loader.EnvironmentLoader.get",
        return_value="DEBUG",
    )
    logger = setup_logging()
    assert logger.getEffectiveLevel() == logging.DEBUG


def test_performance_time_function(mocker):
    mock_debug = mocker.patch("neo4j_framework.utils.performance.logger.debug")

    @Performance.time_function
    def test_func():
        time.sleep(0.01)
        return "result"

    result = test_func()
    assert result == "result"
    mock_debug.assert_called_once()


def test_performance_time_function_error(mocker):
    mock_error = mocker.patch("neo4j_framework.utils.performance.logger.error")

    @Performance.time_function
    def test_func():
        raise ValueError("error")

    with pytest.raises(ValueError):
        test_func()
    mock_error.assert_called_once()


def test_validators_not_none():
    Validators.validate_not_none("value", "name")
    with pytest.raises(ValueError, match="cannot be None"):
        Validators.validate_not_none(None, "name")


def test_validators_string_not_empty():
    Validators.validate_string_not_empty("value", "name")
    with pytest.raises(ValueError, match="cannot be empty"):
        Validators.validate_string_not_empty("", "name")


def test_validators_positive_int():
    Validators.validate_positive_int(1, "name")
    with pytest.raises(ValueError, match="must be a positive integer"):
        Validators.validate_positive_int(0, "name")


def test_validators_int():
    Validators.validate_int(5, "name", min_val=1, max_val=10)
    with pytest.raises(ValueError, match="must be an integer"):
        Validators.validate_int("a", "name")
    with pytest.raises(ValueError, match="must be >= 1"):
        Validators.validate_int(0, "name", min_val=1)


def test_validators_float():
    Validators.validate_float(5.0, "name", min_val=1.0, max_val=10.0)
    with pytest.raises(ValueError, match="must be a number"):
        Validators.validate_float("a", "name")
    with pytest.raises(ValueError, match="must be >= 1.0"):
        Validators.validate_float(0.5, "name", min_val=1.0)
