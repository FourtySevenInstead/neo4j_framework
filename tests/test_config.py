import os
from typing import Any, Dict
import pytest
from dotenv import load_dotenv
from ..src.neo4j_framework.config.env_loader import EnvironmentLoader
from ..src.neo4j_framework.config.db_config import get_db_config


@pytest.fixture
def mock_env(mocker):
    mocker.patch.object(load_dotenv, "__call__")  # Mock dotenv load
    mocker.patch.object(os, "getenv")  # Mock getenv


def test_environment_loader_init():
    loader = EnvironmentLoader(env_file="test.env", env_prefix="TEST_")
    assert loader.env_file == "test.env"
    assert loader.env_prefix == "TEST_"
    assert not loader._loaded


def test_environment_loader_load(mock_env, mocker):
    mocker.patch("os.path.exists", return_value=True)
    loader = EnvironmentLoader()
    assert loader.load()  # Should call load_dotenv
    assert loader._loaded


def test_environment_loader_load_not_found(mock_env, mocker):
    mocker.patch("os.path.exists", return_value=False)
    loader = EnvironmentLoader()
    assert loader.load()  # Still marks as loaded (attempted)
    assert loader._loaded


def test_get_required_missing(mock_env, mocker):
    mocker.patch("os.getenv", return_value=None)
    loader = EnvironmentLoader()
    with pytest.raises(ValueError, match="Required environment variable not set"):
        loader.get("URI", required=True)


def test_get_int_valid(mock_env, mocker):
    mocker.patch("os.getenv", return_value="100")
    loader = EnvironmentLoader()
    assert loader.get_int("MAX_CONNECTION_POOL_SIZE", default=50) == 100


def test_get_int_invalid(mock_env, mocker):
    mocker.patch("os.getenv", return_value="abc")
    loader = EnvironmentLoader()
    with pytest.raises(ValueError, match="must be an integer"):
        loader.get_int("MAX_CONNECTION_POOL_SIZE")


def test_get_int_bounds(mock_env, mocker):
    mocker.patch("os.getenv", return_value="0")
    loader = EnvironmentLoader()
    with pytest.raises(ValueError, match="must be >= 1"):
        loader.get_int("MAX_CONNECTION_POOL_SIZE", min_val=1, max_val=500)


def test_get_float_valid(mock_env, mocker):
    mocker.patch("os.getenv", return_value="30.0")
    loader = EnvironmentLoader()
    assert loader.get_float("CONNECTION_TIMEOUT", default=10.0) == 30.0


def test_get_float_bounds(mock_env, mocker):
    mocker.patch("os.getenv", return_value="0.05")
    loader = EnvironmentLoader()
    with pytest.raises(ValueError, match="must be >= 0.1"):
        loader.get_float("CONNECTION_TIMEOUT", min_val=0.1)


def test_get_bool_valid(mock_env, mocker):
    mocker.patch("os.getenv", return_value="true")
    loader = EnvironmentLoader()
    assert loader.get_bool("ENCRYPTED", default=False)


def test_get_bool_false(mock_env, mocker):
    mocker.patch("os.getenv", return_value="false")
    loader = EnvironmentLoader()
    assert not loader.get_bool("ENCRYPTED", default=True)


def test_get_config(mock_env, mocker):
    mocker.patch(
        "os.getenv",
        side_effect=lambda k, d: {
            "NEO4J_URI": "neo4j://test",
            "NEO4J_PASSWORD": "pass",
            "NEO4J_USERNAME": "user",
            "NEO4J_DATABASE": "testdb",
            "NEO4J_ENCRYPTED": "true",
            "NEO4J_MAX_CONNECTION_POOL_SIZE": "100",
            "NEO4J_CONNECTION_TIMEOUT": "30.0",
        }.get(k, d),
    )
    config = EnvironmentLoader().get_config()
    assert config["uri"] == "neo4j://test"
    assert config["encrypted"]
    assert config["max_connection_pool_size"] == 100


def test_get_db_config(mock_env, mocker):
    mocker.patch(
        "src.neo4j_framework.config.env_loader.EnvironmentLoader.get_config",
        return_value={"uri": "test"},
    )
    config: Dict[str, Any] = get_db_config()
    assert config["uri"] == "test"
