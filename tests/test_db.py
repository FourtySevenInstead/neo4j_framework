from typing import Any, Dict
import pytest
from neo4j import GraphDatabase, Auth
from neo4j_framework.db.connection import Neo4jConnection
from neo4j_framework.db.pool_manager import PoolManager
from neo4j_framework.utils.exceptions import ConnectionError


@pytest.fixture
def mock_driver(mocker):
    mock_driver_instance = mocker.Mock()
    mock_driver_instance.verify_connectivity = mocker.Mock()
    mock_driver_instance.close = mocker.Mock()
    mock_driver_instance.session = mocker.Mock()
    mocker.patch.object(GraphDatabase, "driver", return_value=mock_driver_instance)
    return mock_driver_instance


def test_neo4j_connection_init():
    conn = Neo4jConnection(
        uri="neo4j://test",
        username="user",
        password="pass",
        max_connection_pool_size=50,
    )
    assert conn.uri == "neo4j://test"
    assert conn.max_connection_pool_size == 50
    assert conn.encrypted  # Defaults to True for security


def test_neo4j_connection_init_validation():
    with pytest.raises(ValueError, match="uri cannot be None"):
        Neo4jConnection(uri=None, username="user", password="pass")  # type: ignore
    with pytest.raises(
        ValueError, match="max_connection_pool_size must be between 1 and 500"
    ):
        Neo4jConnection(
            uri="test", username="user", password="pass", max_connection_pool_size=0
        )


def test_connect_basic(mock_driver):
    conn = Neo4jConnection(uri="neo4j://test", username="user", password="pass")
    driver = conn.connect()
    assert conn.is_connected()
    assert conn.get_driver() == driver
    mock_driver.verify_connectivity.assert_called_once()


def test_connect_kerberos(mock_driver):
    conn = Neo4jConnection(uri="neo4j://test", username="user", password="pass")
    conn.connect(auth_type="kerberos", ticket="test_ticket")
    Auth.kerberos.assert_called_with("test_ticket")  # type: ignore


def test_connect_invalid_auth():
    conn = Neo4jConnection(uri="neo4j://test", username="user", password="pass")
    with pytest.raises(ValueError, match="Unknown auth type"):
        conn.connect(auth_type="invalid")


def test_connect_with_mtls(mocker, mock_driver):
    mocker.patch("os.path.exists", return_value=True)
    conn = Neo4jConnection(uri="neo4j://test", username="user", password="pass")
    conn.connect_with_mtls(cert_path="cert.pem", key_path="key.pem")


def test_connect_with_mtls_missing_file(mocker):
    mocker.patch("os.path.exists", return_value=False)
    conn = Neo4jConnection(uri="neo4j://test", username="user", password="pass")
    with pytest.raises(ValueError, match="Certificate file not found"):
        conn.connect_with_mtls(cert_path="missing.pem")


def test_close(mock_driver):
    conn = Neo4jConnection(uri="neo4j://test", username="user", password="pass")
    conn.connect()
    conn.close()
    assert not conn.is_connected()
    mock_driver.close.assert_called_once()


def test_run_in_session(mock_driver):
    conn = Neo4jConnection(uri="neo4j://test", username="user", password="pass")
    conn.connect()

    def func(session):
        return "result"

    result = conn.run_in_session(func)
    assert result == "result"


def test_run_in_session_not_connected():
    conn = Neo4jConnection(uri="neo4j://test", username="user", password="pass")
    with pytest.raises(RuntimeError, match="Not connected"):
        conn.run_in_session(lambda s: None)


def test_get_pool_stats(mock_driver):
    conn = Neo4jConnection(uri="neo4j://test", username="user", password="pass")
    conn.connect()
    mock_driver._pool = mocker.Mock(_in_use=[1, 2], _available=[3])  # type: ignore
    stats = conn.get_pool_stats()
    assert stats["in_use"] == 2
    assert stats["available"] == 1


def test_pool_manager_init(mock_driver):
    pm = PoolManager(mock_driver)
    assert pm.driver == mock_driver


def test_pool_manager_get_stats(mocker):
    pm = PoolManager(mocker.Mock())
    stats = pm.get_pool_stats()
    assert "note" in stats
