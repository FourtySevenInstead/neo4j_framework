import pytest

from neo4j_framework.transactions.transaction_manager import TransactionManager
from neo4j_framework.db.connection import Neo4jConnection


@pytest.fixture
def mock_connection(mocker):
    conn = mocker.Mock()
    conn.get_driver = mocker.Mock(return_value=mocker.Mock())
    conn.database = "testdb"
    return conn


def test_transaction_manager_init(mock_connection):
    tm = TransactionManager(mock_connection)
    assert tm.connection == mock_connection


def test_run_in_transaction(mocker, mock_connection):
    mock_session = mocker.Mock()
    mock_session.execute_write = mocker.Mock(return_value="result")
    mock_session_mock = mocker.MagicMock()
    mock_session_mock.__enter__.return_value = mock_session
    mock_session_mock.__exit__.return_value = False
    mock_connection.get_driver.return_value.session.return_value = mock_session_mock
    tm = TransactionManager(mock_connection)

    def tx_func(tx):
        return "result"

    result = tm.run_in_transaction(tx_func)
    assert result == "result"


def test_explicit_transaction(mocker, mock_connection):
    mocker.patch.object(TransactionManager, "run_in_transaction", return_value="result")
    tm = TransactionManager(mock_connection)
    result = tm.explicit_transaction("CREATE (n)")
    assert result == "result"


def test_context_manager(mocker, mock_connection):
    mock_session = mocker.Mock()
    mock_session.close = mocker.Mock()
    mock_connection.get_driver.return_value.session.return_value = mock_session
    tm = TransactionManager(mock_connection)
    with tm as session:
        assert session == mock_session
    mock_session.close.assert_called_once()


def test_context_manager_exception(mocker, mock_connection):
    mock_session = mocker.Mock()
    mock_session.close = mocker.Mock()
    mock_connection.get_driver.return_value.session.return_value = mock_session
    tm = TransactionManager(mock_connection)
    with pytest.raises(ValueError):
        with tm:
            raise ValueError("test error")
    mock_session.close.assert_called_once()
