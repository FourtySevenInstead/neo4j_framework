import pytest

from neo4j_framework.queries.base_query import BaseQuery
from neo4j_framework.queries.query_manager import QueryManager
from neo4j_framework.queries.query_templates import QueryTemplates
from neo4j_framework.db.connection import Neo4jConnection


@pytest.fixture
def mock_connection(mocker):
    conn = mocker.Mock()
    conn.get_driver = mocker.Mock(return_value=mocker.Mock())
    conn.database = "testdb"
    return conn


def test_base_query_init():
    query = BaseQuery("MATCH (n) RETURN n", {"param": 1})
    assert query.params == {"param": 1}


def test_base_query_init_none():
    with pytest.raises(ValueError, match="query_str cannot be None"):
        BaseQuery(None)  # type: ignore


def test_base_query_execute(mocker, mock_connection):
    mock_session = mocker.Mock()
    mock_result = mocker.Mock(data=mocker.Mock(return_value={"key": "value"}))
    mock_session.run = mocker.Mock(return_value=[mock_result])
    mock_session_mock = mocker.MagicMock()
    mock_session_mock.__enter__.return_value = mock_session
    mock_session_mock.__exit__.return_value = False
    mock_connection.get_driver.return_value.session.return_value = mock_session_mock
    query = BaseQuery("MATCH (n) RETURN n")
    result = query.execute(mock_connection)
    assert result == [{"key": "value"}]


def test_query_manager_init(mock_connection):
    qm = QueryManager(mock_connection)
    assert qm.connection == mock_connection


def test_execute_read(mocker, mock_connection):
    mock_session = mocker.Mock()
    mock_result = mocker.Mock(data=mocker.Mock(return_value={"key": "value"}))
    mock_session.execute_read = mocker.Mock(
        return_value=mocker.Mock(__iter__=lambda self: iter([mock_result]))
    )
    mock_session_mock = mocker.MagicMock()
    mock_session_mock.__enter__.return_value = mock_session
    mock_session_mock.__exit__.return_value = False
    mock_connection.get_driver.return_value.session.return_value = mock_session_mock
    qm = QueryManager(mock_connection)
    result = qm.execute_read("MATCH (n) RETURN n")
    assert result == [{"key": "value"}]


def test_execute_write(mocker, mock_connection):
    mock_session = mocker.Mock()
    mock_session.execute_write = mocker.Mock(return_value="result")
    mock_session_mock = mocker.MagicMock()
    mock_session_mock.__enter__.return_value = mock_session
    mock_session_mock.__exit__.return_value = False
    mock_connection.get_driver.return_value.session.return_value = mock_session_mock
    qm = QueryManager(mock_connection)
    result = qm.execute_write("CREATE (n)")
    assert result == "result"


def test_execute_query(mocker, mock_connection):
    mocker.patch(
        "neo4j_framework.queries.base_query.BaseQuery.execute",
        return_value=[{"key": "value"}],
    )
    qm = QueryManager(mock_connection)
    result = qm.execute_query("MATCH (n) RETURN n")
    assert result == [{"key": "value"}]


def test_query_templates():
    assert QueryTemplates.CREATE_NODE == "CREATE (n:Node {name: $name}) RETURN n"
    assert QueryTemplates.get_template("CREATE_NODE") == QueryTemplates.CREATE_NODE
    assert QueryTemplates.get_template("INVALID") is None
