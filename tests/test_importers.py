import pytest
from pathlib import Path

from neo4j_framework.importers.csv_importer import CSVImporter
from neo4j_framework.db.connection import Neo4jConnection


@pytest.fixture
def mock_connection(mocker):
    conn = mocker.Mock()
    conn.get_driver = mocker.Mock(return_value=mocker.Mock())
    conn.database = "testdb"
    return conn


def test_csv_importer_init(mock_connection):
    importer = CSVImporter(mock_connection, allowed_dir="/allowed")
    assert importer.allowed_dir == Path("/allowed").resolve()


def test_validate_file_path(mocker, mock_connection):
    mocker.patch.object(Path, "exists", return_value=True)
    mocker.patch("os.access", return_value=True)
    importer = CSVImporter(mock_connection, allowed_dir="/allowed")
    path = importer._validate_file_path("/allowed/test.csv")
    assert str(path) == "/allowed/test.csv"


def test_validate_file_path_not_found(mocker, mock_connection):
    mocker.patch.object(Path, "exists", return_value=False)
    importer = CSVImporter(mock_connection)
    with pytest.raises(ValueError, match="not found"):
        importer._validate_file_path("missing.csv")


def test_validate_file_path_outside_allowed(mocker, mock_connection):
    mocker.patch.object(Path, "exists", return_value=True)
    mocker.patch("os.access", return_value=True)
    importer = CSVImporter(mock_connection, allowed_dir="/allowed")
    with pytest.raises(ValueError, match="must be within"):
        importer._validate_file_path("/disallowed/test.csv")


def test_import_csv(mocker, mock_connection):
    mocker.patch.object(Path, "exists", return_value=True)
    mocker.patch("os.access", return_value=True)
    mocker.patch.object(Path, "as_uri", return_value="file:///test.csv")
    mock_session = mocker.Mock()
    mock_session.run = mocker.Mock(return_value="result")
    mock_session_mock = mocker.MagicMock()
    mock_session_mock.__enter__.return_value = mock_session
    mock_session_mock.__exit__.return_value = False
    mock_connection.get_driver.return_value.session.return_value = mock_session_mock
    importer = CSVImporter(mock_connection)
    result = importer.import_csv(
        "test.csv", "LOAD CSV FROM $file_url", params={"param": 1}
    )
    assert result == "result"
    mock_session.run.assert_called_with(
        "LOAD CSV FROM $file_url", {"param": 1, "file_url": "file:///test.csv"}
    )
