"""
Type stubs for neo4j package - provides type hints for Pyright
"""

from typing import (
    Any,
    Iterator,
    LiteralString,
    Optional,
    Dict,
    List,
    Callable,
    Union,
    Tuple,
)

class Neo4jException(Exception):
    """Base exception for Neo4j driver."""

    pass

class QueryError(Neo4jException):
    """Raised when a query fails."""

    pass

class ClientError(Neo4jException):
    """Raised for client-side errors."""

    pass

class TransientError(Neo4jException):
    """Raised for transient errors (e.g., connection timeouts)."""

    pass

class DatabaseError(Neo4jException):
    """Raised for database-side errors."""

    pass

class Auth:
    """Neo4j authentication class with static factory methods."""

    @staticmethod
    def basic(username: str, password: str, realm: Optional[str] = None) -> "Auth":
        """Create basic authentication."""
        ...

    @staticmethod
    def kerberos(ticket: str) -> "Auth":
        """Create Kerberos authentication."""
        ...

    @staticmethod
    def bearer(token: str) -> "Auth":
        """Create Bearer token authentication."""
        ...

    @staticmethod
    def custom(
        scheme: str,
        principal: str,
        credentials: str,
        realm: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> "Auth":
        """Create custom authentication."""
        ...

class Record:
    """Neo4j Record type - represents a single result row."""

    def __getitem__(self, key: Union[str, int]) -> Any:
        """Get a value by key (string) or index (int)."""
        ...

    def __iter__(self) -> Iterator[Any]:
        """Iterate over values in the record."""
        ...

    def keys(self) -> List[str]:
        """Get all keys in the record."""
        ...

    def values(self) -> List[Any]:
        """Get all values in the record."""
        ...

    def data(self) -> Dict[str, Any]:
        """Get record data as a dictionary."""
        ...

class Result:
    """Neo4j Result type - represents query results."""

    def __iter__(self) -> Iterator[Record]:
        """Iterate over records in result."""
        ...

    def fetch(self, n: int = 1) -> List[Record]:
        """Fetch up to n records."""
        ...

    def single(self) -> Optional[Record]:
        """Get a single record or None."""
        ...

    def consume(self) -> Any:
        """Consume the result and return summary."""
        ...

class Transaction:
    """Neo4j Transaction type."""

    def run(
        self, query: str, parameters: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> Result:
        """Execute a query within the transaction."""
        ...

    def commit(self) -> None:
        """Commit the transaction."""
        ...

    def rollback(self) -> None:
        """Rollback the transaction."""
        ...

    def close(self) -> None:
        """Close the transaction."""
        ...

class Session:
    """Neo4j Session type."""

    def run(
        self, query: str, parameters: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> Result:
        """Execute a query in the session."""
        ...

    def begin_transaction(self) -> Transaction:
        """Begin an explicit transaction."""
        ...

    def read_transaction(
        self, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        """Execute a function in a read transaction."""
        ...

    def write_transaction(
        self, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        """Execute a function in a write transaction."""
        ...

    def execute_read(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute a read operation (Neo4j 4.4+)."""
        ...

    def execute_write(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute a write operation (Neo4j 4.4+)."""
        ...

    def close(self) -> None:
        """Close the session."""
        ...

    def __enter__(self) -> "Session":
        """Context manager entry."""
        ...

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        ...

class Driver:
    """Neo4j Driver type."""

    def session(self, database: Optional[str] = None, **kwargs: Any) -> Session:
        """Create a new session."""
        ...

    def verify_connectivity(self) -> None:
        """Verify the connection."""
        ...

    def close(self) -> None:
        """Close the driver and all connections."""
        ...

    def __enter__(self) -> "Driver":
        """Context manager entry."""
        ...

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        ...

class GraphDatabase:
    """Neo4j GraphDatabase class."""

    @staticmethod
    def driver(
        uri: str,
        *,
        auth: Optional[Auth] = None,
        encrypted: Optional[bool] = None,
        trust: Optional[str] = None,
        max_connection_pool_size: int = 100,
        connection_timeout: float = 30.0,
        **kwargs: Any,
    ) -> Driver:
        """Create a new driver."""
        ...

# Type alias for Cypher queries
Query = LiteralString
