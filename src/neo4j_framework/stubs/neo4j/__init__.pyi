"""
Type stubs for neo4j package - provides type hints for Pyright
"""

from typing import Any, Optional, Dict, List, Callable, Union

class Auth:
    """Neo4j authentication class with static factory methods."""
    
    @staticmethod
    def basic(
        username: str,
        password: str,
        realm: Optional[str] = None
    ) -> "Auth":
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
        parameters: Optional[Dict[str, Any]] = None
    ) -> "Auth":
        """Create custom authentication."""
        ...

class Record:
    """Neo4j Record type."""
    def __getitem__(self, key: Union[str, int]) -> Any: ...
    def __iter__(self) -> Any: ...
    def keys(self) -> List[str]: ...
    def values(self) -> List[Any]: ...

class Result:
    """Neo4j Result type."""
    def __iter__(self) -> Any: ...
    def fetch(self, n: int = 1) -> List[Record]: ...
    def single(self) -> Optional[Record]: ...
    def consume(self) -> Any: ...

class Transaction:
    """Neo4j Transaction type."""
    def run(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> Result: ...
    def commit(self) -> None: ...
    def rollback(self) -> None: ...
    def close(self) -> None: ...

class Session:
    """Neo4j Session type."""
    def run(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> Result: ...
    def begin_transaction(self) -> Transaction: ...
    def read_transaction(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any: ...
    def write_transaction(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any: ...
    def close(self) -> None: ...
    def __enter__(self) -> "Session": ...
    def __exit__(self, *args: Any) -> None: ...

class Driver:
    """Neo4j Driver type."""
    def session(
        self,
        database: Optional[str] = None,
        **kwargs: Any
    ) -> Session: ...
    def verify_connectivity(self) -> None: ...
    def close(self) -> None: ...
    def __enter__(self) -> "Driver": ...
    def __exit__(self, *args: Any) -> None: ...

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
        **kwargs: Any
    ) -> Driver: ...
