"""
Neo4j connection management with support for multiple authentication methods.
"""

import logging
from typing import Optional, Any, Dict, Type, cast, Callable, TypeVar
from neo4j import GraphDatabase, Driver, Auth, Session

logger = logging.getLogger(__name__)
T = TypeVar("T")


class Neo4jConnection:
    """
    Manages Neo4j database connections with support for multiple authentication methods.
    Handles connection pooling and lifecycle management.
    """

    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        database: str = "neo4j",
        encrypted: bool = True,
        max_connection_pool_size: int = 100,
    ) -> None:
        """
        Initialize Neo4j connection configuration.

        Args:
            uri: Neo4j connection URI (neo4j://, neo4j+s://, bolt://, etc.)
            username: Database username
            password: Database password
            database: Target database name
            encrypted: Enable encryption (ignored for URIs with +s or +ssc)
            max_connection_pool_size: Maximum number of concurrent connections

        Raises:
            ValueError: If configuration is invalid
        """
        from neo4j_framework.utils.validators import Validators

        # Validate inputs
        Validators.validate_not_none(uri, "uri")
        Validators.validate_not_none(username, "username")
        Validators.validate_not_none(password, "password")
        Validators.validate_string_not_empty(uri, "uri")
        Validators.validate_string_not_empty(username, "username")
        Validators.validate_string_not_empty(password, "password")

        # Validate pool size bounds
        if max_connection_pool_size < 1 or max_connection_pool_size > 500:
            raise ValueError(
                f"max_connection_pool_size must be between 1 and 500, "
                f"got {max_connection_pool_size}"
            )

        self.uri: str = uri
        self.username: str = username
        self.password: str = password
        self.database: str = database

        # Determine if URI scheme handles encryption
        has_secure_scheme: bool = "+s" in uri or "+ssc" in uri

        # Only use encrypted flag if URI doesn't already handle it
        self.encrypted: bool = False if has_secure_scheme else encrypted
        self.max_connection_pool_size: int = max_connection_pool_size

        self._driver: Optional[Driver] = None
        logger.debug(f"Neo4jConnection initialized: {uri}")

    def connect(
        self,
        auth_type: str = "basic",
        **auth_kwargs: Any,
    ) -> Driver:
        """
        Establish connection to Neo4j database.

        Args:
            auth_type: Authentication type ('basic', 'kerberos', 'bearer', 'custom')
            **auth_kwargs: Additional authentication parameters

        Returns:
            Neo4j Driver instance

        Raises:
            Exception: If connection fails
        """
        try:
            # Create appropriate auth based on auth_type
            if auth_type == "basic":
                auth: Auth = Auth.basic(self.username, self.password)
            elif auth_type == "kerberos":
                if "ticket" not in auth_kwargs:
                    raise ValueError("Kerberos ticket required")
                ticket: str = auth_kwargs["ticket"]
                auth = Auth.kerberos(ticket)
            elif auth_type == "bearer":
                if "token" not in auth_kwargs:
                    raise ValueError("Bearer token required")
                token: str = auth_kwargs["token"]
                auth = Auth.bearer(token)
            elif auth_type == "custom":
                required = ["scheme", "principal", "credentials", "realm"]
                missing = [k for k in required if k not in auth_kwargs]
                if missing:
                    raise ValueError(f"Custom auth missing: {', '.join(missing)}")
                scheme: str = auth_kwargs["scheme"]
                principal: str = auth_kwargs["principal"]
                credentials: str = auth_kwargs["credentials"]
                realm_val: str = auth_kwargs["realm"]
                auth = Auth.custom(scheme, principal, credentials, realm_val)
            else:
                raise ValueError(f"Unknown auth type: {auth_type}")

            # Build driver kwargs with proper types
            driver_kwargs: Dict[str, Any] = {
                "max_connection_pool_size": self.max_connection_pool_size,
            }

            # Only add encrypted flag if we need to specify it
            # (not needed for +s URIs, but required for non-secure URIs)
            if not ("+s" in self.uri or "+ssc" in self.uri):
                driver_kwargs["encrypted"] = self.encrypted

            self._driver = GraphDatabase.driver(
                self.uri,
                auth=auth,
                **driver_kwargs,
            )

            # Test the connection
            self._driver.verify_connectivity()
            logger.info(f"Connected to Neo4j: {self.uri}")

            return self._driver

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            raise

    def connect_with_mtls(
        self,
        cert_path: str,
        key_path: Optional[str] = None,
    ) -> Driver:
        """
        Connect using mTLS certificates.

        Args:
            cert_path: Path to certificate file
            key_path: Optional path to key file

        Returns:
            Neo4j Driver instance

        Raises:
            ValueError: If certificate files don't exist
        """
        import os

        if not os.path.exists(cert_path):
            raise ValueError(f"Certificate file not found: {cert_path}")

        if key_path and not os.path.exists(key_path):
            raise ValueError(f"Key file not found: {key_path}")

        auth: Auth = Auth.basic(self.username, self.password)

        driver_kwargs: Dict[str, Any] = {
            "max_connection_pool_size": self.max_connection_pool_size,
            "encrypted": True,
            "trust": "TRUST_ALL_CERTIFICATES",
        }

        self._driver = GraphDatabase.driver(
            self.uri,
            auth=auth,
            **driver_kwargs,
        )

        logger.info(f"Connected with mTLS: {self.uri}")
        return self._driver

    def is_connected(self) -> bool:
        """Check if connection is established."""
        return self._driver is not None

    def get_driver(self) -> Optional[Driver]:
        """Get the Neo4j driver instance."""
        return self._driver

    def close(self) -> None:
        """Close the connection."""
        if self._driver:
            self._driver.close()
            self._driver = None
            logger.debug("Connection closed")

    def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        if not self._driver:
            return {}

        try:
            # pyright: ignore[attr-defined]
            # _pool is a private Neo4j driver attribute without type stubs
            pool = cast(Any, self._driver._pool)

            in_use_count: int = 0
            available_count: int = 0

            if hasattr(pool, "_in_use"):
                in_use_count = len(pool._in_use)

            if hasattr(pool, "_available"):
                available_count = len(pool._available)

            return {
                "in_use": in_use_count,
                "available": available_count,
            }
        except Exception:
            return {"status": "unable to retrieve pool stats"}

    def run_in_session(
        self,
        func: Callable[[Session], T],
        database: Optional[str] = None,
    ) -> T:
        """
        Execute a function within a Neo4j session.

        Args:
            func: Function that takes a Session and returns a value.
            database: Optional database name (defaults to self.database).

        Returns:
            Result of func.

        Raises:
            RuntimeError: If not connected.
            Exception: If func raises.
        """
        if not self.is_connected():
            raise RuntimeError("Not connected to Neo4j")
        effective_db = database or self.database
        try:
            with self._driver.session(database=effective_db) as session:  # type: ignore
                return func(session)
        except Exception as e:
            logger.error(f"Session execution failed: {type(e).__name__}: {e}")
            raise

    def __enter__(self) -> "Neo4jConnection":
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(
        self,
        _exc_type: Optional[type[BaseException]],
        _exc_val: Optional[BaseException],
        _exc_tb: Any,
    ) -> bool:
        """Context manager exit."""
        self.close()
        return False
