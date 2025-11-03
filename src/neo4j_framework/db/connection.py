"""
Neo4j connection management with support for multiple authentication methods.
"""

import logging
from typing import Optional
from neo4j import GraphDatabase, Driver, Auth

logger = logging.getLogger(__name__)


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
    ):
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
        Validators.validate_int(
            max_connection_pool_size,
            "max_connection_pool_size",
            min_val=1,
            max_val=500,
        )

        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.config = {
            "uri": uri,
            "username": username,
            "password": password,
            "database": database,
            "max_connection_pool_size": max_connection_pool_size,
        }

        # Determine if URI scheme handles encryption
        # URIs with +s or +ssc already include encryption
        has_secure_scheme = "+s" in uri or "+ssc" in uri

        # Only pass encrypted flag if URI doesn't already handle it
        if not has_secure_scheme:
            self.config["encrypted"] = encrypted
        else:
            self.config["encrypted"] = False

        self._driver: Optional[Driver] = None
        logger.debug(f"Neo4jConnection initialized: {uri}")

    def connect(
        self,
        auth_type: str = "basic",
        **auth_kwargs,
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
                auth = Auth.basic(self.username, self.password)
            elif auth_type == "kerberos":
                if "ticket" not in auth_kwargs:
                    raise ValueError("Kerberos ticket required")
                auth = Auth.kerberos(auth_kwargs["ticket"])
            elif auth_type == "bearer":
                if "token" not in auth_kwargs:
                    raise ValueError("Bearer token required")
                auth = Auth.bearer(auth_kwargs["token"])
            elif auth_type == "custom":
                required = ["scheme", "principal", "credentials", "realm"]
                missing = [k for k in required if k not in auth_kwargs]
                if missing:
                    raise ValueError(f"Custom auth missing: {', '.join(missing)}")
                auth = Auth.custom(
                    auth_kwargs["scheme"],
                    auth_kwargs["principal"],
                    auth_kwargs["credentials"],
                    auth_kwargs["realm"],
                )
            else:
                raise ValueError(f"Unknown auth type: {auth_type}")

            # Build driver kwargs
            driver_kwargs = {
                "max_connection_pool_size": self.config["max_connection_pool_size"],
            }

            # Only add encrypted flag if it's False (True is default, but can't be used with +s URIs)
            if not self.config["encrypted"] or "+s" not in self.uri:
                driver_kwargs["encrypted"] = self.config.get("encrypted", True)

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

        auth = Auth.basic(self.username, self.password)

        driver_kwargs = {
            "max_connection_pool_size": self.config["max_connection_pool_size"],
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

    def get_pool_stats(self) -> dict:
        """Get connection pool statistics."""
        if not self._driver:
            return {}

        # Get pool info from driver (if available in your neo4j version)
        try:
            pool = self._driver._pool
            return {
                "in_use": len(pool._in_use),
                "available": len(pool._available),
            }
        except Exception:
            return {"status": "unable to retrieve pool stats"}

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
