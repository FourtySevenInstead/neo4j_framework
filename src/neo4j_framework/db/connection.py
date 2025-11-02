"""
Neo4j database connection management with multiple authentication methods.
"""

import os
import logging
from typing import Optional, Union

from neo4j import GraphDatabase, Driver, Auth
from neo4j import basic_auth, kerberos_auth, bearer_auth, custom_auth
from neo4j.auth_management import (
    ClientCertificate,
    ClientCertificateProviders,
    ClientCertificateProvider,
)
from neo4j.exceptions import (
    ServiceUnavailable,
    AuthError,
    ClientError,
    DriverError,
)

logger = logging.getLogger(__name__)


class Neo4jConnection:
    """
    Manages Neo4j database connections with support for multiple
    authentication methods and secure configuration.
    """

    def __init__(
        self,
        uri: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: str = "neo4j",
        encrypted: bool = True,
        max_connection_pool_size: int = 100,
        connection_timeout: float = 30.0,
        max_transaction_retry_time: float = 30.0,
        connection_acquisition_timeout: float = 60.0,
        max_connection_lifetime: float = 3600.0,
        keep_alive: bool = True,
        liveness_check_timeout: Optional[float] = None,
        **kwargs,
    ):
        """
        Initialize Neo4j connection.

        Args:
            uri: Neo4j connection URI (neo4j://, neo4j+s://, bolt://, etc.)
            username: Username for basic authentication
            password: Password for basic authentication
            database: Target database name (default: neo4j)
            encrypted: Whether to use encryption (default: True for security)
            max_connection_pool_size: Maximum connections in pool (1-500, default: 100)
            connection_timeout: TCP connection timeout in seconds (default: 30.0)
            max_transaction_retry_time: Max retry time for transactions (default: 30.0)
            connection_acquisition_timeout: Timeout for acquiring connection (default: 60.0)
            max_connection_lifetime: Max lifetime of connection (default: 3600.0)
            keep_alive: Enable TCP keep-alive (default: True)
            liveness_check_timeout: Connection liveness check timeout

        Raises:
            ValueError: If pool size is out of valid range
        """
        # Validate pool size
        if not (1 <= max_connection_pool_size <= 500):
            raise ValueError(
                f"max_connection_pool_size must be between 1 and 500, "
                f"got {max_connection_pool_size}"
            )

        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.encrypted = encrypted

        # Driver configuration
        self.config = {
            "max_connection_pool_size": max_connection_pool_size,
            "connection_timeout": connection_timeout,
            "max_transaction_retry_time": max_transaction_retry_time,
            "connection_acquisition_timeout": connection_acquisition_timeout,
            "max_connection_lifetime": max_connection_lifetime,
            "keep_alive": keep_alive,
            "encrypted": encrypted,
        }

        if liveness_check_timeout is not None:
            self.config["liveness_check_timeout"] = liveness_check_timeout

        self.config.update(kwargs)
        self._driver: Optional[Driver] = None

    def _create_basic_auth(self) -> Auth:
        """Create basic authentication."""
        if not self.username or not self.password:
            raise ValueError("Username and password required for basic authentication")
        return basic_auth(self.username, self.password)

    def _create_kerberos_auth(self, ticket: str) -> Auth:
        """Create Kerberos authentication."""
        return kerberos_auth(ticket)

    def _create_bearer_auth(self, token: str) -> Auth:
        """Create Bearer token authentication."""
        return bearer_auth(token)

    def _create_custom_auth(
        self, principal: str, credentials: str, realm: str, scheme: str, **parameters
    ) -> Auth:
        """Create custom authentication."""
        return custom_auth(principal, credentials, realm, scheme, **parameters)

    def connect(
        self,
        auth: Optional[Auth] = None,
        auth_type: str = "basic",
        client_certificate: Optional[ClientCertificateProvider] = None,
        **auth_params,
    ) -> Driver:
        """
        Establish connection to Neo4j database.

        Args:
            auth: Pre-configured Auth object
            auth_type: Authentication type ('basic', 'kerberos', 'bearer', 'custom')
            client_certificate: Client certificate provider for mTLS
            **auth_params: Additional authentication parameters

        Returns:
            Driver: Neo4j driver instance

        Raises:
            ValueError: If authentication parameters are invalid
            ServiceUnavailable: If Neo4j service is unavailable
            AuthError: If authentication fails
            ClientError: If there's a client-side error
        """
        if self._driver is not None:
            logger.warning("Driver already exists. Closing existing driver.")
            self.close()

        # Determine authentication
        if auth is None:
            if auth_type == "basic":
                auth = self._create_basic_auth()
            elif auth_type == "kerberos":
                ticket = auth_params.get("ticket")
                if not ticket:
                    raise ValueError("Kerberos ticket required")
                auth = self._create_kerberos_auth(ticket)
            elif auth_type == "bearer":
                token = auth_params.get("token")
                if not token:
                    raise ValueError("Bearer token required")
                auth = self._create_bearer_auth(token)
            elif auth_type == "custom":
                required_params = ["principal", "credentials", "realm", "scheme"]
                for param in required_params:
                    if param not in auth_params:
                        raise ValueError(
                            f"'{param}' required for custom authentication"
                        )
                auth = self._create_custom_auth(**auth_params)
            else:
                raise ValueError(f"Unknown auth_type: {auth_type}")

        driver_config = self.config.copy()
        if client_certificate:
            driver_config["client_certificate"] = client_certificate

        try:
            logger.info(f"Connecting to Neo4j at {self.uri}")
            self._driver = GraphDatabase.driver(self.uri, auth=auth, **driver_config)

            # Verify connectivity with timeout handling
            self._driver.verify_connectivity()
            logger.info(f"Successfully connected to Neo4j at {self.uri}")

            return self._driver

        except AuthError as e:
            logger.error("Authentication failed. Check credentials.")
            raise
        except ServiceUnavailable as e:
            logger.error(f"Neo4j service unavailable at {self.uri}")
            raise
        except ClientError as e:
            logger.error(f"Client error during connection: {e}")
            raise
        except DriverError as e:
            logger.error(f"Driver error: {e}")
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error connecting to Neo4j: {type(e).__name__}: {e}"
            )
            raise

    def connect_with_mtls(
        self,
        cert_path: str,
        key_path: Optional[str] = None,
        key_password: Optional[Union[str, callable]] = None,
        auth: Optional[Auth] = None,
    ) -> Driver:
        """
        Connect with mutual TLS (client certificate authentication).

        Args:
            cert_path: Path to client certificate file
            key_path: Path to private key file (if separate)
            key_password: Password to decrypt private key (string or callable)
            auth: Authentication (still required with mTLS)

        Returns:
            Driver: Neo4j driver instance

        Raises:
            ValueError: If certificate files don't exist or are not readable
            FileNotFoundError: If certificate paths are invalid
        """
        # Validate certificate file exists and is readable
        if not os.path.exists(cert_path):
            raise ValueError(f"Certificate file not found: {cert_path}")
        if not os.access(cert_path, os.R_OK):
            raise ValueError(f"Certificate file not readable: {cert_path}")

        if key_path and not os.path.exists(key_path):
            raise ValueError(f"Key file not found: {key_path}")
        if key_path and not os.access(key_path, os.R_OK):
            raise ValueError(f"Key file not readable: {key_path}")

        logger.debug(f"Using mTLS with certificate: {cert_path}")

        cert_provider = ClientCertificateProviders.static(
            ClientCertificate(
                certfile=cert_path, keyfile=key_path, password=key_password
            )
        )

        return self.connect(
            auth=auth or self._create_basic_auth(), client_certificate=cert_provider
        )

    def get_driver(self) -> Driver:
        """
        Get the current driver instance.

        Returns:
            Driver: Neo4j driver instance

        Raises:
            RuntimeError: If not connected
        """
        if self._driver is None:
            raise RuntimeError("Not connected. Call connect() first.")
        return self._driver

    def is_connected(self) -> bool:
        """Check if driver is connected."""
        return self._driver is not None

    def close(self):
        """Close the driver connection."""
        if self._driver is not None:
            try:
                self._driver.close()
                logger.info("Neo4j connection closed")
            except Exception as e:
                logger.error(f"Error closing connection: {e}")
            finally:
                self._driver = None

    def __enter__(self):
        """Context manager entry."""
        if not self.is_connected():
            self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
