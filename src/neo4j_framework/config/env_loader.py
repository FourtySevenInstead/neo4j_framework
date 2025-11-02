"""
Environment variable loader with validation and security features.
"""

import os
import logging
from typing import Optional, Dict, Any

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Valid range for connection pool size
POOL_SIZE_MIN = 1
POOL_SIZE_MAX = 500


class EnvironmentLoader:
    """
    Loads and manages environment variables securely.

    Supports custom .env file paths and environment variable prefixes
    for flexible integration into multiple projects.

    **Important:** override=False means existing environment variables
    (from the system/shell) take precedence over values in the .env file.
    This is intentional—don't override host system configuration.
    """

    def __init__(
        self,
        env_file: str = ".env",
        env_prefix: str = "NEO4J_",
    ):
        """
        Initialize the environment loader.

        Args:
            env_file: Path to the .env file (default: .env)
            env_prefix: Prefix for environment variables (default: NEO4J_)
        """
        self.env_file = env_file
        self.env_prefix = env_prefix
        self._loaded = False

    def load(self) -> bool:
        """Load environment variables from .env file."""
        if not self._loaded:
            if os.path.exists(self.env_file):
                self._loaded = load_dotenv(self.env_file, override=False)
                logger.debug(f"Environment variables loaded from {self.env_file}")
            else:
                logger.debug(f".env file not found at {self.env_file}")
                self._loaded = True  # Mark as attempted
        return self._loaded

    def get(
        self, key: str, default: Optional[str] = None, required: bool = False
    ) -> Optional[str]:
        """
        Get an environment variable value.

        Automatically prepends the configured prefix to the key.

        Args:
            key: Environment variable name (without prefix)
            default: Default value if not found
            required: If True, raises ValueError when key is not found

        Returns:
            Environment variable value or default

        Raises:
            ValueError: If required variable is not found
        """
        if not self._loaded:
            self.load()

        # Build the full key with prefix
        full_key = f"{self.env_prefix}{key}"
        value = os.getenv(full_key, default)

        if required and value is None:
            raise ValueError(
                f"Required environment variable not set. Please configure {full_key}"
            )

        return value

    def get_int(
        self,
        key: str,
        default: Optional[int] = None,
        min_val: Optional[int] = None,
        max_val: Optional[int] = None,
    ) -> Optional[int]:
        """
        Get an integer environment variable with optional bounds checking.

        Args:
            key: Environment variable name
            default: Default value if not found
            min_val: Minimum allowed value (inclusive)
            max_val: Maximum allowed value (inclusive)

        Returns:
            Integer value or default

        Raises:
            ValueError: If value is not a valid integer or out of bounds
        """
        value = self.get(key)
        if value is None:
            return default

        try:
            int_value = int(value)
        except ValueError:
            raise ValueError(
                f"Environment variable {self.env_prefix}{key} must be an integer, "
                f"got '{value}'"
            )

        # Validate bounds
        if min_val is not None and int_value < min_val:
            raise ValueError(
                f"Environment variable {self.env_prefix}{key} must be >= {min_val}, "
                f"got {int_value}"
            )
        if max_val is not None and int_value > max_val:
            raise ValueError(
                f"Environment variable {self.env_prefix}{key} must be <= {max_val}, "
                f"got {int_value}"
            )

        return int_value

    def get_float(
        self,
        key: str,
        default: Optional[float] = None,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ) -> Optional[float]:
        """
        Get a float environment variable with optional bounds checking.

        Args:
            key: Environment variable name
            default: Default value if not found
            min_val: Minimum allowed value (inclusive)
            max_val: Maximum allowed value (inclusive)

        Returns:
            Float value or default

        Raises:
            ValueError: If value is not a valid float or out of bounds
        """
        value = self.get(key)
        if value is None:
            return default

        try:
            float_value = float(value)
        except ValueError:
            raise ValueError(
                f"Environment variable {self.env_prefix}{key} must be a float, "
                f"got '{value}'"
            )

        # Validate bounds
        if min_val is not None and float_value < min_val:
            raise ValueError(
                f"Environment variable {self.env_prefix}{key} must be >= {min_val}, "
                f"got {float_value}"
            )
        if max_val is not None and float_value > max_val:
            raise ValueError(
                f"Environment variable {self.env_prefix}{key} must be <= {max_val}, "
                f"got {float_value}"
            )

        return float_value

    def get_bool(self, key: str, default: bool = False) -> bool:
        """
        Get a boolean environment variable.

        Accepts: true, 1, yes, on (case-insensitive)

        Args:
            key: Environment variable name
            default: Default value if not found

        Returns:
            Boolean value or default
        """
        value = self.get(key)
        if value is None:
            return default
        return value.lower() in ("true", "1", "yes", "on")

    def get_config(self) -> Dict[str, Any]:
        """
        Get all Neo4j configuration from environment variables.

        Returns:
            Dictionary containing all configuration values

        Raises:
            ValueError: If required variables are missing or invalid
        """
        self.load()

        return {
            "uri": self.get("URI", "neo4j://localhost:7687", required=True),
            "username": self.get("USERNAME", "neo4j"),
            "password": self.get("PASSWORD", required=True),
            "database": self.get("DATABASE", "neo4j"),
            # Default to True for security
            "encrypted": self.get_bool("ENCRYPTED", True),
            "max_connection_pool_size": self.get_int(
                "MAX_CONNECTION_POOL_SIZE",
                100,
                min_val=POOL_SIZE_MIN,
                max_val=POOL_SIZE_MAX,
            ),
            "connection_timeout": self.get_float(
                "CONNECTION_TIMEOUT", 30.0, min_val=0.1, max_val=300.0
            ),
            "max_transaction_retry_time": self.get_float(
                "MAX_TRANSACTION_RETRY_TIME", 30.0, min_val=1.0, max_val=300.0
            ),
            "kerberos_ticket": self.get("KERBEROS_TICKET"),
            "bearer_token": self.get("BEARER_TOKEN"),
            "client_cert_path": self.get("CLIENT_CERT_PATH"),
            "client_key_path": self.get("CLIENT_KEY_PATH"),
            "client_key_password": self.get("CLIENT_KEY_PASSWORD"),
        }
