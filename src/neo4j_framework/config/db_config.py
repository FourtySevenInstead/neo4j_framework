from typing import Dict, Any
from .env_loader import EnvironmentLoader


def get_db_config(env_file: str = ".env", env_prefix: str = "NEO4J_") -> Dict[str, Any]:
    """
    Get database configuration from environment variables.

    Args:
        env_file: Path to .env file
        env_prefix: Prefix for environment variables

    Returns:
        Dictionary with database configuration
    """
    loader = EnvironmentLoader(env_file, env_prefix)
    return loader.get_config()
