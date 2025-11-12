"""
Logging configuration utilities.
"""

import logging


def setup_logging(env_file: str = ".env", env_prefix: str = "NEO4J_"):
    """
    Set up logging with configuration from environment variables.

    Args:
        env_file: Path to .env file
        env_prefix: Prefix for environment variables

    Returns:
        Logger instance
    """
    from ..config.env_loader import EnvironmentLoader

    loader = EnvironmentLoader(env_file, env_prefix)
    log_level = loader.get("LOG_LEVEL", "INFO")

    # Configure root logger
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    return logging.getLogger(__name__)
