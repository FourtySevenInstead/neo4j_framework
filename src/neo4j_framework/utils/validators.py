"""
Input validation utilities.
"""

import logging

logger = logging.getLogger(__name__)


class Validators:
    """
    Input validation utilities.
    """

    @staticmethod
    def validate_not_none(value, name: str):
        """
        Validate that a value is not None.

        Args:
            value: Value to validate
            name: Name of the parameter for error messages

        Raises:
            ValueError: If value is None
        """
        if value is None:
            logger.error(f"Validation failed: {name} cannot be None")
            raise ValueError(f"{name} cannot be None")

    @staticmethod
    def validate_string_not_empty(value: str, name: str):
        """
        Validate that a string is not empty.

        Args:
            value: String to validate
            name: Name of the parameter for error messages

        Raises:
            ValueError: If string is empty
        """
        if not value or not value.strip():
            logger.error(f"Validation failed: {name} cannot be empty")
            raise ValueError(f"{name} cannot be empty")

    @staticmethod
    def validate_positive_int(value: int, name: str):
        """
        Validate that a value is a positive integer.

        Args:
            value: Value to validate
            name: Name of the parameter for error messages

        Raises:
            ValueError: If value is not a positive integer
        """
        if not isinstance(value, int) or value <= 0:
            logger.error(f"Validation failed: {name} must be a positive integer")
            raise ValueError(f"{name} must be a positive integer")

    @staticmethod
    def validate_int(
        value: int,
        name: str,
        min_val: int | None = None,
        max_val: int | None = None,
    ) -> None:
        """Validate an integer with optional bounds checking."""
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{name} must be an integer, got {type(value).__name__}")

        if min_val is not None and value < min_val:
            raise ValueError(f"{name} must be >= {min_val}, got {value}")

        if max_val is not None and value > max_val:
            raise ValueError(f"{name} must be <= {max_val}, got {value}")

    @staticmethod
    def validate_float(
        value: float,
        name: str,
        min_val: float | None = None,
        max_val: float | None = None,
    ) -> None:
        """Validate a float with optional bounds checking."""
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{name} must be a number, got {type(value).__name__}")

        if min_val is not None and value < min_val:
            raise ValueError(f"{name} must be >= {min_val}, got {value}")

        if max_val is not None and value > max_val:
            raise ValueError(f"{name} must be <= {max_val}, got {value}")
