"""
Performance monitoring utilities.
"""

import time
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


class Performance:
    """
    Performance utilities (e.g., timing operations).
    """

    @staticmethod
    def time_function(func: Callable[..., Any]) -> Callable[..., Any]:
        """
        Decorator to measure function execution time.

        Uses logging instead of print() for better control and integration
        with logging systems.

        Args:
            func: Function to measure

        Returns:
            Wrapped function
        """

        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result: Any = func(*args, **kwargs)
                end = time.time()
                elapsed = end - start
                logger.debug(f"{func.__name__} execution time: {elapsed:.3f} seconds")
                return result
            except Exception as e:
                end = time.time()
                elapsed = end - start
                logger.error(f"{func.__name__} failed after {elapsed:.3f} seconds: {e}")
                raise

        return wrapper
