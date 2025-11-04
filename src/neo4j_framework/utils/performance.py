"""
Performance monitoring utilities.
"""

import time
import logging
from typing import Callable, ParamSpec, TypeVar
import functools

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


class Performance:
    """
    Performance utilities (e.g., timing operations).
    """

    @staticmethod
    def time_function(func: Callable[P, R]) -> Callable[P, R]:
        """
        Decorator to measure function execution time.

        Uses logging instead of print() for better control and integration
        with logging systems.

        Args:
            func: Function to measure

        Returns:
            Wrapped function
        """

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start = time.time()
            try:
                result: R = func(*args, **kwargs)
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
