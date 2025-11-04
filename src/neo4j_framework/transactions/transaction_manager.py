"""
Transaction management with context manager support.
"""

from __future__ import annotations

import logging
from typing import Callable, Any, Dict, Optional, TYPE_CHECKING, cast, TypeVar


logger = logging.getLogger(__name__)

T = TypeVar("T")

if TYPE_CHECKING:
    from neo4j_framework.stubs.neo4j import (
        Driver,
        ManagedTransaction,
        Result,
        Session,
    )  # noqa: F401


class TransactionManager:
    """
    Handles both managed and explicit transactions with context manager support.
    """

    def __init__(self, connection: Any):
        """
        Initialize transaction manager.

        Args:
            connection: Neo4j connection instance

        Raises:
            ValueError: If connection is None
        """
        if connection is None:
            raise ValueError("connection cannot be None")
        self.connection = connection
        self._session: Optional[Session] = None
        logger.debug("TransactionManager initialized")

    def run_in_transaction(
        self,
        tx_function: Callable[[ManagedTransaction], T],
        database: Optional[str] = None,
    ) -> T:
        """
        Execute a function within a managed transaction.

        Args:
            tx_function: Function that takes transaction object and executes logic
            database: Target database (optional)

        Returns:
            Result of tx_function

        Raises:
            Exception: If transaction fails
        """

        logger.debug("Starting managed transaction...")

        driver = cast(Driver, self.connection.get_driver())
        effective_db = database or cast(str, self.connection.database)

        try:
            with driver.session(database=effective_db) as session:
                result = session.execute_write(tx_function)
                logger.debug("Transaction completed successfully")
                return result
        except Exception as e:
            logger.error(f"Transaction failed: {type(e).__name__}: {e}")
            raise

    def explicit_transaction(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None,
    ) -> Result:
        """
        Execute an explicit transaction with a single query.

        Args:
            query: Cypher query string
            params: Query parameters
            database: Target database (optional)

        Returns:
            Query result

        Raises:
            ValueError: If query is None
            Exception: If transaction fails
        """
        if not query:
            raise ValueError("query cannot be None")

        logger.debug("Executing explicit transaction...")

        def tx_func(tx: ManagedTransaction) -> Result:
            return tx.run(query, params or {})

        return self.run_in_transaction(tx_func, database)

    def __enter__(self) -> Session:
        """
        Context manager entry.

        Opens a session for the transaction block.

        Returns:
            Session object
        """
        logger.debug("Entering transaction context...")
        driver = cast(Driver, self.connection.get_driver())
        self._session = driver.session(database=cast(str, self.connection.database))
        return self._session

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: object,
    ) -> bool:
        """
        Context manager exit.

        Closes the session and handles errors if necessary.

        Returns:
            False to not suppress exceptions
        """
        if self._session:
            try:
                self._session.close()
                if exc_type:
                    logger.error(
                        f"Transaction context exited with exception: {exc_type.__name__}"
                    )
                else:
                    logger.debug("Transaction context closed successfully")
            except Exception as e:
                logger.error(f"Error closing transaction session: {e}")
            finally:
                self._session = None

        # Don't suppress exceptions
        return False
