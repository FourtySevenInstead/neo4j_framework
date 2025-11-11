"""
Query execution engine with read/write differentiation and logging.
"""

from __future__ import annotations

import logging

from typing import Any, Dict, LiteralString, Optional, cast, List

from neo4j import Query

from .base_query import BaseQuery

from stubs.neo4j import (
    Driver,
    ManagedTransaction,
    Record,
    Result,
    Session,
)  # noqa: F401

logger = logging.getLogger(__name__)


class QueryManager:
    """
    Query execution engine with parameterization and read/write optimization.

    Differentiates between read and write operations for automatic retry
    semantics and connection pool optimization.
    """

    def __init__(self, connection: Any):
        """
        Initialize query manager.

        Args:
            connection: Neo4j connection instance

        Raises:
            ValueError: If connection is None
        """
        if connection is None:
            raise ValueError("connection cannot be None")
        self.connection = connection
        logger.debug("QueryManager initialized")

    def execute_read(
        self,
        query_str: LiteralString | Query,
        params: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None,
    ) -> List[dict[str, Any]]:
        """
        Execute a read query with automatic retry semantics.

        Uses session.execute_read() for read optimization and automatic
        retries on transient failures.

        Args:
            query_str: Cypher query string (LiteralString for type safety)
            params: Query parameters
            database: Target database (optional)

        Returns:
            List of records as dictionaries

        Raises:
            Exception: If query execution fails
        """
        logger.debug("Executing read query...")

        def _read(tx: ManagedTransaction) -> Result:
            return tx.run(query_str, params or {})

        try:
            driver = cast(Driver, self.connection.get_driver())
            effective_db = database or cast(str, self.connection.database)
            with driver.session(database=effective_db) as session:
                result: Result = session.execute_read(_read)
                records: List[dict[str, Any]] = [record.data() for record in result]
                logger.debug(f"Read query returned {len(records)} records")
                return records
        except Exception as e:
            logger.error(f"Read query failed: {type(e).__name__}: {e}")
            raise

    def execute_write(
        self,
        query_str: LiteralString | Query,
        params: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None,
    ) -> Result:
        """
        Execute a write query with automatic retry semantics.

        Uses session.execute_write() for write optimization and automatic
        retries on transient failures.

        Args:
            query_str: Cypher query string (LiteralString for type safety)
            params: Query parameters
            database: Target database (optional)

        Returns:
            Query result

        Raises:
            Exception: If query execution fails
        """
        logger.debug("Executing write query...")

        def _write(tx: ManagedTransaction) -> Result:
            return tx.run(query_str, params or {})

        try:
            driver = cast(Driver, self.connection.get_driver())
            effective_db = database or cast(str, self.connection.database)
            with driver.session(database=effective_db) as session:
                result: Result = session.execute_write(_write)
                logger.debug("Write query executed successfully")
                return result
        except Exception as e:
            logger.error(f"Write query failed: {type(e).__name__}: {e}")
            raise

    def execute_query(
        self,
        query_str: LiteralString | Query,
        params: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None,
    ) -> List[dict[str, Any]]:
        """
        Execute a generic query without optimization.

        Use execute_read() or execute_write() for better performance.
        This method is provided for backward compatibility.

        Args:
            query_str: Cypher query string
            params: Query parameters
            database: Target database (optional)

        Returns:
            List of records as dictionaries
        """
        logger.warning(
            "execute_query() called. For better performance, use "
            "execute_read() for read operations or execute_write() for write operations."
        )
        query = BaseQuery(query_str, params)
        return query.execute(self.connection, database)
