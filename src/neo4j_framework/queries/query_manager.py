"""
Query execution engine with read/write differentiation and logging.
"""

import logging
from typing import Dict, Any, LiteralString

from neo4j import Query


logger = logging.getLogger(__name__)


class QueryManager:
    """
    Query execution engine with parameterization and read/write optimization.

    Differentiates between read and write operations for automatic retry
    semantics and connection pool optimization.
    """

    def __init__(self, connection):
        """
        Initialize query manager.

        Args:
            connection: Neo4j connection instance
        """
        if connection is None:
            raise ValueError("connection cannot be None")
        self.connection = connection
        logger.debug("QueryManager initialized")

    def execute_read(
        self,
        query_str: LiteralString | Query,
        params: Dict[str, Any] | None = None,
        database: str | None = None,
    ) -> list:
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

        def _read(tx):
            return tx.run(query_str, params or {})

        try:
            with self.connection.get_driver().session(
                database=database or self.connection.database
            ) as session:
                result = session.execute_read(_read)
                records = [record.data() for record in result]
                logger.debug(f"Read query returned {len(records)} records")
                return records
        except Exception as e:
            logger.error(f"Read query failed: {type(e).__name__}: {e}")
            raise

    def execute_write(
        self,
        query_str: LiteralString | Query,
        params: Dict[str, Any] | None = None,
        database: str | None = None,
    ):
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

        def _write(tx):
            return tx.run(query_str, params or {})

        try:
            with self.connection.get_driver().session(
                database=database or self.connection.database
            ) as session:
                result = session.execute_write(_write)
                logger.debug("Write query executed successfully")
                return result
        except Exception as e:
            logger.error(f"Write query failed: {type(e).__name__}: {e}")
            raise

    def execute_query(
        self,
        query_str: LiteralString | Query,
        params: Dict[str, Any] | None = None,
        database: str | None = None,
    ):
        """
        Execute a generic query without optimization.

        Use execute_read() or execute_write() for better performance.
        This method is provided for backward compatibility.

        Args:
            query_str: Cypher query string
            params: Query parameters
            database: Target database (optional)

        Returns:
            Query result
        """
        logger.warning(
            "execute_query() called. For better performance, use "
            "execute_read() for read operations or execute_write() for write operations."
        )
        from src.neo4j_framework.queries.base_query import BaseQuery

        query = BaseQuery(query_str, params)
        return query.execute(self.connection, database)
