"""
Base query class with parameter safety and logging.
"""

import logging
from typing import Any, Dict, LiteralString, Optional, Callable, TypeVar

from ..db.connection import Neo4jConnection
from stubs.neo4j import Session, Result  # noqa: F401

from neo4j import Query

logger = logging.getLogger(__name__)


class BaseQuery:
    """
    Base class for queries with parameterization and logging.

    Enforces use of parameterized queries to prevent injection attacks.
    """

    def __init__(
        self,
        query_str: LiteralString | Query,
        params: Dict[str, Any] | None = None,
    ):
        """
        Initialize query.

        Args:
            query_str: Query string (LiteralString for type safety)
                or Query object
            params: Query parameters

        Raises:
            ValueError: If query_str is None or invalid
        """
        if not query_str:
            raise ValueError("query_str cannot be None")
        self.query_str = query_str
        self.params = params or {}
        logger.debug(f"Query initialized with {len(self.params)} parameters")

    def execute(
        self,
        connection: "Neo4jConnection",
        database: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Execute the query.

        Args:
            connection: Neo4j connection instance
            database: Target database (optional, uses connection
                default if not specified)

        Returns:
            Query result

        Raises:
            RuntimeError: If connection is not established
        """
        logger.debug("Executing query...")

        try:
            with connection.get_driver().session(
                database=database or connection.database
            ) as session:
                result = session.run(self.query_str, self.params)
                logger.debug("Query executed successfully")
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Query execution failed: {type(e).__name__}: {e}")
            raise
