"""
CSV import functionality with path validation and logging.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, cast

from stubs.neo4j import (
    Driver,
    Query,
    Result,
)  # noqa: F401

logger = logging.getLogger(__name__)


class CSVImporter:
    """
    Efficient bulk data loading from CSV files with security validation.

    Validates file paths to prevent directory traversal attacks.
    """

    def __init__(self, connection: Any, allowed_dir: Optional[str] = None):
        """
        Initialize CSV importer.

        Args:
            connection: Neo4j connection instance
            allowed_dir: Optional directory to restrict CSV imports to.
                        If set, files must be within this directory.
        """
        if connection is None:
            raise ValueError("connection cannot be None")
        self.connection = connection
        self.allowed_dir = Path(allowed_dir).resolve() if allowed_dir else None
        logger.debug("CSVImporter initialized")

    def _validate_file_path(self, file_path: str) -> Path:
        """
        Validate CSV file path to prevent directory traversal.

        Args:
            file_path: Path to CSV file

        Returns:
            Resolved Path object

        Raises:
            ValueError: If path is invalid or outside allowed directory
        """
        try:
            path = Path(file_path).resolve()
        except Exception as e:
            raise ValueError(f"Invalid file path: {file_path}") from e

        # Check if file exists
        if not path.exists():
            raise ValueError(f"CSV file not found: {file_path}")

        # Check if file is readable
        if not os.access(path, os.R_OK):
            raise ValueError(f"CSV file not readable: {file_path}")

        # If allowed_dir is set, ensure file is within it
        if self.allowed_dir:
            try:
                path.relative_to(self.allowed_dir)
            except ValueError:
                raise ValueError(
                    f"CSV file must be within {self.allowed_dir}, " f"got {file_path}"
                )

        logger.debug(f"File path validated: {path}")
        return path

    def import_csv(
        self,
        file_path: str,
        query: Query,
        params: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None,
    ) -> Result:
        """
        Import CSV file using Neo4j LOAD CSV.

        Example query:
            LOAD CSV WITH HEADERS FROM $file_url AS row
            CREATE (:Node {name: row.name})

        Args:
            file_path: Path to CSV file
            query: Cypher query with LOAD CSV
            params: Additional query parameters
            database: Target database (optional)

        Returns:
            Query result

        Raises:
            ValueError: If file_path is invalid or outside allowed directory
            Exception: If query execution fails
        """
        if not file_path:
            raise ValueError("file_path cannot be None")
        if not query:
            raise ValueError("query cannot be None")

        # Validate and normalize path
        validated_path = self._validate_file_path(file_path)

        logger.info(f"Importing CSV file: {validated_path}")

        # Prepare parameters
        full_params = params or {}
        # Use file:/// URL scheme for Neo4j LOAD CSV
        full_params["file_url"] = validated_path.as_uri()

        try:
            driver = cast(Driver, self.connection.get_driver())
            effective_db = database or cast(str, self.connection.database)
            with driver.session(database=effective_db) as session:
                result = session.run(query, full_params)
                logger.info(f"CSV import completed: {validated_path}")
                return result
        except Exception as e:
            logger.error(f"CSV import failed: {type(e).__name__}: {e}")
            raise
