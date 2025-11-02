"""
Example usage of the Neo4j framework with correct import paths.
"""

from src.neo4j_framework.config.db_config import get_db_config
from src.neo4j_framework.db.connection import Neo4jConnection
from src.neo4j_framework.queries.query_manager import QueryManager
from src.neo4j_framework.utils.logger import setup_logging

logger = setup_logging()

# Load configuration
config = get_db_config()

# Create connection
conn = Neo4jConnection(
    uri=config["uri"],
    username=config["username"],
    password=config["password"],
    database=config["database"],
    encrypted=config["encrypted"],
)

try:
    # Connect to database
    conn.connect()

    # Create query manager
    query_manager = QueryManager(conn)

    # Execute a read query
    result = query_manager.execute_read("MATCH (n) RETURN n LIMIT 1")
    logger.info(f"Query result: {result}")

except Exception as e:
    logger.error(f"Error: {e}")
    raise
finally:
    # Always close connection
    conn.close()
