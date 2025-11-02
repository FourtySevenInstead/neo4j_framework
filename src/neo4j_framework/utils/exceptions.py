"""
Custom exception classes for Neo4j framework.
Enables targeted error handling and recovery strategies.
"""


class Neo4jFrameworkException(Exception):
    """Base exception for all framework errors."""

    pass


class ConnectionError(Neo4jFrameworkException):
    """Raised when connection to Neo4j fails."""

    pass


class AuthenticationError(Neo4jFrameworkException):
    """Raised when authentication fails."""

    pass


class ConfigurationError(Neo4jFrameworkException):
    """Raised when configuration is invalid."""

    pass


class ValidationError(Neo4jFrameworkException):
    """Raised when input validation fails."""

    pass


class QueryError(Neo4jFrameworkException):
    """Raised when query execution fails."""

    pass


class TransactionError(Neo4jFrameworkException):
    """Raised when transaction execution fails."""

    pass
