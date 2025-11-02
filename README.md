# Neo4j Framework

A **production-ready, type-safe Python library** for seamless Neo4j database integration. Built for reliability, security, and ease of use across any project.

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![Tests Passing](https://img.shields.io/badge/tests-113%2F113%20passing-brightgreen)](tests/)
[![Type Checking](https://img.shields.io/badge/type%20checking-pyright-blue)](pyrightconfig.json)
[![MIT License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## Overview

The Neo4j Framework provides a complete abstraction layer over Neo4j's Python driver, eliminating boilerplate while maintaining full power and flexibility. It's designed specifically for developers who want safety guarantees through type hints, security by default, and zero configuration complexity.

### Key Characteristics

- **Type-Safe** - Full type hints enable IDE autocomplete and catch errors before runtime
- **Security-First** - Parameterized queries, path validation, and bounds checking built-in
- **Zero Boilerplate** - Environment-based config, automatic connection pooling, clean APIs
- **Production Ready** - 113 comprehensive tests covering edge cases and error scenarios
- **Framework Agnostic** - Drop into Flask, FastAPI, Django, or standalone scripts

---

## Installation

### From Source

```bash
git clone https://github.com/yourusername/neo4j_framework.git
cd neo4j_framework

# Install in development mode
pip install -e .

# Or standard install
pip install .
```

### Dependencies

```
neo4j>=5.0.0,<6.0.0
python-dotenv>=1.0.0
typing-extensions>=4.0.0
```

---

## Quick Start

### 1. Configure Environment

Create a `.env` file in your project:

```bash
# .env
NEO4J_URI=neo4j://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_secure_password
NEO4J_DATABASE=neo4j
NEO4J_ENCRYPTED=true
NEO4J_MAX_CONNECTION_POOL_SIZE=50
```

### 2. Initialize & Execute

```python
from src.neo4j_framework.config.env_loader import EnvironmentLoader
from src.neo4j_framework.db.connection import Neo4jConnection
from src.neo4j_framework.queries.query_manager import QueryManager

# Load configuration
loader = EnvironmentLoader(env_prefix="NEO4J_")
config = loader.get_config()

# Create connection
conn = Neo4jConnection(
    uri=config["uri"],
    username=config["username"],
    password=config["password"],
    database=config["database"],
    encrypted=config["encrypted"],
)

# Connect
conn.connect()

# Execute query
qm = QueryManager(conn)
results = qm.execute_read(
    "MATCH (p:Person) WHERE p.age > $min_age RETURN p.name, p.age",
    params={"min_age": 18}
)

for record in results:
    print(f"{record['name']}: {record['age']} years old")

# Cleanup
conn.close()
```

### 3. Using Context Managers (Recommended)

```python
with Neo4jConnection(
    uri="neo4j://localhost:7687",
    username="neo4j",
    password="password"
) as conn:
    qm = QueryManager(conn)
    results = qm.execute_read("MATCH (n) RETURN COUNT(n) as count")
    print(f"Total nodes: {results[0]['count']}")
    # Connection automatically closed
```

---

## Core Components

### Environment Loader (`config.env_loader`)

Manages configuration with validation and security:

```python
from src.neo4j_framework.config.env_loader import EnvironmentLoader

loader = EnvironmentLoader(env_prefix="NEO4J_")

# Simple value
uri = loader.get("URI")

# With validation
pool_size = loader.get_int("MAX_CONNECTION_POOL_SIZE", min_val=1, max_val=500)
timeout = loader.get_float("CONNECTION_TIMEOUT", min_val=0.1, max_val=300.0)
encrypted = loader.get_bool("ENCRYPTED", default=True)

# Complete config
config = loader.get_config()
```

### Connection Manager (`db.connection`)

Handles all connection logic with multiple auth methods:

```python
from src.neo4j_framework.db.connection import Neo4jConnection

# Basic connection
conn = Neo4jConnection(
    uri="neo4j://localhost:7687",
    username="neo4j",
    password="password",
    database="neo4j",
    encrypted=True,
    max_connection_pool_size=50
)
conn.connect()

# mTLS connection
conn.connect_with_mtls(
    cert_path="/path/to/client.crt",
    key_path="/path/to/client.key"
)

# Kerberos authentication
conn.connect(auth_type="kerberos", ticket=kerberos_ticket)

# Check status
if conn.is_connected():
    print("Connected!")
```

### Query Manager (`queries.query_manager`)

Execute parameterized queries safely:

```python
from src.neo4j_framework.queries.query_manager import QueryManager

qm = QueryManager(conn)

# Read operation (optimized for read semantics)
results = qm.execute_read(
    "MATCH (n:Person) RETURN n LIMIT 10"
)

# Write operation (optimized for write semantics)
result = qm.execute_write(
    "CREATE (p:Person {name: $name}) RETURN p",
    params={"name": "Alice"}
)

# Parameterized queries prevent injection
user_input = "Alice'; DROP TABLE users; --"
results = qm.execute_read(
    "MATCH (p:Person) WHERE p.name = $name RETURN p",
    params={"name": user_input}  # Safe!
)
```

### Transaction Manager (`transactions.transaction_manager`)

Handle complex multi-step operations:

```python
from src.neo4j_framework.transactions.transaction_manager import TransactionManager

txm = TransactionManager(conn)

def create_relationship(tx):
    """Multi-step transaction function."""
    # Create two people
    tx.run("CREATE (p1:Person {name: $n1})", {"n1": "Alice"})
    tx.run("CREATE (p2:Person {name: $n2})", {"n2": "Bob"})
    # Create relationship
    tx.run(
        "MATCH (p1:Person {name: $n1}), (p2:Person {name: $n2}) "
        "CREATE (p1)-[:KNOWS]->(p2)",
        {"n1": "Alice", "n2": "Bob"}
    )

# Execute transaction
txm.run_in_transaction(create_relationship)
```

### CSV Importer (`importers.csv_importer`)

Bulk import data securely:

```python
from src.neo4j_framework.importers.csv_importer import CSVImporter

# Restrict imports to specific directory for security
importer = CSVImporter(conn, allowed_dir="/data/csv_imports")

# Import CSV
query = """
LOAD CSV WITH HEADERS FROM $file_url AS row
CREATE (p:Person {
    name: row.name,
    email: row.email,
    age: toInteger(row.age)
})
"""

importer.import_csv("/data/csv_imports/people.csv", query)

# Attempting directory traversal fails
importer.import_csv("/../../../etc/passwd", query)  # Raises ValueError
```

---

## Security Features

### Parameterized Queries

All queries use parameter binding to prevent Cypher injection:

```python
# ✅ SAFE - Uses parameter binding
qm.execute_read(
    "MATCH (n) WHERE n.name = $name RETURN n",
    params={"name": user_input}
)

# ❌ UNSAFE - Never do this!
qm.execute_read(
    f"MATCH (n) WHERE n.name = '{user_input}' RETURN n"
)
```

### Input Validation

All inputs are validated with bounds checking:

```python
from src.neo4j_framework.utils.validators import Validators

# Validate not None
Validators.validate_not_none(value, "parameter_name")

# Validate bounds
Validators.validate_int(pool_size, "pool_size", min_val=1, max_val=500)
Validators.validate_float(timeout, "timeout", min_val=0.1, max_val=300.0)
```

### Path Traversal Prevention

CSV imports are restricted to allowed directories:

```python
# Directory traversal attempts are blocked
importer = CSVImporter(conn, allowed_dir="/safe/directory")
importer.import_csv("/etc/passwd", query)  # ValueError: outside allowed directory
```

### Secure Defaults

- Encryption enabled by default
- Credentials required for all connections
- Pool sizes bounded (1-500)
- Timeouts enforced (0.1s - 300s)

---

## Architecture

### Directory Structure

```
neo4j_framework/
├── src/neo4j_framework/          # The installable package
│   ├── __init__.py               # Package exports
│   ├── py.typed                  # Type hints marker
│   ├── config/                   # Configuration management
│   ├── db/                       # Connection management
│   ├── queries/                  # Query execution
│   ├── transactions/             # Transaction handling
│   ├── importers/                # CSV bulk import
│   └── utils/                    # Utilities & exceptions
├── tests/                        # 113 comprehensive tests
├── examples/                     # Usage examples
├── README.md                     # This file
├── requirements.txt              # Dependencies
├── pyproject.toml               # Build configuration
└── pyrightconfig.json           # Type checking config
```

### Component Relationships

```
┌─────────────────────────────────────────┐
│   Environment Configuration             │
│   (env_loader.py)                       │
│   - Loads from .env                     │
│   - Validates types & bounds            │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│   Connection Manager                    │
│   (connection.py)                       │
│   - Multiple auth methods               │
│   - Connection pooling                  │
└──────────────────┬──────────────────────┘
                   │
                   ├─────────────────────────┐
                   │                         │
                   ▼                         ▼
        ┌────────────────────┐    ┌──────────────────┐
        │  Query Manager     │    │ Transaction      │
        │  (query_manager)   │    │ Manager          │
        │  - Read/Write      │    │ (tx_manager)     │
        │  - Parameterized   │    │ - Multi-step     │
        └────────────────────┘    │ - Rollback       │
                                  └──────────────────┘
                   │
                   ├─────────────────────────┐
                   │                         │
                   ▼                         ▼
        ┌────────────────────┐    ┌──────────────────┐
        │  CSV Importer      │    │  Query Templates │
        │  (csv_importer)    │    │  (base_query.py) │
        │  - Bulk load       │    │  - Reusable      │
        │  - Path validation │    │  - Type-safe     │
        └────────────────────┘    └──────────────────┘
```

---

## Testing

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test class
pytest tests/test_framework.py::TestEnvironmentLoader -v

# Specific test
pytest tests/test_framework.py::TestEnvironmentLoader::test_initialization -v

# With coverage
pytest tests/ --cov=src/neo4j_framework --cov-report=html
```

### Test Coverage

- **113 total tests** - 100% passing
- **20 unit tests** - Edge cases and boundary conditions
- **39 core tests** - Framework functionality
- **28 integration tests** - Real-world workflows
- **12 security tests** - Injection prevention, validation
- **14 additional tests** - Performance, concurrency

---

## Examples

### Example 1: Simple Query

```python
from src.neo4j_framework.config.db_config import get_db_config
from src.neo4j_framework.db.connection import Neo4jConnection
from src.neo4j_framework.queries.query_manager import QueryManager

config = get_db_config()
conn = Neo4jConnection(**config)
conn.connect()

qm = QueryManager(conn)
results = qm.execute_read("MATCH (p:Person) RETURN p LIMIT 5")

for record in results:
    print(record)

conn.close()
```

### Example 2: Transaction

```python
from src.neo4j_framework.transactions.transaction_manager import TransactionManager

txm = TransactionManager(conn)

def create_users(tx):
    for i in range(10):
        tx.run(
            "CREATE (u:User {id: $id, name: $name})",
            {"id": f"user_{i}", "name": f"User {i}"}
        )

txm.run_in_transaction(create_users)
```

### Example 3: Multi-Database

```python
# Create separate connections
primary = Neo4jConnection(
    uri="neo4j://primary:7687",
    username="neo4j",
    password="password",
    database="production"
)

analytics = Neo4jConnection(
    uri="neo4j://analytics:7687",
    username="neo4j",
    password="password",
    database="analytics"
)

primary.connect()
analytics.connect()

qm_prod = QueryManager(primary)
qm_analytics = QueryManager(analytics)

# Use independently
prod_data = qm_prod.execute_read("MATCH (n) RETURN COUNT(n)")
analytics_data = qm_analytics.execute_read("MATCH (n) RETURN COUNT(n)")
```

---

## Configuration Reference

### Environment Variables

| Variable | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `NEO4J_URI` | str | - | ✅ | Connection URI (neo4j://host:port) |
| `NEO4J_USERNAME` | str | - | ✅ | Authentication username |
| `NEO4J_PASSWORD` | str | - | ✅ | Authentication password |
| `NEO4J_DATABASE` | str | neo4j | - | Target database |
| `NEO4J_ENCRYPTED` | bool | true | - | Enable SSL/TLS |
| `NEO4J_MAX_CONNECTION_POOL_SIZE` | int | 100 | - | Pool size (1-500) |
| `NEO4J_CONNECTION_TIMEOUT` | float | 30.0 | - | Connection timeout (0.1-300s) |
| `NEO4J_MAX_TRANSACTION_RETRY_TIME` | float | 30.0 | - | Transaction retry time (1-300s) |

### Custom Prefixes

Use different prefixes for multiple projects:

```bash
# Project A
PROJECT_A_URI=neo4j://server-a:7687
PROJECT_A_USERNAME=neo4j
PROJECT_A_PASSWORD=password_a

# Project B
PROJECT_B_URI=neo4j://server-b:7687
PROJECT_B_USERNAME=neo4j
PROJECT_B_PASSWORD=password_b
```

```python
loader_a = EnvironmentLoader(env_prefix="PROJECT_A_")
loader_b = EnvironmentLoader(env_prefix="PROJECT_B_")
```

---

## Troubleshooting

### Connection Issues

```python
# Check URI format
# ✅ neo4j://localhost:7687
# ✅ neo4j+s://localhost:7687 (encrypted)
# ❌ localhost:7687 (missing protocol)

# Verify credentials
assert os.getenv("NEO4J_USERNAME") is not None
assert os.getenv("NEO4J_PASSWORD") is not None

# Test connectivity
import socket
socket.create_connection(("localhost", 7687), timeout=5)
```

### Query Issues

```python
# Use parameters, not string concatenation
# ✅
results = qm.execute_read(
    "MATCH (n) WHERE n.name = $name RETURN n",
    params={"name": user_input}
)

# ❌
results = qm.execute_read(f"MATCH (n) WHERE n.name = '{user_input}' RETURN n")
```

### Performance

```python
# Add database indexes
qm.execute_write("CREATE INDEX ON :Person(id)")

# Use LIMIT for large result sets
results = qm.execute_read("MATCH (n) RETURN n LIMIT 1000")

# Increase pool size for concurrency
conn = Neo4jConnection(
    uri="...",
    username="...",
    password="...",
    max_connection_pool_size=200
)
```

---

## API Reference

### EnvironmentLoader

```python
loader = EnvironmentLoader(env_file=".env", env_prefix="NEO4J_")

# Get values
value = loader.get("KEY", default=None, required=False)
int_val = loader.get_int("INT_KEY", default=0, min_val=None, max_val=None)
float_val = loader.get_float("FLOAT_KEY", default=0.0, min_val=None, max_val=None)
bool_val = loader.get_bool("BOOL_KEY", default=False)

# Get complete config
config = loader.get_config()  # Returns all Neo4j config
```

### Neo4jConnection

```python
conn = Neo4jConnection(
    uri="neo4j://...",
    username="neo4j",
    password="...",
    database="neo4j",
    encrypted=True,
    max_connection_pool_size=100
)

conn.connect()
conn.connect_with_mtls(cert_path="...", key_path="...")
driver = conn.get_driver()
connected = conn.is_connected()
conn.close()
```

### QueryManager

```python
qm = QueryManager(conn)

# Read query
results = qm.execute_read(query, params=None, database=None)

# Write query
result = qm.execute_write(query, params=None, database=None)

# Generic query (use execute_read/execute_write instead)
result = qm.execute_query(query, params=None, database=None)
```

### TransactionManager

```python
txm = TransactionManager(conn)

# Execute in transaction
result = txm.run_in_transaction(func, database=None)

# Context manager
with txm as session:
    session.execute_write(lambda tx: tx.run(query))
```

### CSVImporter

```python
importer = CSVImporter(conn, allowed_dir="/safe/directory")

# Import CSV
result = importer.import_csv(file_path, query, params=None, database=None)

# Validate path
validated_path = importer._validate_file_path(file_path)
```

---

## Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/your-feature`)
3. **Write tests** for new functionality
4. **Run tests** (`pytest tests/ -v`)
5. **Type check** (`pyright`)
6. **Commit** with clear messages
7. **Push** to your fork
8. **Submit** a Pull Request

### Development Setup

```bash
# Clone and setup
git clone https://github.com/yourusername/neo4j_framework.git
cd neo4j_framework

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install with dev dependencies
pip install -e .
pip install pytest pytest-cov pyright

# Run tests
pytest tests/ -v

# Type checking
pyright
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Support

- 📖 **Documentation**: See examples and usage sections above
- 🐛 **Issues**: Report bugs on GitHub Issues
- 💬 **Discussions**: Ask questions on GitHub Discussions
- 📧 **Email**: Contact maintainers for direct questions

---

## Changelog

### Version 2.0.0 (Current)

✅ **Complete package reorganization** with `src/` layout
✅ **113 comprehensive tests** (100% passing)
✅ **Pyright integration** for strict type checking
✅ **Full IDE support** (gd/gI/K navigation in LazyVim/VSCode)
✅ **Security hardened** with injection prevention and path validation
✅ **Multi-database support** with explicit database selection
✅ **mTLS authentication** for enterprise deployments
✅ **Connection pooling** with configurable bounds (1-500)
✅ **Transaction management** with automatic retry semantics
✅ **CSV bulk import** with directory restriction
✅ **Query templates** for reusable patterns
✅ **Custom exceptions** for targeted error handling
✅ **Performance monitoring** with built-in timing
✅ **Comprehensive documentation** with examples

---

**Built with ❤️ for developers who value type safety and security.**
