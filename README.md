# Home library

This project is my experimental playground for building a personal knowledge base on top of my digital library. I’m currently exploring how to apply modern technologies like RAG (Retrieval-Augmented Generation) to my EPUB collection, adding a simple UI to search and reference interesting content. Over time, I plan to extend it with agentic patterns, MCP, and other emerging approaches to see how they work in practice.

## Features

- **EPUB Processing**: Parse and extract metadata, chapters, and text content from EPUB files
- **Text Chunking**: Intelligent text segmentation for optimal RAG performance
- **Vector Embeddings**: Generate embeddings using sentence-transformers models
- **Database Storage**: PostgreSQL database with SQLAlchemy ORM for scalable storage
- **CLI Interface**: Command-line tools for database management and EPUB uploads
- **Docker Support**: Easy database setup with Docker Compose
- **Comprehensive Testing**: Full test coverage for all components

## Architecture

The system is built around these core components:

- **Models**: Database schema for EPUBs, chapters, text chunks, and embeddings
- **Database Service**: Connection management and session handling
- **Vectorizer**: Text chunking and embedding generation
- **CLI Commands**: Database migration and EPUB upload tools

## Prerequisites
- Python 3.13 or newer
- uv installed: https://docs.astral.sh/uv/getting-started/installation/
- Docker and Docker Compose (for PostgreSQL database)

## Initialization

### Quick Start
For a complete development setup:

```bash
make dev-setup
```

This will install dependencies, start the database, and create the initial schema.

### Manual Setup
Set up the project and install all dependencies (including dev tools like pytest and ruff):

```bash
uv sync --dev
```

Or via Makefile helper:
```bash
make install-dev
```

This will create a local virtual environment (by default at .venv) and generate/update uv.lock.

### Database Setup
Start the PostgreSQL database:

```bash
make db-start
```

Create the database schema:

```bash
make migrate-create
```

## Configuration
Settings are loaded using pydantic-settings from environment variables and an optional .env file at the project root.

- Copy the example file and adjust values:
```bash
cp env.example .env
```

- Example .env values (see env.example):
```
APP_NAME=home-library
DEBUG=false
LOG_LEVEL=INFO
ENVIRONMENT=development

# Database settings
DATABASE_URL=postgresql://home_library_user:home_library_pass@localhost:5432/home_library
DATABASE_ECHO=false
```

- Access settings in code:
```python
from home_library.settings import settings

print(settings.app_name)  # "home-library" by default or value from .env
print(settings.database_url)  # Database connection string
```

Notes:
- Environment variables override .env values; defaults are used if neither is present.
- .env is read from the project root. If you run commands from the root, this will be picked up automatically.
- The system will automatically fall back to SQLite if PostgreSQL is not available (useful for development).

## Usage

### Database Management
Check database status:
```bash
make migrate-status
```

Create database tables:
```bash
make migrate-create
```

Reset database (drop and recreate):
```bash
make migrate-reset
```

### EPUB Management
Upload an EPUB file to the database:
```bash
make upload-epub FILE=path/to/your/book.epub
```

List all EPUBs in the database:
```bash
make upload-list
```

### CLI Commands
All commands are also available as direct CLI tools:

```bash
# Database management
uv run db-migrate status
uv run db-migrate create
uv run db-migrate reset

# EPUB management
uv run db-upload upload path/to/book.epub
uv run db-upload list
```

### Available Commands
- `db-migrate`: Database schema management
  - `create`: Create all tables
  - `drop`: Drop all tables
  - `reset`: Reset database (drop and recreate)
  - `status`: Check database health and status

- `db-upload`: EPUB file management
  - `upload <file>`: Upload EPUB to database
  - `list`: List all EPUBs in database

## Testing
Run tests with pytest:
```bash
uv run pytest -q
```

Or via Makefile helper:
```bash
make test
```

Run tests with coverage:
```bash
make test-cov
```

Run a specific test:
```bash
uv run pytest -q tests/test_main.py::test_hello_function_returns_expected_string
```

## Linting
Use ruff for linting and formatting:
```bash
uv run ruff check .
uv run ruff format .
```

Or via Makefile helper (runs both)
```bash
make lint
```

## Adding dependencies
Add a runtime dependency (written to [project.dependencies] in pyproject.toml and locked in uv.lock):
```bash
uv add requests
```

Add a development-only dependency (written to [dependency-groups.dev]):
```bash
uv add --dev black
```

Remove a dependency:
```bash
uv remove requests
```

Upgrade dependencies to latest allowed versions and update the lockfile:
```bash
uv sync --upgrade
```

Install exactly as pinned in uv.lock (reproducible):
```bash
uv sync --frozen --dev
```

## Project Layout
- `src/home_library/` — Main package source code
  - `models.py` — Database models and schema definitions
  - `database.py` — Database service and connection management
  - `settings.py` — Application configuration via pydantic-settings
  - `vectorizer.py` — Text chunking and vectorization utilities
  - `epub_processor.py` — EPUB parsing and processing
  - `embeddings.py` — Embedding model management
  - `cli/` — Command-line interface modules
    - `migrate.py` — Database migration commands
    - `upload.py` — EPUB upload and management commands
- `tests/` — Comprehensive test suite covering all components
- `docker-compose.yml` — PostgreSQL database setup
- `Makefile` — Development and deployment automation
- `pyproject.toml` — Build config and console script mappings

## Notes
- This project uses the src/ layout. Using `uv run` ensures imports resolve without manually setting PYTHONPATH.
- Python 3.13+ is required as specified in pyproject.toml.
- The system automatically falls back to SQLite if PostgreSQL is unavailable, making it easy to develop without Docker.
- All database operations are wrapped in transactions with automatic rollback on errors.
- The Makefile provides convenient shortcuts for common development tasks.

