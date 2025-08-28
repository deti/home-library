# home-library

A minimal home library project using uv for environment and dependency management. It features a simple, testable "Hello, World!" example and a src/ package layout.

## Prerequisites
- Python 3.13 or newer
- uv installed: https://docs.astral.sh/uv/getting-started/installation/

## Initialization
Set up the project and install all dependencies (including dev tools like pytest and ruff):

```bash
uv sync --dev
```
Or via Makefile helper
```bash
make init
```

This will create a local virtual environment (by default at .venv) and generate/update uv.lock.

## Running
There are multiple ways to run the app (all print a greeting):

1) Console script entry point (defined in pyproject):
```bash
uv run home-library
```

2) Run the package module directly:
```bash
uv run python -m home_library.main
```

3) Run the root script:
```bash
uv run python main.py
```

Expected outputs:
- home-library and python -m home_library.main -> "Hello, World!"
- python main.py -> "Hello from home-library!"

## Testing
Run tests with pytest:
```bash
uv run pytest -q
```

# or via Makefile helper
```bash
make test
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

## Project layout
- src/home_library/main.py — package entry points (hello() and main())
- main.py — simple root-level script
- tests/ — pytest tests covering both entry points
- pyproject.toml — build config and console script mapping (home-library -> home_library.main:main)

## Notes
- This project uses the src/ layout. Using `uv run` ensures imports resolve without manually setting PYTHONPATH.
- Python 3.13+ is required as specified in pyproject.toml.

