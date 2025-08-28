.PHONY: init lint test clean help

# Default target
help:
	@echo "Available targets:"
	@echo "  init         - Initialize the development environment (runs install)"
	@echo "  install      - Install dependencies"
	@echo "  lint         - Run ruff linter and formatter"
	@echo "  format       - Format code with ruff"
	@echo "  test         - Run pytest"
	@echo "  check        - Run all checks (lint, type-check, test)"
	@echo "  clean        - Clean cache files"

install:
	uv sync --dev

lint:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -name "*.pyc" -delete
