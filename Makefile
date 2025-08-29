.PHONY: help install install-dev test test-cov lint format clean db-start db-stop db-reset migrate migrate-create migrate-drop migrate-reset migrate-status upload upload-list show-settings epub-info vectorize-epub dev-setup upload-epub

help:  ## Show this help message
	@echo "Home Library Management System"
	@echo "=============================="
	@echo ""
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install production dependencies
	uv pip install -e .

install-dev:  ## Install development dependencies
	uv pip install -e ".[dev]"

test:  ## Run tests
	uv run pytest tests/ -v

test-cov:  ## Run tests with coverage
	uv run pytest tests/ --cov=src/home_library --cov-report=html --cov-report=term-missing

lint:  ## Run linting
	uv run ruff check src/ tests/

format:  ## Format code
	uv run ruff format src/ tests/

clean:  ## Clean up generated files
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -f home_library.db

db-start:  ## Start PostgreSQL database with Docker
	docker-compose up -d postgres
	@echo "Waiting for database to be ready..."
	@until docker-compose exec -T postgres pg_isready -U home_library_user -d home_library; do sleep 1; done
	@echo "Database is ready!"

db-stop:  ## Stop PostgreSQL database
	docker-compose down

db-reset:  ## Reset PostgreSQL database (stop, remove volumes, start)
	docker-compose down -v
	docker-compose up -d postgres
	@echo "Waiting for database to be ready..."
	@until docker-compose exec -T postgres pg_isready -U home_library_user -d home_library; do sleep 1; done
	@echo "Database is ready!"

migrate:  ## Show migration help
	uv run db-migrate --help

migrate-create:  ## Create database tables
	uv run db-migrate create

migrate-drop:  ## Drop database tables
	uv run db-migrate drop

migrate-reset:  ## Reset database (drop and recreate tables)
	uv run db-migrate reset

migrate-status:  ## Check database status
	uv run db-migrate status

upload:  ## Show upload help
	uv run db-upload --help

upload-list:  ## List EPUBs in database
	uv run db-upload list

show-settings:  ## Show current application settings
	uv run show-settings

epub-info:  ## Show EPUB info for test file
	uv run epub-info test.epub

vectorize-epub:  ## Vectorize test EPUB file
	uv run vectorize-epub test.epub

dev-setup: install-dev db-start migrate-create  ## Complete development setup
	@echo "Development environment is ready!"
	@echo "You can now:"
	@echo "  - Upload EPUBs: make upload-epub FILE=path/to/book.epub"
	@echo "  - List EPUBs: make upload-list"
	@echo "  - Check status: make migrate-status"
	@echo "  - Run tests: make test"

upload-epub:  ## Upload an EPUB file (usage: make upload-epub FILE=path/to/book.epub)
	@if [ -z "$(FILE)" ]; then \
		echo "Error: FILE parameter is required"; \
		echo "Usage: make upload-epub FILE=path/to/book.epub"; \
		exit 1; \
	fi
	uv run db-upload upload "$(FILE)"
