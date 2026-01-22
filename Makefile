.PHONY: help install install-dev db-init db-load db-reset db-check test lint format clean

# Default target
help:
	@echo "Retain - Customer Lifecycle Management Platform"
	@echo ""
	@echo "Setup Commands:"
	@echo "  make install       Install production dependencies"
	@echo "  make install-dev   Install all dependencies (including dev)"
	@echo ""
	@echo "Database Commands:"
	@echo "  make db-init       Create database tables"
	@echo "  make db-load       Load CSV data into database"
	@echo "  make db-reset      Drop and recreate all tables"
	@echo "  make db-check      Verify database connection and counts"
	@echo ""
	@echo "Development Commands:"
	@echo "  make test          Run test suite"
	@echo "  make test-cov      Run tests with coverage report"
	@echo "  make lint          Run linters (ruff, mypy)"
	@echo "  make format        Format code (black, isort)"
	@echo "  make clean         Remove build artifacts"
	@echo ""
	@echo "Data Commands:"
	@echo "  make data-generate Generate synthetic dataset"
	@echo ""

# =============================================================================
# SETUP
# =============================================================================

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

# =============================================================================
# DATABASE
# =============================================================================

db-init:
	@echo "Creating database tables..."
	python -m src.data.database init
	@echo "Done!"

db-load:
	@echo "Loading data into database..."
	python -m src.data.load
	@echo "Done!"

db-reset:
	@echo "WARNING: This will delete all data!"
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ]
	python -m src.data.database reset
	@echo "Database reset complete."

db-check:
	@echo "Checking database connection..."
	python -m src.data.database check

# =============================================================================
# TESTING
# =============================================================================

test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing
	@echo "Coverage report: htmlcov/index.html"

# =============================================================================
# CODE QUALITY
# =============================================================================

lint:
	ruff check src/ tests/
	mypy src/

format:
	black src/ tests/
	isort src/ tests/
	ruff check --fix src/ tests/

# =============================================================================
# DATA
# =============================================================================

data-generate:
	@echo "Generating synthetic dataset..."
	python -m src.data.generate --seed 42 --accounts 60000 --output data/raw/
	@echo "Done! Files in data/raw/"

# =============================================================================
# CLEANUP
# =============================================================================

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
