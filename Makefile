.PHONY: help install install-dev test lint format type-check clean run

help:
	@echo "Available commands:"
	@echo "  make install      - Install production dependencies"
	@echo "  make install-dev  - Install development dependencies"
	@echo "  make test        - Run all tests"
	@echo "  make test-fast   - Run tests without slow integration tests"
	@echo "  make lint        - Run linting with ruff"
	@echo "  make format      - Format code with black"
	@echo "  make type-check  - Run type checking with mypy"
	@echo "  make clean       - Clean up cache and build files"
	@echo "  make run         - Run the application"
	@echo "  make ci          - Run all CI checks locally"

install:
	pip install --upgrade pip
	pip install -r requirements.txt

install-dev: install
	pip install -r requirements-dev.txt
	pre-commit install

test:
	pytest tests/ -v --cov=. --cov-report=term-missing --cov-report=html

test-fast:
	pytest tests/ -v -m "not slow" --cov=. --cov-report=term-missing

lint:
	ruff check . --fix
	black --check --diff .

format:
	black .
	ruff check . --fix

type-check:
	mypy . --ignore-missing-imports

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "dist" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "build" -exec rm -rf {} + 2>/dev/null || true

run:
	python main.py

ci: lint type-check test
	@echo "All CI checks passed!"

# Development shortcuts
dev: install-dev
	@echo "Development environment ready!"

quick: format lint test-fast
	@echo "Quick checks passed!"