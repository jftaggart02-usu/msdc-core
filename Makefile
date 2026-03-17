# msdc-core Makefile
# Targets: install, test, lint, type-check, check, clean

.PHONY: install test lint type-check check clean

# Install the package in editable mode with dev dependencies
install:
	pip install -e ".[dev]"

# Run all unit tests with coverage
test:
	pytest tests/ -v --tb=short --cov=msdc_core --cov-report=term-missing

# Run Pylint on package and tests
lint:
	pylint msdc_core tests

# Run Mypy static type checking
type-check:
	mypy msdc_core tests

# Run all checks (lint + type-check + tests)
check: lint type-check test

# Remove build artefacts and caches
clean:
	rm -rf build dist *.egg-info .eggs __pycache__ .mypy_cache .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
