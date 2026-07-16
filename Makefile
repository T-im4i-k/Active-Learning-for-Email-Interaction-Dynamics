.PHONY: help
.PHONY: install install-dev
.PHONY: format
.PHONY: check-format
.PHONY: lint
.PHONY: mypy
.PHONY: check
.PHONY: clean

help:
	@echo "Available targets:"
	@echo ""
	@echo "Installation:"
	@echo "  make install            - Install package (non-editable)"
	@echo "  make install-dev        - Install package in editable mode with dev dependencies"
	@echo ""
	@echo "Formatting:"
	@echo "  make format             - Format source code with isort and black"
	@echo ""
	@echo "Formatting checking:"
	@echo "  make check-format       - Check source code formatting with isort and black"
	@echo ""
	@echo "Linting:"
	@echo "  make lint               - Run pylint on source code"
	@echo ""
	@echo "Type checking:"
	@echo "  make mypy               - Run mypy on source code"
	@echo ""
	@echo "Combined checks:"
	@echo "  make check              - Run all checks on source code (format, lint, mypy, test)"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean              - Remove Python artifacts"

install:
	pip install .

install-dev:
	pip install -e .[dev]


format:
	@echo "Sorting imports in source code with isort..."
	isort src/
	@echo "Formatting source code with black..."
	black src/


check-format:
	@echo "Checking import order in source code with isort..."
	@isort --check-only --diff src/
	@echo "Checking source code formatting with black..."
	@black --check --diff src/


lint:
	@echo "Running pylint on source code..."
	pylint src/


mypy:
	@echo "Running mypy type checking on source code..."
	mypy src/


check: check-format lint mypy
	@echo "✓ All source checks passed!"


clean:
	@echo "Cleaning Python artifacts..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	@echo "✓ Cleanup complete!"