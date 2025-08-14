.PHONY: help install install-dev clean lint format test test-cov type-check pre-commit setup-dev run-api docker-build docker-run

# Default target
help: ## Show this help message
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Environment setup
install: ## Install production dependencies
	uv pip install -r requirements.txt

install-dev: ## Install development dependencies
	uv pip install -r requirements-dev.txt

setup-dev: install-dev ## Complete development setup
	@if command -v pre-commit >/dev/null 2>&1; then \
		pre-commit install; \
	else \
		echo "Warning: pre-commit not found in PATH. Make sure virtual environment is activated."; \
		echo "Run: source .venv/bin/activate && make setup-dev"; \
		exit 1; \
	fi
	@echo "Development environment setup complete!"

# Environment management
clean: ## Clean up temporary files and caches
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	rm -rf .coverage

# Code quality
lint: ## Run ruff linter
	ruff check src tests

format: ## Format code with black and ruff
	black src tests notebooks
	ruff check --fix src tests
	isort src tests

type-check: ## Run mypy type checker
	mypy src

# Testing
test: ## Run tests
	pytest tests/

test-cov: ## Run tests with coverage report
	pytest --cov=src --cov-report=html --cov-report=term-missing tests/

test-watch: ## Run tests in watch mode
	pytest-watch tests/

# Pre-commit hooks
pre-commit: ## Run pre-commit hooks on all files
	pre-commit run --all-files

# Quality check pipeline (recommended before commits)
check-all: clean lint type-check test ## Run all quality checks
	@echo "All quality checks passed!"

# Application
run-api: ## Run the FastAPI application
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

run-jupyter: ## Start Jupyter Lab
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# MLflow
mlflow-ui: ## Start MLflow UI
	mlflow ui --host 0.0.0.0 --port 5000

# Docker commands
docker-build: ## Build Docker image
	docker build -t customer-churn-prediction -f docker/Dockerfile .

docker-run: ## Run Docker container
	docker run -p 8000:8000 customer-churn-prediction

docker-dev: ## Run development environment in Docker
	docker-compose -f docker/docker-compose.yml up --build

# Data processing
process-data: ## Run data preprocessing pipeline
	python src/data/preprocess.py

train-model: ## Train ML model
	python src/models/train.py

# Utility commands
requirements: ## Generate requirements.txt from pyproject.toml
	pip-compile pyproject.toml

update-deps: ## Update all dependencies to latest versions
	pip-compile --upgrade pyproject.toml

# Project initialization
init-notebooks: ## Create initial notebook structure
	mkdir -p notebooks/exploratory notebooks/experiments notebooks/reports
	touch notebooks/exploratory/.gitkeep
	touch notebooks/experiments/.gitkeep
	touch notebooks/reports/.gitkeep

init-config: ## Create initial configuration files
	mkdir -p config
	echo "# Development configuration" > config/dev.yml
	echo "# Production configuration" > config/prod.yml
	echo "# Local configuration (add to .gitignore)" > config/local.yml

# Complete project setup
init-project: init-notebooks init-config setup-dev ## Initialize complete project structure
	@echo "Project initialization complete!"
	@echo "Next steps:"
	@echo "  1. Update author information in pyproject.toml"
	@echo "  2. Review and customize configuration files"
	@echo "  3. Run 'make check-all' to verify setup"