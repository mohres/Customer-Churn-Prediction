.PHONY: help install install-dev clean lint format test test-cov pre-commit setup-dev run-api docker-build docker-run

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
	rm -rf __pycache__/
	rm -rf src/__pycache__/
	rm -rf src/*/__pycache__/
	rm -rf src/*/*/__pycache__/
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf cache/
	rm -rf plots/
	rm -rf reports/
	rm -rf test_dashboard/
	rm -rf test_monitoring_data/
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/

# Code quality
lint: ## Run ruff linter
	ruff check src tests

format: ## Format code with black and ruff
	black src tests notebooks
	ruff check --fix src tests
	isort src tests


# Testing

test-cov: ## Run tests with coverage report
	pytest --cov=src --cov-report=html --cov-report=term-missing tests/

test-watch: ## Run tests in watch mode
	pytest-watch tests/

# Pre-commit hooks
pre-commit: ## Run pre-commit hooks on all files
	pre-commit run --all-files

# Quality check pipeline (recommended before commits)
check-all: clean lint ## Run all quality checks
	@echo "All quality checks passed!"

# Application
run-api: ## Run the FastAPI application
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

run-jupyter: ## Start Jupyter Lab
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# MLflow
mlflow-ui: ## Start MLflow UI
	mlflow ui --host 0.0.0.0 --port 5000

# Monitoring
test-monitoring: ## Test the monitoring system
	python scripts/test_monitoring.py

run-monitoring: ## Generate monitoring dashboard
	python scripts/demo_monitoring.py

# Docker commands
docker-build: ## Build Docker image
	docker build -t customer-churn-prediction -f Dockerfile .

docker-run: ## Run Docker container
	docker run -p 8000:8000 customer-churn-prediction


# Data processing
process-data: ## Run data preprocessing and validation
	python -m src.data.data_validator

train-model: ## Train ML model
	python scripts/train_production_model.py --data-path data/customer_churn_mini.json

# Utility commands
requirements: ## Generate requirements.txt from pyproject.toml
	uv pip compile pyproject.toml -o requirements.txt

update-deps: ## Update all dependencies to latest versions
	uv pip compile --upgrade pyproject.toml -o requirements.txt

# Complete project setup
init-project: setup-dev ## Initialize complete project structure
	@echo "Project initialization complete!"
	@echo "Next steps:"
	@echo "  1. Update author information in pyproject.toml"
	@echo "  2. Review and customize configuration files"
	@echo "  3. Run 'make check-all' to verify setup"
