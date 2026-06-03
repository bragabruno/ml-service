.PHONY: help install dev test lint typecheck serve train dbt eval demo airflow clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	uv sync

dev: ## Install all dependencies (dev + dbt + airflow)
	uv sync --all-extras

test: ## Run tests + lint + typecheck
	ruff check src/ tests/
	ruff format --check src/ tests/
	mypy src/
	pytest tests/ -v --tb=short

lint: ## Run linters
	ruff check src/ tests/
	ruff format --check src/ tests/

typecheck: ## Run mypy
	mypy src/

serve: ## Start the FastAPI server
	uvicorn ml_service.app.main:app --host 0.0.0.0 --port 8000 --reload

train: ## Run the training pipeline
	python -m ml_service.training.train

dbt: ## Run dbt build + test (DuckDB target)
	cd dbt && dbt build --target duckdb && dbt test --target duckdb

eval: ## Run the AI evaluation suite
	python -m eval.runner

demo: ## Run the full end-to-end demo
	bash scripts/demo.sh

airflow: ## Start Airflow standalone
	airflow standalone

clean: ## Remove build artifacts and caches
	rm -rf .mypy_cache .ruff_cache .pytest_cache htmlcov .coverage
	rm -rf dbt/target dbt/logs dbt/dbt_packages
	rm -rf mlruns mlartifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
