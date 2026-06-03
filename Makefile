.PHONY: help install dev test lint typecheck serve train dbt eval demo airflow generate parity clean

# Doppler integration (consistent with root Makefile)
DOPPLER := $(shell command -v doppler 2>/dev/null)
ifdef DOPPLER
  RUN := doppler run --
else
  RUN :=
endif

PYTHON = .venv/bin/python
UV = uv

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	$(UV) sync

dev: ## Install all dependencies (dev + dbt + airflow)
	$(UV) sync --all-extras

test: ## Run tests + lint + typecheck
	$(UV) run ruff check src/ tests/
	$(UV) run ruff format --check src/ tests/
	$(UV) run mypy src/
	$(UV) run pytest tests/ -v --tb=short

lint: ## Run linters
	$(UV) run ruff check src/ tests/
	$(UV) run ruff format --check src/ tests/

typecheck: ## Run mypy
	$(UV) run mypy src/

serve: ## Start the FastAPI server
	$(RUN) $(UV) run uvicorn ml_service.app.main:app --host 0.0.0.0 --port 8000 --reload

generate: ## Generate synthetic data into DuckDB
	$(RUN) $(PYTHON) data/generate_synthetic.py

parity: ## Run train/serve feature parity check
	$(RUN) $(UV) run python -m ml_service.features.parity

train: ## Run the training pipeline
	$(RUN) $(UV) run python -m ml_service.training.train

dbt: ## Run dbt seed + build + test (DuckDB target)
	cd dbt && $(RUN) dbt seed --target duckdb && $(RUN) dbt build --target duckdb && $(RUN) dbt test --target duckdb

eval: ## Run the AI evaluation suite
	$(RUN) $(UV) run python -m eval.runner

demo: ## Run the full end-to-end demo
	$(RUN) bash scripts/demo.sh

airflow: ## Start Airflow standalone
	$(RUN) airflow standalone

clean: ## Remove build artifacts and caches
	rm -rf .mypy_cache .ruff_cache .pytest_cache htmlcov .coverage
	rm -rf dbt/target dbt/logs dbt/dbt_packages
	rm -rf mlruns mlartifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
