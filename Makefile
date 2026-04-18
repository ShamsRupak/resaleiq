.DEFAULT_GOAL := help
SHELL := /bin/bash

.PHONY: help install install-all generate generate-sample db-up db-down db-reset db-load db-migrate sql-list sql-run train-all counterfactual phase3-charts phase3-load phase3-all lot-train dashboard dashboard-lite lint type test check clean

help: ## Show available targets
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-18s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install core deps via uv
	uv sync

install-all: ## Install all deps (core + db + ml + dashboard + dev)
	uv sync --all-extras

generate: ## Generate the full synthetic dataset (~250K transactional rows)
	uv run resaleiq-generate

generate-sample: ## Generate a small sample dataset for smoke testing
	uv run resaleiq-generate --scale sample

db-up: ## Start PostgreSQL in Docker
	docker compose up -d postgres
	@echo "Waiting for postgres to be healthy..."
	@until docker compose exec postgres pg_isready -U resaleiq -d resaleiq > /dev/null 2>&1; do sleep 1; done
	@echo "Postgres ready at localhost:5432"

db-down: ## Stop PostgreSQL
	docker compose down

db-reset: ## Destroy and recreate the database
	docker compose down -v
	rm -rf ./data/postgres
	$(MAKE) db-up

db-load: ## Bulk-load parquet files into Postgres via COPY
	uv run python scripts/load_parquet_to_postgres.py

db-migrate: ## Apply pending SQL migrations
	@for f in migrations/*.sql; do \
		echo "Applying $$f..."; \
		docker compose exec -T postgres psql -U resaleiq -d resaleiq -f - < $$f; \
	done

sql-list: ## List available SQL queries
	uv run python -m resaleiq.sql.runner list

sql-run: ## Run a SQL query by name (use Q=<query_name>)
	@if [ -z "$(Q)" ]; then echo "Usage: make sql-run Q=01_segment_mape_audit"; exit 1; fi
	uv run python -m resaleiq.sql.runner run $(Q)

train-all: ## Train all 5 Phase 3 models on temporal split
	uv run python scripts/train_all_models.py

counterfactual: ## Run counterfactual dollar-impact analysis
	uv run python scripts/counterfactual_analysis.py

phase3-charts: ## Generate Phase 3 verification charts
	uv run python scripts/generate_phase3_charts.py

phase3-load: ## Load Phase 3 predictions into Postgres
	uv run python scripts/load_phase3_predictions.py

phase3-all: ## Run the complete Phase 3 pipeline (train, charts, counterfactual, load)
	$(MAKE) train-all
	$(MAKE) phase3-charts
	$(MAKE) counterfactual
	$(MAKE) db-migrate
	$(MAKE) phase3-load

lot-train: ## Train the lot-level XGBoost model for the optimizer
	uv run python scripts/train_lot_model.py

dashboard: ## Launch the Streamlit dashboard (all 5 pages, full DB)
	PYTHONPATH=.:src uv run streamlit run dashboard/app.py

dashboard-lite: ## Launch the dashboard with only parquet files (no Postgres required)
	PYTHONPATH=.:src RESALEIQ_PG_HOST=__disabled__ uv run streamlit run dashboard/app.py

lint: ## Run ruff
	uv run ruff check src tests
	uv run ruff format --check src tests

type: ## Run mypy
	uv run mypy src

test: ## Run pytest
	uv run pytest

check: lint type test ## Run all quality gates

clean: ## Remove caches and generated artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find data -name '*.parquet' -delete 2>/dev/null || true
