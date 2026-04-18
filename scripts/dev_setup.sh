#!/usr/bin/env bash
# One-shot dev setup: sync deps, generate data, start Postgres, load schema.
# Run from the repo root.

set -euo pipefail

cd "$(dirname "$0")/.."

echo "[1/3] Installing dependencies via uv"
uv sync --all-extras

echo "[2/3] Generating synthetic dataset"
uv run resaleiq-generate

echo "[3/3] Starting Postgres sandbox"
if [ ! -f .env ]; then
    cp .env.example .env
fi
docker compose up -d postgres

echo
echo "Setup complete."
echo "  Parquet files: data/*.parquet"
echo "  Postgres:      localhost:5432 (user=resaleiq, db=resaleiq)"
echo "  Run tests:     make test"
