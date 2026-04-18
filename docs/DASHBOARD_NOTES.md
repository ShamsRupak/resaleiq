# ResaleIQ Dashboard

Five-page Streamlit application showing the Phase 3 pricing work and the lot optimizer from Phase 4. Navy-forward brand palette, Plotly charts, Postgres-first with parquet fallback.

## Pages

| # | Page | Primary audience | What it answers |
|---|---|---|---|
| 1 | Executive Summary | CEO view | Platform GMV, headline MAPE, dollar impact of the Phase 3 fix |
| 2 | Pricing Model Performance | Senior DS view | Calibration, residuals by segment, feature importance, conformal coverage |
| 3 | Auction Lot Optimizer | Both | Given an inventory bag, recommend a partition into auction lots |
| 4 | Segment Audit | Both (diagnostic workflow) | Excess-error ranking with drill-down by category, grade, carrier, season |
| 5 | Experiment Tracker | CEO view (methodology) | Shipped experiments plus 3 proposed with power analysis |

## Running locally

### First-time setup

```bash
# Install dashboard dependencies (Streamlit + Plotly) in addition to core
uv sync --all-extras

# Make sure Phase 1 data exists and Postgres is loaded
make generate
make db-up
make db-load

# Train the lot-level model for Page 3
make lot-train

# Optional: load Phase 3 predictions into Postgres
make phase3-load
```

### Start the dashboard

```bash
make dashboard
```

Opens at `http://localhost:8501`. Navigation is via the left sidebar; pages load on click.

### Parquet-only mode (no Postgres)

If Docker is not running:

```bash
make dashboard-lite
```

Pages gracefully fall back to reading `data/*.parquet` directly. Some aggregation queries that depend on SQL will still work but via pandas.

## Configuration

### Database connection

Priority order:

1. Environment variables: `RESALEIQ_PG_HOST`, `RESALEIQ_PG_PORT`, `RESALEIQ_PG_USER`, `RESALEIQ_PG_PASSWORD`, `RESALEIQ_PG_DB`
2. `.streamlit/secrets.toml` under a `[postgres]` section
3. Standard Docker-Compose defaults (`localhost:5432`, user/pass/db all `resaleiq`)

### Example secrets.toml

```toml
[postgres]
host = "localhost"
port = 5432
user = "resaleiq"
password = "resaleiq"
dbname = "resaleiq"
```

## Streamlit Community Cloud deployment

The dashboard is deploy-ready but intentionally not automated because hosting Postgres publicly is out of scope. For a live demo:

### Option 1: parquet-only (simplest)

Deploy the repo to Streamlit Community Cloud with no database. Every page renders from the parquet files committed to `data/`. The data volume (215K rows) is well within the 1GB repo limit.

Steps:
1. Push the repo to GitHub (already done)
2. On streamlit.io, create a new app pointing at `dashboard/app.py`
3. No secrets needed

### Option 2: with managed Postgres

Deploy with a Supabase or Neon free-tier Postgres for live queries.

Steps:
1. Create a free Supabase project
2. Run `make db-load` against the remote connection
3. Add `postgres` block to Streamlit Cloud secrets
4. Deploy

## File layout

```
dashboard/
  app.py                           # Landing page
  utils.py                         # Brand palette, connection, loaders, formatters
  pages/
    1_Executive_Summary.py
    2_Pricing_Model_Performance.py
    3_Auction_Lot_Optimizer.py
    4_Segment_Audit.py
    5_Experiment_Tracker.py

src/resaleiq/ml/
  lot_model.py                     # Lot-level XGBoost + partition strategies

scripts/
  train_lot_model.py               # One-time lot model training

data/
  lot_model.pkl                    # Trained lot-model artifact
  lot_model_summary.json           # Training metrics
```

## Design principles

Applied consistently across all pages:

- No em dashes in any copy
- Navy (#0B1F3A) headers, blue (#1A4B8C) accents, red (#C1292E) for alerts only
- Every chart has a business-outcome subtitle, not just the axis labels
- Plotly for interactive charts; matplotlib only for things Plotly can not render
- Postgres-first, parquet-fallback so the dashboard never fails on a fresh clone
- Sidebar shows live DB connection status (green dot for live, orange for parquet)

## Known limitations

- The lot-level model is now trained with a time-series holdout split (train = 5,493 lots Oct 2024 to Jan 2026, holdout = 531 lots Feb to Mar 2026). Out-of-sample MAPE is 3.98% versus a naive baseline of 18.11% - a 78.0% relative improvement on data the model never saw during training. Conformal coverage on the holdout comes in at 66.5% vs 80% target, a real distribution-shift finding that the rolling-recalibration experiment on the Experiment Tracker page addresses.
- The landing page does not auto-detect whether the lot model artifact exists. Run `make lot-train` before opening Page 3 or it will show an error panel.
- Parquet fallback will re-read files on every page load. Streamlit's `@st.cache_data` partially mitigates but a busy dashboard may want a Redis or in-memory cache for production.
- Page 5's "Experiment Tracker" uses hardcoded numbers pulled from the Phase 3 run. For a fully-dynamic implementation, those would move into a migration and be queried live.

## Testing

Dashboard utilities and lot-model code are covered by 24 additional tests in:

- `tests/test_lot_model.py` (14 tests)
- `tests/test_dashboard_utils.py` (10 tests)

Run with:

```bash
make test
```
