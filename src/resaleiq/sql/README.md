# SQL analysis layer

Four CTE-based queries over the ResaleIQ schema, written for PostgreSQL 16. Each file is standalone, heavily commented, and ships with a smoke test in `tests/test_sql_queries.py`.

## How to run

All four queries execute via the runner CLI. From the repo root:

```bash
# List available queries
make sql-list

# Run a query and render results as a table
make sql-run Q=01_segment_mape_audit

# Inspect the query plan
uv run python -m resaleiq.sql.runner explain 01_segment_mape_audit
```

First-time setup requires loading Phase 1 parquet files into the Postgres sandbox:

```bash
make db-up       # start Postgres if not already running
make db-load     # bulk-load data/*.parquet via COPY
```

## Query catalog

### 01_segment_mape_audit

**What it answers.** Where is the baseline model's error concentrated, and which segments should we fix first?

**Structure.** Three CTEs (`prediction_context`, `segment_stats`, `ranked`) followed by a final SELECT. Segments are defined as `(device_category, condition_grade, prediction_month)`. We rank by total absolute dollar error rather than MAPE alone, because MAPE ranking tends to pick noisy tails with small N.

**Why this runs first in the audit workflow.** A pricing team with limited engineering bandwidth needs the biggest dollar-impact segments identified before spending the next sprint. The query also reports `median_ape` alongside mean MAPE, which catches cases where a handful of bad rows are distorting the mean.

**Expected result on ResaleIQ data.** Android Mid in September and October 2024/2025 ranks in the top handful by dollar impact. This is the planted segment surfacing naturally out of the audit, without the analyst needing prior knowledge that it exists.

### 02_lot_performance

**What it answers.** How do lots perform on clearing rate, premium over reserve, time-to-clear, and model accuracy, split by auction type?

**Structure.** Three CTEs (`lot_features`, `cleared_with_predictions`, `auction_type_stats` + `cleared_stats`) and a final join. Uses `FILTER (WHERE ...)` for conditional aggregation, `PERCENTILE_CONT` for distribution shape, and a left join on `model_predictions` to keep lots without predictions in the population.

**Why the idioms matter.** `FILTER` is the modern-SQL way to do conditional aggregation, clearer than `SUM(CASE WHEN ...)` and faster in the query planner. Percentiles are the right tool for auction data because the distribution is right-skewed; means alone would mislead.

**Expected result.** Popcorn auctions land at roughly 4.2 percent average premium over reserve; fixed-end auctions at roughly 2.3 percent. Clearing rate sits around 80 percent for both types. Popcorn lots show a nonzero average close delay, matching the extension mechanism.

### 03_popcorn_dynamics

**What it answers.** Does the popcorn mechanism actually extract more value, and can we bound the effect with a confidence interval?

**Structure.** Four CTEs covering (1) per-lot premium computation, (2) per-auction-type mean and variance, (3) the difference and SE for the 95 CI via the two-sample normal approximation, (4) within-popcorn decomposition comparing lots that triggered an extension to lots that did not.

**Why the CI matters.** "Popcorn's premium is higher" without a CI is not a claim that survives review. The normal approximation is more than adequate at N greater than 1000 per group, which ResaleIQ easily clears. For tighter small-sample work, bootstrapping would move to Python.

**Expected result.** Single row. Point estimate of the difference in premium between popcorn and fixed-end auctions is roughly 2 percentage points, with a narrow CI that does not cross zero. Within-popcorn decomposition shows that lots where a bid actually triggered an extension cleared meaningfully higher than those where no extension fired.

### 04_buyer_tier_auction_interaction

**What it answers.** Do Enterprise buyers pay different premiums on popcorn auctions than on fixed-end, relative to Mid-market and SMB?

**Structure.** Two CTEs. The second uses `GROUPING SETS` to compute the 2-by-3 cell matrix plus both marginals and the grand total in a single aggregation pass. The `GROUPING()` function relabels marginal rows with `'ALL'` in the final output.

**Why GROUPING SETS.** The alternative is four separate GROUP BYs unioned together; GROUPING SETS does it in one pass and one grouped scan, which is why it exists. Senior SQL readers will recognize this as the idiomatic pattern for crosstabs.

**Expected result.** Nine rows total (six cells, three tier marginals, one auction-type marginal pair, one grand total). Enterprise buyers land at the highest premium in both auction types; the auction-type main effect should dominate but the interaction (does the Enterprise-vs-SMB gap widen on popcorn?) is the interesting question the query surfaces.

## Testing

Each query has a smoke test in `tests/test_sql_queries.py` that asserts:

- Non-empty result set.
- Expected column names present.
- Planted segment surfaces in query 01 (Android Mid in a launch month appears in the top of the ranking).
- Popcorn premium > fixed-end premium in query 02.
- 95 CI on difference in query 03 does not cross zero.
- All three tier marginals present in query 04.

Run with:

```bash
make test
```

Tests depend on Postgres being up and data being loaded. `make db-up && make db-load` is the prerequisite.
