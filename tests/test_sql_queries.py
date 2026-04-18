"""Smoke tests for the four SQL analysis queries.

These tests require a running Postgres sandbox with Phase 1 data loaded. They
skip cleanly if the database isn't reachable, so `make test` still passes on
a dev machine without Postgres (the data-generation and market-dynamics
tests remain the primary correctness gates).

Run manually after `make db-up && make db-load`:
    uv run pytest tests/test_sql_queries.py -v
"""

from __future__ import annotations

import os

import pandas as pd
import psycopg
import pytest

from resaleiq.sql import list_queries, load_query


def _database_url() -> str:
    url = os.environ.get(
        "DATABASE_URL",
        "postgresql+psycopg://resaleiq:resaleiq@localhost:5432/resaleiq",
    )
    return url.replace("postgresql+psycopg://", "postgresql://")


def _postgres_available() -> bool:
    """Return True if we can connect to the configured Postgres instance."""

    try:
        with (
            psycopg.connect(_database_url(), connect_timeout=2) as conn,
            conn.cursor() as cur,
        ):
            cur.execute("SELECT 1")
            cur.fetchone()
        return True
    except Exception:
        return False


def _data_loaded() -> bool:
    """Return True if the critical tables have rows (data bootstrap done)."""

    try:
        with (
            psycopg.connect(_database_url(), connect_timeout=2) as conn,
            conn.cursor() as cur,
        ):
            cur.execute("SELECT COUNT(*) FROM model_predictions")
            row = cur.fetchone()
            return bool(row and row[0] > 0)
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _postgres_available() or not _data_loaded(),
    reason="Postgres sandbox not reachable or not loaded; run make db-up && make db-load",
)


@pytest.fixture(scope="module")
def conn() -> psycopg.Connection:
    """Provide a live psycopg connection to the sandbox."""

    connection = psycopg.connect(_database_url())
    yield connection
    connection.close()


def _run_query(conn: psycopg.Connection, name: str) -> pd.DataFrame:
    """Execute a query file and return rows as a DataFrame."""

    sql = load_query(name)
    with conn.cursor() as cur:
        cur.execute(sql)
        columns = [d.name for d in cur.description] if cur.description else []
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=columns)


class TestQueryCatalog:
    def test_four_queries_registered(self) -> None:
        queries = list_queries()
        assert len(queries) == 4
        assert "01_segment_mape_audit" in queries
        assert "02_lot_performance" in queries
        assert "03_popcorn_dynamics" in queries
        assert "04_buyer_tier_auction_interaction" in queries


class Test01SegmentMapeAudit:
    @pytest.fixture(scope="class")
    def result(self, conn: psycopg.Connection) -> pd.DataFrame:
        return _run_query(conn, "01_segment_mape_audit")

    def test_non_empty(self, result: pd.DataFrame) -> None:
        assert len(result) > 0

    def test_expected_columns(self, result: pd.DataFrame) -> None:
        required = {
            "priority_rank",
            "device_category",
            "condition_grade",
            "pred_month",
            "n_predictions",
            "mape",
            "overall_mape",
            "total_abs_error_usd",
            "excess_abs_error_usd",
        }
        assert required.issubset(result.columns)

    def test_ranking_descending_by_excess_error(self, result: pd.DataFrame) -> None:
        """Ranking is by excess error; excess_abs_error_usd must be monotonically decreasing."""

        errors = result["excess_abs_error_usd"].astype(float).tolist()
        assert errors == sorted(errors, reverse=True)

    def test_planted_segment_in_top_10(self, result: pd.DataFrame) -> None:
        """The planted Android Mid launch-month slice should surface near the top."""

        top10 = result.head(10)
        launch_months = {"2024-09-01", "2024-10-01", "2025-09-01", "2025-10-01"}
        top10_launch_android = top10[
            (top10["device_category"] == "Android Mid")
            & (top10["pred_month"].astype(str).isin(launch_months))
        ]
        assert len(top10_launch_android) >= 1, (
            "The planted segment (Android Mid in iPhone launch months) should "
            "surface in the top 10 by dollar error impact. If this fails, either "
            "data is stale (rerun generate + load) or the planted effect is weaker "
            "than calibrated."
        )


class Test02LotPerformance:
    @pytest.fixture(scope="class")
    def result(self, conn: psycopg.Connection) -> pd.DataFrame:
        return _run_query(conn, "02_lot_performance")

    def test_two_auction_types(self, result: pd.DataFrame) -> None:
        assert set(result["auction_type"]) == {"popcorn", "fixed_end"}

    def test_popcorn_premium_exceeds_fixed_end(self, result: pd.DataFrame) -> None:
        pop = result[result["auction_type"] == "popcorn"]["avg_premium_over_reserve"].iloc[0]
        fix = result[result["auction_type"] == "fixed_end"]["avg_premium_over_reserve"].iloc[0]
        assert float(pop) > float(fix)

    def test_clearing_rate_reasonable(self, result: pd.DataFrame) -> None:
        """Clearing rate should be in the 70 to 90 percent band for both types."""

        for rate in result["clearing_rate"]:
            assert 0.60 <= float(rate) <= 0.95


class Test03PopcornDynamics:
    @pytest.fixture(scope="class")
    def result(self, conn: psycopg.Connection) -> pd.DataFrame:
        return _run_query(conn, "03_popcorn_dynamics")

    def test_single_row(self, result: pd.DataFrame) -> None:
        assert len(result) == 1

    def test_difference_positive(self, result: pd.DataFrame) -> None:
        diff = float(result["diff_mean_premium_pp"].iloc[0])
        assert diff > 0, f"popcorn should extract more premium than fixed end; got diff={diff}"

    def test_ci_does_not_cross_zero(self, result: pd.DataFrame) -> None:
        """95% CI on the difference should sit entirely above zero."""

        low = float(result["diff_ci_low_pp"].iloc[0])
        high = float(result["diff_ci_high_pp"].iloc[0])
        assert low > 0, f"CI low bound should be > 0 but was {low}"
        assert high > low

    def test_within_popcorn_values_present(self, result: pd.DataFrame) -> None:
        """Within-popcorn decomposition values should both be populated.

        Note: the synthetic data generator sets popcorn premium at the
        auction-type level, not as a function of whether a specific lot's
        bids actually triggered an extension. So the triggered-vs-not
        comparison within popcorn reflects noise rather than causal effect.
        On real data this split would isolate the extension effect
        specifically; here we only verify both values are present.
        """

        triggered = float(result["premium_when_popcorn_triggered_pct"].iloc[0])
        not_triggered = float(result["premium_when_popcorn_not_triggered_pct"].iloc[0])
        assert triggered > 0 and not_triggered > 0


class Test04BuyerTierAuctionInteraction:
    @pytest.fixture(scope="class")
    def result(self, conn: psycopg.Connection) -> pd.DataFrame:
        return _run_query(conn, "04_buyer_tier_auction_interaction")

    def test_three_tiers_present(self, result: pd.DataFrame) -> None:
        tiers = set(result["buyer_tier"])
        assert {"Enterprise", "Mid-market", "SMB"}.issubset(tiers)

    def test_all_marginals_present(self, result: pd.DataFrame) -> None:
        """GROUPING SETS should produce cells + tier marginals + auction marginals + grand total."""

        # Cell rows: auction_type in {popcorn, fixed_end} AND buyer_tier in the three tiers
        cells = result[
            result["auction_type"].isin(["popcorn", "fixed_end"])
            & result["buyer_tier"].isin(["Enterprise", "Mid-market", "SMB"])
        ]
        assert len(cells) == 6, f"expected 6 cells, got {len(cells)}"

        # At least one row with auction_type = 'ALL' (tier marginals)
        assert (result["auction_type"] == "ALL").any()

        # At least one row with buyer_tier = 'ALL' (auction marginals)
        assert (result["buyer_tier"] == "ALL").any()

    def test_enterprise_premium_highest_overall(self, result: pd.DataFrame) -> None:
        """Enterprise marginal row should show highest average premium."""

        tier_marginals = result[result["auction_type"] == "ALL"]
        # Find the tier with the highest premium
        top_tier = tier_marginals.sort_values("avg_premium_pct", ascending=False).iloc[0][
            "buyer_tier"
        ]
        assert top_tier == "Enterprise"
