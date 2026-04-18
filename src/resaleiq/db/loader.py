"""Bulk-load Phase 1 parquet files into the PostgreSQL schema.

Uses psycopg 3's native COPY API, which streams CSV-encoded rows directly
into the server. This is an order of magnitude faster than pandas' to_sql
(which issues per-row INSERTs) and matches the production pattern a real
data engineering team would use.

Usage:
    from resaleiq.db.loader import load_all
    load_all()  # loads from default data_dir, default DATABASE_URL

Environment variables (loaded from .env if present):
    DATABASE_URL - postgresql+psycopg://user:pass@host:port/db
"""

from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path

import pandas as pd
import psycopg
from rich.console import Console

from resaleiq.config import DATA_DIR

console = Console()

# Order matters for foreign-key integrity: reference tables first, then
# transactional tables, then evaluation.
LOAD_ORDER: tuple[str, ...] = (
    "devices",
    "skus",
    "buyers",
    "sku_listings",
    "sku_offers",
    "lots",
    "lot_items",
    "lot_bids",
    "model_predictions",
)


def _database_url() -> str:
    """Resolve DATABASE_URL from environment.

    We strip the SQLAlchemy-style ``+psycopg`` dialect suffix because psycopg
    itself doesn't recognize it; SQLAlchemy uses it to pick a driver. The
    native psycopg connection string is plain ``postgresql://``.
    """

    url = os.environ.get(
        "DATABASE_URL",
        "postgresql+psycopg://resaleiq:resaleiq@localhost:5432/resaleiq",
    )
    return url.replace("postgresql+psycopg://", "postgresql://")


def _prepare_dataframe_for_copy(table: str, df: pd.DataFrame) -> pd.DataFrame:
    """Coerce dtypes so COPY accepts the CSV cleanly.

    pandas NaN becomes an empty CSV field which Postgres reads as NULL when
    we use COPY with NULL ''. Booleans need to be 't'/'f'. Timestamps need
    ISO format. Nullable integer columns (e.g., lots.winning_buyer_id) need
    to round-trip through pandas Int64 so ``283`` doesn't become ``283.0`` in
    the CSV (Postgres BIGINT rejects the decimal).
    """

    out = df.copy()
    for col in out.columns:
        # Nullable integer columns: lift float-with-NaN to pandas Int64 if
        # every non-null value is integer-equivalent (no fractional part).
        if pd.api.types.is_float_dtype(out[col]):
            non_null = out[col].dropna()
            if len(non_null) > 0 and ((non_null % 1) == 0).all():
                # Only attempt the cast if the values fit Int64 cleanly.
                # If not, leave as float; CSV output will carry ".0" and the
                # DB column must be NUMERIC for this to load. Not expected
                # in the schema.
                with contextlib.suppress(TypeError, ValueError):
                    out[col] = out[col].astype("Int64")
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d %H:%M:%S.%f")
        elif pd.api.types.is_bool_dtype(out[col]):
            out[col] = out[col].map({True: "t", False: "f"})
    _ = table
    return out


def _copy_dataframe(conn: psycopg.Connection, table: str, df: pd.DataFrame) -> int:
    """Stream a DataFrame into ``table`` via COPY FROM STDIN. Returns row count."""

    prepared = _prepare_dataframe_for_copy(table, df)
    csv_buffer = io.StringIO()
    prepared.to_csv(csv_buffer, index=False, header=False, na_rep="")
    csv_buffer.seek(0)

    columns = ", ".join(prepared.columns)
    copy_sql = f"COPY {table} ({columns}) FROM STDIN WITH (FORMAT csv, NULL '', HEADER false)"
    with conn.cursor() as cur, cur.copy(copy_sql) as copy:
        for chunk in iter(lambda: csv_buffer.read(65536), ""):
            copy.write(chunk)
    return len(prepared)


def truncate_all_tables(conn: psycopg.Connection) -> None:
    """Truncate every data table. Schema and views are preserved.

    TRUNCATE CASCADE is correct because we're resetting a dev environment.
    In production this would be reckless; here it's the right primitive.
    RESTART IDENTITY resets sequences, but our schema uses explicit IDs
    rather than serial columns, so it's a no-op. Left in for safety.
    """

    with conn.cursor() as cur:
        tables_reversed = ", ".join(reversed(LOAD_ORDER))
        cur.execute(f"TRUNCATE TABLE {tables_reversed} RESTART IDENTITY CASCADE")


def load_all(data_dir: Path = DATA_DIR, *, truncate: bool = True) -> dict[str, int]:
    """Load every parquet file in ``data_dir`` into Postgres.

    Returns a dict mapping table names to row counts loaded.
    """

    url = _database_url()
    console.print(f"[bold cyan]Connecting to[/] {url.split('@')[-1]}")
    row_counts: dict[str, int] = {}

    with psycopg.connect(url) as conn:
        if truncate:
            console.print("[yellow]Truncating existing tables[/]")
            truncate_all_tables(conn)

        for table in LOAD_ORDER:
            parquet_path = data_dir / f"{table}.parquet"
            if not parquet_path.exists():
                raise FileNotFoundError(
                    f"Expected parquet file not found: {parquet_path}. Run 'make generate' first."
                )
            df = pd.read_parquet(parquet_path)
            n_rows = _copy_dataframe(conn, table, df)
            row_counts[table] = n_rows
            console.print(f"  [green]{table:<20}[/] loaded {n_rows:,} rows")
        conn.commit()

    return row_counts


__all__ = ["LOAD_ORDER", "load_all", "truncate_all_tables"]
