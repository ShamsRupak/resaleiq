"""Load Phase 3 predictions into the ``model_predictions`` table.

Keeps the existing synthetic baseline_v1 rows in place and appends the four
real-XGBoost model versions (baseline_v2_xgb through targeted_v1_xgb).

Run order:
    1. First apply the migration: ``migrations/001_add_conformal_intervals.sql``
    2. Then run this script: ``python scripts/load_phase3_predictions.py``
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import psycopg
from psycopg import sql
from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from resaleiq.config import DATA_DIR


def _connect() -> psycopg.Connection:
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = int(os.getenv("POSTGRES_PORT", "5432"))
    user = os.getenv("POSTGRES_USER", "resaleiq")
    password = os.getenv("POSTGRES_PASSWORD", "resaleiq")
    database = os.getenv("POSTGRES_DB", "resaleiq")
    return psycopg.connect(
        host=host, port=port, user=user, password=password, dbname=database
    )


def main() -> None:
    console = Console()
    preds = pd.read_parquet(DATA_DIR / "phase3_predictions.parquet")
    console.print(f"Loading {len(preds):,} Phase 3 predictions across "
                  f"{preds['model_version'].nunique()} model versions")

    with _connect() as conn:
        with conn.cursor() as cur:
            # Ensure columns exist (idempotent safety).
            cur.execute(
                "ALTER TABLE model_predictions "
                "ADD COLUMN IF NOT EXISTS predicted_low  NUMERIC(12,2), "
                "ADD COLUMN IF NOT EXISTS predicted_high NUMERIC(12,2);"
            )

            # Delete any Phase 3 rows that might already exist so re-runs are clean.
            tags = tuple(preds["model_version"].unique())
            cur.execute(
                "DELETE FROM model_predictions WHERE model_version = ANY(%s);",
                (list(tags),),
            )
            deleted = cur.rowcount
            if deleted:
                console.print(f"  deleted {deleted:,} existing Phase 3 rows")

            # Insert in batches using COPY from CSV buffer for speed.
            import io
            buf = io.StringIO()
            # prediction_id has no sequence; compute next ID from current max.
            cur.execute("SELECT COALESCE(MAX(prediction_id), 0) FROM model_predictions;")
            max_id_row = cur.fetchone()
            start_id = int(max_id_row[0]) + 1 if max_id_row is not None else 1
            preds_with_id = preds.copy()
            preds_with_id["prediction_id"] = range(start_id, start_id + len(preds_with_id))
            console.print(f"  assigning prediction_ids starting at {start_id:,}")

            # Insert in batches using COPY from CSV buffer for speed.
            import io
            buf = io.StringIO()
            (
                preds_with_id[
                    [
                        "prediction_id",
                        "target_type",
                        "target_id",
                        "predicted",
                        "actual",
                        "pred_at",
                        "model_version",
                        "predicted_low",
                        "predicted_high",
                    ]
                ]
                .rename(columns={"pred_at": "predicted_at"})
                .to_csv(buf, index=False, header=False, na_rep="\\N")
            )
            buf.seek(0)
            with cur.copy(
                "COPY model_predictions "
                "(prediction_id, target_type, target_id, predicted, actual, predicted_at, model_version, predicted_low, predicted_high) "
                "FROM STDIN (FORMAT CSV, NULL '\\N')"
            ) as copy:
                copy.write(buf.read())
            console.print(f"  inserted {len(preds):,} rows")

            cur.execute(
                "SELECT model_version, COUNT(*) FROM model_predictions "
                "GROUP BY model_version ORDER BY model_version;"
            )
            for row in cur.fetchall():
                console.print(f"    {row[0]:<32}{row[1]:>8,} rows")

        conn.commit()
    console.print("[green]Done.[/green]")


if __name__ == "__main__":
    main()
