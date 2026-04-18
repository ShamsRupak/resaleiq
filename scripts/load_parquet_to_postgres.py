#!/usr/bin/env python
"""Bootstrap script: load data/*.parquet into the running Postgres sandbox.

Assumes ``make db-up`` has been run (Postgres container is healthy) and
``make generate`` has produced the parquet files.
"""

from __future__ import annotations

import sys

from resaleiq.db.loader import load_all


def main() -> int:
    try:
        row_counts = load_all()
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    total = sum(row_counts.values())
    print(f"\nLoaded {total:,} rows across {len(row_counts)} tables.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
