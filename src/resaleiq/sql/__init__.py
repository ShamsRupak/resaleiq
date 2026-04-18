"""SQL analysis layer: CTE-based queries over the ResaleIQ schema."""

from pathlib import Path

SQL_DIR: Path = Path(__file__).parent


def list_queries() -> list[str]:
    """Return the sorted list of query names (file stems)."""

    return sorted(p.stem for p in SQL_DIR.glob("*.sql"))


def load_query(name: str) -> str:
    """Load a query's SQL text by filename stem (without the .sql extension).

    Raises ``FileNotFoundError`` with a helpful message listing available
    queries if the name doesn't resolve.
    """

    path = SQL_DIR / f"{name}.sql"
    if not path.exists():
        available = ", ".join(list_queries())
        raise FileNotFoundError(f"Query '{name}' not found in {SQL_DIR}. Available: {available}")
    return path.read_text(encoding="utf-8")


__all__ = ["SQL_DIR", "list_queries", "load_query"]
