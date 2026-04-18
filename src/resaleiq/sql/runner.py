"""SQL query runner: CLI for executing and inspecting analysis queries.

Usage:
    uv run python -m resaleiq.sql.runner list
    uv run python -m resaleiq.sql.runner run 01_segment_mape_audit
    uv run python -m resaleiq.sql.runner explain 01_segment_mape_audit

Or via the Makefile:
    make sql-list
    make sql-run Q=01_segment_mape_audit
"""

from __future__ import annotations

import os
from typing import Annotated, Any

import pandas as pd
import psycopg
import typer
from rich.console import Console
from rich.table import Table

from resaleiq.sql import list_queries, load_query

app = typer.Typer(
    help="Execute and inspect ResaleIQ SQL analysis queries.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def _database_url() -> str:
    """Resolve the psycopg-compatible connection string."""

    url = os.environ.get(
        "DATABASE_URL",
        "postgresql+psycopg://resaleiq:resaleiq@localhost:5432/resaleiq",
    )
    return url.replace("postgresql+psycopg://", "postgresql://")


def _execute(sql: str) -> pd.DataFrame:
    """Execute ``sql`` and return rows as a DataFrame."""

    with psycopg.connect(_database_url()) as conn, conn.cursor() as cur:
        cur.execute(sql)
        if cur.description is None:
            return pd.DataFrame()
        columns = [d.name for d in cur.description]
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=columns)


def _render_table(df: pd.DataFrame, title: str) -> None:
    """Render a DataFrame via rich, with right-aligned numeric columns."""

    if df.empty:
        console.print(f"[yellow]{title}: no rows[/]")
        return
    table = Table(title=title, show_header=True, header_style="bold", show_lines=False)
    for col in df.columns:
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        table.add_column(col, justify="right" if is_numeric else "left")
    for _, row in df.iterrows():
        table.add_row(*[_format_cell(v) for v in row.tolist()])
    console.print(table)


def _format_cell(value: Any) -> str:
    """Render a cell for the rich table."""

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "[dim]NULL[/]"
    if isinstance(value, float):
        return f"{value:,.4f}" if abs(value) < 100 else f"{value:,.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


@app.command("list")
def cmd_list() -> None:
    """List available SQL queries."""

    queries = list_queries()
    if not queries:
        console.print("[yellow]No queries found.[/]")
        raise typer.Exit(1)
    console.print("[bold]Available queries:[/]")
    for name in queries:
        console.print(f"  [cyan]{name}[/]")


@app.command("run")
def cmd_run(
    name: Annotated[str, typer.Argument(help="Query file stem (no .sql)")],
    rows: Annotated[int, typer.Option(help="Max rows to display")] = 50,
) -> None:
    """Execute a query and render the result as a table."""

    sql = load_query(name)
    console.print(f"[bold]Running[/] [cyan]{name}[/]")
    df = _execute(sql)
    if len(df) > rows:
        console.print(f"[dim]Truncating display to {rows} of {len(df)} rows[/]")
        df = df.head(rows)
    _render_table(df, title=name)


@app.command("explain")
def cmd_explain(
    name: Annotated[str, typer.Argument(help="Query file stem (no .sql)")],
    analyze: Annotated[bool, typer.Option(help="Include ANALYZE (runs query)")] = False,
) -> None:
    """Show the EXPLAIN plan for a query."""

    sql = load_query(name)
    prefix = "EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)" if analyze else "EXPLAIN"
    explain_sql = f"{prefix} {sql}"
    df = _execute(explain_sql)
    console.print(f"[bold]Explain plan for[/] [cyan]{name}[/]")
    for row in df.itertuples(index=False):
        console.print(row[0])


if __name__ == "__main__":
    app()
