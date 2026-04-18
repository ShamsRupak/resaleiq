"""End-to-end orchestrator CLI.

Run via::

    uv run resaleiq-generate
    uv run resaleiq-generate --scale sample

Writes nine parquet files into ``data/``. Reproducible from the master seed
defined in ``config.py``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from resaleiq.config import DATA_DIR, SCALES, Scale
from resaleiq.data_generation.buyers import build_buyers
from resaleiq.data_generation.devices import build_devices
from resaleiq.data_generation.lot_flow import build_lots
from resaleiq.data_generation.predictions import build_predictions
from resaleiq.data_generation.sku_flow import build_sku_listings, build_sku_offers
from resaleiq.data_generation.skus import build_skus, join_device_attrs

app = typer.Typer(
    help="Generate the ResaleIQ synthetic dataset.",
    no_args_is_help=False,
    add_completion=False,
)

console = Console()


def _write_parquet(df: pd.DataFrame, path: Path, name: str) -> int:
    """Write a DataFrame to parquet and return its size in bytes."""

    df.to_parquet(path, index=False, engine="pyarrow", compression="snappy")
    return path.stat().st_size


@app.command()
def generate(
    scale: Annotated[Scale, typer.Option(help="full or sample")] = "full",
    out_dir: Annotated[Path, typer.Option(help="Output directory")] = DATA_DIR,
    quiet: Annotated[bool, typer.Option(help="Suppress summary table")] = False,
) -> None:
    """Generate all nine tables at the requested scale."""

    out_dir.mkdir(parents=True, exist_ok=True)
    scale_cfg = SCALES[scale]
    console.print(f"[bold cyan]ResaleIQ data generation[/] scale=[bold]{scale}[/]")

    timings: dict[str, float] = {}
    sizes: dict[str, int] = {}
    row_counts: dict[str, int] = {}

    # 1. Reference layer.
    t0 = time.perf_counter()
    devices = build_devices()
    timings["devices"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    skus = build_skus(devices)
    skus_flat = join_device_attrs(skus, devices)
    timings["skus"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    buyers = build_buyers(scale_cfg)
    timings["buyers"] = time.perf_counter() - t0

    # 2. Offer-clearing layer.
    t0 = time.perf_counter()
    sku_listings = build_sku_listings(skus_flat, scale_cfg)
    timings["sku_listings"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    sku_offers = build_sku_offers(sku_listings, skus_flat, buyers, scale_cfg)
    timings["sku_offers"] = time.perf_counter() - t0

    # 3. Auction layer.
    t0 = time.perf_counter()
    lots, lot_items, lot_bids = build_lots(skus_flat, buyers, scale_cfg)
    timings["lots_and_bids"] = time.perf_counter() - t0

    # 4. Evaluation layer.
    t0 = time.perf_counter()
    predictions = build_predictions(
        sku_offers=sku_offers,
        sku_listings=sku_listings,
        skus_flat=skus_flat,
        lots=lots,
        lot_items=lot_items,
    )
    timings["predictions"] = time.perf_counter() - t0

    # Write parquet.
    outputs = {
        "devices": devices,
        "skus": skus,
        "buyers": buyers,
        "sku_listings": sku_listings,
        "sku_offers": sku_offers,
        "lots": lots,
        "lot_items": lot_items,
        "lot_bids": lot_bids,
        "model_predictions": predictions,
    }
    for name, df in outputs.items():
        path = out_dir / f"{name}.parquet"
        sizes[name] = _write_parquet(df, path, name)
        row_counts[name] = len(df)

    if not quiet:
        _print_summary(row_counts, sizes, timings, out_dir)


def _print_summary(
    row_counts: dict[str, int],
    sizes: dict[str, int],
    timings: dict[str, float],
    out_dir: Path,
) -> None:
    """Render a summary table of the generation run."""

    table = Table(title=f"Generated output in {out_dir}", show_header=True, header_style="bold")
    table.add_column("Table", style="cyan")
    table.add_column("Rows", justify="right")
    table.add_column("Size", justify="right")
    for name, rows in row_counts.items():
        size_kb = sizes[name] / 1024
        size_str = f"{size_kb:,.1f} KB" if size_kb < 1024 else f"{size_kb / 1024:,.1f} MB"
        table.add_row(name, f"{rows:,}", size_str)
    table.add_section()
    table.add_row(
        "[bold]TOTAL[/]",
        f"[bold]{sum(row_counts.values()):,}[/]",
        f"[bold]{sum(sizes.values()) / 1024 / 1024:,.1f} MB[/]",
    )
    console.print(table)

    timing_table = Table(title="Stage timings", show_header=True, header_style="bold")
    timing_table.add_column("Stage", style="cyan")
    timing_table.add_column("Seconds", justify="right")
    for stage, secs in timings.items():
        timing_table.add_row(stage, f"{secs:.2f}")
    timing_table.add_section()
    timing_table.add_row("[bold]TOTAL[/]", f"[bold]{sum(timings.values()):.2f}[/]")
    console.print(timing_table)


if __name__ == "__main__":
    app()
