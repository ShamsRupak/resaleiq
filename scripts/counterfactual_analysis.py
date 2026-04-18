"""Counterfactual dollar-impact analysis: targeted vs baseline on the planted segment.

Framing: *error reduction in dollars per offer*, not revenue uplift. The
honest question is "how many dollars of pricing error does the targeted
model save per planted-segment offer, relative to the baseline?" Converting
that to revenue requires assumptions about how pricing decisions translate
into sales, which this script does not claim.

Outputs:
    docs/phase3_counterfactual.png   histogram of per-offer error reduction
    data/phase3_counterfactual.parquet   per-offer table with both errors
    console output with summary statistics
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from resaleiq.config import DATA_DIR
from resaleiq.data_generation.market_dynamics import is_iphone_launch_month


BRAND_NAVY = "#0d1b2a"
BRAND_ACCENT = "#1d9bf0"
BRAND_GRAY = "#6b7280"


def main() -> None:
    console = Console()

    preds = pd.read_parquet(DATA_DIR / "phase3_predictions.parquet")
    sku_offers = pd.read_parquet(DATA_DIR / "sku_offers.parquet")
    sku_listings = pd.read_parquet(DATA_DIR / "sku_listings.parquet")
    skus = pd.read_parquet(DATA_DIR / "skus.parquet")
    devices = pd.read_parquet(DATA_DIR / "devices.parquet")

    # Join predictions with device category so we can filter to planted segment.
    offers = (
        sku_offers.dropna(subset=["clearing_price"])[["offer_id", "listing_id", "offer_at"]]
        .merge(sku_listings[["listing_id", "sku_id"]], on="listing_id")
        .merge(skus[["sku_id", "device_id"]], on="sku_id")
        .merge(devices[["device_id", "device_category"]], on="device_id")
    )
    offers["offer_at"] = pd.to_datetime(offers["offer_at"])
    offers["is_launch"] = offers["offer_at"].apply(
        lambda ts: is_iphone_launch_month(ts.date())
    )
    offers["is_planted"] = (offers["device_category"] == "Android Mid") & offers["is_launch"]

    # Keep only the two models we compare.
    wide = (
        preds[preds["model_version"].isin(["baseline_v2_xgb", "targeted_v1_xgb"])]
        .pivot_table(
            index="target_id",
            columns="model_version",
            values="predicted",
        )
        .reset_index()
        .rename(columns={"target_id": "offer_id"})
    )
    actuals = (
        preds[preds["model_version"] == "baseline_v2_xgb"][["target_id", "actual"]]
        .rename(columns={"target_id": "offer_id"})
    )
    df = wide.merge(actuals, on="offer_id").merge(
        offers[["offer_id", "device_category", "is_launch", "is_planted"]], on="offer_id"
    )

    df["abs_err_baseline"] = (df["baseline_v2_xgb"] - df["actual"]).abs()
    df["abs_err_targeted"] = (df["targeted_v1_xgb"] - df["actual"]).abs()
    df["err_reduction"] = df["abs_err_baseline"] - df["abs_err_targeted"]
    df["abs_pct_err_baseline"] = df["abs_err_baseline"] / df["actual"]
    df["abs_pct_err_targeted"] = df["abs_err_targeted"] / df["actual"]

    planted = df[df["is_planted"]].copy()
    overall = df.copy()

    # Summary table.
    table = Table(title="Counterfactual dollar impact: targeted_v1_xgb vs baseline_v2_xgb")
    table.add_column("segment")
    table.add_column("n offers", justify="right")
    table.add_column("avg clearing $", justify="right")
    table.add_column("MAPE base", justify="right")
    table.add_column("MAPE tgt", justify="right")
    table.add_column("$ err reduction / offer", justify="right")
    table.add_column("total $ err reduction", justify="right")
    for name, subset in [("planted (Android Mid x launch)", planted), ("all offers", overall)]:
        n = len(subset)
        avg_actual = subset["actual"].mean()
        mape_b = subset["abs_pct_err_baseline"].mean() * 100
        mape_t = subset["abs_pct_err_targeted"].mean() * 100
        per_offer = subset["err_reduction"].mean()
        total = subset["err_reduction"].sum()
        table.add_row(
            name,
            f"{n:,}",
            f"${avg_actual:,.0f}",
            f"{mape_b:.2f}%",
            f"{mape_t:.2f}%",
            f"${per_offer:+.2f}",
            f"${total:+,.0f}",
        )
    console.print(table)

    # Save the per-offer table.
    out_parquet = DATA_DIR / "phase3_counterfactual.parquet"
    df[
        [
            "offer_id",
            "device_category",
            "is_launch",
            "is_planted",
            "actual",
            "baseline_v2_xgb",
            "targeted_v1_xgb",
            "abs_err_baseline",
            "abs_err_targeted",
            "err_reduction",
        ]
    ].to_parquet(out_parquet, index=False)
    console.print(f"\n[green]Wrote per-offer table to {out_parquet}[/green]")

    # Chart: histogram of per-offer error reduction in the planted segment.
    fig, ax = plt.subplots(1, 1, figsize=(10, 5.5))
    ax.hist(
        planted["err_reduction"],
        bins=40,
        color=BRAND_NAVY,
        edgecolor="white",
        linewidth=0.5,
        alpha=0.85,
    )
    ax.axvline(0, color="red", linestyle="--", linewidth=1, alpha=0.6, label="no improvement")
    ax.axvline(
        float(planted["err_reduction"].mean()),
        color=BRAND_ACCENT,
        linewidth=2,
        label=f"mean: ${planted['err_reduction'].mean():+.2f}/offer",
    )
    ax.set_xlabel("Error reduction per offer (dollars)")
    ax.set_ylabel("Number of offers")
    ax.set_title(
        f"Per-offer error reduction on the planted segment\n"
        f"Android Mid during iPhone launch months, n={len(planted):,} offers in test set"
    )
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    chart_path = Path(__file__).resolve().parents[1] / "docs" / "phase3_counterfactual.png"
    plt.savefig(chart_path, dpi=130, bbox_inches="tight")
    plt.close()
    console.print(f"[green]Wrote chart to {chart_path}[/green]")

    # One-line quotable summary statistic.
    mean_per_offer = float(planted["err_reduction"].mean())
    total = float(planted["err_reduction"].sum())
    n_planted = len(planted)
    console.print(
        f"\n[bold]Quotable statistic:[/bold] the targeted model reduces absolute pricing error "
        f"by an average of [bold]${mean_per_offer:+.2f} per offer[/bold] on the planted "
        f"segment (Android Mid during iPhone launch months), for a total of "
        f"[bold]${total:+,.0f}[/bold] across {n_planted:,} test-set offers."
    )


if __name__ == "__main__":
    main()
