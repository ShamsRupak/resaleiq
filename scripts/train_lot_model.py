"""Train the lot-level XGBoost pricing model with a time-series holdout.

Splits cleared lots on ``opened_at``: lots before Feb 1 2026 train the model,
lots from Feb 1 2026 onward are the held-out test set. Reports both
in-sample and out-of-sample MAPE against a naive baseline.

The in-sample number measures how expressive the features are. The
out-of-sample number measures how the model generalizes to future lots.
Gap between them is honest evidence of overfitting risk.

Run once after Phase 1 data exists. Saves the trained model to disk as a
pickled LotModel plus a JSON summary of training metrics. The dashboard's
lot optimizer page loads this artifact at runtime.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from resaleiq.config import DATA_DIR
from resaleiq.ml.lot_model import (
    assemble_lot_feature_frame,
    select_features,
    train_lot_model_with_holdout,
)

console = Console()


def main() -> None:
    console.print("[bold]Loading lot + sku + device tables...[/bold]")
    lots = pd.read_parquet(DATA_DIR / "lots.parquet")
    lot_items = pd.read_parquet(DATA_DIR / "lot_items.parquet")
    skus = pd.read_parquet(DATA_DIR / "skus.parquet")
    devices = pd.read_parquet(DATA_DIR / "devices.parquet")

    console.print(
        f"  lots: [cyan]{len(lots):,}[/cyan], "
        f"lot_items: [cyan]{len(lot_items):,}[/cyan], "
        f"cleared rate: [cyan]{(lots.status == 'cleared').mean():.2%}[/cyan]"
    )

    console.print("\n[bold]Assembling lot feature frame...[/bold]")
    frame = assemble_lot_feature_frame(lots, lot_items, skus, devices)
    console.print(f"  rows: [cyan]{len(frame):,}[/cyan]")

    console.print(
        "\n[bold]Training lot-level ratio model with time-series holdout...[/bold]"
    )
    console.print(
        "  [dim]Holdout = lots with opened_at >= 2026-02-01 (Feb and Mar 2026)[/dim]"
    )
    result = train_lot_model_with_holdout(
        frame,
        holdout_start=pd.Timestamp("2026-02-01"),
        num_boost_round=500,
        learning_rate=0.03,
    )
    model = result["model"]

    # Report split and metrics
    train_lo, train_hi = result["train_period"]
    hold_lo, hold_hi = result["holdout_period"]

    split_tbl = Table(title="Time-series split")
    split_tbl.add_column("set")
    split_tbl.add_column("n_lots", justify="right")
    split_tbl.add_column("period")
    split_tbl.add_row(
        "train",
        f"{result['n_train']:,}",
        f"{train_lo.date()} to {train_hi.date()}",
    )
    split_tbl.add_row(
        "holdout",
        f"{result['n_holdout']:,}",
        f"{hold_lo.date()} to {hold_hi.date()}",
    )
    console.print(split_tbl)

    metrics_tbl = Table(title="MAPE: in-sample vs out-of-sample vs naive")
    metrics_tbl.add_column("metric")
    metrics_tbl.add_column("value", justify="right")
    metrics_tbl.add_row(
        "in-sample aggregate MAPE", f"{result['in_sample_mape']:.2%}"
    )
    metrics_tbl.add_row(
        "in-sample median APE", f"{result['in_sample_median_ape']:.2%}"
    )
    metrics_tbl.add_row("out-of-sample aggregate MAPE", f"{result['oos_mape']:.2%}")
    metrics_tbl.add_row("out-of-sample median APE", f"{result['oos_median_ape']:.2%}")
    metrics_tbl.add_row("naive baseline MAPE (OOS)", f"{result['naive_mape']:.2%}")
    metrics_tbl.add_row(
        "relative improvement vs naive (OOS)",
        f"{result['relative_improvement_oos']:.1%}",
    )
    metrics_tbl.add_row(
        "holdout conformal coverage (80% target)",
        f"{result['holdout_conformal_coverage']:.1%}",
    )
    console.print(metrics_tbl)

    # Feature importance
    importance = model.booster.get_score(importance_type="gain")
    sorted_imp = sorted(importance.items(), key=lambda x: -x[1])[:10]
    tbl = Table(title="Top-10 lot-model features by gain")
    tbl.add_column("feature")
    tbl.add_column("gain", justify="right")
    for feat, gain in sorted_imp:
        tbl.add_row(feat, f"{gain:.2f}")
    console.print(tbl)

    # Save artifact
    artifact_path = DATA_DIR / "lot_model.pkl"
    with open(artifact_path, "wb") as f:
        pickle.dump(model, f)
    console.print(f"\nWrote lot model artifact to [cyan]{artifact_path}[/cyan]")

    summary = {
        "n_cleared_lots": int((frame["status"] == "cleared").sum()),
        "n_train": result["n_train"],
        "n_holdout": result["n_holdout"],
        "holdout_start": str(result["holdout_start"].date()),
        "train_period_start": str(train_lo.date()),
        "train_period_end": str(train_hi.date()),
        "holdout_period_start": str(hold_lo.date()),
        "holdout_period_end": str(hold_hi.date()),
        "n_features": len(model.feature_names),
        "features": model.feature_names,
        "in_sample_aggregate_mape": result["in_sample_mape"],
        "in_sample_median_ape": result["in_sample_median_ape"],
        "oos_aggregate_mape": result["oos_mape"],
        "oos_median_ape": result["oos_median_ape"],
        "naive_baseline_mape_oos": result["naive_mape"],
        "relative_improvement_vs_naive_oos": result["relative_improvement_oos"],
        "holdout_conformal_coverage": result["holdout_conformal_coverage"],
        "ratio_median": model.ratio_median,
        "ratio_mad": model.ratio_mad,
        "top_features_by_gain": [
            {"feature": k, "gain": float(v)} for k, v in sorted_imp
        ],
    }
    summary_path = DATA_DIR / "lot_model_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    console.print(f"Wrote summary to [cyan]{summary_path}[/cyan]")


if __name__ == "__main__":
    main()
