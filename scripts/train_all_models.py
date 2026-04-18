"""Train the Phase 3 model family: baseline_v2_xgb, two ablations, targeted_v1_xgb.

Temporal split:
    Train        : 2024-10-01 to 2025-07-31   (10 months, includes 1 iPhone launch)
    Calibration  : 2025-08-01 to 2025-08-31   (1 month, for split-conformal)
    Test         : 2025-09-01 to 2026-03-31   (7 months, includes 2nd launch)

Test set includes the second iPhone launch cycle, so we evaluate whether
models that saw the first cycle during training can correctly handle the
next one. A baseline model (no cross-brand features) cannot; the targeted
model should.

Outputs:
    data/phase3_predictions.parquet     rows = test-set offers x 4 models
    data/phase3_summary.parquet          per-model MAPE summary

Loads the 4 new model_predictions rows into Postgres (append to existing
baseline_v1 synthetic rows).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from resaleiq.config import DATA_DIR
from resaleiq.ml.evaluate import (
    compute_planted_segment_mape,
    compute_segment_mape,
    coverage_rate,
    mape,
)
from resaleiq.ml.features import (
    FeatureSet,
    assemble_cleared_offers,
    build_cross_brand_features,
    build_feature_matrix,
)
from resaleiq.ml.train import (
    TrainedModel,
    compute_conformal_intervals,
    feature_importance,
    train_xgb_model,
)

TRAIN_END = pd.Timestamp("2025-08-01")  # exclusive
CAL_END = pd.Timestamp("2025-09-01")    # exclusive
TEST_END = pd.Timestamp("2026-04-01")   # exclusive

FEATURE_SETS: list[tuple[FeatureSet, str]] = [
    ("baseline", "baseline_v2_xgb"),
    ("plus_launch", "plus_launch_xgb"),
    ("plus_launch_days", "plus_launch_days_xgb"),
    ("plus_launch_days_price", "plus_launch_days_price_xgb"),
    ("targeted", "targeted_v1_xgb"),
]

MODEL_TAG_TO_LABEL: dict[str, str] = {
    "baseline_v2_xgb": "Baseline XGBoost (static attributes only)",
    "plus_launch_xgb": "+is_iphone_launch_month",
    "plus_launch_days_xgb": "+days_since_iphone_launch",
    "plus_launch_days_price_xgb": "+iphone_price_change_30d",
    "targeted_v1_xgb": "+cross_category_price_ratio (full targeted)",
}


def _split_dataframe(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return train / calibration / test DataFrames by ``offer_at``."""

    offer_at = pd.to_datetime(df["offer_at"])
    train = df[offer_at < TRAIN_END].copy()
    cal = df[(offer_at >= TRAIN_END) & (offer_at < CAL_END)].copy()
    test = df[(offer_at >= CAL_END) & (offer_at < TEST_END)].copy()
    return train, cal, test


def _fit_one(
    feature_set: FeatureSet,
    train_df: pd.DataFrame,
    cal_df: pd.DataFrame,
) -> TrainedModel:
    """Fit XGBoost with train/cal split for early stopping.

    Uses the price-ratio target (clearing_price / baseline_value_usd) so
    multiplicative effects like the launch-month depression surface as
    first-order signal rather than getting buried under baseline_value's
    dollar magnitude.
    """

    X_train, y_train = build_feature_matrix(train_df, feature_set=feature_set)
    X_cal, y_cal = build_feature_matrix(cal_df, feature_set=feature_set)
    return train_xgb_model(
        X_train, y_train, X_cal, y_cal,
        target_type="ratio",
        baseline_train=train_df["baseline_value_usd"],
        baseline_val=cal_df["baseline_value_usd"],
    )


def _predict_one(
    model: TrainedModel,
    feature_set: FeatureSet,
    df: pd.DataFrame,
) -> np.ndarray:
    X, _ = build_feature_matrix(df, feature_set=feature_set)
    return model.predict(X, baseline_value=df["baseline_value_usd"])


def main() -> None:
    console = Console()
    data_dir = DATA_DIR

    console.print("[bold]Loading parquet tables...[/bold]")
    sku_offers = pd.read_parquet(data_dir / "sku_offers.parquet")
    sku_listings = pd.read_parquet(data_dir / "sku_listings.parquet")
    skus = pd.read_parquet(data_dir / "skus.parquet")
    devices = pd.read_parquet(data_dir / "devices.parquet")

    console.print("[bold]Assembling cleared-offer feature frame...[/bold]")
    df = assemble_cleared_offers(sku_offers, sku_listings, skus, devices)
    df = build_cross_brand_features(df)
    console.print(f"  cleared offers: {len(df):,}")

    train_df, cal_df, test_df = _split_dataframe(df)
    console.print(
        f"  train={len(train_df):,}   cal={len(cal_df):,}   test={len(test_df):,}"
    )
    if len(cal_df) == 0 or len(test_df) == 0:
        raise RuntimeError(
            "Calibration or test fold is empty; check TRAIN_END / CAL_END bounds."
        )

    summary_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    models: dict[str, TrainedModel] = {}

    for feature_set, model_tag in FEATURE_SETS:
        t0 = time.time()
        console.print(f"\n[bold cyan]Training[/bold cyan] {model_tag} ({feature_set})")
        model = _fit_one(feature_set, train_df, cal_df)
        models[model_tag] = model
        console.print(
            f"  best_iteration={model.best_iteration}   elapsed={time.time() - t0:.1f}s"
        )

        # Predict on test set for everyone; use calibration set for conformal
        # on the targeted model only.
        test_pred = _predict_one(model, feature_set, test_df)
        test_mape = mape(test_df["clearing_price"], test_pred)
        planted_mape = compute_planted_segment_mape(
            test_df.assign(_pred=test_pred), pred_col="_pred"
        )

        summary_rows.append(
            {
                "model_version": model_tag,
                "feature_set": feature_set,
                "n_features": len(model.feature_names),
                "best_iteration": model.best_iteration,
                "test_mape": round(test_mape, 4),
                "planted_segment_mape": round(planted_mape, 4),
                "label": MODEL_TAG_TO_LABEL[model_tag],
            }
        )

        # Store predictions for parquet / Postgres write.
        pred_frame = pd.DataFrame(
            {
                "target_type": "sku_offer",
                "target_id": test_df["offer_id"].values,
                "predicted": test_pred,
                "actual": test_df["clearing_price"].values,
                "pred_at": test_df["offer_at"].values,
                "model_version": model_tag,
                "predicted_low": np.nan,
                "predicted_high": np.nan,
            }
        )

        # Conformal intervals: only on the targeted model.
        if model_tag == "targeted_v1_xgb":
            X_cal_feat, y_cal_feat = build_feature_matrix(cal_df, feature_set=feature_set)
            X_test_feat, _ = build_feature_matrix(test_df, feature_set=feature_set)
            conformal = compute_conformal_intervals(
                model,
                X_cal=X_cal_feat,
                y_cal=y_cal_feat,
                X_pred=X_test_feat,
                alpha=0.2,
                baseline_cal=cal_df["baseline_value_usd"],
                baseline_pred=test_df["baseline_value_usd"],
            )
            cov = coverage_rate(
                test_df["clearing_price"].values, conformal.lower, conformal.upper
            )
            pred_frame["predicted_low"] = conformal.lower
            pred_frame["predicted_high"] = conformal.upper
            console.print(
                f"  conformal ratio half-width={conformal.half_width:.4f}   "
                f"test coverage={cov:.3f} (target 0.80)"
            )
            summary_rows[-1]["conformal_ratio_halfwidth"] = round(
                conformal.half_width, 4
            )
            summary_rows[-1]["conformal_test_coverage"] = round(cov, 3)

        prediction_frames.append(pred_frame)

    # Consolidate.
    all_preds = pd.concat(prediction_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows)

    out_preds = data_dir / "phase3_predictions.parquet"
    out_summary = data_dir / "phase3_summary.parquet"
    all_preds.to_parquet(out_preds, index=False)
    summary.to_parquet(out_summary, index=False)

    console.print(f"\n[green]Wrote {len(all_preds):,} predictions to {out_preds}[/green]")
    console.print(f"[green]Wrote per-model summary to {out_summary}[/green]")

    # Pretty-print summary.
    table = Table(title="Phase 3 model summary (test set: Sept 2025 - Mar 2026)")
    table.add_column("model_version")
    table.add_column("n_feat", justify="right")
    table.add_column("test MAPE", justify="right")
    table.add_column("planted MAPE", justify="right")
    table.add_column("delta vs baseline", justify="right")
    baseline_planted = next(
        r["planted_segment_mape"] for r in summary_rows if r["model_version"] == "baseline_v2_xgb"
    )
    baseline_overall = next(
        r["test_mape"] for r in summary_rows if r["model_version"] == "baseline_v2_xgb"
    )
    for row in summary_rows:
        delta_overall = float(row["test_mape"]) - float(baseline_overall)
        delta_planted = float(row["planted_segment_mape"]) - float(baseline_planted)
        table.add_row(
            row["model_version"],
            str(row["n_features"]),
            f"{float(row['test_mape']) * 100:.2f}%",
            f"{float(row['planted_segment_mape']) * 100:.2f}%",
            f"{delta_overall * 100:+.2f}pp / {delta_planted * 100:+.2f}pp planted",
        )
    console.print(table)

    # Feature importance for the targeted model.
    console.print("\n[bold]Feature importance (targeted_v1_xgb, by gain):[/bold]")
    fi = feature_importance(models["targeted_v1_xgb"])
    for _, row in fi.iterrows():
        console.print(f"  {row['feature']:<40s}{row['importance']:>12.1f}")


if __name__ == "__main__":
    main()
