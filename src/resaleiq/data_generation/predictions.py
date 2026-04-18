"""Baseline model predictions for the ``model_predictions`` table.

These predictions simulate what an untuned XGBoost baseline would produce if
it were trained without the cross-brand temporal features that surface the
Android-mid-tier iPhone-launch-month effect. By design:

- Overall MAPE across all predictions lands in the 10 to 12 percent band,
  matching the industry-typical baseline for wholesale secondhand phone
  pricing models disclosed in public product materials.
- MAPE on the Android mid tier during iPhone launch months rises into the 23
  to 26 percent band, reflecting the structural signal the baseline misses.
- Grade D and newly released inventory carry wider prediction variance.

A downstream modeling notebook (Phase 4) will replace these with real model
outputs. In the meantime this table supports the SQL layer's segment audit
and the Streamlit dashboard's MAPE visualizations end-to-end.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from resaleiq.config import child_seed
from resaleiq.data_generation.market_dynamics import (
    is_iphone_launch_month,
    is_newly_released,
)

# Calibration constants for the synthetic baseline. These were selected so the
# overall MAPE across cleared transactions lands near 11 percent and the
# Android-mid iPhone-launch segment lands near 24 percent. Adjust here if the
# underlying shock parameters in ``NoiseConfig`` are tuned.
_BASE_SIGMA = 0.105
# Baseline_v1 represents the broken production model that doesn't know
# about the cross-brand depression. When real Android Mid prices during
# iPhone launches drop ~20% (to ~80% of baseline), this model keeps
# predicting at non-launch levels, i.e., it over-predicts by ~25%
# (= 0.20 / 0.80). That systematic bias plus wider variance is what
# makes its MAPE on the planted segment materially worse than any
# model with launch-aware features.
_ANDROID_MID_LAUNCH_BIAS = 0.25
_ANDROID_MID_LAUNCH_SIGMA = 0.18
_GRADE_D_EXTRA_SIGMA = 0.07
_NEW_RELEASE_EXTRA_SIGMA = 0.05


def _predict_one(
    actual_price: float,
    device_category: str,
    condition_grade: str,
    release_date: pd.Timestamp,
    transaction_date: pd.Timestamp,
    rng: np.random.Generator,
) -> float:
    """Sample a single baseline prediction given the row's context."""

    tx_date = (
        transaction_date.date() if isinstance(transaction_date, pd.Timestamp) else transaction_date
    )
    rel_date = release_date.date() if isinstance(release_date, pd.Timestamp) else release_date

    planted_segment = device_category == "Android Mid" and is_iphone_launch_month(tx_date)
    if planted_segment:
        # Baseline misses the cross-brand depression: predicts roughly what
        # the price "would have been" outside the launch months, hence a
        # positive bias (over-predicts).
        mean = _ANDROID_MID_LAUNCH_BIAS
        sigma = _ANDROID_MID_LAUNCH_SIGMA
    else:
        mean = 0.0
        sigma = _BASE_SIGMA

    if condition_grade == "D":
        sigma = float(np.hypot(sigma, _GRADE_D_EXTRA_SIGMA))
    if is_newly_released(rel_date, tx_date):
        sigma = float(np.hypot(sigma, _NEW_RELEASE_EXTRA_SIGMA))

    multiplier = 1.0 + float(rng.normal(mean, sigma))
    predicted = max(actual_price * multiplier, 1.0)
    return round(predicted, 2)


def build_predictions(
    sku_offers: pd.DataFrame,
    sku_listings: pd.DataFrame,
    skus_flat: pd.DataFrame,
    lots: pd.DataFrame,
    lot_items: pd.DataFrame,
) -> pd.DataFrame:
    """Generate the ``model_predictions`` table across both target types.

    One row per cleared sku_offer (target_type='sku_offer') and one row per
    cleared lot (target_type='lot'). Predictions for non-cleared rows are
    omitted since there is no actual to compare against.
    """

    rng = np.random.default_rng(child_seed("model_predictions"))

    # SKU offer predictions. Only cleared offers (those with a clearing_price).
    cleared_offers = sku_offers.dropna(subset=["clearing_price"]).merge(
        sku_listings[["listing_id", "sku_id"]],
        on="listing_id",
        how="left",
        validate="m:1",
    )
    cleared_offers = cleared_offers.merge(
        skus_flat[["sku_id", "device_category", "condition_grade", "release_date", "manufacturer"]],
        on="sku_id",
        how="left",
        validate="m:1",
    )

    sku_pred_rows: list[dict[str, object]] = []
    for row in cleared_offers.itertuples(index=False):
        pred = _predict_one(
            actual_price=float(row.clearing_price),
            device_category=str(row.device_category),
            condition_grade=str(row.condition_grade),
            release_date=row.release_date,
            transaction_date=row.offer_at,
            rng=rng,
        )
        sku_pred_rows.append(
            {
                "target_type": "sku_offer",
                "target_id": int(row.offer_id),
                "predicted": pred,
                "actual": round(float(row.clearing_price), 2),
                "predicted_at": row.offer_at,
                "model_version": "baseline_v1",
            }
        )

    # Lot predictions: one per cleared lot. For lot-level context we use the
    # modal device_category and grade across the lot's items.
    cleared_lots = lots[lots["status"] == "cleared"].copy()
    lot_item_ctx = lot_items.merge(
        skus_flat[["sku_id", "device_category", "condition_grade", "release_date"]],
        on="sku_id",
        how="left",
        validate="m:1",
    )

    # Modal category and grade per lot, weighted by quantity.
    def _mode_weighted(group: pd.DataFrame, col: str) -> str:
        weights = group.groupby(col)["quantity"].sum()
        return str(weights.idxmax())

    category_by_lot = lot_item_ctx.groupby("lot_id").apply(
        lambda g: _mode_weighted(g, "device_category"), include_groups=False
    )
    grade_by_lot = lot_item_ctx.groupby("lot_id").apply(
        lambda g: _mode_weighted(g, "condition_grade"), include_groups=False
    )
    # Earliest release date in the lot (to approximate "newly released" context).
    release_by_lot = lot_item_ctx.groupby("lot_id")["release_date"].max()

    lot_pred_rows: list[dict[str, object]] = []
    for row in cleared_lots.itertuples(index=False):
        lid = int(row.lot_id)
        pred = _predict_one(
            actual_price=float(row.clearing_price),
            device_category=str(category_by_lot.loc[lid]),
            condition_grade=str(grade_by_lot.loc[lid]),
            release_date=release_by_lot.loc[lid],
            transaction_date=row.scheduled_close,
            rng=rng,
        )
        lot_pred_rows.append(
            {
                "target_type": "lot",
                "target_id": lid,
                "predicted": pred,
                "actual": round(float(row.clearing_price), 2),
                "predicted_at": row.scheduled_close,
                "model_version": "baseline_v1",
            }
        )

    all_rows = sku_pred_rows + lot_pred_rows
    df = pd.DataFrame(all_rows)
    df.insert(0, "prediction_id", np.arange(1, len(df) + 1, dtype=np.int64))
    return df


__all__ = ["build_predictions"]
