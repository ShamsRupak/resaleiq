"""Evaluation metrics for Phase 3 models.

Focused on the metrics that match the Phase 2 SQL query layer: MAPE by
segment, excess error over baseline, and conformal interval coverage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from resaleiq.data_generation.market_dynamics import is_iphone_launch_month


def mape(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Mean absolute percentage error, with safe division."""

    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    non_zero = y_true_arr != 0
    if not non_zero.any():
        return float("nan")
    return float(
        np.mean(np.abs(y_pred_arr[non_zero] - y_true_arr[non_zero]) / y_true_arr[non_zero])
    )


def coverage_rate(
    y_true: np.ndarray | pd.Series,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Fraction of true values inside [lower, upper]."""

    y = np.asarray(y_true, dtype=float)
    return float(((y >= lower) & (y <= upper)).mean())


@dataclass
class SegmentMetrics:
    """Summary metrics per (device_category, is_launch_month) bucket."""

    rows: pd.DataFrame


def compute_segment_mape(
    df: pd.DataFrame,
    pred_col: str,
    actual_col: str = "clearing_price",
) -> SegmentMetrics:
    """Compute MAPE per (device_category, is_launch_month) segment.

    ``df`` must include columns ``device_category``, ``offer_at``, and the
    prediction and actual columns named by the arguments. Returns a
    DataFrame with one row per segment plus the overall aggregate.
    """

    out = df.copy()
    out["is_launch"] = out["offer_at"].apply(
        lambda ts: is_iphone_launch_month(pd.Timestamp(ts).date())
    )
    out["ape"] = (out[pred_col] - out[actual_col]).abs() / out[actual_col]

    grouped = (
        out.groupby(["device_category", "is_launch"])
        .agg(n=("ape", "size"), mape=("ape", "mean"))
        .reset_index()
    )
    overall = pd.DataFrame(
        [
            {
                "device_category": "ALL",
                "is_launch": "ALL",
                "n": len(out),
                "mape": float(out["ape"].mean()),
            }
        ]
    )
    return SegmentMetrics(rows=pd.concat([grouped, overall], ignore_index=True))


def compute_planted_segment_mape(df: pd.DataFrame, pred_col: str) -> float:
    """Return MAPE on Android Mid during iPhone launch months."""

    out = df.copy()
    out["is_launch"] = out["offer_at"].apply(
        lambda ts: is_iphone_launch_month(pd.Timestamp(ts).date())
    )
    planted = out[(out["device_category"] == "Android Mid") & out["is_launch"]]
    if len(planted) == 0:
        return float("nan")
    return mape(planted["clearing_price"], planted[pred_col])


__all__ = [
    "SegmentMetrics",
    "compute_planted_segment_mape",
    "compute_segment_mape",
    "coverage_rate",
    "mape",
]
