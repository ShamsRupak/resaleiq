"""Feature engineering for the offer-clearing pricing model.

Two feature groups:

Base features (used by both baseline and targeted models):
    - device_category, condition_grade, storage_gb, carrier: categorical
    - msrp_new: nominal list price when new
    - device_age_days: days between release and offer
    - offer_month: month index for mild seasonality
    - is_flagship: binary convenience flag
    - baseline_value_usd: the internal anchor the generator uses

Cross-brand features (added only by the targeted model):
    - days_since_latest_iphone_launch
    - iphone_price_change_30d: rolling iPhone-category price change over the
      30 days prior to each offer
    - cross_category_price_ratio: Android price level relative to iPhone
      price level, same 30-day window

The hypothesis: during iPhone launch months, Android Mid prices depress
because of cross-brand substitution. A model with only Android-side features
can't see this. Adding the three cross-brand features lets the model learn
the effect.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from resaleiq.data_generation.market_dynamics import (
    days_since_latest_iphone_launch,
    is_iphone_launch_month,
)

# Ordered categorical encoding preserves the implicit grade ordering.
CONDITION_GRADE_ORDER: dict[str, int] = {"A+": 5, "A": 4, "B": 3, "C": 2, "D": 1}
DEVICE_CATEGORY_ORDER: dict[str, int] = {
    "Apple Flagship": 4,
    "Apple Mid": 3,
    "Android Flagship": 2,
    "Android Mid": 1,
    "Android Budget": 0,
}
CARRIER_ORDER: dict[str, int] = {
    "Unlocked": 4,
    "ATT": 3,
    "Verizon": 2,
    "TMobile": 1,
    "Sprint": 0,
}

FeatureSet = Literal[
    "baseline", "plus_launch", "plus_launch_days", "plus_launch_days_price", "targeted"
]


def assemble_cleared_offers(
    sku_offers: pd.DataFrame,
    sku_listings: pd.DataFrame,
    skus: pd.DataFrame,
    devices: pd.DataFrame,
) -> pd.DataFrame:
    """Join the four tables into a cleared-offer DataFrame.

    Returns one row per SKU offer that cleared (has a clearing price) with
    every attribute needed to engineer features downstream.
    """

    cleared = sku_offers.dropna(subset=["clearing_price"]).copy()
    df = (
        cleared.merge(sku_listings[["listing_id", "sku_id"]], on="listing_id", how="left")
        .merge(
            skus[
                [
                    "sku_id",
                    "device_id",
                    "storage_gb",
                    "carrier",
                    "condition_grade",
                    "baseline_value_usd",
                ]
            ],
            on="sku_id",
            how="left",
        )
        .merge(
            devices[
                [
                    "device_id",
                    "manufacturer",
                    "model_family",
                    "device_category",
                    "release_date",
                    "msrp_new",
                ]
            ],
            on="device_id",
            how="left",
        )
    )
    df["offer_at"] = pd.to_datetime(df["offer_at"])
    df["release_date"] = pd.to_datetime(df["release_date"])
    return df


def _rolling_median_by_date(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    window_days: int,
) -> pd.DataFrame:
    """Rolling median of ``value_col`` over a backward window of ``window_days``.

    Returns a DataFrame with columns [date_col, 'rolling_value'] sorted by
    date and deduplicated so it can be used as the right side of a
    ``merge_asof`` call. We collapse multiple rows at the same timestamp by
    taking the last observed rolling value.
    """

    if len(df) == 0:
        return pd.DataFrame({date_col: pd.to_datetime([]), "rolling_value": []})
    subset = df[[date_col, value_col]].sort_values(date_col).copy()
    indexed = subset.set_index(date_col)[value_col]
    rolled = indexed.rolling(f"{window_days}D", closed="left").median()
    result = rolled.reset_index().rename(columns={value_col: "rolling_value"})
    # Collapse duplicate timestamps to the last observed rolling value.
    result = result.groupby(date_col, as_index=False).last()
    return result


def build_cross_brand_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the three cross-brand features as new columns.

    Expects ``df`` to have columns ``offer_at``, ``device_category``, and
    ``clearing_price`` already populated by ``assemble_cleared_offers``.
    Uses ``pd.merge_asof`` for robust time-based lookups that tolerate
    duplicate timestamps.
    """

    out = df.copy()
    out["offer_at"] = pd.to_datetime(out["offer_at"])

    # Feature 1: days since the latest iPhone launch month began.
    out["days_since_latest_iphone_launch"] = (
        out["offer_at"].apply(lambda ts: days_since_latest_iphone_launch(ts.date())).astype(float)
    )

    # Feature 4 (added first because it's the cleanest signal):
    # binary indicator for whether the offer is within an iPhone launch month.
    # iPhone launch dates are public information; any competent analyst would
    # add this feature before more complex cross-brand signals.
    out["is_iphone_launch_month"] = (
        out["offer_at"].apply(lambda ts: is_iphone_launch_month(ts.date())).astype(int)
    )

    iphone_categories = ["Apple Flagship", "Apple Mid"]
    iphone_df = out[out["device_category"].isin(iphone_categories)][["offer_at", "clearing_price"]]
    android_mid_df = out[out["device_category"] == "Android Mid"][["offer_at", "clearing_price"]]

    roll_30_iphone = _rolling_median_by_date(iphone_df, "offer_at", "clearing_price", 30)
    roll_60_iphone = _rolling_median_by_date(iphone_df, "offer_at", "clearing_price", 60)
    roll_30_android_mid = _rolling_median_by_date(android_mid_df, "offer_at", "clearing_price", 30)

    # Feature 2: 30-day rolling iPhone median change vs the prior 30 days.
    merged = pd.merge_asof(
        roll_30_iphone.rename(columns={"rolling_value": "r30"}),
        roll_60_iphone.rename(columns={"rolling_value": "r60"}),
        on="offer_at",
        direction="backward",
    )
    merged["iphone_price_change_30d"] = (
        (merged["r30"] - merged["r60"]) / merged["r60"].replace(0, np.nan)
    ).fillna(0.0)
    iphone_price_change_lookup = merged[["offer_at", "iphone_price_change_30d"]].sort_values(
        "offer_at"
    )

    # Feature 3: Android Mid median price / iPhone median price over 30 days.
    # Align both rolling series on iPhone timestamps via asof, then merge
    # onto Android timestamps.
    ratio_merged = pd.merge_asof(
        roll_30_android_mid.rename(columns={"rolling_value": "am30"}),
        roll_30_iphone.rename(columns={"rolling_value": "ip30"}),
        on="offer_at",
        direction="backward",
    )
    ratio_merged["cross_category_price_ratio"] = (
        ratio_merged["am30"] / ratio_merged["ip30"].replace(0, np.nan)
    ).fillna(1.0)
    ratio_lookup = ratio_merged[["offer_at", "cross_category_price_ratio"]].sort_values("offer_at")

    # Attach features to every row in ``out`` using asof against the sorted
    # lookups. For offers whose exact timestamp predates the first lookup
    # entry we fill with a neutral default.
    out_sorted = out.sort_values("offer_at").reset_index().rename(columns={"index": "_orig_idx"})

    out_sorted = pd.merge_asof(
        out_sorted,
        iphone_price_change_lookup,
        on="offer_at",
        direction="backward",
    )
    out_sorted["iphone_price_change_30d"] = out_sorted["iphone_price_change_30d"].fillna(0.0)

    out_sorted = pd.merge_asof(
        out_sorted,
        ratio_lookup,
        on="offer_at",
        direction="backward",
    )
    out_sorted["cross_category_price_ratio"] = out_sorted["cross_category_price_ratio"].fillna(1.0)

    # Restore the original row order.
    out_sorted = out_sorted.sort_values("_orig_idx").drop(columns=["_orig_idx"])
    out_sorted.index = out.index
    return out_sorted


def build_feature_matrix(
    df: pd.DataFrame,
    feature_set: FeatureSet,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build the feature matrix X and target y for a given feature set.

    feature_set levels (nested, each adds to the prior):
        'baseline'         -> base features only (9 features)
        'plus_days'        -> base + days_since_latest_iphone_launch (10)
        'plus_days_price'  -> base + days + iphone_price_change_30d (11)
        'targeted'         -> all features including cross_category_ratio (12)

    This enables the ablation study in ``train_all_models.py``.
    """

    base_feature_cols = [
        "device_category_enc",
        "condition_grade_enc",
        "carrier_enc",
        "storage_gb",
        "storage_log2",
        "msrp_new",
        "baseline_value_usd",
        "device_age_days",
        "is_flagship",
    ]

    X = pd.DataFrame(index=df.index)
    X["device_category_enc"] = df["device_category"].map(DEVICE_CATEGORY_ORDER).astype(int)
    X["condition_grade_enc"] = df["condition_grade"].map(CONDITION_GRADE_ORDER).astype(int)
    X["carrier_enc"] = df["carrier"].map(CARRIER_ORDER).astype(int)
    X["storage_gb"] = df["storage_gb"].astype(int)
    X["storage_log2"] = np.log2(df["storage_gb"].astype(float))
    X["msrp_new"] = df["msrp_new"].astype(float)
    X["baseline_value_usd"] = df["baseline_value_usd"].astype(float)
    X["device_age_days"] = (df["offer_at"] - df["release_date"]).dt.total_seconds() / 86400.0
    X["is_flagship"] = (
        df["device_category"].isin(["Apple Flagship", "Android Flagship"]).astype(int)
    )

    # Add launch-related features progressively. These are *all* time-sensitive
    # features: a baseline model with only static device attributes is blind
    # to seasonality. The ladder goes from simplest (publicly-observable launch
    # indicator) to most information-dense (rolling price signals).
    if feature_set in ("plus_launch", "plus_launch_days", "plus_launch_days_price", "targeted"):
        X["is_iphone_launch_month"] = df["is_iphone_launch_month"]
    if feature_set in ("plus_launch_days", "plus_launch_days_price", "targeted"):
        X["days_since_latest_iphone_launch"] = df["days_since_latest_iphone_launch"]
    if feature_set in ("plus_launch_days_price", "targeted"):
        X["iphone_price_change_30d"] = df["iphone_price_change_30d"]
    if feature_set == "targeted":
        X["cross_category_price_ratio"] = df["cross_category_price_ratio"]
        # Explicit launch x Android Mid interaction. Motivated by the segment
        # MAPE audit surfacing Android Mid during iPhone launch months as the
        # dominant error concentration. Encoding the interaction directly
        # helps XGBoost find it despite limited training examples (~300 per
        # launch season).
        X["launch_x_android_mid"] = (
            df["is_iphone_launch_month"] * (df["device_category"] == "Android Mid").astype(int)
        ).astype(int)

    y = df["clearing_price"].astype(float)

    _ = base_feature_cols  # kept for docstring reference
    return X, y


__all__ = [
    "CARRIER_ORDER",
    "CONDITION_GRADE_ORDER",
    "DEVICE_CATEGORY_ORDER",
    "FeatureSet",
    "assemble_cleared_offers",
    "build_cross_brand_features",
    "build_feature_matrix",
]
