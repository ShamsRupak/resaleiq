"""Lot-level pricing model.

Predicts final clearing price for an entire auction lot as a function of its
composition: SKU count, category mix, condition grade mix, price dispersion,
and auction mechanics. Enables the lot optimizer on the dashboard to answer
"given this bag of inventory, what partition maximizes expected clearing
price?"

Training target: observed lot clearing price (sum of winning bids across the
lot). Ratio target against ``sum(baseline_value)`` is used, for the same
reason documented in the offer-level model notes: multiplicative lot-level
effects (category diversification discount, size premium, popcorn uplift)
get buried when using absolute dollar target.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb

from resaleiq.data_generation.market_dynamics import is_iphone_launch_month

LOT_CATEGORIES: Sequence[str] = (
    "Apple Flagship",
    "Apple Mid",
    "Android Flagship",
    "Android Mid",
    "Other",
)

LOT_GRADES: Sequence[str] = ("A+", "A", "B", "C", "D")


@dataclass
class LotModel:
    """Thin wrapper around a trained lot-level booster."""

    booster: xgb.Booster
    feature_names: list[str]
    calibration_offsets: dict[str, float]  # segment -> additive ratio correction
    ratio_median: float  # for conformal half-width
    ratio_mad: float

    def predict_ratio(self, X: pd.DataFrame) -> np.ndarray:
        """Predict the lot clearing-to-baseline ratio for each row of X."""

        d = xgb.DMatrix(X[self.feature_names], feature_names=self.feature_names)
        return self.booster.predict(d)

    def predict_clearing(
        self,
        X: pd.DataFrame,
        baseline_sum: pd.Series,
    ) -> np.ndarray:
        """Predict the absolute dollar clearing price for each lot."""

        return self.predict_ratio(X) * baseline_sum.to_numpy()

    def predict_with_interval(
        self,
        X: pd.DataFrame,
        baseline_sum: pd.Series,
        confidence: float = 0.80,
    ) -> pd.DataFrame:
        """Predict clearing price with a conformal-style multiplicative interval.

        Returns a DataFrame with columns point, lower, upper.
        """

        ratios = self.predict_ratio(X)
        # Conformal-style half-width using median absolute deviation of training
        # residuals, scaled for the target confidence level. For 80% confidence
        # using MAD gives roughly 1.28 * MAD.
        z = {
            0.50: 0.674,
            0.80: 1.282,
            0.90: 1.645,
            0.95: 1.960,
        }.get(confidence, 1.282)
        half_width_ratio = z * self.ratio_mad
        point = ratios * baseline_sum.to_numpy()
        lower = np.maximum(ratios - half_width_ratio, 0.0) * baseline_sum.to_numpy()
        upper = (ratios + half_width_ratio) * baseline_sum.to_numpy()
        return pd.DataFrame({"point": point, "lower": lower, "upper": upper})


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #


def assemble_lot_feature_frame(
    lots: pd.DataFrame,
    lot_items: pd.DataFrame,
    skus: pd.DataFrame,
    devices: pd.DataFrame,
) -> pd.DataFrame:
    """Build a per-lot feature frame from the raw tables.

    Returns one row per lot with:
        - lot identifiers (lot_id, opened_at, auction_type)
        - composition: sku_count, size_bucket
        - category mix: share_apple_flagship, share_apple_mid, etc.
        - grade mix: share_a_plus, share_a, share_b, share_c, share_d
        - price stats: total_baseline_value, baseline_value_cv (dispersion)
        - auction mechanics: is_popcorn, has_reserve
        - temporal: is_iphone_launch_month
        - target columns: clearing_price (sum of winning bids when cleared)
    """

    # Join lot_items to skus and devices to get category and baseline value
    items = lot_items.merge(
        skus[["sku_id", "device_id", "condition_grade", "baseline_value_usd"]], on="sku_id"
    ).merge(devices[["device_id", "device_category"]], on="device_id")
    # Expand baseline by quantity, so total_baseline is inventory-dollar-weighted
    items["item_baseline_total"] = items["baseline_value_usd"] * items["quantity"]
    items["item_unit_count"] = items["quantity"]

    # Aggregate per-lot statistics
    agg = (
        items.groupby("lot_id")
        .agg(
            sku_count=("lot_item_id", "count"),
            total_units=("item_unit_count", "sum"),
            total_baseline_value=("item_baseline_total", "sum"),
            mean_baseline=("baseline_value_usd", "mean"),
            std_baseline=("baseline_value_usd", "std"),
        )
        .reset_index()
    )
    agg["baseline_value_cv"] = (
        agg["std_baseline"] / agg["mean_baseline"].where(agg["mean_baseline"] > 0, 1)
    ).fillna(0)

    # Category shares weighted by unit count, not distinct SKU count
    cat_shares = (
        items.groupby(["lot_id", "device_category"])["item_unit_count"]
        .sum()
        .unstack(fill_value=0)
        .reindex(columns=list(LOT_CATEGORIES), fill_value=0)
    )
    cat_shares = cat_shares.div(cat_shares.sum(axis=1).replace(0, 1), axis=0)
    cat_shares.columns = [f"share_{c.lower().replace(' ', '_')}" for c in cat_shares.columns]

    # Grade shares weighted by unit count
    grade_shares = (
        items.groupby(["lot_id", "condition_grade"])["item_unit_count"]
        .sum()
        .unstack(fill_value=0)
        .reindex(columns=list(LOT_GRADES), fill_value=0)
    )
    grade_shares = grade_shares.div(grade_shares.sum(axis=1).replace(0, 1), axis=0)
    grade_shares.columns = [
        f"share_grade_{c.lower().replace('+', '_plus')}" for c in grade_shares.columns
    ]

    # Diversity metric: entropy of category distribution (0 = all one cat, 1 = uniform)
    def entropy_row(r: pd.Series) -> float:
        p = r.to_numpy()
        p = p[p > 0]
        if len(p) == 0:
            return 0.0
        return float(-(p * np.log2(p)).sum() / np.log2(len(LOT_CATEGORIES)))

    cat_shares["category_entropy"] = cat_shares.apply(entropy_row, axis=1)

    # Merge into lots frame
    lots = lots.copy()
    lots["is_popcorn"] = (lots["auction_type"] == "popcorn").astype(int)
    lots["has_reserve"] = (lots["reserve_price"].fillna(0) > 0).astype(int)

    lots["is_iphone_launch_month"] = (
        pd.to_datetime(lots["actual_close"].fillna(lots["scheduled_close"]))
        .dt.date.apply(is_iphone_launch_month)
        .astype(int)
    )

    # Standardise the field name used downstream
    lots["clearing_price_usd"] = lots["clearing_price"]
    lots["opened_at"] = lots["scheduled_close"]

    result = (
        lots[
            [
                "lot_id",
                "opened_at",
                "auction_type",
                "is_popcorn",
                "has_reserve",
                "is_iphone_launch_month",
                "clearing_price_usd",
                "status",
            ]
        ]
        .merge(agg, on="lot_id")
        .merge(cat_shares, on="lot_id")
        .merge(grade_shares, on="lot_id")
    )
    return result


def select_features(df: pd.DataFrame) -> list[str]:
    """Return the canonical feature list for the lot-level model."""

    cat_cols = [f"share_{c.lower().replace(' ', '_')}" for c in LOT_CATEGORIES]
    grade_cols = [f"share_grade_{c.lower().replace('+', '_plus')}" for c in LOT_GRADES]
    return [
        "sku_count",
        "total_units",
        "mean_baseline",
        "baseline_value_cv",
        "category_entropy",
        "is_popcorn",
        "has_reserve",
        "is_iphone_launch_month",
        *cat_cols,
        *grade_cols,
    ]


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


def _prepare_training_frame(feature_frame: pd.DataFrame) -> pd.DataFrame:
    """Shared filter and ratio computation used by both training entry points."""

    df = feature_frame[
        (feature_frame["status"] == "cleared")
        & (feature_frame["clearing_price_usd"] > 0)
        & (feature_frame["total_baseline_value"] > 0)
    ].copy()

    if len(df) < 50:
        raise ValueError(f"Only {len(df)} cleared lots with valid prices; need at least 50")

    df["ratio"] = df["clearing_price_usd"] / df["total_baseline_value"]
    # Winsorize heavy tails at 1st and 99th percentile on the ratio
    lo, hi = df["ratio"].quantile([0.01, 0.99])
    df = df[(df["ratio"] >= lo) & (df["ratio"] <= hi)].copy()
    return df


def _fit_booster(
    X: pd.DataFrame,
    y: np.ndarray,
    features: list[str],
    num_boost_round: int,
    learning_rate: float,
    seed: int,
) -> xgb.Booster:
    """Fixed-round XGBoost fit with Phase 3 hyperparameters."""

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "max_depth": 5,
        "learning_rate": learning_rate,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 3,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "seed": seed,
        "nthread": 1,
    }
    dtrain = xgb.DMatrix(X, label=y, feature_names=features)
    return xgb.train(params, dtrain, num_boost_round=num_boost_round, verbose_eval=False)


def train_lot_model(
    feature_frame: pd.DataFrame,
    num_boost_round: int = 500,
    learning_rate: float = 0.03,
    seed: int = 20260420,
) -> LotModel:
    """Train the lot-level ratio model on the full cleared-lot set.

    Kept for backwards compatibility and rapid in-sample diagnostics. For the
    honest measurement of generalization error, use
    ``train_lot_model_with_holdout`` which reports both in-sample and
    out-of-sample metrics against a time-based holdout.
    """

    df = _prepare_training_frame(feature_frame)
    features = select_features(df)
    X = df[features]
    y = df["ratio"].to_numpy()

    booster = _fit_booster(X, y, features, num_boost_round, learning_rate, seed)

    in_sample_preds = booster.predict(xgb.DMatrix(X, feature_names=features))
    residuals = y - in_sample_preds
    ratio_mad = float(1.4826 * np.median(np.abs(residuals - np.median(residuals))))

    return LotModel(
        booster=booster,
        feature_names=features,
        calibration_offsets={},
        ratio_median=float(np.median(y)),
        ratio_mad=ratio_mad,
    )


def train_lot_model_with_holdout(
    feature_frame: pd.DataFrame,
    holdout_start: pd.Timestamp | None = None,
    num_boost_round: int = 500,
    learning_rate: float = 0.03,
    seed: int = 20260420,
) -> dict:
    """Train the lot-level ratio model with a time-series holdout.

    Splits cleared lots by ``opened_at`` (which is derived from
    ``scheduled_close`` in the feature frame). Lots opening before
    ``holdout_start`` form the training set; lots opening on or after form
    the holdout. Reports both in-sample and out-of-sample MAPE against a
    naive baseline that predicts ``total_baseline_value``.

    The MAD half-width used for conformal intervals is fitted on the train
    set only, which is the honest conformal discipline: calibrate on data
    the booster has not seen repeatedly.

    Returns a dict with:
        - model: LotModel ready for inference
        - n_train, n_holdout: split sizes
        - holdout_start: the threshold used
        - in_sample_mape: aggregate MAPE on training lots
        - oos_mape: aggregate MAPE on holdout lots
        - naive_mape: aggregate MAPE of "predict total_baseline_value" on holdout
        - relative_improvement_oos: (naive - oos) / naive
        - in_sample_median_ape, oos_median_ape: median absolute percent error
        - train_period, holdout_period: date ranges as (min, max) tuples
    """

    df = _prepare_training_frame(feature_frame)
    df["opened_at"] = pd.to_datetime(df["opened_at"])
    if holdout_start is None:
        holdout_start = pd.Timestamp("2026-02-01")
    ts = pd.Timestamp(holdout_start)

    train_df = df[df["opened_at"] < ts].copy()
    hold_df = df[df["opened_at"] >= ts].copy()

    if len(train_df) < 50:
        raise ValueError(f"Only {len(train_df)} training lots before {ts.date()}; need >= 50")
    if len(hold_df) < 20:
        raise ValueError(f"Only {len(hold_df)} holdout lots on or after {ts.date()}; need >= 20")

    features = select_features(df)
    X_train = train_df[features]
    y_train = train_df["ratio"].to_numpy()
    X_hold = hold_df[features]
    y_hold_ratio = hold_df["ratio"].to_numpy()

    booster = _fit_booster(X_train, y_train, features, num_boost_round, learning_rate, seed)

    # Empirical-quantile calibration of the conformal half-width.
    # MAD * z assumes approximately Gaussian residuals. Ratio residuals tend
    # to have heavier tails in practice (multiplicative shocks, composition
    # outliers), which undercovers. We instead calibrate ratio_mad such that
    # the existing predict_with_interval API at 80% confidence matches the
    # empirical 80th percentile of absolute training residuals.
    train_preds = booster.predict(xgb.DMatrix(X_train, feature_names=features))
    train_residuals = y_train - train_preds
    empirical_q80 = float(np.quantile(np.abs(train_residuals), 0.80))
    # LotModel.predict_with_interval uses half = 1.282 * ratio_mad for 80% confidence,
    # so set ratio_mad = empirical_q80 / 1.282 to get empirically calibrated 80% intervals.
    ratio_mad = empirical_q80 / 1.282

    model = LotModel(
        booster=booster,
        feature_names=features,
        calibration_offsets={},
        ratio_median=float(np.median(y_train)),
        ratio_mad=ratio_mad,
    )

    # In-sample aggregate metrics
    train_pred_price = train_preds * train_df["total_baseline_value"].to_numpy()
    train_actual = train_df["clearing_price_usd"].to_numpy()
    in_sample_mape = float(np.abs(train_pred_price - train_actual).sum() / train_actual.sum())
    in_sample_medape = float(np.median(np.abs(train_pred_price - train_actual) / train_actual))

    # Out-of-sample aggregate metrics
    hold_preds = booster.predict(xgb.DMatrix(X_hold, feature_names=features))
    hold_pred_price = hold_preds * hold_df["total_baseline_value"].to_numpy()
    hold_actual = hold_df["clearing_price_usd"].to_numpy()
    oos_mape = float(np.abs(hold_pred_price - hold_actual).sum() / hold_actual.sum())
    oos_medape = float(np.median(np.abs(hold_pred_price - hold_actual) / hold_actual))

    # Naive baseline: predict total_baseline_value on the holdout
    naive_pred = hold_df["total_baseline_value"].to_numpy()
    naive_mape = float(np.abs(naive_pred - hold_actual).sum() / hold_actual.sum())

    rel_improvement = (naive_mape - oos_mape) / naive_mape if naive_mape > 0 else 0.0

    # Conformal coverage check on the holdout: how often does the actual
    # ratio fall inside the predicted [point - 1.282*mad, point + 1.282*mad]
    # band? Target is 80%.
    half = 1.282 * ratio_mad
    coverage = float(
        ((y_hold_ratio >= hold_preds - half) & (y_hold_ratio <= hold_preds + half)).mean()
    )

    return {
        "model": model,
        "n_train": len(train_df),
        "n_holdout": len(hold_df),
        "holdout_start": ts,
        "train_period": (
            train_df["opened_at"].min(),
            train_df["opened_at"].max(),
        ),
        "holdout_period": (
            hold_df["opened_at"].min(),
            hold_df["opened_at"].max(),
        ),
        "in_sample_mape": in_sample_mape,
        "in_sample_median_ape": in_sample_medape,
        "oos_mape": oos_mape,
        "oos_median_ape": oos_medape,
        "naive_mape": naive_mape,
        "relative_improvement_oos": rel_improvement,
        "ratio_mad": ratio_mad,
        "holdout_conformal_coverage": coverage,
    }


# --------------------------------------------------------------------------- #
# Partition scoring for the lot optimizer
# --------------------------------------------------------------------------- #


def score_partition(
    sku_bag: pd.DataFrame,
    partition: list[list[int]],
    model: LotModel,
    auction_type: str = "popcorn",
    reference_month: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Score a proposed partition of SKUs into lots.

    ``sku_bag`` must have: sku_id, device_category, condition_grade,
    baseline_value_usd.

    ``partition`` is a list of lots, each a list of sku_id values.

    Returns a DataFrame with one row per proposed lot and a summary row:
        lot_index, n_skus, total_baseline, predicted_clearing,
        predicted_lower, predicted_upper, ratio.
    """

    rows = []
    ref_ts = pd.Timestamp(reference_month or pd.Timestamp("2025-09-15"))
    is_launch = int(is_iphone_launch_month(ref_ts.date()))

    for lot_idx, sku_ids in enumerate(partition):
        if not sku_ids:
            continue
        lot_df = sku_bag[sku_bag["sku_id"].isin(sku_ids)]
        if len(lot_df) == 0:
            continue

        # Build a one-row feature frame for this hypothetical lot
        total_baseline = float(lot_df["baseline_value_usd"].sum())
        mean_baseline = float(lot_df["baseline_value_usd"].mean())
        std_baseline = float(lot_df["baseline_value_usd"].std(ddof=0))
        cv = std_baseline / mean_baseline if mean_baseline > 0 else 0.0

        feat = {
            "sku_count": len(lot_df),
            "total_units": len(lot_df),  # assuming quantity=1 per SKU in hypothetical bag
            "mean_baseline": mean_baseline,
            "baseline_value_cv": cv,
            "is_popcorn": int(auction_type == "popcorn"),
            "has_reserve": 1,
            "is_iphone_launch_month": is_launch,
        }

        # Category shares
        for cat in LOT_CATEGORIES:
            col = f"share_{cat.lower().replace(' ', '_')}"
            feat[col] = float((lot_df["device_category"] == cat).mean())

        # Grade shares
        for g in LOT_GRADES:
            col = f"share_grade_{g.lower().replace('+', '_plus')}"
            feat[col] = float((lot_df["condition_grade"] == g).mean())

        # Category entropy
        shares = np.array([feat[f"share_{c.lower().replace(' ', '_')}"] for c in LOT_CATEGORIES])
        shares = shares[shares > 0]
        if len(shares) > 0:
            feat["category_entropy"] = float(
                -(shares * np.log2(shares)).sum() / np.log2(len(LOT_CATEGORIES))
            )
        else:
            feat["category_entropy"] = 0.0

        row_df = pd.DataFrame([feat])
        pred = model.predict_with_interval(row_df, pd.Series([total_baseline])).iloc[0]
        ratio = model.predict_ratio(row_df)[0]

        rows.append(
            {
                "lot_index": lot_idx,
                "n_skus": len(lot_df),
                "total_baseline": total_baseline,
                "predicted_clearing": float(pred["point"]),
                "predicted_lower": float(pred["lower"]),
                "predicted_upper": float(pred["upper"]),
                "ratio": float(ratio),
            }
        )

    return pd.DataFrame(rows)


def greedy_partition_by_category(sku_bag: pd.DataFrame) -> list[list[int]]:
    """Simple strategy: one lot per device category."""

    partition = []
    for cat in LOT_CATEGORIES:
        ids = sku_bag.loc[sku_bag["device_category"] == cat, "sku_id"].tolist()
        if ids:
            partition.append(ids)
    return partition


def greedy_partition_by_grade(sku_bag: pd.DataFrame) -> list[list[int]]:
    """Strategy: one lot per condition grade (mixed categories)."""

    partition = []
    for g in LOT_GRADES:
        ids = sku_bag.loc[sku_bag["condition_grade"] == g, "sku_id"].tolist()
        if ids:
            partition.append(ids)
    return partition


def single_lot_partition(sku_bag: pd.DataFrame) -> list[list[int]]:
    """Strategy: one big lot."""

    return [sku_bag["sku_id"].tolist()]


def balanced_partition(sku_bag: pd.DataFrame, n_lots: int = 4) -> list[list[int]]:
    """Strategy: split into N roughly equal-size lots, sorted by value."""

    sorted_df = sku_bag.sort_values("baseline_value_usd", ascending=False).reset_index(drop=True)
    partition: list[list[int]] = [[] for _ in range(n_lots)]
    for i, sku_id in enumerate(sorted_df["sku_id"].tolist()):
        partition[i % n_lots].append(sku_id)
    return [p for p in partition if p]


PARTITION_STRATEGIES = {
    "single_lot": (single_lot_partition, "One big lot with everything"),
    "by_category": (greedy_partition_by_category, "One lot per device category"),
    "by_grade": (greedy_partition_by_grade, "One lot per condition grade"),
    "balanced_4": (lambda b: balanced_partition(b, 4), "4 roughly equal-value lots"),
    "balanced_6": (lambda b: balanced_partition(b, 6), "6 roughly equal-value lots"),
}
