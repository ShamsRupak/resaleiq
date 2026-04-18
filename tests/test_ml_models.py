"""Tests for the Phase 3 ML package.

Smoke tests for feature engineering, training, conformal interval coverage,
and segment MAPE computation. Uses the already-generated parquet data so
the generator doesn't run in every test.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from resaleiq.config import DATA_DIR
from resaleiq.ml.evaluate import (
    compute_segment_mape,
    coverage_rate,
    mape,
)
from resaleiq.ml.features import (
    assemble_cleared_offers,
    build_cross_brand_features,
    build_feature_matrix,
)
from resaleiq.ml.train import (
    compute_conformal_intervals,
    feature_importance,
    train_xgb_model,
)


def _need_generated_data() -> None:
    """Skip if required parquet files are missing."""

    for fname in ("sku_offers.parquet", "sku_listings.parquet", "skus.parquet", "devices.parquet"):
        if not (DATA_DIR / fname).exists():
            pytest.skip(f"{fname} not generated; run ``make generate`` first")


@pytest.fixture(scope="module")
def cleared_offers() -> pd.DataFrame:
    _need_generated_data()
    sku_offers = pd.read_parquet(DATA_DIR / "sku_offers.parquet")
    sku_listings = pd.read_parquet(DATA_DIR / "sku_listings.parquet")
    skus = pd.read_parquet(DATA_DIR / "skus.parquet")
    devices = pd.read_parquet(DATA_DIR / "devices.parquet")
    df = assemble_cleared_offers(sku_offers, sku_listings, skus, devices)
    return build_cross_brand_features(df)


class TestFeatureEngineering:
    def test_assemble_cleared_offers_has_all_joins(self, cleared_offers: pd.DataFrame) -> None:
        required = {
            "offer_id",
            "listing_id",
            "sku_id",
            "device_id",
            "clearing_price",
            "offer_at",
            "device_category",
            "condition_grade",
            "storage_gb",
            "carrier",
            "release_date",
            "msrp_new",
            "baseline_value_usd",
        }
        assert required.issubset(cleared_offers.columns)
        assert len(cleared_offers) > 0
        assert cleared_offers["clearing_price"].gt(0).all()

    def test_cross_brand_features_bounded(self, cleared_offers: pd.DataFrame) -> None:
        # days_since_latest_iphone_launch: 0 to ~365 (launches annual).
        assert cleared_offers["days_since_latest_iphone_launch"].between(0, 400).all()
        # iphone_price_change_30d: bounded pct change, should lie in [-0.5, 0.5].
        assert cleared_offers["iphone_price_change_30d"].between(-0.5, 0.5).all()
        # cross_category_price_ratio: positive by construction.
        assert cleared_offers["cross_category_price_ratio"].gt(0).all()

    def test_feature_matrix_baseline_has_no_launch_features(
        self, cleared_offers: pd.DataFrame
    ) -> None:
        X, y = build_feature_matrix(cleared_offers, feature_set="baseline")
        assert "is_iphone_launch_month" not in X.columns
        assert "days_since_latest_iphone_launch" not in X.columns
        assert "iphone_price_change_30d" not in X.columns
        assert "cross_category_price_ratio" not in X.columns
        assert "launch_x_android_mid" not in X.columns
        assert len(X) == len(y)

    def test_targeted_has_all_launch_features(self, cleared_offers: pd.DataFrame) -> None:
        X, _ = build_feature_matrix(cleared_offers, feature_set="targeted")
        for name in (
            "is_iphone_launch_month",
            "days_since_latest_iphone_launch",
            "iphone_price_change_30d",
            "cross_category_price_ratio",
            "launch_x_android_mid",
        ):
            assert name in X.columns, name

    def test_feature_count_grows_with_ladder(self, cleared_offers: pd.DataFrame) -> None:
        counts = {}
        for level in (
            "baseline",
            "plus_launch",
            "plus_launch_days",
            "plus_launch_days_price",
            "targeted",
        ):
            X, _ = build_feature_matrix(cleared_offers, feature_set=level)
            counts[level] = X.shape[1]
        values = list(counts.values())
        # Monotone non-decreasing.
        for a, b in pairwise(values):
            assert a < b


class TestMetrics:
    def test_mape_zero_for_perfect_predictions(self) -> None:
        y = pd.Series([10.0, 20.0, 30.0])
        assert mape(y, y) == pytest.approx(0.0)

    def test_mape_matches_definition(self) -> None:
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 180.0, 330.0])
        # |10/100| + |20/200| + |30/300| = 0.1 + 0.1 + 0.1
        assert mape(y_true, y_pred) == pytest.approx(0.1)

    def test_coverage_rate_bounded(self) -> None:
        y = np.array([1.0, 2.0, 3.0, 4.0])
        lo = np.array([0.0, 0.0, 0.0, 5.0])
        hi = np.array([2.0, 3.0, 2.5, 6.0])
        # 1 in [0,2]=yes, 2 in [0,3]=yes, 3 in [0,2.5]=no, 4 in [5,6]=no
        assert coverage_rate(y, lo, hi) == pytest.approx(0.5)

    def test_compute_segment_mape_has_overall_row(self, cleared_offers: pd.DataFrame) -> None:
        df = cleared_offers.head(500).copy()
        df["pred"] = df["clearing_price"] * 1.1
        result = compute_segment_mape(df, pred_col="pred")
        assert "ALL" in result.rows["device_category"].values


class TestTraining:
    def test_xgb_trains_and_predicts(self, cleared_offers: pd.DataFrame) -> None:
        df = cleared_offers.sample(n=min(2000, len(cleared_offers)), random_state=0)
        df_train = df.iloc[:1500]
        df_val = df.iloc[1500:]
        X_train, y_train = build_feature_matrix(df_train, feature_set="baseline")
        X_val, y_val = build_feature_matrix(df_val, feature_set="baseline")
        model = train_xgb_model(
            X_train,
            y_train,
            X_val,
            y_val,
            num_boost_round=50,
            early_stopping_rounds=10,
            target_type="ratio",
            baseline_train=df_train["baseline_value_usd"],
            baseline_val=df_val["baseline_value_usd"],
        )
        preds = model.predict(X_val, baseline_value=df_val["baseline_value_usd"])
        assert preds.shape == (len(X_val),)
        assert (preds > 0).all()
        # Trained model should beat a constant-mean prediction.
        constant = np.full_like(y_val.values, y_val.values.mean(), dtype=float)
        assert mape(y_val, preds) < mape(y_val, constant)

    def test_feature_importance_sums_to_positive(self, cleared_offers: pd.DataFrame) -> None:
        df = cleared_offers.sample(n=min(2000, len(cleared_offers)), random_state=0)
        df_train = df.iloc[:1500]
        df_val = df.iloc[1500:]
        X_train, y_train = build_feature_matrix(df_train, feature_set="baseline")
        X_val, y_val = build_feature_matrix(df_val, feature_set="baseline")
        model = train_xgb_model(
            X_train,
            y_train,
            X_val,
            y_val,
            num_boost_round=50,
            early_stopping_rounds=10,
            target_type="ratio",
            baseline_train=df_train["baseline_value_usd"],
            baseline_val=df_val["baseline_value_usd"],
        )
        fi = feature_importance(model)
        assert list(fi.columns) == ["feature", "importance"]
        assert fi["importance"].sum() > 0


class TestConformal:
    def test_conformal_covers_at_target_level(self, cleared_offers: pd.DataFrame) -> None:
        df = cleared_offers.sample(n=min(3000, len(cleared_offers)), random_state=0)
        df_train = df.iloc[:1800]
        df_cal = df.iloc[1800:2400]
        df_test = df.iloc[2400:]
        X_train, y_train = build_feature_matrix(df_train, feature_set="baseline")
        X_val, y_val = build_feature_matrix(df_cal, feature_set="baseline")
        X_test, y_test = build_feature_matrix(df_test, feature_set="baseline")
        model = train_xgb_model(
            X_train,
            y_train,
            X_val,
            y_val,
            num_boost_round=100,
            early_stopping_rounds=20,
            target_type="ratio",
            baseline_train=df_train["baseline_value_usd"],
            baseline_val=df_cal["baseline_value_usd"],
        )
        conformal = compute_conformal_intervals(
            model,
            X_cal=X_val,
            y_cal=y_val,
            X_pred=X_test,
            alpha=0.2,
            baseline_cal=df_cal["baseline_value_usd"],
            baseline_pred=df_test["baseline_value_usd"],
        )
        coverage = coverage_rate(y_test.values, conformal.lower, conformal.upper)
        # 80% target with small calibration fold; tolerate +/- 8 percentage points.
        assert 0.72 <= coverage <= 0.92, coverage
        assert conformal.half_width > 0

    def test_conformal_lower_bound_non_negative(self, cleared_offers: pd.DataFrame) -> None:
        df = cleared_offers.sample(n=min(1500, len(cleared_offers)), random_state=0)
        df_train = df.iloc[:1000]
        df_cal = df.iloc[1000:1250]
        df_test = df.iloc[1250:]
        X_train, y_train = build_feature_matrix(df_train, feature_set="baseline")
        X_cal, y_cal = build_feature_matrix(df_cal, feature_set="baseline")
        X_test, _ = build_feature_matrix(df_test, feature_set="baseline")
        model = train_xgb_model(
            X_train,
            y_train,
            X_cal,
            y_cal,
            num_boost_round=50,
            early_stopping_rounds=10,
            target_type="ratio",
            baseline_train=df_train["baseline_value_usd"],
            baseline_val=df_cal["baseline_value_usd"],
        )
        conformal = compute_conformal_intervals(
            model,
            X_cal=X_cal,
            y_cal=y_cal,
            X_pred=X_test,
            alpha=0.2,
            baseline_cal=df_cal["baseline_value_usd"],
            baseline_pred=df_test["baseline_value_usd"],
        )
        assert (conformal.lower >= 0).all()
        assert (conformal.upper >= conformal.lower).all()
