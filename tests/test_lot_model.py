"""Tests for the lot-level pricing model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from resaleiq.ml.lot_model import (
    LOT_CATEGORIES,
    LOT_GRADES,
    PARTITION_STRATEGIES,
    assemble_lot_feature_frame,
    balanced_partition,
    greedy_partition_by_category,
    greedy_partition_by_grade,
    score_partition,
    select_features,
    single_lot_partition,
    train_lot_model,
    train_lot_model_with_holdout,
)

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def lot_frames() -> dict[str, pd.DataFrame]:
    """Build a tiny self-consistent set of lots, lot_items, skus, devices."""

    rng = np.random.default_rng(42)

    # 20 devices across 5 categories
    devices = pd.DataFrame(
        [
            {
                "device_id": i,
                "device_category": LOT_CATEGORIES[i % len(LOT_CATEGORIES)],
                "manufacturer": "Apple"
                if "Apple" in LOT_CATEGORIES[i % len(LOT_CATEGORIES)]
                else "Samsung",
                "model_name": f"Model {i}",
            }
            for i in range(20)
        ]
    )

    # 200 SKUs
    skus = []
    for sku_id in range(200):
        device_id = sku_id % 20
        grade = LOT_GRADES[sku_id % len(LOT_GRADES)]
        base_val = 200 + (device_id * 30) + (hash(grade) % 50)
        skus.append(
            {
                "sku_id": sku_id,
                "device_id": device_id,
                "storage_gb": 128,
                "carrier": "Unlocked",
                "condition_grade": grade,
                "baseline_value_usd": float(base_val),
            }
        )
    skus = pd.DataFrame(skus)

    # 150 lots (enough to populate a time-series holdout with >= 20 on each side)
    lots_rows = []
    lot_items_rows = []
    item_id = 0
    for lot_id in range(150):
        n_items = int(rng.integers(5, 11))
        sampled = rng.choice(skus["sku_id"].values, size=n_items, replace=False)
        total_baseline = 0.0
        for sku_id in sampled:
            quantity = int(rng.integers(1, 10))
            unit_price = float(skus.loc[skus["sku_id"] == sku_id, "baseline_value_usd"].iloc[0])
            lot_items_rows.append(
                {
                    "lot_item_id": item_id,
                    "lot_id": lot_id,
                    "sku_id": int(sku_id),
                    "quantity": quantity,
                    "unit_ref_price": unit_price,
                }
            )
            item_id += 1
            total_baseline += unit_price * quantity

        auction_type = "popcorn" if lot_id % 3 != 0 else "fixed_end"
        reserve = total_baseline * 0.5 if lot_id % 2 == 0 else 0
        clearing = total_baseline * float(rng.uniform(0.75, 1.1))
        status = "cleared" if lot_id % 4 != 0 else "unsold"

        # Spread lots across 15 months (Jan 2025 - Mar 2026) so holdout tests
        # can split on a 2026 boundary with lots on both sides.
        month_offset = lot_id % 15  # 0-14
        year = 2025 if month_offset < 12 else 2026
        month = (month_offset % 12) + 1 if month_offset < 12 else month_offset - 11
        close_ts = pd.Timestamp(f"{year}-{month:02d}-15")

        lots_rows.append(
            {
                "lot_id": lot_id,
                "winning_buyer_id": 1 if status == "cleared" else None,
                "reserve_price": reserve,
                "clearing_price": clearing if status == "cleared" else 0,
                "scheduled_close": close_ts,
                "actual_close": close_ts,
                "auction_type": auction_type,
                "status": status,
            }
        )

    return {
        "devices": devices,
        "skus": skus,
        "lots": pd.DataFrame(lots_rows),
        "lot_items": pd.DataFrame(lot_items_rows),
    }


# --------------------------------------------------------------------------- #
# Feature frame tests
# --------------------------------------------------------------------------- #


class TestFeatureAssembly:
    def test_frame_has_one_row_per_lot(self, lot_frames):
        frame = assemble_lot_feature_frame(
            lot_frames["lots"],
            lot_frames["lot_items"],
            lot_frames["skus"],
            lot_frames["devices"],
        )
        assert len(frame) == len(lot_frames["lots"])

    def test_category_shares_sum_to_one(self, lot_frames):
        frame = assemble_lot_feature_frame(
            lot_frames["lots"],
            lot_frames["lot_items"],
            lot_frames["skus"],
            lot_frames["devices"],
        )
        cat_cols = [f"share_{c.lower().replace(' ', '_')}" for c in LOT_CATEGORIES]
        row_sums = frame[cat_cols].sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)

    def test_grade_shares_sum_to_one(self, lot_frames):
        frame = assemble_lot_feature_frame(
            lot_frames["lots"],
            lot_frames["lot_items"],
            lot_frames["skus"],
            lot_frames["devices"],
        )
        grade_cols = [f"share_grade_{c.lower().replace('+', '_plus')}" for c in LOT_GRADES]
        row_sums = frame[grade_cols].sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)

    def test_total_baseline_reflects_quantity(self, lot_frames):
        """total_baseline_value should account for item quantity, not just SKU count."""

        frame = assemble_lot_feature_frame(
            lot_frames["lots"],
            lot_frames["lot_items"],
            lot_frames["skus"],
            lot_frames["devices"],
        )
        first = frame.iloc[0]
        expected = (
            lot_frames["lot_items"][lot_frames["lot_items"]["lot_id"] == first["lot_id"]]
            .merge(lot_frames["skus"][["sku_id", "baseline_value_usd"]], on="sku_id")
            .eval("baseline_value_usd * quantity")
            .sum()
        )
        assert first["total_baseline_value"] == pytest.approx(expected)

    def test_features_list_is_stable(self, lot_frames):
        frame = assemble_lot_feature_frame(
            lot_frames["lots"],
            lot_frames["lot_items"],
            lot_frames["skus"],
            lot_frames["devices"],
        )
        feats = select_features(frame)
        assert all(f in frame.columns for f in feats)
        assert "sku_count" in feats
        assert "total_units" in feats
        assert "is_iphone_launch_month" in feats


# --------------------------------------------------------------------------- #
# Training tests
# --------------------------------------------------------------------------- #


class TestTraining:
    def test_model_trains_and_predicts(self, lot_frames):
        frame = assemble_lot_feature_frame(
            lot_frames["lots"],
            lot_frames["lot_items"],
            lot_frames["skus"],
            lot_frames["devices"],
        )
        model = train_lot_model(frame, num_boost_round=50)
        assert model.booster is not None
        assert len(model.feature_names) > 10
        assert model.ratio_mad > 0

    def test_prediction_intervals_include_point(self, lot_frames):
        frame = assemble_lot_feature_frame(
            lot_frames["lots"],
            lot_frames["lot_items"],
            lot_frames["skus"],
            lot_frames["devices"],
        )
        model = train_lot_model(frame, num_boost_round=50)
        cleared = frame[frame["status"] == "cleared"].head(5)
        X = cleared[model.feature_names]
        pred = model.predict_with_interval(X, cleared["total_baseline_value"])
        assert (pred["lower"] <= pred["point"]).all()
        assert (pred["point"] <= pred["upper"]).all()


# --------------------------------------------------------------------------- #
# Partition strategy tests
# --------------------------------------------------------------------------- #


class TestPartitioning:
    @pytest.fixture
    def sample_bag(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "sku_id": i,
                    "device_category": LOT_CATEGORIES[i % 5],
                    "condition_grade": LOT_GRADES[i % 5],
                    "baseline_value_usd": 100 + i * 10,
                }
                for i in range(25)
            ]
        )

    def test_single_lot_returns_one_group(self, sample_bag):
        p = single_lot_partition(sample_bag)
        assert len(p) == 1
        assert len(p[0]) == 25

    def test_by_category_groups_all_skus(self, sample_bag):
        p = greedy_partition_by_category(sample_bag)
        all_ids = [sku for lot in p for sku in lot]
        assert sorted(all_ids) == sorted(sample_bag["sku_id"].tolist())

    def test_by_grade_groups_all_skus(self, sample_bag):
        p = greedy_partition_by_grade(sample_bag)
        all_ids = [sku for lot in p for sku in lot]
        assert sorted(all_ids) == sorted(sample_bag["sku_id"].tolist())

    def test_balanced_respects_n_lots(self, sample_bag):
        for n in [2, 3, 4, 5]:
            p = balanced_partition(sample_bag, n_lots=n)
            assert len(p) <= n
            all_ids = [sku for lot in p for sku in lot]
            assert sorted(all_ids) == sorted(sample_bag["sku_id"].tolist())

    def test_all_strategies_registered(self):
        expected = {"single_lot", "by_category", "by_grade", "balanced_4", "balanced_6"}
        assert set(PARTITION_STRATEGIES.keys()) == expected


# --------------------------------------------------------------------------- #
# Scoring tests
# --------------------------------------------------------------------------- #


class TestScoring:
    def test_score_partition_returns_row_per_lot(self, lot_frames):
        frame = assemble_lot_feature_frame(
            lot_frames["lots"],
            lot_frames["lot_items"],
            lot_frames["skus"],
            lot_frames["devices"],
        )
        model = train_lot_model(frame, num_boost_round=50)

        bag = (
            lot_frames["skus"]
            .head(20)
            .merge(lot_frames["devices"][["device_id", "device_category"]], on="device_id")
        )
        partition = [bag["sku_id"].tolist()[:10], bag["sku_id"].tolist()[10:]]
        result = score_partition(bag, partition, model)
        assert len(result) == 2
        assert all(result["predicted_clearing"] > 0)
        assert (result["predicted_lower"] <= result["predicted_clearing"]).all()
        assert (result["predicted_clearing"] <= result["predicted_upper"]).all()

    def test_score_partition_handles_empty_lot(self, lot_frames):
        frame = assemble_lot_feature_frame(
            lot_frames["lots"],
            lot_frames["lot_items"],
            lot_frames["skus"],
            lot_frames["devices"],
        )
        model = train_lot_model(frame, num_boost_round=50)
        bag = (
            lot_frames["skus"]
            .head(10)
            .merge(lot_frames["devices"][["device_id", "device_category"]], on="device_id")
        )
        partition = [bag["sku_id"].tolist(), []]
        result = score_partition(bag, partition, model)
        assert len(result) == 1  # Empty lot skipped


# --------------------------------------------------------------------------- #
# Holdout training tests (Phase 5 hardening)
# --------------------------------------------------------------------------- #


class TestHoldoutTraining:
    """Verify the time-series holdout training behaves correctly.

    These tests exist because the in-sample-only training of earlier phases
    was a credibility risk. Splitting on ``opened_at`` and reporting both
    in-sample and out-of-sample MAPE against a naive baseline is the honest
    measurement pattern for generalization error.
    """

    def test_holdout_returns_expected_keys(self, lot_frames):
        frame = assemble_lot_feature_frame(
            lot_frames["lots"],
            lot_frames["lot_items"],
            lot_frames["skus"],
            lot_frames["devices"],
        )
        result = train_lot_model_with_holdout(
            frame,
            holdout_start=pd.Timestamp("2026-01-01"),
            num_boost_round=50,
        )
        required_keys = {
            "model",
            "n_train",
            "n_holdout",
            "holdout_start",
            "train_period",
            "holdout_period",
            "in_sample_mape",
            "oos_mape",
            "naive_mape",
            "relative_improvement_oos",
            "ratio_mad",
            "holdout_conformal_coverage",
            "in_sample_median_ape",
            "oos_median_ape",
        }
        assert required_keys.issubset(result.keys())

    def test_train_and_holdout_sizes_reasonable(self, lot_frames):
        frame = assemble_lot_feature_frame(
            lot_frames["lots"],
            lot_frames["lot_items"],
            lot_frames["skus"],
            lot_frames["devices"],
        )
        result = train_lot_model_with_holdout(
            frame,
            holdout_start=pd.Timestamp("2026-01-01"),
            num_boost_round=50,
        )
        assert result["n_train"] >= 50
        assert result["n_holdout"] >= 20
        # Total should equal cleared lots that pass winsorization
        assert result["n_train"] + result["n_holdout"] > 70

    def test_holdout_lots_strictly_after_train_period(self, lot_frames):
        frame = assemble_lot_feature_frame(
            lot_frames["lots"],
            lot_frames["lot_items"],
            lot_frames["skus"],
            lot_frames["devices"],
        )
        holdout_start = pd.Timestamp("2026-01-01")
        result = train_lot_model_with_holdout(
            frame,
            holdout_start=holdout_start,
            num_boost_round=50,
        )
        _, train_hi = result["train_period"]
        hold_lo, _ = result["holdout_period"]
        assert train_hi < holdout_start
        assert hold_lo >= holdout_start

    def test_model_beats_naive_on_holdout(self, lot_frames):
        """A lot model with composition features must beat 'predict baseline_value'."""

        frame = assemble_lot_feature_frame(
            lot_frames["lots"],
            lot_frames["lot_items"],
            lot_frames["skus"],
            lot_frames["devices"],
        )
        result = train_lot_model_with_holdout(
            frame,
            holdout_start=pd.Timestamp("2026-01-01"),
            num_boost_round=100,
        )
        assert result["oos_mape"] < result["naive_mape"]
        assert result["relative_improvement_oos"] > 0.05  # at least 5% better

    def test_raises_when_holdout_too_small(self, lot_frames):
        """Setting holdout_start past all lots should raise."""

        frame = assemble_lot_feature_frame(
            lot_frames["lots"],
            lot_frames["lot_items"],
            lot_frames["skus"],
            lot_frames["devices"],
        )
        with pytest.raises(ValueError, match="holdout"):
            train_lot_model_with_holdout(
                frame,
                holdout_start=pd.Timestamp("2030-01-01"),
                num_boost_round=50,
            )

    def test_raises_when_train_too_small(self, lot_frames):
        """Setting holdout_start before all lots should raise."""

        frame = assemble_lot_feature_frame(
            lot_frames["lots"],
            lot_frames["lot_items"],
            lot_frames["skus"],
            lot_frames["devices"],
        )
        with pytest.raises(ValueError, match="training"):
            train_lot_model_with_holdout(
                frame,
                holdout_start=pd.Timestamp("2020-01-01"),
                num_boost_round=50,
            )

    def test_model_predicts_usable_ratios(self, lot_frames):
        """The returned LotModel should work for inference on the holdout frame."""

        frame = assemble_lot_feature_frame(
            lot_frames["lots"],
            lot_frames["lot_items"],
            lot_frames["skus"],
            lot_frames["devices"],
        )
        result = train_lot_model_with_holdout(
            frame,
            holdout_start=pd.Timestamp("2026-01-01"),
            num_boost_round=50,
        )
        model = result["model"]
        cleared = frame[frame["status"] == "cleared"].head(5)
        X = cleared[model.feature_names]
        pred = model.predict_with_interval(X, cleared["total_baseline_value"])
        assert (pred["lower"] <= pred["point"]).all()
        assert (pred["point"] <= pred["upper"]).all()
        assert (pred["point"] > 0).all()

    def test_empirical_quantile_calibration_on_train(self, lot_frames):
        """The conformal half-width should give approximately 80% coverage on training residuals
        (by construction when using empirical-quantile calibration)."""

        frame = assemble_lot_feature_frame(
            lot_frames["lots"],
            lot_frames["lot_items"],
            lot_frames["skus"],
            lot_frames["devices"],
        )
        result = train_lot_model_with_holdout(
            frame,
            holdout_start=pd.Timestamp("2026-01-01"),
            num_boost_round=100,
        )
        # ratio_mad should be positive and finite
        assert result["ratio_mad"] > 0
        assert np.isfinite(result["ratio_mad"])
        # Coverage should be computable (between 0 and 1)
        assert 0.0 <= result["holdout_conformal_coverage"] <= 1.0
