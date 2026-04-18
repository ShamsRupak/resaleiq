"""Smoke tests for the reference layer (devices, skus, buyers)."""

from __future__ import annotations

import pandas as pd

from resaleiq.config import (
    BUYER_REGIONS,
    BUYER_TIERS,
    BUYER_TYPES,
    CARRIERS,
    CONDITION_GRADES,
    DEVICE_CATEGORIES,
)


class TestDevices:
    def test_catalog_non_empty(self, devices: pd.DataFrame) -> None:
        assert len(devices) >= 60

    def test_unique_device_ids(self, devices: pd.DataFrame) -> None:
        assert devices["device_id"].is_unique

    def test_all_categories_represented(self, devices: pd.DataFrame) -> None:
        assert set(devices["device_category"].unique()).issubset(set(DEVICE_CATEGORIES))
        # Every category should appear at least once.
        for cat in DEVICE_CATEGORIES:
            assert (devices["device_category"] == cat).any(), f"missing: {cat}"

    def test_msrp_positive(self, devices: pd.DataFrame) -> None:
        assert (devices["msrp_new"] > 0).all()


class TestSkus:
    def test_unique_sku_ids(self, skus: pd.DataFrame) -> None:
        assert skus["sku_id"].is_unique

    def test_vocabulary_adherence(self, skus: pd.DataFrame) -> None:
        assert set(skus["carrier"].unique()).issubset(set(CARRIERS))
        assert set(skus["condition_grade"].unique()).issubset(set(CONDITION_GRADES))

    def test_baseline_floor(self, skus: pd.DataFrame) -> None:
        # Config floor is 15 USD.
        assert (skus["baseline_value_usd"] >= 15.0).all()

    def test_grade_ordering(self, skus: pd.DataFrame) -> None:
        # For any given device_id, storage, and carrier, Grade A+ should
        # carry a higher baseline value than Grade D. This is a core invariant
        # of the depreciation curve and guards against accidental re-weighting.
        pivot = (
            skus.groupby(["device_id", "storage_gb", "carrier", "condition_grade"])[
                "baseline_value_usd"
            ]
            .first()
            .unstack("condition_grade")
        )
        # Keep only rows where both A+ and D are present (all should be).
        common = pivot.dropna(subset=["A+", "D"])
        assert (common["A+"] > common["D"]).all()


class TestBuyers:
    def test_unique_buyer_ids(self, buyers: pd.DataFrame) -> None:
        assert buyers["buyer_id"].is_unique

    def test_vocabulary(self, buyers: pd.DataFrame) -> None:
        assert set(buyers["buyer_type"].unique()).issubset(set(BUYER_TYPES))
        assert set(buyers["region"].unique()).issubset(set(BUYER_REGIONS))
        assert set(buyers["tier"].unique()).issubset(set(BUYER_TIERS))

    def test_all_tiers_present(self, buyers: pd.DataFrame) -> None:
        for tier in BUYER_TIERS:
            assert (buyers["tier"] == tier).any(), f"tier missing: {tier}"
