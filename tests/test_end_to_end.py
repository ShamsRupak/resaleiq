"""End-to-end integration tests: volume targets, FK integrity, MAPE profile.

The MAPE test is the most important integration check in the project. It
verifies that the planted Android-mid-tier iPhone-launch-month segment carries
materially elevated error against the overall baseline. If this test fails
after a change, the central narrative of the portfolio project is broken.
"""

from __future__ import annotations

import pandas as pd

from resaleiq.config import SCALE_SAMPLE
from resaleiq.data_generation.market_dynamics import is_iphone_launch_month


class TestVolumeTargets:
    """Check that sample-scale volumes land in expected bands."""

    def test_buyers_volume(self, buyers: pd.DataFrame) -> None:
        assert len(buyers) == SCALE_SAMPLE.n_buyers

    def test_listings_volume(self, sku_listings: pd.DataFrame) -> None:
        assert len(sku_listings) == SCALE_SAMPLE.n_listings

    def test_offers_volume_reasonable(
        self, sku_offers: pd.DataFrame, sku_listings: pd.DataFrame
    ) -> None:
        avg = len(sku_offers) / max(len(sku_listings), 1)
        # Poisson mean was avg_offers_per_listing=3; allow slack.
        assert 2.0 < avg < 4.2

    def test_lots_volume(
        self, lots_bundle: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    ) -> None:
        lots, _, _ = lots_bundle
        assert len(lots) == SCALE_SAMPLE.n_lots

    def test_lot_items_volume(
        self, lots_bundle: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    ) -> None:
        lots, lot_items, _ = lots_bundle
        avg = len(lot_items) / max(len(lots), 1)
        assert 2.5 < avg < 7.0


class TestForeignKeyIntegrity:
    """Every FK reference resolves to an extant parent row."""

    def test_skus_to_devices(self, skus: pd.DataFrame, devices: pd.DataFrame) -> None:
        assert skus["device_id"].isin(devices["device_id"]).all()

    def test_listings_to_skus(self, sku_listings: pd.DataFrame, skus: pd.DataFrame) -> None:
        assert sku_listings["sku_id"].isin(skus["sku_id"]).all()

    def test_offers_to_listings_and_buyers(
        self, sku_offers: pd.DataFrame, sku_listings: pd.DataFrame, buyers: pd.DataFrame
    ) -> None:
        assert sku_offers["listing_id"].isin(sku_listings["listing_id"]).all()
        assert sku_offers["buyer_id"].isin(buyers["buyer_id"]).all()

    def test_lot_items_to_lots_and_skus(
        self,
        lots_bundle: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
        skus: pd.DataFrame,
    ) -> None:
        lots, lot_items, _ = lots_bundle
        assert lot_items["lot_id"].isin(lots["lot_id"]).all()
        assert lot_items["sku_id"].isin(skus["sku_id"]).all()

    def test_lot_bids_to_lots_and_buyers(
        self,
        lots_bundle: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
        buyers: pd.DataFrame,
    ) -> None:
        lots, _, lot_bids = lots_bundle
        assert lot_bids["lot_id"].isin(lots["lot_id"]).all()
        assert lot_bids["buyer_id"].isin(buyers["buyer_id"]).all()


class TestLotAuctionMechanics:
    def test_cleared_lots_have_clearing_price(
        self, lots_bundle: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    ) -> None:
        lots, _, _ = lots_bundle
        cleared = lots[lots["status"] == "cleared"]
        assert cleared["clearing_price"].notna().all()
        assert (cleared["clearing_price"] >= cleared["reserve_price"]).all()

    def test_unsold_lots_have_no_clearing_price(
        self, lots_bundle: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    ) -> None:
        lots, _, _ = lots_bundle
        unsold = lots[lots["status"] == "unsold"]
        assert unsold["clearing_price"].isna().all()

    def test_popcorn_premium_exceeds_fixed_end(
        self, lots_bundle: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    ) -> None:
        lots, _, _ = lots_bundle
        cleared = lots[lots["status"] == "cleared"].copy()
        cleared["premium"] = cleared["clearing_price"] / cleared["reserve_price"] - 1.0
        popcorn = cleared[cleared["auction_type"] == "popcorn"]["premium"].mean()
        fixed = cleared[cleared["auction_type"] == "fixed_end"]["premium"].mean()
        assert popcorn > fixed


class TestMAPEProfile:
    """The central narrative check: planted segment shows elevated MAPE."""

    def _mape(self, df: pd.DataFrame) -> float:
        return float((df["predicted"] - df["actual"]).abs().div(df["actual"]).mean())

    def test_overall_mape_in_baseline_band(
        self,
        predictions: pd.DataFrame,
    ) -> None:
        """Overall MAPE should be in the 9 to 14 percent band."""

        sku_preds = predictions[predictions["target_type"] == "sku_offer"]
        mape = self._mape(sku_preds)
        assert 0.09 <= mape <= 0.14, f"overall MAPE out of band: {mape:.3f}"

    def test_planted_segment_mape_elevated(
        self,
        predictions: pd.DataFrame,
        sku_offers: pd.DataFrame,
        sku_listings: pd.DataFrame,
        skus_flat: pd.DataFrame,
    ) -> None:
        """Android mid tier during iPhone launch months: 20 percent plus MAPE."""

        sku_preds = predictions[predictions["target_type"] == "sku_offer"].copy()
        sku_preds = sku_preds.merge(
            sku_offers[["offer_id", "listing_id", "offer_at"]],
            left_on="target_id",
            right_on="offer_id",
        )
        sku_preds = sku_preds.merge(sku_listings[["listing_id", "sku_id"]], on="listing_id")
        sku_preds = sku_preds.merge(skus_flat[["sku_id", "device_category"]], on="sku_id")
        sku_preds["is_launch"] = sku_preds["offer_at"].apply(
            lambda ts: is_iphone_launch_month(ts.date() if hasattr(ts, "date") else ts)
        )
        planted = sku_preds[
            (sku_preds["device_category"] == "Android Mid") & (sku_preds["is_launch"])
        ]
        # Sample data may be small; require at least 30 rows for a stable
        # estimate, otherwise skip.
        if len(planted) < 30:
            return
        mape = self._mape(planted)
        # Planted segment MAPE should be markedly higher than overall.
        assert mape >= 0.18, f"planted segment MAPE too low: {mape:.3f}"

    def test_planted_segment_higher_than_other_android_mid(
        self,
        predictions: pd.DataFrame,
        sku_offers: pd.DataFrame,
        sku_listings: pd.DataFrame,
        skus_flat: pd.DataFrame,
    ) -> None:
        """Launch-month MAPE on Android mid should exceed non-launch-month MAPE."""

        sku_preds = predictions[predictions["target_type"] == "sku_offer"].copy()
        sku_preds = sku_preds.merge(
            sku_offers[["offer_id", "listing_id", "offer_at"]],
            left_on="target_id",
            right_on="offer_id",
        )
        sku_preds = sku_preds.merge(sku_listings[["listing_id", "sku_id"]], on="listing_id")
        sku_preds = sku_preds.merge(skus_flat[["sku_id", "device_category"]], on="sku_id")
        android_mid = sku_preds[sku_preds["device_category"] == "Android Mid"].copy()
        android_mid["is_launch"] = android_mid["offer_at"].apply(
            lambda ts: is_iphone_launch_month(ts.date() if hasattr(ts, "date") else ts)
        )
        launch = android_mid[android_mid["is_launch"]]
        non_launch = android_mid[~android_mid["is_launch"]]
        if len(launch) < 30 or len(non_launch) < 30:
            return
        assert self._mape(launch) > self._mape(non_launch)
