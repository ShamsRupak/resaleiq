"""Shared fixtures. Tests use the 'sample' scale for speed."""

from __future__ import annotations

import pandas as pd
import pytest

from resaleiq.config import SCALE_SAMPLE
from resaleiq.data_generation.buyers import build_buyers
from resaleiq.data_generation.devices import build_devices
from resaleiq.data_generation.lot_flow import build_lots
from resaleiq.data_generation.predictions import build_predictions
from resaleiq.data_generation.sku_flow import build_sku_listings, build_sku_offers
from resaleiq.data_generation.skus import build_skus, join_device_attrs


@pytest.fixture(scope="session")
def devices() -> pd.DataFrame:
    return build_devices()


@pytest.fixture(scope="session")
def skus(devices: pd.DataFrame) -> pd.DataFrame:
    return build_skus(devices)


@pytest.fixture(scope="session")
def skus_flat(skus: pd.DataFrame, devices: pd.DataFrame) -> pd.DataFrame:
    return join_device_attrs(skus, devices)


@pytest.fixture(scope="session")
def buyers() -> pd.DataFrame:
    return build_buyers(SCALE_SAMPLE)


@pytest.fixture(scope="session")
def sku_listings(skus_flat: pd.DataFrame) -> pd.DataFrame:
    return build_sku_listings(skus_flat, SCALE_SAMPLE)


@pytest.fixture(scope="session")
def sku_offers(
    sku_listings: pd.DataFrame,
    skus_flat: pd.DataFrame,
    buyers: pd.DataFrame,
) -> pd.DataFrame:
    return build_sku_offers(sku_listings, skus_flat, buyers, SCALE_SAMPLE)


@pytest.fixture(scope="session")
def lots_bundle(
    skus_flat: pd.DataFrame, buyers: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return build_lots(skus_flat, buyers, SCALE_SAMPLE)


@pytest.fixture(scope="session")
def predictions(
    sku_offers: pd.DataFrame,
    sku_listings: pd.DataFrame,
    skus_flat: pd.DataFrame,
    lots_bundle: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> pd.DataFrame:
    lots, lot_items, _ = lots_bundle
    return build_predictions(
        sku_offers=sku_offers,
        sku_listings=sku_listings,
        skus_flat=skus_flat,
        lots=lots,
        lot_items=lot_items,
    )
