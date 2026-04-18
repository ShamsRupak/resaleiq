"""Central configuration: seed, paths, date range, scale presets, and market constants.

All random draws in the project flow from ``MASTER_SEED`` through child seeds
derived deterministically by ``child_seed``. Anyone cloning the repo and running
``make generate`` should get byte-identical parquet output.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Literal

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------

ROOT_DIR: Path = Path(__file__).resolve().parent.parent.parent
DATA_DIR: Path = ROOT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Master seed
# ----------------------------------------------------------------------------

MASTER_SEED: int = 20260420  # Fixed seed for deterministic data generation.


def child_seed(namespace: str) -> int:
    """Deterministic child seed from ``MASTER_SEED`` and a namespace string.

    Using ``hashlib`` rather than simple string concatenation keeps child seeds
    well distributed in the uint32 space, avoiding any accidental correlation
    across generation steps.
    """

    digest = hashlib.sha256(f"{MASTER_SEED}:{namespace}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


# ----------------------------------------------------------------------------
# Date range
# ----------------------------------------------------------------------------

START_DATE: date = date(2024, 10, 1)
END_DATE: date = date(2026, 3, 31)

# iPhone launch months (Sept and Oct each year) drive the planted segment.
# These months are where Android mid-tier SKUs carry elevated MAPE in the
# baseline model; see ``market_dynamics.iphone_launch_factor``.
IPHONE_LAUNCH_MONTHS: tuple[tuple[int, int], ...] = (
    (2024, 9),
    (2024, 10),
    (2025, 9),
    (2025, 10),
)

# ----------------------------------------------------------------------------
# Scale presets
# ----------------------------------------------------------------------------

Scale = Literal["full", "sample"]


@dataclass(frozen=True)
class ScaleConfig:
    """Volume targets per table for a given scale preset."""

    name: Scale
    n_buyers: int
    n_listings: int
    avg_offers_per_listing: float
    n_lots: int
    avg_items_per_lot: float
    avg_bids_per_lot: float


SCALE_FULL = ScaleConfig(
    name="full",
    n_buyers=400,
    n_listings=20_000,
    avg_offers_per_listing=3.25,
    n_lots=8_000,
    avg_items_per_lot=5.0,
    avg_bids_per_lot=5.6,
)

SCALE_SAMPLE = ScaleConfig(
    name="sample",
    n_buyers=80,
    n_listings=1_500,
    avg_offers_per_listing=3.0,
    n_lots=600,
    avg_items_per_lot=4.5,
    avg_bids_per_lot=5.0,
)

SCALES: dict[Scale, ScaleConfig] = {"full": SCALE_FULL, "sample": SCALE_SAMPLE}


# ----------------------------------------------------------------------------
# Market-structure constants (used by market_dynamics)
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class NoiseConfig:
    """Structural parameters governing the noise profile of clearing prices.

    Calibrated so an untuned gradient-boosted baseline lands around 10 to 12
    percent overall MAPE, with one concentrated segment (Android mid-tier in
    iPhone launch months) sitting in the 23 to 26 percent band that a targeted
    feature-engineering fix reduces to roughly 16 percent.
    """

    base_sigma: float = 0.085  # gaussian noise on log-price, baseline level
    grade_variance_multiplier: dict[str, float] = field(
        default_factory=lambda: {
            "A+": 0.6,
            "A": 0.85,
            "B": 1.0,
            "C": 1.5,
            "D": 3.0,
        }
    )
    new_release_variance_multiplier: float = 2.0  # first 60 days on market
    new_release_window_days: int = 60
    # The planted segment effect: Android mid-tier clearing-price depression
    # during iPhone launch months. Magnitude tuned so baseline MAPE on this
    # slice sits around 23 to 26 percent.
    # iPhone launch months depress Android Mid clearing prices due to
    # cross-brand substitution: buyers defer mid-tier Android purchases to
    # wait for or switch to new iPhones. Industry analyses of 2020-2024
    # Counterpoint Research data put the effect in the -10 to -25% range
    # with ~8-12% idiosyncratic variance across devices and markets.
    # These values match the middle of that range (a plausible moderate
    # launch cycle like the 2024 iPhone 15 Pro refresh).
    android_mid_iphone_launch_depression_mean: float = -0.20
    android_mid_iphone_launch_depression_sigma: float = 0.08
    # Lot auction premiums by auction type (fraction over reserve).
    lot_premium_mean_popcorn: float = 0.042
    lot_premium_mean_fixed_end: float = 0.018
    lot_unsold_rate: float = 0.18


NOISE = NoiseConfig()


# ----------------------------------------------------------------------------
# Business vocabulary (kept in one place so tests can assert invariants)
# ----------------------------------------------------------------------------

CONDITION_GRADES: tuple[str, ...] = ("A+", "A", "B", "C", "D")
CARRIERS: tuple[str, ...] = ("Unlocked", "ATT", "Verizon", "TMobile", "Sprint")
BUYER_TYPES: tuple[str, ...] = ("distributor", "reseller", "refurbisher", "carrier")
BUYER_REGIONS: tuple[str, ...] = ("US-East", "US-West", "EU", "APAC", "LATAM")
BUYER_TIERS: tuple[str, ...] = ("Enterprise", "Mid-market", "SMB")
AUCTION_TYPES: tuple[str, ...] = ("popcorn", "fixed_end")
LOT_STATUSES: tuple[str, ...] = ("cleared", "unsold", "cancelled")
SKU_OFFER_OUTCOMES: tuple[str, ...] = ("accepted", "rejected", "countered_accepted", "expired")

DEVICE_CATEGORIES: tuple[str, ...] = (
    "Apple Flagship",
    "Apple Mid",
    "Android Flagship",
    "Android Mid",
    "Android Budget",
)
