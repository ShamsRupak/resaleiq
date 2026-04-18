"""SKU generation: expand devices into saleable variants.

A SKU is the unit of inventory the marketplace actually prices. It is defined
by device x storage x carrier x condition grade, with a baseline value in USD
that downstream flows use as the fair-value anchor.

Storage options vary by device tier (budget phones don't ship at 1TB); carrier
and grade fan out uniformly. The ``baseline_value_usd`` formula applies a
depreciation curve from MSRP based on months since release, plus multipliers
for condition grade and carrier (unlocked carries a premium).
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd

from resaleiq.config import CARRIERS, CONDITION_GRADES, child_seed

# Reference date used when computing the baseline value. The value of a phone
# evolves over the 18 months of the simulation, but the SKU-level anchor needs
# a single number for downstream flows to reason about. Downstream dynamics
# apply time-varying adjustments on top of this.
_BASELINE_REFERENCE_DATE = date(2025, 6, 30)  # Midpoint of the 18-month window.


def _storage_options_for(manufacturer: str, msrp: float) -> tuple[int, ...]:
    """Return plausible storage tiers for a device given its manufacturer and MSRP.

    Rules applied:
    - Budget phones (MSRP under 300) get 64 and 128 GB only.
    - Mid-tier (300 to 800) gets 128 and 256.
    - Flagship Apple adds 512 and (for >=1099) 1024.
    - Flagship Android (>=900) adds 512; 1024 only for S23 Ultra and up.
    """

    if msrp < 300:
        return (64, 128)
    if msrp < 800:
        return (128, 256)
    if manufacturer == "Apple":
        return (128, 256, 512, 1024) if msrp >= 1099 else (128, 256, 512)
    # Android flagship
    return (256, 512, 1024) if msrp >= 1199 else (256, 512)


def _carrier_premium(carrier: str) -> float:
    """Multiplier applied to baseline value by carrier."""

    return {
        "Unlocked": 1.05,
        "ATT": 1.00,
        "Verizon": 1.00,
        "TMobile": 0.98,
        "Sprint": 0.90,  # legacy carrier, heavier haircut
    }[carrier]


def _storage_multiplier(storage_gb: int) -> float:
    """Multiplier applied to baseline value by storage tier.

    The curve is sub-linear: doubling storage adds roughly 15 percent, matching
    observed pricing in the wholesale secondhand market rather than retail.
    """

    return 1.0 + 0.15 * math.log2(storage_gb / 128)


def _grade_multiplier(grade: str) -> float:
    """Multiplier applied to baseline value by condition grade."""

    return {
        "A+": 1.00,  # Like new, original packaging, sealed.
        "A": 0.88,
        "B": 0.72,
        "C": 0.55,
        "D": 0.25,  # Parts only.
    }[grade]


def _depreciation_factor(release_date: date, reference_date: date) -> float:
    """Return a depreciation factor from MSRP based on months since release.

    The curve is exponential with a half-life of approximately 24 months.
    Devices released after the reference date (i.e. future-dated in the
    simulation) carry no depreciation and are anchored to MSRP.
    """

    if release_date >= reference_date:
        return 1.0
    months_elapsed = (reference_date.year - release_date.year) * 12 + (
        reference_date.month - release_date.month
    )
    # 24-month half-life: factor = 0.5 ^ (months / 24).
    return float(0.5 ** (months_elapsed / 24))


def _baseline_value(
    msrp: float,
    storage_gb: int,
    carrier: str,
    grade: str,
    release_date: pd.Timestamp,
) -> float:
    """Compute a deterministic fair-value anchor for a SKU."""

    release_d = release_date.date() if isinstance(release_date, pd.Timestamp) else release_date
    depreciation = _depreciation_factor(release_d, _BASELINE_REFERENCE_DATE)
    value = (
        msrp
        * depreciation
        * _storage_multiplier(storage_gb)
        * _carrier_premium(carrier)
        * _grade_multiplier(grade)
    )
    # Floor at 15 USD: no inventory clears below salvage.
    return round(max(value, 15.0), 2)


def build_skus(devices: pd.DataFrame) -> pd.DataFrame:
    """Expand ``devices`` into the full SKU catalog.

    Rather than a pure cartesian product, storage options depend on device tier
    (budget phones don't come in 1TB). Carriers and grades fan out uniformly.
    """

    _ = child_seed("skus")  # Reserved for future stochastic SKU attributes.
    rows: list[dict[str, object]] = []
    sku_id = 1
    # Iterate in a fixed order to keep sku_id assignment deterministic.
    for device in devices.sort_values("device_id").itertuples(index=False):
        storage_tiers = _storage_options_for(device.manufacturer, device.msrp_new)
        for storage_gb in storage_tiers:
            for carrier in CARRIERS:
                for grade in CONDITION_GRADES:
                    rows.append(
                        {
                            "sku_id": sku_id,
                            "device_id": device.device_id,
                            "storage_gb": storage_gb,
                            "carrier": carrier,
                            "condition_grade": grade,
                            "baseline_value_usd": _baseline_value(
                                device.msrp_new,
                                storage_gb,
                                carrier,
                                grade,
                                device.release_date,
                            ),
                        }
                    )
                    sku_id += 1
    df = pd.DataFrame(rows)
    assert (df["baseline_value_usd"] >= 15.0).all()
    return df


def join_device_attrs(skus: pd.DataFrame, devices: pd.DataFrame) -> pd.DataFrame:
    """Join ``devices`` onto ``skus`` for downstream flows that need both.

    Kept as a helper so flow modules can work with a single flattened frame
    without re-doing the join in every function.
    """

    joined = skus.merge(devices, on="device_id", how="left", validate="m:1")
    assert len(joined) == len(skus)
    return joined


# Expose a symbol for callers who want the reference date without importing
# the private constant directly.
def baseline_reference_date() -> date:
    """Return the reference date used to anchor SKU baseline values."""

    return _BASELINE_REFERENCE_DATE


__all__ = ["baseline_reference_date", "build_skus", "join_device_attrs"]


# Trivial test hook to keep numpy import non-dead.
_ = np.asarray([])
