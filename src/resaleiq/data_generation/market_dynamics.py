"""Market dynamics: seasonality, noise, and the planted cross-brand segment.

This is the most important module in the data-generation pipeline. It controls:

1. The time-varying fair value of a SKU on a given date (base depreciation plus
   seasonal adjustments).
2. The planted structural error: Android mid-tier clearing-price depression
   during iPhone launch months (September and October of each year). A baseline
   model that ignores cross-brand temporal features will carry elevated MAPE on
   this slice. A model that adds three features (days since latest iPhone
   launch, recent iPhone price change, cross-category price ratio) will
   recover most of that error.
3. The high-variance tails: Grade D inventory and newly released devices carry
   wider clearing-price dispersion than the rest of the catalog.

All of the above are disclosed openly in the project README. The intent is
transparency about the generative process, not an obscured pattern waiting
to be found. The point is to provide a realistic segment-diagnostics
workflow on reproducible data.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import NamedTuple

import numpy as np
import pandas as pd

from resaleiq.config import IPHONE_LAUNCH_MONTHS, NOISE


class FairValueContext(NamedTuple):
    """Inputs the fair-value function needs per row.

    Kept as a NamedTuple (rather than a dataclass) so it can be constructed
    inside tight numpy-vectorized paths without per-call allocation overhead.
    """

    baseline_value_usd: float
    manufacturer: str
    device_category: str
    release_date: date
    grade: str
    transaction_date: date


def is_iphone_launch_month(d: date) -> bool:
    """Whether ``d`` falls in one of the configured iPhone launch months."""

    return (d.year, d.month) in IPHONE_LAUNCH_MONTHS


def days_since_latest_iphone_launch(d: date) -> int:
    """Number of days since the most recent iPhone launch month began.

    Returns a large positive integer if no launch has occurred before ``d``
    (i.e. the value is meaningful on and after Sept 1, 2024).
    """

    candidates = [
        date(year, month, 1) for (year, month) in IPHONE_LAUNCH_MONTHS if date(year, month, 1) <= d
    ]
    if not candidates:
        return 10_000
    return (d - max(candidates)).days


def seasonality_factor(d: date, device_category: str) -> float:
    """General seasonality adjustment to fair value.

    - Post-holiday trade-in surge (January and February) softens prices about
      2 percent across the board.
    - Summer doldrums (June through August) soften prices about 1 percent.
    - December holiday demand lifts prices 1.5 percent.
    These are independent of the iPhone-launch planted segment effect.
    """

    month = d.month
    if month in (1, 2):
        return -0.02
    if month in (6, 7, 8):
        return -0.01
    if month == 12:
        return 0.015
    # iPhone launch months carry their own effect; handled separately so the
    # caller can reason about general seasonality and the planted segment
    # independently.
    _ = device_category
    return 0.0


def iphone_launch_effect(
    d: date,
    device_category: str,
    rng: np.random.Generator,
) -> float:
    """Return the planted cross-brand substitution shock for a transaction.

    Applied only to Android mid-tier SKUs cleared during an iPhone launch
    month. The shock is sampled from a gaussian with negative mean so that on
    average the clearing price depresses by several percent, with enough
    dispersion to guarantee the baseline model carries 23 to 26 percent MAPE
    on this slice. A targeted feature-engineering fix using the three signals
    documented in the README recovers most of this variance.
    """

    if device_category != "Android Mid" or not is_iphone_launch_month(d):
        return 0.0
    return float(
        rng.normal(
            NOISE.android_mid_iphone_launch_depression_mean,
            NOISE.android_mid_iphone_launch_depression_sigma,
        )
    )


def grade_variance(grade: str) -> float:
    """Multiplier on the base gaussian noise sigma by condition grade."""

    return NOISE.grade_variance_multiplier[grade]


def is_newly_released(release_date: date, transaction_date: date) -> bool:
    """Whether the device was released within the new-release variance window."""

    if release_date > transaction_date:
        return False
    age_days = (transaction_date - release_date).days
    return age_days < NOISE.new_release_window_days


def noise_sigma(context: FairValueContext) -> float:
    """Compute the effective noise sigma for a single transaction.

    Combines the base sigma with the condition-grade variance multiplier and,
    for newly released devices, an additional multiplier. The result is the
    standard deviation of the multiplicative noise term applied to the fair
    value.
    """

    sigma = NOISE.base_sigma * grade_variance(context.grade)
    if is_newly_released(context.release_date, context.transaction_date):
        sigma *= NOISE.new_release_variance_multiplier
    return sigma


def clearing_price_from_fair_value(
    context: FairValueContext,
    rng: np.random.Generator,
    aggression_multiplier: float = 1.0,
) -> float:
    """Sample a realistic clearing price for a transaction.

    Components, applied multiplicatively to ``baseline_value_usd``:

    1. General seasonality factor (holiday, summer, post-holiday).
    2. Planted iPhone-launch cross-brand effect (Android mid only, Sept and
       Oct of launch years).
    3. Gaussian noise with sigma scaled by grade and release recency.
    4. Buyer aggression multiplier (Enterprise buyers bid higher).

    The result is floored at 1 USD to keep downstream log-space math safe.
    """

    season = seasonality_factor(context.transaction_date, context.device_category)
    iphone_shock = iphone_launch_effect(context.transaction_date, context.device_category, rng)
    sigma = noise_sigma(context)
    eps = float(rng.normal(0.0, sigma))
    # All factors are additive on the fractional scale, then applied to the
    # anchor. Using (1 + sum) rather than product(1 + x) keeps the shocks from
    # compounding in ways that make MAPE unstable to analyze.
    factor = 1.0 + season + iphone_shock + eps
    factor *= aggression_multiplier
    price = context.baseline_value_usd * max(factor, 0.05)
    return round(max(price, 1.0), 2)


def month_index(d: date) -> int:
    """Return a monotonic month index, for use as a simple seasonality feature.

    Months since January 2024 inclusive; this keeps the values compact and the
    first simulation month (October 2024) at index 9.
    """

    return (d.year - 2024) * 12 + (d.month - 1)


def draw_random_dates(
    rng: np.random.Generator,
    start: date,
    end: date,
    n: int,
    iphone_launch_weight: float = 1.8,
) -> np.ndarray:
    """Sample ``n`` dates between ``start`` and ``end`` with mild up-weighting
    on iPhone launch months (since trade-in volume spikes then in real life).

    Returns a numpy array of ``datetime64[D]``.
    """

    total_days = (end - start).days + 1
    all_days = np.array(
        [start + timedelta(days=i) for i in range(total_days)], dtype="datetime64[D]"
    )
    # Build weights: launch months get ``iphone_launch_weight``, others 1.
    weights = np.ones(total_days, dtype=float)
    for i, day_np in enumerate(all_days):
        py_date = _np_to_date(day_np)
        if is_iphone_launch_month(py_date):
            weights[i] = iphone_launch_weight
    weights /= weights.sum()
    idx = rng.choice(total_days, size=n, p=weights)
    return all_days[idx]


def _np_to_date(day_np: np.datetime64) -> date:
    """Convert a numpy datetime64[D] to a python ``date``."""

    py = day_np.astype("datetime64[D]").astype(datetime)
    if isinstance(py, datetime):
        return py.date()
    # numpy can return date-like for [D] precision on some platforms.
    return py


def np_to_date_array(days_np: np.ndarray) -> list[date]:
    """Vectorized helper: convert a numpy datetime64[D] array to list[date]."""

    return [_np_to_date(d) for d in days_np]


def rolling_median_clearing_price(
    df: pd.DataFrame,
    group_cols: list[str],
    date_col: str,
    value_col: str,
    window_days: int,
) -> pd.Series:
    """Compute a per-group rolling median within ``window_days`` prior to each row.

    Used only for feature construction in downstream modeling phases; included
    here so the data-generation-side notion of "recent price" is centralized.
    """

    out = df.sort_values(date_col).copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out = out.set_index(date_col)
    rolled = (
        out.groupby(group_cols)[value_col]
        .rolling(f"{window_days}D", min_periods=1, closed="left")
        .median()
    )
    return rolled.reset_index(level=group_cols, drop=True)


__all__ = [
    "FairValueContext",
    "clearing_price_from_fair_value",
    "days_since_latest_iphone_launch",
    "draw_random_dates",
    "grade_variance",
    "iphone_launch_effect",
    "is_iphone_launch_month",
    "is_newly_released",
    "month_index",
    "noise_sigma",
    "np_to_date_array",
    "rolling_median_clearing_price",
    "seasonality_factor",
]
