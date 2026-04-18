"""Buyer generation: accounts for the sku_offers and lot_bids flows.

Buyers are characterized by three attributes that flows downstream use for
pricing dynamics:

- ``buyer_type`` (distributor, reseller, refurbisher, carrier) drives what kind
  of inventory they pursue. Distributors take large lots, refurbishers take
  lower grades, carriers take unlocked flagship inventory.
- ``region`` is a simple geographic tag used in the SQL layer to slice
  clearing-rate analysis.
- ``tier`` (Enterprise, Mid-market, SMB) governs bidding aggression: Enterprise
  buyers submit higher absolute bids and win more often, matching the real
  distribution of wholesale activity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from resaleiq.config import (
    BUYER_REGIONS,
    BUYER_TIERS,
    BUYER_TYPES,
    ScaleConfig,
    child_seed,
)

# Target mix for buyer attributes. These are eyeballed from industry commentary
# rather than ground truth, and are configurable here for anyone who wants to
# tune the simulation.
_TIER_WEIGHTS: dict[str, float] = {"Enterprise": 0.15, "Mid-market": 0.45, "SMB": 0.40}
_TYPE_WEIGHTS: dict[str, float] = {
    "distributor": 0.40,
    "reseller": 0.35,
    "refurbisher": 0.20,
    "carrier": 0.05,
}
_REGION_WEIGHTS: dict[str, float] = {
    "US-East": 0.35,
    "US-West": 0.25,
    "EU": 0.20,
    "APAC": 0.12,
    "LATAM": 0.08,
}


def _weighted_choice(
    rng: np.random.Generator, options: tuple[str, ...], weights: dict[str, float], size: int
) -> np.ndarray:
    """Weighted sampler that keeps option ordering deterministic."""

    p = np.array([weights[opt] for opt in options], dtype=float)
    p = p / p.sum()
    return rng.choice(np.array(options), size=size, p=p)


def build_buyers(scale: ScaleConfig) -> pd.DataFrame:
    """Generate the buyers table at the requested scale."""

    rng = np.random.default_rng(child_seed(f"buyers:{scale.name}"))
    n = scale.n_buyers
    df = pd.DataFrame(
        {
            "buyer_id": np.arange(1, n + 1, dtype=np.int64),
            "buyer_type": _weighted_choice(rng, BUYER_TYPES, _TYPE_WEIGHTS, n),
            "region": _weighted_choice(rng, BUYER_REGIONS, _REGION_WEIGHTS, n),
            "tier": _weighted_choice(rng, BUYER_TIERS, _TIER_WEIGHTS, n),
        }
    )
    # Invariant: every configured tier appears at least once. At scale=400 this
    # is effectively certain, but the assertion documents intent and catches
    # any future tier-weight tuning that would violate it.
    for tier in BUYER_TIERS:
        assert (df["tier"] == tier).any(), f"tier missing in generated buyers: {tier}"
    return df


def bidding_aggression(tier: str) -> float:
    """Return a multiplier applied to a buyer's offer / bid price.

    Calibrated so the top-tier buyers land roughly 5 percent higher than the
    bottom tier on equivalent inventory, matching the published 3 to 5 percent
    "storefront uplift" typical in wholesale secondhand marketplaces for
    enterprise buyer flow.
    """

    return {"Enterprise": 1.03, "Mid-market": 1.00, "SMB": 0.97}[tier]


__all__ = ["bidding_aggression", "build_buyers"]
