"""SKU flow: generate ``sku_listings`` and the ``sku_offers`` against them.

A listing is a single-SKU posting on the storefront with a list price and
quantity. Offers arrive from buyers with negotiation dynamics:

- Some offers land close to list and clear immediately.
- Some land far below list and are rejected.
- Some land in the middle and result in a counter that may or may not clear.
- Some listings receive no offers at all before they expire.

The outcome distribution targets roughly 55 percent of listings receiving at
least one accepted or countered-accepted offer, matching commentary from the
wholesale secondhand space.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from resaleiq.config import END_DATE, SKU_OFFER_OUTCOMES, START_DATE, ScaleConfig, child_seed
from resaleiq.data_generation.buyers import bidding_aggression
from resaleiq.data_generation.market_dynamics import (
    FairValueContext,
    clearing_price_from_fair_value,
    draw_random_dates,
)

_ = SKU_OFFER_OUTCOMES  # re-exported vocabulary invariant


def _sample_listing_skus(
    rng: np.random.Generator,
    skus: pd.DataFrame,
    n_listings: int,
) -> np.ndarray:
    """Sample SKUs for listings with realistic popularity skew.

    Popularity roughly follows device_category: flagship Apple gets the most
    listings, budget Android the fewest. The skew is deliberately mild (ratio
    of 3:1 between extremes) so every segment has enough data for robust
    modeling.
    """

    category_weight = {
        "Apple Flagship": 3.0,
        "Apple Mid": 2.5,
        "Android Flagship": 2.0,
        "Android Mid": 1.8,
        "Android Budget": 1.0,
    }
    w = skus["device_category"].map(category_weight).to_numpy(dtype=float)
    # Additional haircut for Grade D: parts-only doesn't sell as actively.
    w = np.where(skus["condition_grade"].to_numpy() == "D", w * 0.4, w)
    w /= w.sum()
    return rng.choice(skus["sku_id"].to_numpy(), size=n_listings, p=w)


def _list_price_markup(rng: np.random.Generator, n: int) -> np.ndarray:
    """Listing markup over baseline value.

    Sellers list slightly above fair value and absorb an occasional below-fair
    listing when they need to move aging inventory.
    """

    return 1.0 + rng.normal(0.08, 0.05, size=n)


def build_sku_listings(
    skus_flat: pd.DataFrame,
    scale: ScaleConfig,
) -> pd.DataFrame:
    """Generate the ``sku_listings`` table.

    ``skus_flat`` must already be joined against ``devices`` via
    ``skus.join_device_attrs`` so this function can read ``device_category``,
    ``release_date``, and other device-level attributes directly.
    """

    rng = np.random.default_rng(child_seed(f"sku_listings:{scale.name}"))
    n = scale.n_listings
    sampled_skus = _sample_listing_skus(rng, skus_flat, n)
    sku_lookup = skus_flat.set_index("sku_id")
    sampled = sku_lookup.loc[sampled_skus].reset_index()

    listed_at_np = draw_random_dates(rng, START_DATE, END_DATE, n)
    listed_at = pd.to_datetime(listed_at_np)
    markup = _list_price_markup(rng, n)
    list_price = np.maximum(sampled["baseline_value_usd"].to_numpy() * markup, 5.0).round(2)
    # Small quantities are typical for storefront postings (large quantities go
    # to the auction side). Draw from a 1-to-12 distribution skewed low.
    quantity = (rng.integers(1, 12, size=n)).astype(np.int32)

    df = pd.DataFrame(
        {
            "listing_id": np.arange(1, n + 1, dtype=np.int64),
            "sku_id": sampled["sku_id"].to_numpy(),
            "list_price": list_price,
            "quantity": quantity,
            "listed_at": listed_at,
            "seller_name": rng.choice(
                np.array(
                    [
                        "Summit Wireless",
                        "Crown Distributors",
                        "Ridgeline Mobile",
                        "Parallel Trade Co",
                        "Meridian Wholesale",
                    ]
                ),
                size=n,
            ),
        }
    )
    return df


def _offers_per_listing(rng: np.random.Generator, n: int, mean: float) -> np.ndarray:
    """Draw a plausible number of offers per listing.

    Uses a Poisson centered on ``mean``; clip to [0, 12] so the tail does not
    generate absurd counts.
    """

    return np.clip(rng.poisson(mean, size=n), 0, 12).astype(np.int32)


def build_sku_offers(
    listings: pd.DataFrame,
    skus_flat: pd.DataFrame,
    buyers: pd.DataFrame,
    scale: ScaleConfig,
) -> pd.DataFrame:
    """Generate ``sku_offers`` with realistic outcome distributions.

    For each listing, draw a number of offers (Poisson around
    ``scale.avg_offers_per_listing``). Each offer's price is derived from a
    fair-value sample and the buyer's aggression multiplier. Outcome is then
    determined by the ratio of offer price to fair value.
    """

    rng = np.random.default_rng(child_seed(f"sku_offers:{scale.name}"))
    n_listings = len(listings)
    offer_counts = _offers_per_listing(rng, n_listings, scale.avg_offers_per_listing)
    total_offers = int(offer_counts.sum())

    # Pre-allocate output arrays for speed.
    offer_id = np.arange(1, total_offers + 1, dtype=np.int64)
    listing_id = np.empty(total_offers, dtype=np.int64)
    buyer_id = np.empty(total_offers, dtype=np.int64)
    offer_price = np.empty(total_offers, dtype=np.float64)
    outcome = np.empty(total_offers, dtype=object)
    clearing_price = np.full(total_offers, np.nan, dtype=np.float64)
    offer_at = np.empty(total_offers, dtype="datetime64[ns]")

    sku_lookup = skus_flat.set_index("sku_id")
    buyer_lookup = buyers.set_index("buyer_id")[["tier"]]

    # Pre-shuffle buyer IDs in chunks for offer assignment; this is faster than
    # a sample per offer.
    sampled_buyers = rng.choice(buyers["buyer_id"].to_numpy(), size=total_offers, replace=True)

    idx = 0
    for row_idx, n_offers in enumerate(offer_counts):
        if n_offers == 0:
            continue
        lst = listings.iloc[row_idx]
        sku = sku_lookup.loc[int(lst["sku_id"])]
        baseline = float(sku["baseline_value_usd"])
        listed_date = lst["listed_at"].to_pydatetime().date()
        # Offers arrive within 21 days of listing.
        arrival_offsets = rng.integers(0, 22, size=int(n_offers))

        for k in range(int(n_offers)):
            buyer = int(sampled_buyers[idx])
            tier = str(buyer_lookup.loc[buyer, "tier"])
            aggression = bidding_aggression(tier)

            offer_date = listed_date + timedelta(days=int(arrival_offsets[k]))
            # Clamp offers to the simulation window.
            if offer_date > END_DATE:
                offer_date = END_DATE

            ctx = FairValueContext(
                baseline_value_usd=baseline,
                manufacturer=str(sku["manufacturer"]),
                device_category=str(sku["device_category"]),
                release_date=sku["release_date"].date(),
                grade=str(sku["condition_grade"]),
                transaction_date=offer_date,
            )
            fair = clearing_price_from_fair_value(ctx, rng, aggression_multiplier=aggression)
            # Buyers don't always offer fair: sample a percentage-of-fair band.
            # Majority of offers cluster near fair; a long lower tail of lowballs.
            offer_mult = float(rng.normal(0.94, 0.08))
            offer = max(fair * offer_mult, 1.0)

            listing_id[idx] = int(lst["listing_id"])
            buyer_id[idx] = buyer
            offer_price[idx] = round(offer, 2)
            offer_at[idx] = np.datetime64(offer_date, "ns")

            # Outcome logic: ratio to fair determines path.
            ratio = offer / fair if fair > 0 else 0.0
            if ratio >= 0.98:
                outcome[idx] = "accepted"
                clearing_price[idx] = round(offer, 2)
            elif ratio >= 0.90:
                # Counter + accept: seller nudges up by a few percent.
                if rng.random() < 0.55:
                    outcome[idx] = "countered_accepted"
                    clearing_price[idx] = round(offer * float(rng.uniform(1.01, 1.04)), 2)
                else:
                    outcome[idx] = "rejected"
            elif ratio >= 0.75:
                outcome[idx] = "rejected"
            else:
                outcome[idx] = "expired"
            idx += 1

    df = pd.DataFrame(
        {
            "offer_id": offer_id,
            "listing_id": listing_id,
            "buyer_id": buyer_id,
            "offer_price": offer_price,
            "outcome": outcome.astype(str),
            "clearing_price": clearing_price,
            "offer_at": offer_at,
        }
    )
    assert df["outcome"].isin(SKU_OFFER_OUTCOMES).all()
    return df


__all__ = ["build_sku_listings", "build_sku_offers"]
