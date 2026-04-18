"""Lot flow: generate ``lots``, ``lot_items``, and ``lot_bids``.

The auction side of the marketplace. A lot is a collection of SKUs sold as a
single unit. Key mechanics:

- Two auction types: ``popcorn`` (auto-extending end-of-auction bidding) and
  ``fixed_end`` (hard close time).
- Popcorn lots tend to clear at approximately 4.2 percent over reserve; fixed
  end lots at about 1.8 percent over reserve. Calibrated in
  ``config.NoiseConfig``.
- Approximately 18 percent of lots fail to clear, concentrated in low-grade,
  high-quantity compositions with thin buyer interest.
- Bids carry flags for proxy bidding and popcorn extension trigger, so the
  SQL layer can quantify their effect separately.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from resaleiq.config import (
    AUCTION_TYPES,
    END_DATE,
    LOT_STATUSES,
    NOISE,
    START_DATE,
    ScaleConfig,
    child_seed,
)
from resaleiq.data_generation.market_dynamics import (
    draw_random_dates,
)


def _sample_lot_sku_weights(skus_flat: pd.DataFrame) -> np.ndarray:
    """Compute per-SKU weights for inclusion in a lot.

    Lots tend to be built around flagship-heavy inventory for enterprise
    buyers, with condition grades skewed toward A and B. Grade D gets a small
    positive bump because parts-only lots are a real sub-market (though they
    tend to fail to clear more often, handled in the clearing logic).
    """

    w_category = (
        skus_flat["device_category"]
        .map(
            {
                "Apple Flagship": 2.5,
                "Apple Mid": 2.0,
                "Android Flagship": 1.8,
                "Android Mid": 1.5,
                "Android Budget": 1.0,
            }
        )
        .to_numpy(dtype=float)
    )
    w_grade = (
        skus_flat["condition_grade"]
        .map({"A+": 1.2, "A": 1.5, "B": 1.2, "C": 0.8, "D": 0.6})
        .to_numpy(dtype=float)
    )
    weights = w_category * w_grade
    return weights / weights.sum()


def _build_lot_items(
    rng: np.random.Generator,
    skus_flat: pd.DataFrame,
    n_lots: int,
    avg_items_per_lot: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate lot_items and a per-lot summary frame used by lot generation.

    Returns the ``lot_items`` DataFrame and a per-lot aggregate frame holding
    reference prices and grade mix, which the lots generator needs to compute
    reserves and clearing outcomes.
    """

    # Draw a number of distinct SKUs per lot. Poisson around mean, clipped to
    # [2, 12] so the tails don't produce pathological one-item or 30-item lots.
    sku_counts = np.clip(rng.poisson(avg_items_per_lot, size=n_lots), 2, 12).astype(np.int32)

    sku_weights = _sample_lot_sku_weights(skus_flat)
    sku_ids = skus_flat["sku_id"].to_numpy()
    sku_baseline = dict(
        zip(skus_flat["sku_id"].to_numpy(), skus_flat["baseline_value_usd"], strict=True)
    )
    sku_grade = dict(zip(skus_flat["sku_id"].to_numpy(), skus_flat["condition_grade"], strict=True))

    rows: list[dict[str, object]] = []
    lot_summary: list[dict[str, object]] = []

    lot_item_id = 1
    for lot_idx in range(n_lots):
        n_items = int(sku_counts[lot_idx])
        chosen = rng.choice(sku_ids, size=n_items, replace=False, p=sku_weights)
        # Quantity per line item. Lots typically carry 1 to 20 units per SKU.
        quantities = rng.integers(1, 21, size=n_items).astype(np.int32)
        total_ref_price = 0.0
        grade_share_d = 0
        for sku_id, qty in zip(chosen, quantities, strict=True):
            unit_ref = float(sku_baseline[int(sku_id)])
            rows.append(
                {
                    "lot_item_id": lot_item_id,
                    "lot_id": lot_idx + 1,
                    "sku_id": int(sku_id),
                    "quantity": int(qty),
                    "unit_ref_price": round(unit_ref, 2),
                }
            )
            total_ref_price += unit_ref * int(qty)
            if sku_grade[int(sku_id)] == "D":
                grade_share_d += int(qty)
            lot_item_id += 1
        total_quantity = int(quantities.sum())
        lot_summary.append(
            {
                "lot_id": lot_idx + 1,
                "total_ref_price": round(total_ref_price, 2),
                "total_quantity": total_quantity,
                "grade_d_share": grade_share_d / total_quantity if total_quantity > 0 else 0.0,
                "n_unique_skus": n_items,
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(lot_summary)


def _unsold_probability(grade_d_share: float, total_quantity: int) -> float:
    """Probability that a lot fails to clear.

    Concentrated in low-grade, high-quantity compositions. The base rate is
    calibrated against ``NOISE.lot_unsold_rate``; the modifiers push
    low-quality lots higher and small high-grade lots lower.
    """

    base = NOISE.lot_unsold_rate
    # Grade-D weight: pure Grade D lots roughly 2x base unsold rate.
    grade_d_factor = 1.0 + grade_d_share
    # Large lots (>80 units) are harder to place; small lots easier.
    size_factor = 1.0 if total_quantity < 30 else 1.0 + (total_quantity - 30) * 0.005
    # Small high-grade lots: drop unsold rate toward 8 percent.
    small_clean_discount = 0.5 if (grade_d_share < 0.05 and total_quantity < 20) else 1.0
    return float(min(base * grade_d_factor * size_factor * small_clean_discount, 0.6))


def _pick_winning_buyer(rng: np.random.Generator, buyers: pd.DataFrame) -> tuple[int, str]:
    """Select a winning buyer weighted by tier aggression."""

    tier_weights = buyers["tier"].map({"Enterprise": 2.0, "Mid-market": 1.0, "SMB": 0.5})
    p = tier_weights.to_numpy(dtype=float)
    p = p / p.sum()
    idx = int(rng.choice(len(buyers), p=p))
    buyer_row = buyers.iloc[idx]
    return int(buyer_row["buyer_id"]), str(buyer_row["tier"])


def build_lots(
    skus_flat: pd.DataFrame,
    buyers: pd.DataFrame,
    scale: ScaleConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate lots, lot_items, and lot_bids in one pass.

    The three tables are produced together because clearing logic depends on
    bid arrival dynamics, and bid dynamics depend on lot reserve prices.
    """

    rng = np.random.default_rng(child_seed(f"lots:{scale.name}"))
    n_lots = scale.n_lots

    # Draw lot-level attributes first.
    scheduled_close_np = draw_random_dates(rng, START_DATE, END_DATE, n_lots)
    scheduled_close_ts = pd.to_datetime(scheduled_close_np)
    auction_type = rng.choice(np.array(AUCTION_TYPES), size=n_lots, p=[0.55, 0.45])

    # Build lot_items and per-lot aggregates.
    lot_items, lot_summary = _build_lot_items(rng, skus_flat, n_lots, scale.avg_items_per_lot)

    # Reserve price: 0.82 of total reference price, with noise. This ensures
    # reserves sit below fair so most lots clear.
    reserve_multiplier = rng.normal(0.82, 0.04, size=n_lots)
    reserve_price = np.maximum(
        lot_summary["total_ref_price"].to_numpy() * reserve_multiplier, 15.0
    ).round(2)

    # Per-lot attributes used for bidding and clearing.
    total_qty = lot_summary["total_quantity"].to_numpy()
    grade_d_share = lot_summary["grade_d_share"].to_numpy()

    # Decide clearing vs unsold vs cancelled for each lot.
    rand_status = rng.random(n_lots)
    unsold_probs = np.array(
        [_unsold_probability(grade_d_share[i], int(total_qty[i])) for i in range(n_lots)]
    )
    # A small fraction is cancelled (seller pulled listing). Draw independently.
    cancelled_mask = rng.random(n_lots) < 0.015
    unsold_mask = (rand_status < unsold_probs) & (~cancelled_mask)
    cleared_mask = ~(unsold_mask | cancelled_mask)

    # Compose status column.
    status = np.where(cleared_mask, "cleared", np.where(unsold_mask, "unsold", "cancelled"))

    # Clearing price per lot.
    clearing_price = np.full(n_lots, np.nan, dtype=np.float64)
    winning_buyer_id = np.full(n_lots, -1, dtype=np.int64)
    actual_close = np.full(n_lots, np.datetime64("NaT"), dtype="datetime64[ns]")

    # Premium over reserve depends on auction type.
    for i in range(n_lots):
        if not cleared_mask[i]:
            continue
        premium_mean = (
            NOISE.lot_premium_mean_popcorn
            if auction_type[i] == "popcorn"
            else NOISE.lot_premium_mean_fixed_end
        )
        # Additional variance for high Grade D share (parts lots are noisier).
        premium_sigma = 0.025 + 0.03 * float(grade_d_share[i])
        premium = float(rng.normal(premium_mean, premium_sigma))
        cp = float(reserve_price[i]) * (1.0 + premium)
        clearing_price[i] = round(max(cp, float(reserve_price[i])), 2)
        buyer_id, _ = _pick_winning_buyer(rng, buyers)
        winning_buyer_id[i] = buyer_id
        # Popcorn lots close slightly after scheduled.
        close_offset = rng.integers(0, 180) if auction_type[i] == "popcorn" else 0  # minutes
        close_ts = scheduled_close_ts[i] + pd.Timedelta(minutes=int(close_offset))
        actual_close[i] = close_ts.to_datetime64()

    lots = pd.DataFrame(
        {
            "lot_id": np.arange(1, n_lots + 1, dtype=np.int64),
            "winning_buyer_id": winning_buyer_id,
            "reserve_price": reserve_price,
            "clearing_price": clearing_price,
            "scheduled_close": scheduled_close_ts,
            "actual_close": actual_close,
            "auction_type": auction_type.astype(str),
            "status": status.astype(str),
        }
    )
    lots["winning_buyer_id"] = lots["winning_buyer_id"].where(
        lots["winning_buyer_id"] > 0, other=pd.NA
    )
    assert lots["status"].isin(LOT_STATUSES).all()

    # Build lot_bids for cleared and unsold lots (cancelled lots have no bids).
    lot_bids = _build_lot_bids(
        rng,
        lots=lots,
        lot_summary=lot_summary,
        buyers=buyers,
        scale=scale,
    )
    return lots, lot_items, lot_bids


def _build_lot_bids(
    rng: np.random.Generator,
    lots: pd.DataFrame,
    lot_summary: pd.DataFrame,
    buyers: pd.DataFrame,
    scale: ScaleConfig,
) -> pd.DataFrame:
    """Generate bids for every cleared and unsold lot.

    Cleared lots have a sequence of ascending bids terminating at the clearing
    price. Unsold lots have one or two tentative bids that never reach reserve.
    """

    # Pre-allocate with an upper bound; trim at the end.
    max_bids_per_lot = 15
    upper = int(len(lots) * max_bids_per_lot)
    bid_id_arr = np.empty(upper, dtype=np.int64)
    lot_id_arr = np.empty(upper, dtype=np.int64)
    buyer_id_arr = np.empty(upper, dtype=np.int64)
    bid_amount_arr = np.empty(upper, dtype=np.float64)
    bid_at_arr = np.empty(upper, dtype="datetime64[ns]")
    is_proxy_arr = np.empty(upper, dtype=bool)
    popcorn_arr = np.empty(upper, dtype=bool)

    idx = 0
    bid_id_counter = 1
    avg_bids = scale.avg_bids_per_lot
    buyer_ids = buyers["buyer_id"].to_numpy()

    # Iterate deterministically by lot_id.
    for row in lots.sort_values("lot_id").itertuples(index=False):
        lot_id = int(row.lot_id)
        if row.status == "cancelled":
            continue

        reserve = float(row.reserve_price)
        if row.status == "cleared":
            final_price = float(row.clearing_price)
            n_bids = max(2, int(rng.poisson(avg_bids)))
            n_bids = min(n_bids, max_bids_per_lot)
            # Bid sequence: geometric climb from roughly 0.85 * reserve to clearing price.
            ladder_raw = np.linspace(0.88, 1.0, n_bids) + rng.normal(0, 0.02, size=n_bids)
            ladder = np.clip(ladder_raw, 0.85, 1.0)
            ladder = np.maximum.accumulate(ladder)  # monotonic
            # Scale so final bid equals clearing price.
            if ladder[-1] != ladder[0]:
                ladder = ladder / ladder[-1]
            amounts = ladder * final_price
            # Time distribution: bids cluster near close, especially popcorn.
            close_ts = (
                pd.Timestamp(row.actual_close)
                if pd.notna(row.actual_close)
                else pd.Timestamp(row.scheduled_close)
            )
            start_ts = close_ts - pd.Timedelta(days=int(rng.integers(1, 7)))
            # Bid arrival offsets as fractions of the window, skewed late.
            fractions = np.sort(rng.beta(2.5, 1.2, size=n_bids))
            bid_times = [start_ts + (close_ts - start_ts) * float(f) for f in fractions]
            # Proxy and popcorn flags. Popcorn only applies to popcorn auctions
            # and only for bids in the final 10 minutes.
            is_proxy = rng.random(n_bids) < 0.25
            if row.auction_type == "popcorn":
                popcorn_triggers = np.array(
                    [(close_ts - t).total_seconds() < 600 and rng.random() < 0.5 for t in bid_times]
                )
            else:
                popcorn_triggers = np.zeros(n_bids, dtype=bool)
            # Bid-by-bid fill.
            chosen_buyers = rng.choice(buyer_ids, size=n_bids, replace=True)
            for k in range(n_bids):
                bid_id_arr[idx] = bid_id_counter
                lot_id_arr[idx] = lot_id
                buyer_id_arr[idx] = int(chosen_buyers[k])
                bid_amount_arr[idx] = round(float(amounts[k]), 2)
                bid_at_arr[idx] = bid_times[k].to_datetime64()
                is_proxy_arr[idx] = bool(is_proxy[k])
                popcorn_arr[idx] = bool(popcorn_triggers[k])
                idx += 1
                bid_id_counter += 1
            # Ensure the winning buyer placed the final bid.
            if int(row.winning_buyer_id) > 0:
                buyer_id_arr[idx - 1] = int(row.winning_buyer_id)
        else:
            # Unsold: one or two tentative sub-reserve bids.
            n_bids = int(rng.integers(1, 3))
            for _ in range(n_bids):
                bid_id_arr[idx] = bid_id_counter
                lot_id_arr[idx] = lot_id
                buyer_id_arr[idx] = int(rng.choice(buyer_ids))
                # 60 to 92 percent of reserve.
                amt = reserve * float(rng.uniform(0.60, 0.92))
                bid_amount_arr[idx] = round(max(amt, 1.0), 2)
                close_ts = pd.Timestamp(row.scheduled_close)
                t = close_ts - pd.Timedelta(days=int(rng.integers(1, 5)))
                bid_at_arr[idx] = t.to_datetime64()
                is_proxy_arr[idx] = bool(rng.random() < 0.15)
                popcorn_arr[idx] = False
                idx += 1
                bid_id_counter += 1

    # Trim to actual size.
    return pd.DataFrame(
        {
            "bid_id": bid_id_arr[:idx],
            "lot_id": lot_id_arr[:idx],
            "buyer_id": buyer_id_arr[:idx],
            "bid_amount": bid_amount_arr[:idx],
            "bid_at": bid_at_arr[:idx],
            "is_proxy": is_proxy_arr[:idx],
            "popcorn": popcorn_arr[:idx],
        }
    )


__all__ = ["build_lots"]
