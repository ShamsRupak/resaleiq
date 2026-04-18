"""Page 3: Auction Lot Optimizer.

Given a bag of inventory, recommend a partition into auction lots that
maximizes predicted total clearing value. Uses a lot-level XGBoost trained
with a time-series holdout.
"""
# ruff: noqa: E402  -- bootstrap must run before third-party imports

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the repo root is on sys.path so ``from dashboard.utils import ...``
# resolves when Streamlit runs this page directly (e.g. on Streamlit
# Community Cloud, which does not set PYTHONPATH). Locally the Makefile
# sets PYTHONPATH=.:src and this block is a no-op.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.utils import (
    BRAND_BLUE,
    BRAND_GRAY,
    BRAND_GREEN,
    BRAND_LIGHT,
    BRAND_NAVY,
    BRAND_RED,
    PLOTLY_LAYOUT,
    fmt_dollars,
    fmt_int,
    fmt_pct,
    load_parquet,
    page_header,
    section,
    setup_page,
    sidebar_brand,
)

from resaleiq.ml.lot_model import (
    LOT_CATEGORIES,
    LOT_GRADES,
    PARTITION_STRATEGIES,
    score_partition,
)


setup_page("Auction Lot Optimizer", icon="📦")
sidebar_brand()
page_header(
    "Auction Lot Optimizer",
    "Given a bag of SKU inventory, the optimizer recommends a partition into "
    "auction lots that maximizes predicted total clearing price. Uses a "
    "lot-level XGBoost trained on 5,493 lots (Oct 2024 to Jan 2026) with a "
    "531-lot held-out test set (Feb to Mar 2026).",
)


# --------------------------------------------------------------------------- #
# Load artifacts
# --------------------------------------------------------------------------- #


@st.cache_resource
def load_lot_model():
    path = Path(__file__).resolve().parent.parent.parent / "data" / "lot_model.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data(ttl=3600)
def load_sku_catalog() -> pd.DataFrame:
    """SKU + device catalog. Uses `model_family`, not `model_name`."""

    skus = load_parquet("skus")
    devices = load_parquet("devices")
    return skus.merge(
        devices[["device_id", "device_category", "manufacturer", "model_family"]],
        on="device_id",
    )


model = load_lot_model()
catalog = load_sku_catalog()

if model is None:
    st.error(
        "Lot model artifact not found. Run `make lot-train` "
        "(or `uv run python scripts/train_lot_model.py`) to train it.",
        icon="⚠️",
    )
    st.stop()


# --------------------------------------------------------------------------- #
# Preset bags
# --------------------------------------------------------------------------- #


PRESET_BAGS = {
    "Launch season dump (Android Mid heavy)": {
        "description": "60 Android Mid + 20 Apple Flagship + 10 Android Flagship. "
                       "Mixed grades. Tests partition quality during iPhone launch.",
        "composition": {
            "Android Mid": {"n": 60, "grade_mix": {"A": 0.3, "B": 0.4, "C": 0.25, "D": 0.05}},
            "Apple Flagship": {"n": 20, "grade_mix": {"A+": 0.2, "A": 0.4, "B": 0.3, "C": 0.1}},
            "Android Flagship": {"n": 10, "grade_mix": {"A": 0.3, "B": 0.5, "C": 0.2}},
        },
        "reference_month": pd.Timestamp("2025-09-15"),
    },
    "Premium fresh trade-ins (all flagship)": {
        "description": "40 Apple Flagship + 30 Android Flagship, A-grade dominant. "
                       "Tests partition quality for high-value inventory.",
        "composition": {
            "Apple Flagship": {"n": 40, "grade_mix": {"A+": 0.4, "A": 0.4, "B": 0.15, "C": 0.05}},
            "Android Flagship": {"n": 30, "grade_mix": {"A+": 0.3, "A": 0.5, "B": 0.15, "C": 0.05}},
        },
        "reference_month": pd.Timestamp("2025-11-15"),
    },
    "Mid-tier diversified (Q1 ramp)": {
        "description": "Mixed-tier inventory typical of a wholesaler's Q1 mix.",
        "composition": {
            "Apple Mid": {"n": 25, "grade_mix": {"A": 0.3, "B": 0.4, "C": 0.25, "D": 0.05}},
            "Android Mid": {"n": 25, "grade_mix": {"A": 0.25, "B": 0.4, "C": 0.3, "D": 0.05}},
            "Apple Flagship": {"n": 15, "grade_mix": {"A+": 0.3, "A": 0.4, "B": 0.3}},
            "Android Flagship": {"n": 15, "grade_mix": {"A+": 0.2, "A": 0.5, "B": 0.3}},
        },
        "reference_month": pd.Timestamp("2026-02-15"),
    },
    "Low-grade wholesale liquidation": {
        "description": "Heavy in C and D grades across categories. Tests "
                       "partition behavior for discount inventory.",
        "composition": {
            "Android Mid": {"n": 40, "grade_mix": {"B": 0.1, "C": 0.5, "D": 0.4}},
            "Android Flagship": {"n": 15, "grade_mix": {"B": 0.2, "C": 0.5, "D": 0.3}},
            "Apple Mid": {"n": 15, "grade_mix": {"B": 0.2, "C": 0.5, "D": 0.3}},
        },
        "reference_month": pd.Timestamp("2026-01-15"),
    },
}


def materialize_bag(composition: dict, catalog: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[pd.DataFrame] = []
    for cat, spec in composition.items():
        cat_catalog = catalog[catalog["device_category"] == cat]
        if cat_catalog.empty:
            continue
        n = spec["n"]
        for grade, pct in spec["grade_mix"].items():
            count = max(1, int(round(n * pct)))
            pool = cat_catalog[cat_catalog["condition_grade"] == grade]
            if pool.empty:
                continue
            sample_size = min(count, len(pool))
            sample = pool.sample(n=sample_size, random_state=rng.integers(0, 10000))
            rows.append(sample)
    if not rows:
        return pd.DataFrame(columns=["sku_id", "device_category", "condition_grade", "baseline_value_usd"])
    bag = pd.concat(rows, ignore_index=True)
    return bag[["sku_id", "device_category", "condition_grade", "baseline_value_usd"]].copy()


# --------------------------------------------------------------------------- #
# Sidebar controls
# --------------------------------------------------------------------------- #


with st.sidebar:
    st.markdown("#### Optimizer controls")
    preset_name = st.selectbox("Preset inventory bag", list(PRESET_BAGS.keys()))
    bag_seed = st.number_input("Sampling seed", min_value=1, max_value=9999, value=42)
    auction_type = st.radio(
        "Auction mechanism",
        options=["popcorn", "fixed_end"],
        index=0,
        help="Popcorn auctions auto-extend in the final minutes to drive bidding up.",
    )

preset = PRESET_BAGS[preset_name]
sku_bag = materialize_bag(preset["composition"], catalog, seed=int(bag_seed))
reference_month = preset["reference_month"]


# --------------------------------------------------------------------------- #
# Bag overview
# --------------------------------------------------------------------------- #


section(preset_name, preset["description"])

bag_cols = st.columns(4)
bag_cols[0].metric("Units in bag", fmt_int(len(sku_bag)))
bag_cols[1].metric("Total baseline value", fmt_dollars(float(sku_bag["baseline_value_usd"].sum()), 0))
bag_cols[2].metric("Reference month", reference_month.strftime("%b %Y"))
bag_cols[3].metric("Auction type", auction_type)

# Composition chart
bag_summary = (
    sku_bag.groupby(["device_category", "condition_grade"])
    .size().reset_index(name="count")
)

fig_bag = go.Figure()
for cat in LOT_CATEGORIES:
    sub = bag_summary[bag_summary["device_category"] == cat]
    if sub.empty:
        continue
    fig_bag.add_trace(
        go.Bar(
            x=sub["condition_grade"],
            y=sub["count"],
            name=cat,
            hovertemplate=f"<b>{cat}</b><br>Grade %{{x}}: %{{y}} units<extra></extra>",
        )
    )
fig_bag.update_layout(
    **PLOTLY_LAYOUT,
    barmode="stack",
    height=280,
    xaxis=dict(title="Condition grade", categoryorder="array", categoryarray=list(LOT_GRADES)),
    yaxis=dict(title="Units", showgrid=True, gridcolor=BRAND_LIGHT),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig_bag, width="stretch")

st.divider()


# --------------------------------------------------------------------------- #
# Score partition strategies
# --------------------------------------------------------------------------- #


section(
    "Partition strategy comparison",
    "Each strategy assigns bag SKUs to lots differently. The lot-level XGBoost "
    "scores each resulting lot, and the chart shows total predicted clearing "
    "price with 80% confidence intervals.",
)

strategy_results = []
for strategy_name, (strategy_fn, description) in PARTITION_STRATEGIES.items():
    partition = strategy_fn(sku_bag)
    if not partition:
        continue
    scored = score_partition(
        sku_bag, partition, model,
        auction_type=auction_type, reference_month=reference_month,
    )
    if scored.empty:
        continue
    strategy_results.append({
        "strategy": strategy_name,
        "description": description,
        "n_lots": len(partition),
        "total_predicted": float(scored["predicted_clearing"].sum()),
        "total_lower": float(scored["predicted_lower"].sum()),
        "total_upper": float(scored["predicted_upper"].sum()),
        "avg_ratio": float(scored["ratio"].mean()),
        "detail": scored,
    })

if not strategy_results:
    st.warning("No valid partitions produced.")
    st.stop()

strategy_results.sort(key=lambda r: -r["total_predicted"])
best = strategy_results[0]
worst = strategy_results[-1]
uplift = best["total_predicted"] - worst["total_predicted"]
uplift_pct = uplift / worst["total_predicted"] if worst["total_predicted"] > 0 else 0.0


# Chart
fig_cmp = go.Figure()
for i, r in enumerate(strategy_results):
    color = BRAND_GREEN if i == 0 else (BRAND_RED if i == len(strategy_results) - 1 else BRAND_GRAY)
    fig_cmp.add_trace(
        go.Bar(
            x=[r["strategy"]],
            y=[r["total_predicted"]],
            error_y=dict(
                type="data", symmetric=False,
                array=[r["total_upper"] - r["total_predicted"]],
                arrayminus=[r["total_predicted"] - r["total_lower"]],
                color=BRAND_GRAY, thickness=1, width=6,
            ),
            marker_color=color,
            text=[f"<b>{fmt_dollars(r['total_predicted'], 0)}</b><br>{r['n_lots']} lots"],
            textposition="outside",
            hovertemplate=(
                f"<b>{r['strategy']}</b><br>{r['description']}<br>"
                f"Predicted: {fmt_dollars(r['total_predicted'], 0)}<br>"
                f"80% CI: [{fmt_dollars(r['total_lower'], 0)}, {fmt_dollars(r['total_upper'], 0)}]<extra></extra>"
            ),
            showlegend=False,
        )
    )

fig_cmp.update_layout(
    **PLOTLY_LAYOUT,
    height=450,
    yaxis=dict(title="Predicted total clearing price ($)", showgrid=True, gridcolor=BRAND_LIGHT),
    xaxis=dict(title=None, showgrid=False, tickangle=-10),
)
st.plotly_chart(fig_cmp, width="stretch")


# Recommendation card
rec_cols = st.columns(3)
rec_cols[0].metric(
    "Recommended strategy",
    best["strategy"],
    delta=best["description"],
    delta_color="off",
)
rec_cols[1].metric(
    "Expected clearing",
    fmt_dollars(best["total_predicted"], 0),
    delta=f"80% CI: {fmt_dollars(best['total_lower'], 0)} to {fmt_dollars(best['total_upper'], 0)}",
    delta_color="off",
)
rec_cols[2].metric(
    "Uplift vs worst strategy",
    fmt_dollars(uplift, 0),
    delta=f"{fmt_pct(uplift_pct, 1)} better",
)


st.divider()


# --------------------------------------------------------------------------- #
# Detailed per-lot breakdown for the best strategy
# --------------------------------------------------------------------------- #


section(
    f"Lot-by-lot detail: {best['strategy']}",
    f"{len(best['detail'])} lots in this partition. "
    f"Each row is a single auction lot with its predicted clearing and interval.",
)

detail = best["detail"].copy()
detail["Lot"] = [f"Lot {i+1}" for i in range(len(detail))]
detail_display = detail[["Lot", "n_skus", "total_baseline", "predicted_clearing",
                         "predicted_lower", "predicted_upper", "ratio"]].copy()
detail_display.columns = ["Lot", "Units", "Baseline $", "Predicted $", "CI low", "CI high", "Ratio"]
for col in ["Baseline $", "Predicted $", "CI low", "CI high"]:
    detail_display[col] = detail_display[col].apply(lambda v: fmt_dollars(float(v), 0))
detail_display["Ratio"] = detail_display["Ratio"].apply(lambda v: f"{float(v):.3f}")
st.dataframe(detail_display, width="stretch", hide_index=True)


with st.expander("All strategies compared (detailed table)"):
    full_cmp = pd.DataFrame([
        {
            "Strategy": r["strategy"],
            "Description": r["description"],
            "Lots": r["n_lots"],
            "Predicted $": fmt_dollars(r["total_predicted"], 0),
            "CI low": fmt_dollars(r["total_lower"], 0),
            "CI high": fmt_dollars(r["total_upper"], 0),
            "Avg ratio": f"{r['avg_ratio']:.3f}",
        }
        for r in strategy_results
    ])
    st.dataframe(full_cmp, width="stretch", hide_index=True)


st.divider()

# Honest model-performance callout that loads from the summary JSON
try:
    import json as _json
    _summary_path = (
        Path(__file__).resolve().parent.parent.parent
        / "data" / "lot_model_summary.json"
    )
    _s = _json.loads(_summary_path.read_text())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Out-of-sample MAPE",
        fmt_pct(_s["oos_aggregate_mape"], 2),
        delta=f"{_s['n_holdout']} held-out lots",
        delta_color="off",
    )
    m2.metric(
        "In-sample MAPE",
        fmt_pct(_s["in_sample_aggregate_mape"], 2),
        delta=f"{_s['n_train']} train lots",
        delta_color="off",
    )
    m3.metric(
        "Naive baseline MAPE (OOS)",
        fmt_pct(_s["naive_baseline_mape_oos"], 2),
        delta="predict `total_baseline_value`",
        delta_color="off",
    )
    m4.metric(
        "Improvement vs naive (OOS)",
        fmt_pct(_s["relative_improvement_vs_naive_oos"], 1),
        delta="on unseen Feb-Mar 2026 data",
        delta_color="normal",
    )
except Exception:
    pass

st.caption(
    "The lot-level model uses 18 composition features (SKU count, category mix, "
    "grade mix, price dispersion, auction mechanics, seasonality). On a "
    "time-series holdout covering Feb to Mar 2026 (531 lots the model never saw "
    "during training), out-of-sample MAPE is 3.98% versus a naive baseline of "
    "18.11% - a 78.0% relative improvement. The in-sample to out-of-sample gap "
    "is 0.85 percentage points (3.13% vs 3.98%), which is honest evidence that "
    "the 18 composition features are capturing generalizable signal rather than "
    "memorizing training lots. Intervals derive from empirical-quantile "
    "calibration on training residuals. Holdout coverage comes in at 66.5% vs "
    "the 80% target because Feb-Mar residuals are approximately 27% wider "
    "than training-period residuals - a real distribution-shift finding that "
    "rolling-window recalibration would fix in production, and is the third "
    "proposed experiment on the Experiment Tracker page."
)
