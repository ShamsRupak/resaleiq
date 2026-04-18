"""Page 1: Executive Summary.

CEO-level view. Headline KPIs + GMV trend + MAPE ladder. Uses Streamlit
native components and correct schema: sku_offers.outcome (not status) and
lots.status.
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

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.utils import (
    BRAND_BLUE,
    BRAND_GREEN,
    BRAND_LIGHT,
    BRAND_NAVY,
    BRAND_RED,
    MODEL_LABELS,
    MODEL_ORDER,
    PLOTLY_LAYOUT,
    counterfactual_summary,
    filter_cleared_lots,
    filter_cleared_offers,
    fmt_dollars,
    fmt_int,
    fmt_pct,
    load_parquet,
    load_phase3_summary,
    page_header,
    section,
    setup_page,
    sidebar_brand,
)


setup_page("Executive Summary", icon="📊")
sidebar_brand()
page_header(
    "Executive Summary",
    "Platform-wide pricing and marketplace KPIs. A 42.3% relative MAPE reduction "
    "on the concentrated error segment (Android Mid during iPhone launch months) "
    "translates to $13.7K in pricing error recovered across 801 test-set offers.",
)


# --------------------------------------------------------------------------- #
# Data loaders (parquet-first; SQL would work too but parquet is simpler)
# --------------------------------------------------------------------------- #


@st.cache_data(ttl=600)
def load_gmv_monthly() -> pd.DataFrame:
    """Monthly GMV from cleared offers plus cleared lots."""

    offers = load_parquet("sku_offers")
    lots = load_parquet("lots")

    offers_c = filter_cleared_offers(offers)
    lots_c = filter_cleared_lots(lots)

    offers_c["month"] = pd.to_datetime(offers_c["offer_at"]).dt.to_period("M").dt.to_timestamp()
    lots_c["month"] = pd.to_datetime(lots_c["actual_close"]).dt.to_period("M").dt.to_timestamp()

    o = offers_c.groupby("month").agg(
        gmv_offers=("clearing_price", "sum"),
        n_offers_cleared=("offer_id", "count"),
    ).reset_index()
    l = lots_c.groupby("month").agg(
        gmv_lots=("clearing_price", "sum"),
        n_lots_cleared=("lot_id", "count"),
    ).reset_index()

    monthly = o.merge(l, on="month", how="outer").fillna(0).sort_values("month")
    monthly["gmv_total"] = monthly["gmv_offers"] + monthly["gmv_lots"]
    return monthly


@st.cache_data(ttl=600)
def headline_numbers() -> dict:
    monthly = load_gmv_monthly()
    cf = counterfactual_summary()
    lots = load_parquet("lots")
    total_lots = int(len(lots))
    cleared_lots = int((lots["status"] == "cleared").sum())

    return {
        "gmv_total": float(monthly["gmv_total"].sum()),
        "n_offers_cleared": int(monthly["n_offers_cleared"].sum()),
        "n_lots_cleared": int(monthly["n_lots_cleared"].sum()),
        "lot_clearing_rate": cleared_lots / total_lots if total_lots > 0 else 0.0,
        "overall_base_mape": cf["all_offers"]["mape_base"],
        "overall_tgt_mape": cf["all_offers"]["mape_tgt"],
        "planted_base_mape": cf["planted"]["mape_base"],
        "planted_tgt_mape": cf["planted"]["mape_tgt"],
        "planted_relative_reduction": (
            (cf["planted"]["mape_base"] - cf["planted"]["mape_tgt"])
            / cf["planted"]["mape_base"]
            if cf["planted"]["mape_base"] > 0
            else 0
        ),
        "planted_dollars_total": cf["planted"]["total_dollars"],
        "planted_dollars_per_offer": cf["planted"]["dollars_per_offer"],
    }


k = headline_numbers()


# --------------------------------------------------------------------------- #
# Top KPI row
# --------------------------------------------------------------------------- #


section("Platform metrics", "18-month simulated window: Oct 2024 to Mar 2026")

c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Total marketplace GMV",
    fmt_dollars(k["gmv_total"], 1),
    delta=f"{fmt_int(k['n_offers_cleared'])} offers + {fmt_int(k['n_lots_cleared'])} lots",
    delta_color="off",
)
c2.metric(
    "Lot clearing rate",
    fmt_pct(k["lot_clearing_rate"], 1),
    delta=f"{fmt_int(k['n_lots_cleared'])} of {fmt_int(k['n_lots_cleared'] + (8000 - k['n_lots_cleared']))} total",
    delta_color="off",
)
c3.metric(
    "Overall MAPE (targeted)",
    fmt_pct(k["overall_tgt_mape"], 2),
    delta=f"vs {fmt_pct(k['overall_base_mape'], 2)} baseline",
    delta_color="inverse",
)
c4.metric(
    "Planted MAPE reduction",
    fmt_pct(k["planted_relative_reduction"], 1),
    delta=f"{fmt_dollars(k['planted_dollars_total'], 0)} recovered",
)

st.divider()

# --------------------------------------------------------------------------- #
# GMV trend
# --------------------------------------------------------------------------- #


section(
    "Monthly GMV and seasonality",
    "Stacked by product line. Red-shaded months are iPhone launch windows "
    "(Sept–Oct) where the pricing model's work matters most.",
)

monthly = load_gmv_monthly()

fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=monthly["month"],
        y=monthly["gmv_offers"] / 1e6,
        name="SKU offers",
        marker_color=BRAND_BLUE,
        hovertemplate="<b>%{x|%b %Y}</b><br>Offers GMV: $%{y:.2f}M<extra></extra>",
    )
)
fig.add_trace(
    go.Bar(
        x=monthly["month"],
        y=monthly["gmv_lots"] / 1e6,
        name="Lots",
        marker_color=BRAND_NAVY,
        hovertemplate="<b>%{x|%b %Y}</b><br>Lots GMV: $%{y:.2f}M<extra></extra>",
    )
)

for m in monthly["month"]:
    if pd.Timestamp(m).month in (9, 10):
        fig.add_vrect(
            x0=pd.Timestamp(m) - pd.Timedelta(days=15),
            x1=pd.Timestamp(m) + pd.Timedelta(days=15),
            fillcolor=BRAND_RED, opacity=0.06, layer="below", line_width=0,
        )

fig.update_layout(
    **PLOTLY_LAYOUT,
    barmode="stack",
    height=360,
    xaxis=dict(title=None, showgrid=False, tickformat="%b %Y"),
    yaxis=dict(title="GMV ($M)", showgrid=True, gridcolor=BRAND_LIGHT),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig, width="stretch")

st.divider()

# --------------------------------------------------------------------------- #
# MAPE: baseline vs targeted across 5 models
# --------------------------------------------------------------------------- #


left, right = st.columns([3, 2])

with left:
    section(
        "MAPE progression across model versions",
        "Planted segment = Android Mid devices during iPhone launch months. "
        "Overall MAPE is nearly flat; planted MAPE drops sharply once launch "
        "features are added.",
    )

    try:
        summary = load_phase3_summary()
        summary = summary[summary["model_version"].isin(MODEL_ORDER)].copy()
        summary["_order"] = summary["model_version"].map({m: i for i, m in enumerate(MODEL_ORDER)})
        summary = summary.sort_values("_order")
        summary["label"] = summary["model_version"].map(MODEL_LABELS)

        fig2 = go.Figure()
        fig2.add_trace(
            go.Bar(
                x=summary["label"],
                y=summary["test_mape"] * 100,
                name="Overall MAPE",
                marker_color=BRAND_BLUE,
                text=[f"{v*100:.2f}%" for v in summary["test_mape"]],
                textposition="outside",
            )
        )
        fig2.add_trace(
            go.Bar(
                x=summary["label"],
                y=summary["planted_segment_mape"] * 100,
                name="Planted MAPE",
                marker_color=BRAND_RED,
                text=[f"{v*100:.1f}%" for v in summary["planted_segment_mape"]],
                textposition="outside",
            )
        )
        fig2.update_layout(
            **PLOTLY_LAYOUT,
            barmode="group",
            height=400,
            yaxis=dict(title="MAPE (%)", showgrid=True, gridcolor=BRAND_LIGHT, range=[0, 32]),
            xaxis=dict(title=None, showgrid=False, tickangle=-12),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig2, width="stretch")
    except FileNotFoundError:
        st.warning("Phase 3 summary missing. Run `make phase3-all`.")

with right:
    section("Headline story", None)
    with st.container(border=True):
        st.caption("BASELINE (PLANTED)")
        st.markdown(
            f"<h3 style='color: {BRAND_NAVY}; margin: 0;'>{fmt_pct(k['planted_base_mape'], 2)}</h3>",
            unsafe_allow_html=True,
        )
    with st.container(border=True):
        st.caption("TARGETED (PLANTED)")
        st.markdown(
            f"<h3 style='color: {BRAND_GREEN}; margin: 0;'>{fmt_pct(k['planted_tgt_mape'], 2)}</h3>",
            unsafe_allow_html=True,
        )
    with st.container(border=True):
        st.caption("RELATIVE REDUCTION")
        st.markdown(
            f"<h2 style='color: {BRAND_NAVY}; margin: 0;'>{fmt_pct(k['planted_relative_reduction'], 1)}</h2>",
            unsafe_allow_html=True,
        )
        over_target_pp = (k["planted_relative_reduction"] - 0.30) * 100
        st.caption(f"{over_target_pp:+.1f} pp vs 30% reference target")
    with st.container(border=True):
        st.caption("OVERALL MAPE (small by design)")
        st.markdown(
            f"<h3 style='color: {BRAND_NAVY}; margin: 0;'>"
            f"{fmt_pct(k['overall_base_mape'], 2)} -> {fmt_pct(k['overall_tgt_mape'], 2)}</h3>",
            unsafe_allow_html=True,
        )
        st.caption(
            "The fix concentrates where error concentrates. Population-wide "
            "improvements that move overall MAPE often hurt segment calibration."
        )

st.divider()

st.caption(
    "**Source:** 5 XGBoost model versions trained on 17,631 cleared offers "
    "(Oct 2024 to Jul 2025), validated on 1,670 (Aug 2025), tested on 14,374 "
    "(Sept 2025 to Mar 2026). Training deterministic under seed 20260420 with "
    "`nthread=1`. See `docs/PHASE3_TECHNICAL_NOTES.md`."
)
