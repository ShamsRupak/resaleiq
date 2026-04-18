"""ResaleIQ dashboard — landing page.

Uses Streamlit-native components (st.metric, st.container, st.columns) for
reliable, readable rendering. No custom HTML KPI cards.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the repo root is on sys.path so ``from dashboard.utils import ...``
# works when Streamlit runs this file directly (e.g. on Streamlit Community
# Cloud, which does not set PYTHONPATH). Locally, the Makefile sets
# PYTHONPATH=.:src and this block is a no-op.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st  # noqa: E402

from dashboard.utils import (  # noqa: E402
    counterfactual_summary,
    fmt_dollars,
    fmt_int,
    fmt_pct,
    setup_page,
    sidebar_brand,
)


setup_page("Overview", icon="📱")
sidebar_brand()

st.title("ResaleIQ")
st.markdown(
    "**Pricing, clearing, and lot-optimization analytics for a synthetic "
    "wholesale secondhand smartphone marketplace.**"
)
st.caption(
    "215,787 transactional rows across 18 months · 5 Phase 3 model versions + 1 lot-composition model · "
    "5 interactive dashboard views"
)

st.divider()

# ----- Headline metrics -----
try:
    cf = counterfactual_summary()
    planted = cf["planted"]
    relative_reduction = (
        (planted["mape_base"] - planted["mape_tgt"]) / planted["mape_base"]
        if planted["mape_base"] > 0
        else 0
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Planted MAPE reduction",
        fmt_pct(relative_reduction, 1),
        delta=f"from {fmt_pct(planted['mape_base'], 1)} baseline",
        delta_color="inverse",
    )
    c2.metric(
        "Dollar impact recovered",
        fmt_dollars(planted["total_dollars"], 0),
        delta=f"${planted['dollars_per_offer']:.2f} per offer",
    )
    c3.metric(
        "Planted offers in test",
        fmt_int(planted["n"]),
        delta="Android Mid × launch",
        delta_color="off",
    )
    c4.metric(
        "Overall MAPE (targeted)",
        fmt_pct(cf["all_offers"]["mape_tgt"], 2),
        delta=f"from {fmt_pct(cf['all_offers']['mape_base'], 2)} baseline",
        delta_color="inverse",
    )
    st.caption(
        "The headline reduction is concentrated on the high-volatility segment. "
        "Overall MAPE moved "
        f"{(cf['all_offers']['mape_base'] - cf['all_offers']['mape_tgt']) * 100:.2f} "
        "percentage points — small by design. Targeted feature engineering is "
        "measured by where error concentrates, not by population-wide averages."
    )
except FileNotFoundError:
    st.warning(
        "Headline metrics unavailable: run `make phase3-all` to generate "
        "Phase 3 artifacts.",
        icon="⚠️",
    )

st.divider()

# ----- Navigation cards -----
left, right = st.columns([3, 2])

with left:
    st.subheader("What's inside")
    st.markdown(
        "This dashboard walks through the four deliverables of a data scientist "
        "role: fluency in the data model, a pricing-accuracy improvement, an "
        "auction-lot recommender, and an experimentation cadence. Each page "
        "is a working artifact, not a mockup."
    )

    st.markdown("")

    with st.container(border=True):
        st.markdown("**📊 Executive Summary**")
        st.caption(
            "CEO-level platform metrics: GMV, clearing rates, headline MAPE "
            "reduction. Single-glance story."
        )

    with st.container(border=True):
        st.markdown("**🔬 Pricing Model Performance**")
        st.caption(
            "Interactive diagnostics across all 5 model versions. Calibration, "
            "residuals, feature importance, conformal coverage."
        )

    with st.container(border=True):
        st.markdown("**📦 Auction Lot Optimizer**")
        st.caption(
            "Given a bag of inventory, recommend a partition into auction lots "
            "that maximizes predicted total clearing value."
        )

    with st.container(border=True):
        st.markdown("**🔎 Segment Audit**")
        st.caption(
            "Interactive version of the Phase 2 excess-error query. Dollar-"
            "weighted ranking of which segments to fix next."
        )

    with st.container(border=True):
        st.markdown("**🧪 Experiment Tracker**")
        st.caption(
            "Shipped work as a real experiment record, plus three proposed "
            "next experiments with power analysis."
        )

with right:
    st.subheader("Methodology highlights")
    with st.container(border=True):
        st.markdown("**Ratio target, not absolute price**")
        st.caption(
            "Training XGBoost on `clearing_price / baseline_value` rather than "
            "absolute `clearing_price` makes multiplicative effects (like the "
            "20% iPhone launch depression on Android Mid) first-class signals."
        )

    with st.container(border=True):
        st.markdown("**Fixed-round deterministic training**")
        st.caption(
            "500 boosting rounds with `nthread=1` and no early stopping. "
            "Reproducible bit-for-bit across XGBoost versions."
        )

    with st.container(border=True):
        st.markdown("**Conformal prediction intervals**")
        st.caption(
            "Split-conformal on ratio residuals. Pricing model hits 80.3% on "
            "the held-out offer test set against an 80% target. Lot model "
            "holdout coverage is 66% because Feb-Mar lot residuals are wider "
            "than training-period residuals - a distribution-shift finding "
            "that rolling recalibration fixes in production."
        )

    with st.container(border=True):
        st.markdown("**Segment-first diagnostic discipline**")
        st.caption(
            "Excess-error ranking identifies dollar-weighted priority. No model "
            "changes until the audit surfaces where error concentrates."
        )

st.divider()

# ----- Stack footer -----
with st.container(border=False):
    st.caption(
        "**Stack:** Python 3.11 · pandas · NumPy · XGBoost 2.1 · PostgreSQL 16 · "
        "Docker · Streamlit · Plotly · 104 passing tests"
    )
    st.caption(
        "**Repository:** [github.com/ShamsRupak/resaleiq](https://github.com/ShamsRupak/resaleiq) · "
        "**Docs:** `docs/TECHNICAL_WRITEUP.md` · `docs/PHASE3_TECHNICAL_NOTES.md` · "
        "`docs/DASHBOARD_NOTES.md`"
    )
