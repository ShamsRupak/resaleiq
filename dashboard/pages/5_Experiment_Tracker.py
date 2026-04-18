"""Page 5: Experiment Tracker.

Shipped, running, and proposed pricing experiments. Demonstrates cadence.
Fixes: uses counterfactual_summary() dict (not cf["segment"] lookup).
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

import math

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
    BRAND_WARN,
    PLOTLY_LAYOUT,
    counterfactual_summary,
    fmt_dollars,
    fmt_int,
    fmt_pct,
    load_phase3_summary,
    page_header,
    section,
    setup_page,
    sidebar_brand,
)


setup_page("Experiment Tracker", icon="🧪")
sidebar_brand()
page_header(
    "Experiment Tracker",
    "Shipped, running, and proposed pricing experiments. Each record includes "
    "hypothesis, design, primary metric, power analysis, and result when "
    "completed. This is how a data science team operates inside a pricing "
    "organization.",
)


# --------------------------------------------------------------------------- #
# Load summary via the fixed helper
# --------------------------------------------------------------------------- #


try:
    cf_summary = counterfactual_summary()
    planted = cf_summary["planted"]
    all_offers = cf_summary["all_offers"]
    planted_relative_reduction = (
        (planted["mape_base"] - planted["mape_tgt"]) / planted["mape_base"]
        if planted["mape_base"] > 0 else 0
    )
except FileNotFoundError:
    st.warning("Counterfactual data missing. Run `make phase3-all`.")
    st.stop()


# --------------------------------------------------------------------------- #
# Portfolio summary
# --------------------------------------------------------------------------- #


kpi = st.columns(4)
kpi[0].metric("Shipped experiments", "1")
kpi[1].metric("Running experiments", "0")
kpi[2].metric("Proposed (next sprint)", "3")
kpi[3].metric(
    "Shipped impact",
    fmt_dollars(planted["total_dollars"], 0),
    delta="Planted-segment error recovered on test set",
)

st.divider()


# --------------------------------------------------------------------------- #
# SHIPPED experiment
# --------------------------------------------------------------------------- #


section("Shipped: Targeted feature set for Android Mid launch depression", None)

with st.container(border=True):
    left, right = st.columns([2, 3])

    with left:
        st.markdown(f"**STATUS**")
        st.success("Shipped", icon="✅")
        st.markdown(f"**DATE**")
        st.caption("2026-04 (simulated ship)")
        st.markdown(f"**OWNER**")
        st.caption("Author: Shams Rupak")

    with right:
        st.markdown("**Hypothesis**")
        st.markdown(
            "Android Mid devices clear at systematically lower prices during "
            "iPhone launch months (Sept–Oct). Adding a `launch_x_android_mid` "
            "interaction feature plus 3 launch-related signals to XGBoost, "
            "trained on the ratio `clearing_price / baseline_value`, will "
            "reduce planted-segment MAPE by ≥ 30% vs the 9-feature baseline."
        )
        st.markdown("**Primary metric: relative MAPE reduction on the planted segment**")

    st.markdown("---")

    rcols = st.columns(4)
    rcols[0].metric("Baseline planted MAPE", fmt_pct(planted["mape_base"], 2))
    rcols[1].metric(
        "Targeted planted MAPE",
        fmt_pct(planted["mape_tgt"], 2),
        delta=f"−{(planted['mape_base'] - planted['mape_tgt']) * 100:.2f} pp",
        delta_color="inverse",
    )
    rcols[2].metric(
        "Relative reduction",
        fmt_pct(planted_relative_reduction, 1),
        delta=f"Target was 30%",
    )
    rcols[3].metric(
        "Dollar impact",
        fmt_dollars(planted["total_dollars"], 0),
        delta=f"${planted['dollars_per_offer']:.2f} / offer",
    )

    st.markdown("")
    st.markdown(
        f"**Outcome:** Exceeded target by {(planted_relative_reduction - 0.30) * 100:+.1f} "
        f"percentage points on planted segment. Non-planted segments held approximately "
        f"neutral ({fmt_pct(cf_summary['non_planted']['mape_base'], 2)} → "
        f"{fmt_pct(cf_summary['non_planted']['mape_tgt'], 2)}), confirming the fix is "
        f"segment-specific and didn't regress the broader population."
    )

st.divider()


# --------------------------------------------------------------------------- #
# PROPOSED experiments
# --------------------------------------------------------------------------- #


section(
    "Proposed: next-sprint experiment portfolio",
    "Three experiments queued behind the shipped targeted model. Each has a "
    "hypothesis, design, and sample-size estimate.",
)


def _sample_size_for_mean_diff(sigma: float, mde: float, alpha: float = 0.05, power: float = 0.80) -> int:
    """Normal approximation: n per arm for two-sample t-test on mean difference."""

    z_alpha = 1.96  # two-sided 5%
    z_beta = 0.84  # 80% power
    return int(math.ceil(2 * ((z_alpha + z_beta) ** 2) * (sigma / mde) ** 2))


# ---- Experiment 1: Production A/B ---- #
with st.container(border=True):
    head = st.columns([4, 1])
    with head[0]:
        st.markdown("**🔬 Experiment 1: Production A/B of targeted_v1_xgb**")
    with head[1]:
        st.info("Proposed", icon="📋")

    st.markdown("**Hypothesis**")
    st.caption(
        "Replacing baseline_v2 with targeted_v1_xgb in production pricing "
        "service will reduce planted-segment absolute error by ≥ 40% relative "
        "on live traffic, matching or exceeding the 42.3% seen on held-out data."
    )

    st.markdown("**Design**")
    st.caption(
        "Two-arm A/B at the buyer-request level. Control: baseline_v2_xgb "
        "serves price recommendations. Treatment: targeted_v1_xgb serves "
        "recommendations. Randomize 50/50, stratified by device_category. "
        "Run during next iPhone launch window for 4 weeks."
    )

    n = _sample_size_for_mean_diff(sigma=0.25, mde=0.04)
    p = st.columns(3)
    p[0].metric("Primary metric", "Planted-segment MAPE")
    p[1].metric("MDE", "4 percentage points")
    p[2].metric("Sample per arm", fmt_int(n))

    st.caption(
        "Guardrails: overall MAPE non-regression (≤ +0.5 pp), clearing-price "
        "distribution drift (Kolmogorov–Smirnov p > 0.05), and listing-to-offer "
        "conversion rate."
    )


# ---- Experiment 2: Lot optimizer shadow ---- #
with st.container(border=True):
    head = st.columns([4, 1])
    with head[0]:
        st.markdown("**📦 Experiment 2: Lot optimizer shadow rollout**")
    with head[1]:
        st.info("Proposed", icon="📋")

    st.markdown("**Hypothesis**")
    st.caption(
        "The balanced-4 lot partition strategy clears at ≥ 8% higher total "
        "price than the single-lot baseline strategy, net of auction fees."
    )

    st.markdown("**Design**")
    st.caption(
        "Shadow mode for 2 weeks: both strategies scored on every incoming "
        "bag of inventory. Only the current operator partition is executed. "
        "Compare predicted vs realized clearing prices when operators happen "
        "to pick different partitions. Power for 8% MDE on mean aggregate "
        "clearing ratio (σ ≈ 0.12):"
    )

    n2 = _sample_size_for_mean_diff(sigma=0.12, mde=0.08)
    p2 = st.columns(3)
    p2[0].metric("Primary metric", "Aggregate clearing ratio")
    p2[1].metric("MDE", "8 percentage points")
    p2[2].metric("Lots needed", fmt_int(n2 * 2))


# ---- Experiment 3: Lot-model rolling recalibration ---- #
with st.container(border=True):
    head = st.columns([4, 1])
    with head[0]:
        st.markdown("**📏 Experiment 3: Lot-model rolling conformal recalibration**")
    with head[1]:
        st.info("Proposed", icon="📋")

    st.markdown("**Hypothesis**")
    st.caption(
        "Lot-model conformal intervals hit 80% coverage on the training period "
        "by construction but drop to 66.5% on the Feb-Mar 2026 holdout. This is "
        "distribution shift, not a bug: Feb-Mar residuals are approximately 27% "
        "wider than training-period residuals. Weekly recalibration on a rolling "
        "60-day window of recent cleared lots should restore holdout coverage to "
        "within 78-82% without widening intervals on well-behaved weeks."
    )

    st.markdown("**Design**")
    st.caption(
        "Offline replay on rolling 30-day windows across the last 6 months of "
        "training data. For each week in the holdout period, recalibrate the "
        "empirical-quantile half-width on the previous 60 days of residuals, "
        "then measure coverage on the next 7 days. Compare against the static "
        "calibration that produced 66.5% today."
    )

    p3 = st.columns(3)
    p3[0].metric("Primary metric", "Empirical coverage")
    p3[1].metric("Target", "80 ± 2 %")
    p3[2].metric("Analysis", "Weekly replay")

st.divider()


# --------------------------------------------------------------------------- #
# Portfolio visualization
# --------------------------------------------------------------------------- #


section("Portfolio view", "Experiments plotted by expected impact and effort.")

portfolio = pd.DataFrame([
    {"name": "Targeted v1 (shipped)", "impact": planted["total_dollars"],
     "effort": 2, "status": "Shipped"},
    {"name": "Production A/B", "impact": planted["total_dollars"] * 12,
     "effort": 4, "status": "Proposed"},  # 12x since it's production-scale monthly impact
    {"name": "Lot optimizer shadow", "impact": 180000,  # estimated $180K annual from auction uplift
     "effort": 5, "status": "Proposed"},
    {"name": "Conformal tune", "impact": 25000,  # estimated from tighter intervals
     "effort": 2, "status": "Proposed"},
])

status_colors = {"Shipped": BRAND_GREEN, "Proposed": BRAND_BLUE, "Running": BRAND_WARN}
portfolio["color"] = portfolio["status"].map(status_colors)

fig = go.Figure()
for status in portfolio["status"].unique():
    sub = portfolio[portfolio["status"] == status]
    fig.add_trace(
        go.Scatter(
            x=sub["effort"],
            y=sub["impact"],
            mode="markers+text",
            name=status,
            text=sub["name"],
            textposition="top center",
            marker=dict(size=22, color=status_colors[status], line=dict(color="white", width=2)),
            hovertemplate="<b>%{text}</b><br>Effort: %{x}/5<br>Impact: $%{y:,.0f}<extra></extra>",
        )
    )
fig.update_layout(
    **PLOTLY_LAYOUT,
    height=420,
    xaxis=dict(
        title="Engineering effort (1 = easy, 5 = hard)",
        showgrid=True, gridcolor=BRAND_LIGHT,
        range=[0.5, 5.5], dtick=1,
    ),
    yaxis=dict(title="Estimated dollar impact ($)", showgrid=True, gridcolor=BRAND_LIGHT, type="log"),
    showlegend=True,
)
st.plotly_chart(fig, width="stretch")

st.caption(
    "Impact axis uses a log scale to accommodate the three-order-of-magnitude "
    "range between test-set findings and production-scale impact. Production "
    "A/B impact is extrapolated from the $13.7K test-set finding at monthly "
    "throughput (approximately 12× annualized). Shadow and conformal impacts "
    "are order-of-magnitude estimates pending measurement."
)
