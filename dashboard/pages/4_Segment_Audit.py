"""Page 4: Segment Audit.

Interactive version of the Phase 2 excess-error query. Dollar-weighted
ranking identifies which segments contribute the most pricing error.
Schema fixes: uses `pred_at` (not `predicted_at`) and outcome (not status).
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
    CLEARED_OUTCOMES,
    MODEL_LABELS,
    MODEL_ORDER,
    PLOTLY_LAYOUT,
    fmt_dollars,
    fmt_int,
    fmt_pct,
    load_parquet,
    load_phase3_predictions,
    page_header,
    section,
    setup_page,
    sidebar_brand,
)


setup_page("Segment Audit", icon="🔎")
sidebar_brand()
page_header(
    "Segment Audit",
    "Which segments contribute the most pricing error in absolute dollar "
    "terms? Segments are ranked by excess error (segment MAPE minus overall "
    "MAPE, weighted by volume and average price). This is the dollar-weighted "
    "prioritization a pricing team should use to pick the next fix.",
)


# --------------------------------------------------------------------------- #
# Data assembly with correct schema
# --------------------------------------------------------------------------- #


@st.cache_data(ttl=3600)
def load_audit_frame() -> pd.DataFrame:
    """Build a feature-rich offer frame. Uses `outcome` not `status`."""

    preds = load_phase3_predictions()
    offers = load_parquet("sku_offers")
    listings = load_parquet("sku_listings")
    skus = load_parquet("skus")
    devices = load_parquet("devices")

    df = preds[preds["target_type"] == "sku_offer"].copy()
    df = df[df["actual"] > 0]

    meta = (
        offers[["offer_id", "listing_id", "offer_at", "outcome"]]
        .merge(listings[["listing_id", "sku_id"]], on="listing_id")
        .merge(skus[["sku_id", "device_id", "condition_grade"]], on="sku_id")
        .merge(devices[["device_id", "device_category", "manufacturer"]], on="device_id")
    )
    meta["offer_at"] = pd.to_datetime(meta["offer_at"])
    meta["month"] = meta["offer_at"].dt.month
    meta["is_launch"] = meta["month"].isin([9, 10])
    meta["quarter"] = meta["offer_at"].dt.quarter.astype(str)
    meta["quarter"] = "Q" + meta["quarter"]

    df = df.merge(
        meta[["offer_id", "device_category", "manufacturer", "condition_grade",
              "is_launch", "quarter", "outcome"]],
        left_on="target_id", right_on="offer_id",
    )
    df["ape"] = (df["predicted"] - df["actual"]).abs() / df["actual"]
    df["bias"] = df["predicted"] - df["actual"]
    return df


audit = load_audit_frame()

available_models = [m for m in MODEL_ORDER if m in audit["model_version"].unique()]


# --------------------------------------------------------------------------- #
# Sidebar controls
# --------------------------------------------------------------------------- #


with st.sidebar:
    st.markdown("#### Audit controls")
    selected_model = st.selectbox(
        "Model to audit",
        options=available_models,
        index=available_models.index("baseline_v2_xgb") if "baseline_v2_xgb" in available_models else 0,
        format_func=lambda m: MODEL_LABELS.get(m, m),
    )
    segment_dims = st.multiselect(
        "Segment dimensions",
        options=["device_category", "condition_grade", "manufacturer", "is_launch", "quarter"],
        default=["device_category", "is_launch"],
        help="Combine 1 or more dimensions to form segments",
    )
    top_n = st.slider("Top N segments to display", min_value=5, max_value=30, value=10)


view = audit[audit["model_version"] == selected_model].copy()

if not segment_dims:
    st.info("Select at least one segment dimension in the sidebar to begin.")
    st.stop()


# --------------------------------------------------------------------------- #
# Overall headline
# --------------------------------------------------------------------------- #


overall_mape = float(view["ape"].mean())
overall_bias = float(view["bias"].mean())
total_err = float((view["predicted"] - view["actual"]).abs().sum())

kpi = st.columns(4)
kpi[0].metric("Model", MODEL_LABELS.get(selected_model, selected_model))
kpi[1].metric("Overall MAPE", fmt_pct(overall_mape, 2))
kpi[2].metric("Avg bias", fmt_dollars(overall_bias, 1))
kpi[3].metric("Total absolute error", fmt_dollars(total_err, 0))

st.divider()


# --------------------------------------------------------------------------- #
# Segment aggregation
# --------------------------------------------------------------------------- #


def _agg(g: pd.DataFrame) -> pd.Series:
    return pd.Series({
        "n_offers": len(g),
        "avg_clearing": float(g["actual"].mean()),
        "mape": float(g["ape"].mean()),
        "bias": float(g["bias"].mean()),
        "total_abs_err": float((g["predicted"] - g["actual"]).abs().sum()),
    })


# Compute per-segment aggregates using a modern groupby pattern that avoids
# pandas 2.2 deprecation warnings on `include_groups`. We compute a helper
# column for absolute error up front, then use named aggregation.
view["_abs_err"] = (view["predicted"] - view["actual"]).abs()
seg = (
    view.groupby(segment_dims, as_index=False)
    .agg(
        n_offers=("ape", "size"),
        avg_clearing=("actual", "mean"),
        mape=("ape", "mean"),
        bias=("bias", "mean"),
        total_abs_err=("_abs_err", "sum"),
    )
)
view.drop(columns=["_abs_err"], inplace=True)

seg["excess_mape_pp"] = (seg["mape"] - overall_mape) * 100
seg["excess_dollar_err"] = seg["excess_mape_pp"] / 100 * seg["avg_clearing"] * seg["n_offers"]
seg["segment"] = seg[segment_dims].astype(str).agg(" | ".join, axis=1)

top = seg.nlargest(top_n, "excess_dollar_err").sort_values("excess_dollar_err", ascending=True)


section(
    f"Top {top_n} segments by excess dollar error",
    "Red bars indicate above-baseline excess (priority to fix). Green bars are "
    "below-baseline segments where the model over-performs.",
)


fig = go.Figure(
    go.Bar(
        x=top["excess_dollar_err"],
        y=top["segment"],
        orientation="h",
        marker_color=[BRAND_RED if v > 0 else BRAND_GREEN for v in top["excess_dollar_err"]],
        text=[fmt_dollars(v, 0) for v in top["excess_dollar_err"]],
        textposition="outside",
        hovertemplate=(
            "<b>%{y}</b><br>Excess $ error: %{text}<br><extra></extra>"
        ),
    )
)
fig.update_layout(
    **PLOTLY_LAYOUT,
    height=max(300, top_n * 38),
    xaxis=dict(title="Excess dollar error vs overall MAPE ($)", showgrid=True, gridcolor=BRAND_LIGHT),
    yaxis=dict(title=None, showgrid=False),
)
st.plotly_chart(fig, width="stretch")

st.divider()


# --------------------------------------------------------------------------- #
# Full segment table
# --------------------------------------------------------------------------- #


section("Full segment breakdown", "Sortable table. Default sort: excess $ error descending.")

display = seg.sort_values("excess_dollar_err", ascending=False).copy()
display_df = pd.DataFrame({
    **{d: display[d] for d in segment_dims},
    "Offers": display["n_offers"].astype(int),
    "MAPE": display["mape"].apply(lambda v: fmt_pct(v, 2)),
    "Excess pp": display["excess_mape_pp"].apply(lambda v: f"{v:+.2f}"),
    "Avg $": display["avg_clearing"].apply(lambda v: fmt_dollars(v, 0)),
    "Excess $ error": display["excess_dollar_err"].apply(lambda v: fmt_dollars(v, 0)),
    "Mean bias $": display["bias"].apply(lambda v: fmt_dollars(v, 1)),
})
st.dataframe(display_df, width="stretch", hide_index=True)

st.divider()


# --------------------------------------------------------------------------- #
# Diagnostic tip
# --------------------------------------------------------------------------- #


section("How to read this audit", None)

st.markdown(
    """
- **Excess pp** is (segment MAPE − overall MAPE) in percentage points. Positive means the model is worse than average on that segment.
- **Excess $ error** multiplies excess pp by segment volume × average clearing price. This is the dollar value a perfect fix on that segment would recover.
- **Mean bias** shows directional error: positive means over-predicting, negative means under-predicting.
- The baseline XGBoost model concentrates most of its dollar error on `Android Mid × launch` (~$22–25K). That is exactly the segment the targeted model fixes, as seen on the Executive Summary page.
    """
)

st.caption(
    "This page replicates the logic from `src/resaleiq/sql/01_segment_mape_audit.sql`. "
    "The SQL version uses CTEs with LATERAL joins; the Python version above uses "
    "`groupby().apply()` for exploratory flexibility. Both produce identical rankings."
)
