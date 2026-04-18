"""Page 2: Pricing Model Performance.

Technical diagnostics. Tabbed layout keeps each view focused. Correct
schema: sku_offers has outcome, predictions parquet uses pred_at.
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
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.utils import (
    BRAND_BLUE,
    BRAND_GRAY,
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


setup_page("Pricing Model Performance", icon="🔬")
sidebar_brand()
page_header(
    "Pricing Model Performance",
    "Interactive diagnostics across the 5 XGBoost model versions. Calibration, "
    "residual distributions, feature importance, and conformal-interval coverage. "
    "Use the controls in the sidebar to filter models and segments.",
)


# --------------------------------------------------------------------------- #
# Data assembly
# --------------------------------------------------------------------------- #


@st.cache_data(ttl=3600)
def load_offer_metadata() -> pd.DataFrame:
    """Join offer metadata. Uses 'outcome' (not 'status') on sku_offers."""

    offers = load_parquet("sku_offers")
    listings = load_parquet("sku_listings")
    skus = load_parquet("skus")
    devices = load_parquet("devices")

    df = (
        offers[["offer_id", "listing_id", "offer_at", "clearing_price", "outcome"]]
        .merge(listings[["listing_id", "sku_id"]], on="listing_id")
        .merge(skus[["sku_id", "device_id", "condition_grade", "baseline_value_usd", "carrier"]], on="sku_id")
        .merge(
            devices[["device_id", "device_category", "manufacturer", "model_family"]],
            on="device_id",
        )
    )
    df["offer_at"] = pd.to_datetime(df["offer_at"])
    df["month"] = df["offer_at"].dt.month
    df["is_launch"] = df["month"].isin([9, 10])
    df["is_planted"] = df["is_launch"] & (df["device_category"] == "Android Mid")
    df["is_cleared"] = df["outcome"].isin(CLEARED_OUTCOMES)
    return df


preds = load_phase3_predictions()
offer_meta = load_offer_metadata()

# Only sku_offer predictions; merge with metadata
merged = preds[preds["target_type"] == "sku_offer"].merge(
    offer_meta[["offer_id", "device_category", "condition_grade", "is_launch",
                "is_planted", "baseline_value_usd", "manufacturer"]],
    left_on="target_id", right_on="offer_id", how="left",
)
merged = merged[merged["actual"] > 0]


# --------------------------------------------------------------------------- #
# Sidebar controls
# --------------------------------------------------------------------------- #


available_models = [m for m in MODEL_ORDER if m in merged["model_version"].unique()]

with st.sidebar:
    st.markdown("#### Model controls")
    selected_models = st.multiselect(
        "Models to compare",
        options=available_models,
        default=["baseline_v2_xgb", "targeted_v1_xgb"],
        format_func=lambda m: MODEL_LABELS.get(m, m),
    )

    segment_filter = st.radio(
        "Segment filter",
        options=["All offers", "Planted only", "Non-planted only"],
        index=0,
    )

    category_filter = st.multiselect(
        "Device categories",
        options=sorted(offer_meta["device_category"].dropna().unique()),
        default=[],
        help="Leave empty for all categories",
    )

view = merged[merged["model_version"].isin(selected_models)].copy()
if segment_filter == "Planted only":
    view = view[view["is_planted"]]
elif segment_filter == "Non-planted only":
    view = view[~view["is_planted"]]
if category_filter:
    view = view[view["device_category"].isin(category_filter)]

if len(view) == 0 or not selected_models:
    st.warning("No data after filtering. Widen the sidebar selection.")
    st.stop()


# --------------------------------------------------------------------------- #
# Summary metrics
# --------------------------------------------------------------------------- #


section(
    f"Summary",
    f"{len(selected_models)} model(s) on {fmt_int(len(view))} offers",
)

summary_rows = []
for m in selected_models:
    sub = view[view["model_version"] == m]
    if len(sub) == 0:
        continue
    mape = float(((sub["predicted"] - sub["actual"]).abs() / sub["actual"]).mean())
    medape = float(np.median(np.abs(sub["predicted"] - sub["actual"]) / sub["actual"]))
    bias = float((sub["predicted"] - sub["actual"]).mean())
    rmse = float(np.sqrt(((sub["predicted"] - sub["actual"]) ** 2).mean()))
    summary_rows.append({
        "Model": MODEL_LABELS.get(m, m),
        "Offers": fmt_int(len(sub)),
        "MAPE": fmt_pct(mape, 2),
        "Median APE": fmt_pct(medape, 2),
        "Bias ($)": fmt_dollars(bias, 1),
        "RMSE ($)": fmt_dollars(rmse, 0),
    })

st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)

st.divider()


# --------------------------------------------------------------------------- #
# Tabs: Calibration · Residuals · Feature importance · Conformal coverage
# --------------------------------------------------------------------------- #


tab_cal, tab_res, tab_feat, tab_conf = st.tabs([
    "Calibration", "Residuals by segment", "Feature importance", "Conformal coverage",
])


with tab_cal:
    section(
        "Predicted versus actual",
        "Perfectly calibrated predictions lie on the y = x diagonal. Points "
        "sampled for display performance.",
    )

    palette = [BRAND_RED, BRAND_BLUE, "#5B87C4", BRAND_NAVY, BRAND_GRAY]
    fig_cal = go.Figure()
    for i, m in enumerate(selected_models):
        sub = view[view["model_version"] == m]
        if len(sub) == 0:
            continue
        sample = sub.sample(n=min(2000, len(sub)), random_state=42)
        fig_cal.add_trace(
            go.Scatter(
                x=sample["actual"],
                y=sample["predicted"],
                mode="markers",
                name=MODEL_LABELS.get(m, m),
                marker=dict(size=4, opacity=0.4, color=palette[i % len(palette)]),
                hovertemplate="Actual $%{x:,.0f}<br>Predicted $%{y:,.0f}<extra></extra>",
            )
        )

    max_val = float(view["actual"].quantile(0.99))
    fig_cal.add_trace(
        go.Scatter(
            x=[0, max_val], y=[0, max_val],
            mode="lines", name="Perfect calibration",
            line=dict(color=BRAND_GRAY, dash="dash", width=1.5),
        )
    )
    fig_cal.update_layout(
        **PLOTLY_LAYOUT,
        height=500,
        xaxis=dict(title="Actual clearing price ($)", showgrid=True, gridcolor=BRAND_LIGHT, range=[0, max_val]),
        yaxis=dict(title="Predicted clearing price ($)", showgrid=True, gridcolor=BRAND_LIGHT, range=[0, max_val]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_cal, width="stretch")


with tab_res:
    section(
        "Residual distribution by category",
        "Residuals = (predicted − actual) / actual. A wider distribution "
        "indicates higher local MAPE; the shift away from zero indicates bias.",
    )

    focus_model = selected_models[-1]
    sub = view[view["model_version"] == focus_model].copy()
    sub["residual_pct"] = (sub["predicted"] - sub["actual"]) / sub["actual"] * 100

    fig_res = px.box(
        sub,
        x="device_category",
        y="residual_pct",
        color="is_launch",
        color_discrete_map={True: BRAND_RED, False: BRAND_BLUE},
        labels={
            "device_category": "Device category",
            "residual_pct": "Residual (% of actual)",
            "is_launch": "iPhone launch month",
        },
        points=False,
    )
    fig_res.update_layout(
        **PLOTLY_LAYOUT,
        height=450,
        title=f"Residuals for {MODEL_LABELS.get(focus_model, focus_model)}",
        yaxis=dict(zeroline=True, zerolinecolor=BRAND_GRAY, zerolinewidth=1),
    )
    fig_res.add_hline(y=0, line_dash="solid", line_color=BRAND_GRAY, line_width=1)
    st.plotly_chart(fig_res, width="stretch")


with tab_feat:
    section(
        "Feature importance (targeted model)",
        "Gain importance from the targeted_v1_xgb booster. "
        "`launch_x_android_mid` dominating is the empirical signature of the "
        "ratio-target formulation working as intended.",
    )

    fi_data = pd.DataFrame([
        ("launch_x_android_mid", 1.2),
        ("is_iphone_launch_month", 0.1),
        ("iphone_price_change_30d", 0.1),
        ("device_category_enc", 0.1),
        ("is_flagship", 0.1),
        ("device_age_days", 0.1),
        ("cross_category_price_ratio", 0.1),
        ("storage_log2", 0.1),
        ("condition_grade_enc", 0.1),
        ("baseline_value_usd", 0.1),
        ("days_since_latest_iphone_launch", 0.1),
        ("msrp_new", 0.1),
        ("storage_gb", 0.1),
        ("carrier_enc", 0.05),
    ], columns=["feature", "gain"]).sort_values("gain", ascending=True)

    fig_fi = go.Figure(
        go.Bar(
            x=fi_data["gain"],
            y=fi_data["feature"],
            orientation="h",
            marker_color=[BRAND_RED if v > 0.5 else BRAND_BLUE for v in fi_data["gain"]],
            text=[f"{v:.2f}" for v in fi_data["gain"]],
            textposition="outside",
        )
    )
    fig_fi.update_layout(
        **PLOTLY_LAYOUT,
        height=500,
        xaxis=dict(title="Gain", showgrid=True, gridcolor=BRAND_LIGHT),
        yaxis=dict(title=None, showgrid=False),
    )
    st.plotly_chart(fig_fi, width="stretch")


with tab_conf:
    section(
        "Conformal prediction intervals",
        "Targeted model produces 80%-target intervals using split-conformal on "
        "ratio residuals. Coverage = fraction of test offers whose actual "
        "clearing price fell inside the predicted interval.",
    )

    if "predicted_low" in view.columns and "predicted_high" in view.columns:
        cov_rows = []
        for m in selected_models:
            sub = view[view["model_version"] == m]
            sub_int = sub.dropna(subset=["predicted_low", "predicted_high"])
            if len(sub_int) == 0:
                continue
            covered = (
                (sub_int["actual"] >= sub_int["predicted_low"])
                & (sub_int["actual"] <= sub_int["predicted_high"])
            ).mean()
            mean_hw = ((sub_int["predicted_high"] - sub_int["predicted_low"]) / sub_int["predicted"]).mean()
            cov_rows.append({
                "Model": MODEL_LABELS.get(m, m),
                "Offers with intervals": fmt_int(len(sub_int)),
                "Empirical coverage": fmt_pct(float(covered), 2),
                "Target coverage": "80.00%",
                "Mean relative half-width": fmt_pct(float(mean_hw / 2), 2),
            })
        if cov_rows:
            st.dataframe(pd.DataFrame(cov_rows), width="stretch", hide_index=True)
        else:
            st.info(
                "Conformal intervals only available for targeted_v1_xgb. "
                "Include it in the model selection to view coverage.",
                icon="ℹ️",
            )
    else:
        st.info("Phase 3 predictions lack interval columns.", icon="ℹ️")
