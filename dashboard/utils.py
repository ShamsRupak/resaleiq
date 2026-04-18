"""Shared utilities for the ResaleIQ Streamlit dashboard.

Centralises database access, brand palette, cached loaders, and formatters.
Prefers Streamlit-native components over custom HTML for reliable rendering.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------- #
# Brand palette (used for Plotly charts; global theme is in .streamlit/config.toml)
# --------------------------------------------------------------------------- #

BRAND_NAVY = "#0B1F3A"
BRAND_BLUE = "#1A4B8C"
BRAND_LIGHT_BLUE = "#5B87C4"
BRAND_LIGHT = "#F4F6FA"
BRAND_RED = "#C1292E"
BRAND_GREEN = "#1F7A3A"
BRAND_GRAY = "#5A6472"
BRAND_WARN = "#D98E04"
BRAND_SURFACE = "#FFFFFF"
BRAND_BORDER = "#D5DCE5"

SEGMENT_COLORS = {
    "Android Mid": "#1A4B8C",
    "Apple Flagship": "#0B1F3A",
    "Apple Mid": "#5B87C4",
    "Android Flagship": "#1F7A3A",
    "Android Budget": "#D98E04",
    "Other": "#8A94A6",
}

MODEL_ORDER = [
    "baseline_v1",
    "baseline_v2_xgb",
    "plus_launch_xgb",
    "plus_launch_days_xgb",
    "plus_launch_days_price_xgb",
    "targeted_v1_xgb",
]

MODEL_LABELS = {
    "baseline_v1": "Synthetic baseline v1",
    "baseline_v2_xgb": "XGBoost baseline",
    "plus_launch_xgb": "+ launch month",
    "plus_launch_days_xgb": "+ days since launch",
    "plus_launch_days_price_xgb": "+ price change",
    "targeted_v1_xgb": "Targeted (14 features)",
}

CLEARED_OUTCOMES = ("accepted", "countered_accepted")

PLOTLY_LAYOUT = dict(
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(color=BRAND_NAVY, family="sans-serif"),
    margin=dict(l=10, r=10, t=40, b=10),
    hoverlabel=dict(bgcolor="white", bordercolor=BRAND_BORDER, font_size=12),
)


# --------------------------------------------------------------------------- #
# Data root
# --------------------------------------------------------------------------- #


def _repo_root() -> Path:
    here = Path(__file__).resolve().parent
    return here.parent


def _data_dir() -> Path:
    return _repo_root() / "data"


# --------------------------------------------------------------------------- #
# Postgres connection
# --------------------------------------------------------------------------- #


def _pg_config() -> Optional[dict[str, str]]:
    """Return connection config, or None if user has explicitly disabled."""

    if os.environ.get("RESALEIQ_PG_HOST") == "__disabled__":
        return None

    env_map = {
        "host": os.environ.get("RESALEIQ_PG_HOST"),
        "port": os.environ.get("RESALEIQ_PG_PORT"),
        "user": os.environ.get("RESALEIQ_PG_USER"),
        "password": os.environ.get("RESALEIQ_PG_PASSWORD"),
        "dbname": os.environ.get("RESALEIQ_PG_DB"),
    }
    if all(env_map.values()):
        return {k: str(v) for k, v in env_map.items()}

    try:
        secrets = st.secrets.get("postgres", {})  # type: ignore[attr-defined]
        if secrets:
            return {
                "host": secrets.get("host", "localhost"),
                "port": str(secrets.get("port", 5432)),
                "user": secrets.get("user", "resaleiq"),
                "password": secrets.get("password", "resaleiq"),
                "dbname": secrets.get("dbname", "resaleiq"),
            }
    except Exception:
        pass

    return {
        "host": "localhost",
        "port": "5432",
        "user": "resaleiq",
        "password": "resaleiq",
        "dbname": "resaleiq",
    }


@st.cache_resource
def get_pg_engine():
    """Return a SQLAlchemy engine or None if unavailable."""

    cfg = _pg_config()
    if cfg is None:
        return None
    try:
        from sqlalchemy import create_engine, text

        uri = (
            f"postgresql+psycopg://{cfg['user']}:{cfg['password']}"
            f"@{cfg['host']}:{cfg['port']}/{cfg['dbname']}"
        )
        engine = create_engine(uri, connect_args={"connect_timeout": 3})
        with engine.connect() as c:
            c.execute(text("SELECT 1"))
        return engine
    except Exception as e:
        print(f"[dashboard] Postgres unreachable: {e}. Falling back to parquet.")
        return None


@st.cache_data(ttl=600)
def run_sql(query: str, params: Optional[dict] = None) -> pd.DataFrame:
    """Execute a query against Postgres, return a DataFrame."""

    engine = get_pg_engine()
    if engine is None:
        raise RuntimeError("Postgres is not reachable; use parquet fallbacks.")
    return pd.read_sql(query, engine, params=params or {})


# --------------------------------------------------------------------------- #
# Parquet loaders
# --------------------------------------------------------------------------- #


@st.cache_data(ttl=3600)
def load_parquet(name: str) -> pd.DataFrame:
    """Load a parquet file from data/ by stem name."""

    path = _data_dir() / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing: {path}. Run `make generate` and `make phase3-all` first."
        )
    return pd.read_parquet(path)


@st.cache_data(ttl=3600)
def load_phase3_predictions() -> pd.DataFrame:
    return load_parquet("phase3_predictions")


@st.cache_data(ttl=3600)
def load_phase3_summary() -> pd.DataFrame:
    return load_parquet("phase3_summary")


@st.cache_data(ttl=3600)
def load_counterfactual() -> pd.DataFrame:
    """Per-offer counterfactual table (not pre-aggregated)."""

    return load_parquet("phase3_counterfactual")


@st.cache_data(ttl=3600)
def counterfactual_summary() -> dict:
    """Aggregate the per-offer counterfactual into segment-level stats."""

    cf = load_counterfactual()
    planted = cf[cf["is_planted"]]
    non_planted = cf[~cf["is_planted"]]

    def _mape(df: pd.DataFrame, model_col: str) -> float:
        if len(df) == 0:
            return 0.0
        mask = df["actual"] > 0
        return float((df.loc[mask, f"abs_err_{model_col}"] / df.loc[mask, "actual"]).mean())

    def _totals(df: pd.DataFrame) -> dict:
        return {
            "n": int(len(df)),
            "avg_clearing": float(df["actual"].mean()) if len(df) else 0.0,
            "mape_base": _mape(df, "baseline"),
            "mape_tgt": _mape(df, "targeted"),
            "dollars_per_offer": float(df["err_reduction"].mean()) if len(df) else 0.0,
            "total_dollars": float(df["err_reduction"].sum()) if len(df) else 0.0,
        }

    return {
        "planted": _totals(planted),
        "all_offers": _totals(cf),
        "non_planted": _totals(non_planted),
    }


# --------------------------------------------------------------------------- #
# Schema helpers: sku_offers uses `outcome`, not `status`
# --------------------------------------------------------------------------- #


def filter_cleared_offers(df: pd.DataFrame) -> pd.DataFrame:
    """Return only rows representing cleared SKU offers.

    An offer is cleared when the outcome is accepted or countered_accepted
    AND the clearing price is positive.
    """

    mask = df["outcome"].isin(CLEARED_OUTCOMES) & (df["clearing_price"].fillna(0) > 0)
    return df[mask].copy()


def filter_cleared_lots(df: pd.DataFrame) -> pd.DataFrame:
    """Return only rows representing cleared auction lots."""

    mask = (df["status"] == "cleared") & (df["clearing_price"].fillna(0) > 0)
    return df[mask].copy()


# --------------------------------------------------------------------------- #
# Formatters
# --------------------------------------------------------------------------- #


def fmt_pct(x: float, digits: int = 2) -> str:
    if pd.isna(x):
        return "-"
    return f"{x * 100:.{digits}f}%"


def fmt_dollars(x: float, digits: int = 0) -> str:
    if pd.isna(x):
        return "-"
    if abs(x) >= 1e9:
        return f"${x / 1e9:.{digits}f}B"
    if abs(x) >= 1e6:
        return f"${x / 1e6:.{digits}f}M"
    if abs(x) >= 1e3:
        return f"${x / 1e3:,.{digits}f}K"
    return f"${x:,.{digits}f}"


def fmt_int(x) -> str:
    if pd.isna(x):
        return "-"
    return f"{int(x):,}"


# --------------------------------------------------------------------------- #
# Page setup helpers
# --------------------------------------------------------------------------- #


def setup_page(title: str, icon: str = "📊") -> None:
    """Apply consistent page config."""

    st.set_page_config(
        page_title=f"ResaleIQ · {title}",
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )


def page_header(title: str, subtitle: str) -> None:
    """Standard page header."""

    st.markdown(f"## {title}")
    st.caption(subtitle)
    st.divider()


def sidebar_brand() -> None:
    """Consistent sidebar branding across pages."""

    st.sidebar.markdown("### ResaleIQ")
    engine = get_pg_engine()
    if engine is not None:
        st.sidebar.success("Live database", icon="🟢")
    else:
        st.sidebar.warning("Parquet mode", icon="📁")
    st.sidebar.caption("Synthetic wholesale smartphone marketplace")
    st.sidebar.link_button(
        "View on GitHub",
        "https://github.com/ShamsRupak/resaleiq",
        width="stretch",
    )
    st.sidebar.divider()


def section(title: str, subtitle: Optional[str] = None) -> None:
    """Section header inside a page."""

    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)
