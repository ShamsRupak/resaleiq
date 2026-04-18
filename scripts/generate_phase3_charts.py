"""Generate the three Phase 3 verification charts.

    docs/phase3_mape_comparison.png   MAPE by segment across all 5 models
    docs/phase3_ablation.png          Per-feature MAPE contribution
    docs/phase3_conformal.png         Conformal interval widths vs actual errors
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from resaleiq.config import DATA_DIR
from resaleiq.data_generation.market_dynamics import is_iphone_launch_month

BRAND_NAVY = "#0d1b2a"
BRAND_ACCENT = "#1d9bf0"
BRAND_GREEN = "#2a9d8f"
BRAND_RED = "#dc2626"
BRAND_GRAY = "#6b7280"
DOCS_DIR = Path(__file__).resolve().parents[1] / "docs"


def _load_predictions_with_segments() -> pd.DataFrame:
    preds = pd.read_parquet(DATA_DIR / "phase3_predictions.parquet")
    sku_offers = pd.read_parquet(DATA_DIR / "sku_offers.parquet")
    sku_listings = pd.read_parquet(DATA_DIR / "sku_listings.parquet")
    skus = pd.read_parquet(DATA_DIR / "skus.parquet")
    devices = pd.read_parquet(DATA_DIR / "devices.parquet")
    offers = (
        sku_offers.dropna(subset=["clearing_price"])[["offer_id", "listing_id", "offer_at"]]
        .merge(sku_listings[["listing_id", "sku_id"]], on="listing_id")
        .merge(skus[["sku_id", "device_id"]], on="sku_id")
        .merge(devices[["device_id", "device_category"]], on="device_id")
    )
    offers["offer_at"] = pd.to_datetime(offers["offer_at"])
    offers["is_launch"] = offers["offer_at"].apply(
        lambda ts: is_iphone_launch_month(ts.date())
    )
    offers["is_planted"] = (offers["device_category"] == "Android Mid") & offers["is_launch"]
    preds = preds.merge(
        offers[["offer_id", "device_category", "is_launch", "is_planted"]],
        left_on="target_id",
        right_on="offer_id",
    )
    preds["ape"] = (preds["predicted"] - preds["actual"]).abs() / preds["actual"]
    return preds


def mape_comparison_chart(preds: pd.DataFrame) -> None:
    """Bar chart: overall MAPE vs planted-segment MAPE across models."""

    model_order = [
        "baseline_v2_xgb",
        "plus_launch_xgb",
        "plus_launch_days_xgb",
        "plus_launch_days_price_xgb",
        "targeted_v1_xgb",
    ]
    short_labels = [
        "baseline\n(9 static feats)",
        "+is_launch",
        "+days_since",
        "+price_change",
        "targeted\n(+ratio + interaction)",
    ]

    overall = preds.groupby("model_version")["ape"].mean().to_dict()
    planted = (
        preds[preds["is_planted"]].groupby("model_version")["ape"].mean().to_dict()
    )
    overall_vals = [overall[m] * 100 for m in model_order]
    planted_vals = [planted[m] * 100 for m in model_order]

    x = np.arange(len(model_order))
    width = 0.38

    fig, ax = plt.subplots(1, 1, figsize=(11, 5.8))
    bar_overall = ax.bar(
        x - width / 2, overall_vals, width, color=BRAND_NAVY, label="All test offers",
    )
    bar_planted = ax.bar(
        x + width / 2, planted_vals, width, color=BRAND_ACCENT, label="Planted segment only",
    )
    for bar, val in zip(bar_overall, overall_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.2, f"{val:.2f}%",
                ha="center", va="bottom", fontsize=9, color=BRAND_NAVY, fontweight="bold")
    for bar, val in zip(bar_planted, planted_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.2, f"{val:.2f}%",
                ha="center", va="bottom", fontsize=9, color=BRAND_ACCENT, fontweight="bold")

    # Theoretical MAPE floor for planted segment. With generator sigma=0.08
    # and mean=-0.20, E[|N(-0.20, 0.08)|]/0.80 ~ 0.105. Compound with base
    # noise of 0.105 gives ~0.13 effective floor in practice.
    ax.axhline(13.0, color=BRAND_RED, linestyle="--", linewidth=1.2, alpha=0.7)
    ax.text(
        len(model_order) - 0.5, 13.4, "planted noise floor (~13%)",
        ha="right", color=BRAND_RED, fontsize=9, style="italic",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=9)
    ax.set_ylabel("MAPE (%)")
    ax.set_title(
        "MAPE by feature ladder: progressive improvement toward the noise floor\n"
        "Test set: Sept 2025 – Mar 2026 (includes 2nd iPhone launch cycle)"
    )
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25, axis="y")
    ax.set_ylim(0, max(planted_vals) * 1.15)
    plt.tight_layout()
    out = DOCS_DIR / "phase3_mape_comparison.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"wrote {out}")


def ablation_chart(preds: pd.DataFrame) -> None:
    """Horizontal bar chart: per-feature marginal MAPE contribution."""

    model_order = [
        "baseline_v2_xgb",
        "plus_launch_xgb",
        "plus_launch_days_xgb",
        "plus_launch_days_price_xgb",
        "targeted_v1_xgb",
    ]
    planted = (
        preds[preds["is_planted"]].groupby("model_version")["ape"].mean().to_dict()
    )
    overall = preds.groupby("model_version")["ape"].mean().to_dict()

    feature_labels = [
        "is_iphone_launch_month",
        "days_since_latest_iphone_launch",
        "iphone_price_change_30d",
        "cross_category_price_ratio + launch_x_android_mid",
    ]
    marginal_overall = [
        (overall[model_order[i]] - overall[model_order[i + 1]]) * 100
        for i in range(len(model_order) - 1)
    ]
    marginal_planted = [
        (planted[model_order[i]] - planted[model_order[i + 1]]) * 100
        for i in range(len(model_order) - 1)
    ]

    y = np.arange(len(feature_labels))
    height = 0.38

    fig, ax = plt.subplots(1, 1, figsize=(11, 5.2))
    ax.barh(y - height / 2, marginal_overall, height, color=BRAND_NAVY, label="Overall MAPE reduction")
    ax.barh(y + height / 2, marginal_planted, height, color=BRAND_ACCENT, label="Planted-segment MAPE reduction")

    for i, (mo, mp) in enumerate(zip(marginal_overall, marginal_planted)):
        ax.text(mo + (0.02 if mo >= 0 else -0.02), i - height / 2,
                f"{mo:+.2f}pp", va="center",
                ha="left" if mo >= 0 else "right", fontsize=9,
                color=BRAND_NAVY, fontweight="bold")
        ax.text(mp + (0.02 if mp >= 0 else -0.02), i + height / 2,
                f"{mp:+.2f}pp", va="center",
                ha="left" if mp >= 0 else "right", fontsize=9,
                color=BRAND_ACCENT, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(feature_labels, fontsize=9)
    ax.set_xlabel("Marginal MAPE reduction vs previous feature level (percentage points)")
    ax.set_title(
        "Feature ablation: MAPE contribution of each cross-brand feature\n"
        "Negative values mean the feature *increased* MAPE (variance-inflating)"
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.25, axis="x")
    plt.tight_layout()
    out = DOCS_DIR / "phase3_ablation.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"wrote {out}")


def conformal_chart(preds: pd.DataFrame) -> None:
    """Show the conformal interval widths vs the actual errors.

    Proves empirical coverage matches the target level and the intervals are
    not grossly over- or under-wide.
    """

    tgt = preds[preds["model_version"] == "targeted_v1_xgb"].copy()
    tgt = tgt.dropna(subset=["predicted_low", "predicted_high"])
    tgt["err"] = tgt["predicted"] - tgt["actual"]
    tgt["in_interval"] = (tgt["actual"] >= tgt["predicted_low"]) & (tgt["actual"] <= tgt["predicted_high"])
    coverage = float(tgt["in_interval"].mean())
    half_width = float((tgt["predicted_high"] - tgt["predicted_low"]).mean() / 2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: scatter of predicted vs actual with intervals.
    sample = tgt.sample(n=min(1500, len(tgt)), random_state=20260420)
    ax1.errorbar(
        sample["predicted"], sample["actual"],
        yerr=[sample["predicted"] - sample["predicted_low"],
              sample["predicted_high"] - sample["predicted"]],
        fmt="o", markersize=2.5, alpha=0.35, color=BRAND_NAVY,
        ecolor=BRAND_ACCENT, elinewidth=0.5, capsize=0,
    )
    lim = max(sample["predicted"].max(), sample["actual"].max()) * 1.05
    ax1.plot([0, lim], [0, lim], color=BRAND_RED, linestyle="--", linewidth=1.2, alpha=0.7)
    ax1.set_xlabel("Predicted clearing price ($)")
    ax1.set_ylabel("Actual clearing price ($)")
    ax1.set_title(
        f"Conformal prediction intervals (80% target)\n"
        f"Empirical coverage: {coverage:.1%}   half-width: ${half_width:.2f}"
    )
    ax1.grid(True, alpha=0.25)
    ax1.set_xlim(0, lim)
    ax1.set_ylim(0, lim)

    # Right: histogram of errors with half-width markers.
    ax2.hist(
        tgt["err"], bins=60, color=BRAND_NAVY, edgecolor="white", linewidth=0.4, alpha=0.8,
    )
    ax2.axvline(half_width, color=BRAND_ACCENT, linestyle="--", linewidth=1.3, label=f"+{half_width:.0f}")
    ax2.axvline(-half_width, color=BRAND_ACCENT, linestyle="--", linewidth=1.3, label=f"-{half_width:.0f}")
    ax2.axvline(0, color=BRAND_RED, linewidth=1.0, alpha=0.6)
    ax2.set_xlabel("Prediction error: predicted minus actual ($)")
    ax2.set_ylabel("Number of offers")
    ax2.set_title(
        "Error distribution with conformal half-width boundaries\n"
        f"80% of offers should fall within ±${half_width:.0f}"
    )
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.25)

    plt.tight_layout()
    out = DOCS_DIR / "phase3_conformal.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"wrote {out}")


def main() -> None:
    preds = _load_predictions_with_segments()
    mape_comparison_chart(preds)
    ablation_chart(preds)
    conformal_chart(preds)


if __name__ == "__main__":
    main()
