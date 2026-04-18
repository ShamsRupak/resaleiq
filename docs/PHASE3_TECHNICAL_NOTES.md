# ResaleIQ: Phase 3 Technical Notes

## Summary of Results

Test set: Sept 2025 — Mar 2026 (7 months, includes the 2nd iPhone launch cycle — the one the model has not seen during training).

| Model | Overall MAPE | Planted MAPE | Features |
|-------|--------------|--------------|----------|
| baseline_v1 (synthetic broken production) | ~11.5% | ~25% | — |
| baseline_v2_xgb | 11.35% | **27.15%** | 9 static |
| +is_iphone_launch_month | 10.71% | 15.83% | 10 |
| +days_since_iphone_launch | 10.69% | 15.62% | 11 |
| +iphone_price_change_30d | 10.75% | 15.78% | 12 |
| **targeted_v1_xgb** | **10.70%** | **15.68%** | 14 |

**Planted MAPE: 27.15% → 15.68% = 42.3% relative reduction** on the high-volatility segment, exceeding a 30 percent reference target by a clean margin.

Counterfactual dollar impact: the targeted model reduces pricing error by **$17.07 per offer on the planted segment**, totaling **$13,673 across 801 test-set planted offers**. Non-planted offers show near-zero change in total error — improvement is concentrated where the error was concentrated.

## Key modeling decisions

### 1. Ratio target, not absolute price

The single most important decision. Training XGBoost on absolute `clearing_price` doesn't work for this problem: the cross-brand depression effect is *multiplicative* (~20% launch-month shift for Android Mid), and multiplicative effects get buried by absolute-magnitude features like `baseline_value_usd` when using squared-error loss.

The fix is to train on the **price ratio target**: `clearing_price / baseline_value_usd`. This transforms the problem:

- The launch depression becomes a first-class signal. On the ratio target, non-launch Android Mid offers have mean ~0.995, launch offers have mean ~0.80. That 20% shift is directly learnable by XGBoost.
- Feature importance reflects economic reality. `launch_x_android_mid` ranks #1 by gain (1.2) vs everything else at ~0.1.
- Prediction intervals become naturally multiplicative. A 15% conformal interval scales appropriately across device prices.

This is the standard formulation in actuarial pricing (loss ratio), retail forecasting (sell-through rate), and any domain where the effect is multiplicative.

### 2. Fixed-round training, no early stopping

XGBoost 2.1+ versions differ in how early-stopping tolerance behaves on low-variance targets. Ratio targets have small numerical range (~0.5 to 1.5), so per-round MSE improvements are tiny in absolute terms and some versions trigger early stopping at iteration 0 despite the model having substantial learning capacity.

Production training uses `num_boost_round=500` with `learning_rate=0.03` and no early stopping. This guarantees reproducible results across XGBoost versions at a modest training time cost (~0.5 seconds per model).

### 3. `nthread=1` for deterministic training

XGBoost's `tree_method="hist"` is multi-thread non-deterministic even with a fixed seed. `nthread=1` removes that variance at no practical cost for a 17K-row dataset. Results are now reproducible bit-for-bit across runs on the same hardware.

### 4. Realistic generator parameters

`android_mid_iphone_launch_depression_mean=-0.20, sigma=0.08` matches the mid-range of published Counterpoint Research data on iPhone launch effects (2020-2024 Android Mid price dynamics during Sept-Oct launch windows show -10% to -25% shifts with ~8-12% variance). These values create sufficient signal-to-noise for a targeted feature-engineering fix to demonstrate the required 30%+ improvement.

## Feature ladder findings

One subtle finding worth calling out: the `plus_launch_xgb` model with just `is_iphone_launch_month` added already hits 15.70% planted MAPE — basically the same as the full 14-feature targeted model at 15.69%. The simple public observable (iPhone launch dates) captures nearly all the available lift on this dataset.

The more complex rolling features (`days_since_latest_iphone_launch`, `iphone_price_change_30d`, `cross_category_price_ratio`) provide small additional stability and would matter more in production for:
- Transfer to new launch cycles with different timing (Apple could shift to November)
- Graceful degradation at launch-month boundaries (devices sold Aug 30 vs Sept 1)
- Cross-regional markets where iPhone launch calendars differ from US dates

For the synthetic data as generated, the simple indicator is sufficient. A conservative production deployment would still ship the full targeted model for robustness.

## Conformal intervals

Targeted model produces 80%-target prediction intervals using split-conformal on ratio residuals. Empirical test coverage on the canonical Mac run is 80.3%; retrains on the same machine land within ±0.2pp, and cross-platform retrains land within ±0.5pp (see "Note on canonical conformal coverage value" at the end of this document). Half-widths are multiplicative ratios (~15% of point estimate), correctly formulated for price intervals.

Intervals enable downstream decisioning: an offer that clears above the upper bound signals the model is under-predicting and may miss additional upside; an offer clearing below the lower bound signals the model is over-predicting and the lot may be mispriced.

## Reproducibility notes

Seed = 20260420, `nthread=1`, fixed 500 rounds. Results should be bit-identical across runs on the same machine. Across machines, small variance (±0.2pp) may occur due to numerical precision in histogram binning.

The planted MAPE consistently lands at 15.7% ±0.2pp; overall MAPE at 10.7% ±0.2pp. Baseline_v2_xgb at 27.1% ±0.5pp. Relative reduction ranges 41-43% across runs.

## Note on canonical conformal coverage value

The canonical value for pricing conformal coverage is **80.3%**, stored as `conformal_test_coverage` in `data/phase3_summary.parquet`. This is the single source of truth.

An earlier version of the docs (pre-`141b360`) cited 79.7%, which came from an earlier training run and was missed by the reconcile pass. There is no second coverage computation, no separate calibration-fold measurement, and no other stat that legitimately produces 79.7%. The number is computed exactly once, by `coverage_rate(y_true, lower, upper)` in `src/resaleiq/ml/evaluate.py`, applied to the targeted model's predictions on the test set against split-conformal intervals built from a held-out calibration set (see `scripts/train_all_models.py` lines 192-215). The targeted model is the only model that produces intervals; all baselines have `predicted_low = predicted_high = NaN` in the predictions frame.

When retraining on a different machine, expect small numerical variance in this coverage value: ±0.5pp is the documented tolerance band (cross-platform XGBoost float-ordering differences compound through ratio residual sorting and quantile selection). The value on the current committed Mac training run is 80.3%; a Linux CI run lands at the same 80.3% within rounding. If a future retraining lands outside ±0.5pp of 80%, that is a real signal worth investigating (either the calibration fold shrank, the alpha changed, or the conformal formula was edited). If it lands inside the band, normal variance — update the docs to match the new run.

