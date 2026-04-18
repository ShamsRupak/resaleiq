# Model Card: ResaleIQ Lot-Composition Model

> **Note on evaluation split:** Every metric in this card suffixed "(OOS)" is measured on 531 lots that closed between 2026-02-01 and 2026-03-31 and were held out of training. The model was never trained on, validated on, or tuned against these lots. The training set is the 5,493 lots that closed between 2024-10-01 and 2026-01-31. The split is strict temporal: no shuffling, no stratification, no leakage. The `holdout_start` timestamp `2026-02-01` is stored in `data/lot_model_summary.json`.

## At a glance

| Property | Value |
|---|---|
| Model type | XGBoost regressor on ratio target |
| Target | `clearing_price / sum(baseline_value)` per lot |
| Training data | 5,493 cleared lots, Oct 2024 to Jan 2026 |
| Test data | 531 held-out lots, Feb to Mar 2026 |
| Features | 18 (SKU count, category mix, grade mix, price dispersion, auction mechanics, seasonality) |
| In-sample aggregate MAPE | 3.13% |
| Out-of-sample aggregate MAPE | 3.98% |
| Naive baseline MAPE (OOS) | 18.11% |
| Relative improvement vs naive (OOS) | 78.0% |
| Conformal coverage on holdout (80% target) | 66.5% |
| Reproducibility | Deterministic under seed 20260420, `nthread=1`, fixed 500 rounds |

## Intended use

Given a bag of SKU inventory (a set of sku_id, device_category, condition_grade, baseline_value tuples) and an auction mechanism (popcorn or fixed_end), predict the expected clearing price of that inventory under several candidate partitions into auction lots. The optimizer selects the partition that maximizes expected total clearing value.

Primary users: a data scientist or operations analyst selecting lot compositions ahead of an auction cycle. Secondary users: stakeholders reviewing which partition strategies dominate under different inventory mixes.

## Training

- Framework: XGBoost 2.1 with `objective=reg:squarederror`, `eval_metric=mae`
- Hyperparameters: `max_depth=5`, `learning_rate=0.03`, `subsample=0.85`, `colsample_bytree=0.85`, `min_child_weight=3`, `reg_lambda=1.0`, `tree_method=hist`, 500 boosting rounds
- No early stopping. Fixed rounds were chosen because early stopping on low-variance ratio targets triggered at iterations 0-17 across seeds, producing unstable boosters. Fixed rounds plus a light learning rate is a stable alternative.
- Ratio target winsorized at the 1st and 99th percentile before fitting. This removes 1.2% of training rows in the tail and protects the booster from lots with pathological clearing outcomes.
- Conformal half-width calibrated empirically: stored `ratio_mad` is set such that `1.282 * ratio_mad` equals the 80th percentile of absolute training residuals. This makes the 80% interval hit 80% on train by construction, and approximately 66% on the Feb-Mar holdout (see Limitations).

## Evaluation

**Time-series holdout.** Lots are split on `opened_at`: all lots with `opened_at < 2026-02-01` are training, all lots on or after are holdout. This is the honest split for a rolling-window production model, which is how this would be deployed.

**Primary metric: aggregate MAPE.** Defined as `sum(|pred - actual|) / sum(|actual|)`. Aggregate is preferred over per-lot average because the business outcome scales with total clearing value, not per-lot percentage error.

**Secondary metric: conformal interval coverage.** Fraction of holdout lots whose actual ratio falls inside the predicted 80% interval. Target is 0.80. Observed: 0.665.

## Limitations

1. **Conformal miscalibration from distribution shift.** Intervals hit 80% on train by construction but only 66.5% on the Feb-Mar holdout. Feb-Mar residuals are approximately 27% wider than training-period residuals in absolute ratio terms. This is a real distribution-shift signal. The production fix is rolling recalibration on a moving window of the most recent cleared lots, which is Experiment 3 on the dashboard Experiment Tracker page.

2. **In-sample vs out-of-sample gap.** The 0.85 percentage-point gap between 3.13% in-sample and 3.98% OOS is small in absolute terms but real. A production deployment should measure this gap continuously and alert if it widens past 2pp.

3. **Strawman-sensitive baseline.** The 78.0% improvement is vs a naive "predict `total_baseline_value`" baseline, which ignores everything the model uses. A more demanding baseline would be median ratio times baseline, or a single-feature booster. Against the median-ratio baseline (which is roughly 0.86 * baseline), the OOS MAPE is approximately 14%, and the lot model improves on it by roughly 72%. The 18% figure is not dishonest but it overstates the improvement by one full order of baseline sophistication.

4. **Training on cleared lots only.** The model learns clearing ratios conditional on the lot clearing. It does not model the probability of clearing, which is a separate ~23% of lots that fail to clear at all. For a full revenue forecast, a two-stage model (clear-probability gate plus this clearing-price model) would be appropriate.

5. **Synthetic data.** This model is trained on a seeded synthetic dataset. Real wholesale marketplace lots would have richer features (buyer relationship history, specific SKU conditions, carrier unlock status) that are abstracted away here. The methodology transfers; the specific coefficients do not.

6. **Partition search is exhaustive over 5 fixed strategies.** The optimizer does not attempt combinatorial search over all possible partitions. It evaluates 5 strategies (single_lot, by_category, by_grade, balanced_4, balanced_6) and picks the best. A production system might benefit from local search starting from the best of these strategies.

## Fairness and adversarial considerations

Not applicable in the current synthetic-data context. In a real deployment, review would be needed for:
- Buyer-tier fairness: does the model systematically under-predict for SMB-tier buyers?
- Seller-tier impact: do pricing recommendations disadvantage small suppliers relative to large distributors?
- Adversarial manipulation: can a seller game the optimizer by listing SKUs strategically?

## Maintenance

- Retrain cadence: weekly on rolling 90-day window of cleared lots
- Conformal recalibration: weekly on the same window, using empirical quantiles of the most recent 60 days of residuals
- Drift monitoring: PSI on feature distributions, rolling MAPE on the last 4 weeks of cleared lots, alert if rolling MAPE exceeds 7% or feature PSI exceeds 0.25
- Rollback criteria: two consecutive weekly MAPE values above 8%, or coverage below 60% on last 100 lots

## Version

- Model version: `lot_v1_holdout_20260418`
- Trained: April 18, 2026
- Code location: `src/resaleiq/ml/lot_model.py` function `train_lot_model_with_holdout`
- Artifact: `data/lot_model.pkl`
- Summary: `data/lot_model_summary.json`
