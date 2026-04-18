# ResaleIQ Production Monitoring Specification

A production monitoring sketch for the pricing and lot-composition models. Assumes deployment on real wholesale marketplace data behind an existing decision-tree-based ensemble. Follows the pattern used in SentinelBoard: Prometheus instrumentation, PSI drift detection, Kolmogorov-Smirnov secondary validation, WebSocket live-streaming to an operator dashboard.

---

## Monitoring objective

Three-tier monitoring, each with a different time horizon and owner:

1. **Real-time operator alerts** (minutes to hours): obvious breaks in the serving path. Latency spikes, error rates, missing features. Owned by on-call ML engineer.
2. **Rolling health metrics** (daily): drift detection, calibration tracking, segment-level MAPE regressions. Owned by the DS team via a morning-standup dashboard.
3. **Portfolio health review** (monthly): model retraining decisions, A/B result synthesis, experiment cadence. Owned by the DS lead with business-stakeholder review.

---

## Tier 1 — real-time operator alerts

### Latency and throughput

- Prometheus histogram: `resaleiq_prediction_latency_seconds{model=...,version=...}` with buckets at 50ms, 100ms, 250ms, 500ms, 1s, 2.5s. Report p50, p95, p99.
- Alert thresholds: p95 > 150ms for 5 consecutive minutes, p99 > 500ms for 5 consecutive minutes. Page the on-call.
- Throughput gauge: `resaleiq_predictions_total{model=...,version=...}`. Alert on drops > 50% from trailing 7-day baseline for a given hour of the week.

### Error rates

- Counter: `resaleiq_prediction_errors_total{model=...,error_type=...}` with error types: `feature_missing`, `schema_mismatch`, `model_load_failed`, `inference_exception`, `interval_invalid`.
- Alert: `feature_missing` rate > 0.5% or any other error type > 0.1% for a rolling 15-minute window.

### Schema and feature integrity

- Per-prediction: assert all 14 pricing features (or 18 lot features) are present and within expected domain ranges (`baseline_value_usd > 0`, `condition_grade in {A+,A,B,C,D}`, etc.). Emit to `feature_missing` counter on failure.
- Fallback path: when a prediction cannot be made, the serving layer returns the relationship-specific floor price (or a business-defined default) with an explicit "fallback=true" flag attached to the response.

### Interval sanity

- Every returned prediction must have `lower <= point <= upper` and all values positive. This is a hard invariant; violations page immediately because they indicate a data corruption or model pickle issue.

---

## Tier 2 — rolling health (daily batch)

### Distribution drift (PSI per feature)

For both the pricing and lot models, compute Population Stability Index daily on the previous 24 hours of predictions against a 30-day rolling training-distribution baseline. PSI buckets at 10 quantiles.

- PSI < 0.10: no action.
- PSI 0.10 to 0.25: yellow flag in the morning dashboard. DS team investigates during the day.
- PSI > 0.25: red flag. Auto-create a ticket, DS team reviews within 24 hours.

Features to monitor:

**Pricing model (14 features):**
- `baseline_value_usd`, `msrp_new`, `device_age_days`, `storage_gb` — continuous, PSI important
- `condition_grade_enc`, `device_category_enc`, `carrier_enc` — categorical, PSI on category shares
- `iphone_price_change_30d`, `cross_category_price_ratio`, `days_since_latest_iphone_launch` — time-varying, expected to shift month-over-month; baseline should be adjusted to rolling 30-day window

**Lot model (18 features):**
- 5 category shares and 5 grade shares — categorical drift, KS-test secondary
- `sku_count`, `total_units`, `mean_baseline`, `baseline_value_cv` — continuous distribution
- `is_popcorn`, `has_reserve` — binary, watch for auction-mechanism policy changes

### Segment-level MAPE monitoring

The core defensive pattern: even if overall MAPE is stable, a single segment can be drifting badly. Daily batch:

1. Bucket all predictions with ground truth available from the previous 7 days by (device_category x is_iphone_launch_month x condition_grade).
2. Compute segment MAPE for each bucket with at least 30 predictions.
3. Compute excess-error versus overall MAPE and flag the top-5 segments by dollar-weighted excess error.
4. Compare each flagged segment to the same segment from 7 days prior and 30 days prior. Alert if any segment's MAPE has degraded by more than 3 percentage points week-over-week.

This is the diagnostic muscle that caught the planted effect in the first place, now run continuously.

### Calibration drift

For the pricing model's conformal intervals:

1. Daily: compute empirical coverage of the 80% interval on yesterday's predictions (requires ground truth, which for this problem comes in 24 to 48 hours after prediction when offers clear or expire).
2. Rolling 14-day coverage window. If it falls below 75% or above 85%, the conformal half-width is recalibrated on the rolling 30-day residuals.
3. If empirical coverage has been outside the 77-83% band for 3 consecutive recalibrations, escalate for a retraining review.

For the lot model, the same pattern on a weekly cadence since lots clear less frequently than offers.

### Bias monitoring

Daily:
- Signed mean residual by segment. If consistently positive or negative for a segment over 7 days, the model is systematically over- or under-predicting. Flag for investigation.
- Target: signed bias within +/- 1% of the mean clearing price per segment.

---

## Tier 3 — monthly portfolio review

### Retraining cadence

- **Pricing model**: retrain monthly on the rolling trailing 12 months. The retrain goes through a shadow A/B for 2 weeks before traffic shifts. Promotion criteria: no regression in overall MAPE and no segment regression > 2pp, conformal coverage within 77-83% band.
- **Lot model**: retrain weekly on the rolling trailing 6 months. Lower confidence threshold for promotion because lot volume is lower (roughly 8,000 lots over 18 months means thin weekly samples).

### Experiment tracker review

Walk through the 3 proposed experiments currently on the Experiment Tracker page:

1. **Production A/B of targeted vs baseline pricing model**: has it run yet, what was the MDE achieved, what's the next iteration?
2. **Lot optimizer shadow rollout**: if shipped, compare shadow-predicted clearing to actual. If not shipped, prioritize.
3. **Conformal recalibration**: confirm the rolling-window recalibration is running and coverage is staying in-band.

### Metric portfolio

Monthly review of the full metric portfolio, not just MAPE:

| Metric | Target | How measured |
|---|---|---|
| Pricing MAPE overall | <= 11% | Weekly batch |
| Pricing MAPE on top-10 dollar-weighted segments | each <= 2x overall | Weekly batch |
| Pricing conformal coverage | 78-82% | 14-day rolling |
| Lot model OOS MAPE | <= 5% | Monthly holdout |
| Lot model coverage | 75-85% (target 80%) | Rolling |
| Feature PSI (max across features) | < 0.25 | Daily |
| Prediction latency p99 | < 500ms | Real-time |
| Serving error rate | < 0.1% | Real-time |

### Incident review

If any Tier-1 alert has paged on-call, or any Tier-2 flag has created a ticket, conduct a brief postmortem at the monthly review. Not as formal as a full incident review: 10 minutes per incident, focusing on what was detected, how fast, and whether the monitoring spec needs adjustment.

---

## Deployment

### Serving infrastructure

- FastAPI service wrapping the pickled LotModel and targeted_v1_xgb booster. Running behind a load balancer with 3 replicas. Autoscaling on CPU and request queue depth.
- Horizontal scaling: models are stateless, easy to scale. Main bottleneck is model load time on cold start (~2-3 seconds to unpickle the booster). Use a warm pool.
- Version hygiene: every model artifact is tagged with `model_version` and `trained_at`. The serving layer always echoes these in the response headers so the monitoring pipeline can attribute predictions to specific versions.

### Feature store integration

- The 14 pricing-model features come from a feature store that joins SKU reference data (device, storage, carrier, condition grade), time-varying features (iphone_price_change_30d, days_since_latest_iphone_launch), and derived interaction features (cross_category_price_ratio, launch_x_android_mid).
- Feature freshness monitoring: each feature has a "max tolerable staleness" (iphone_price_change_30d tolerates hours of lag; baseline_value_usd tolerates days). Violations surface in Tier-1 alerts.

### Rollout playbook

Standard 4-stage promotion:

1. **Shadow mode** (week 1-2): new version runs in parallel with current, predictions logged but not served. Compare distributions, segment MAPE, interval coverage.
2. **Canary** (week 3): new version serves 5% of traffic. Watch Tier-1 alerts closely.
3. **Ramp** (week 4): 5% -> 25% -> 50% -> 100% over 5 days, with 24-hour hold at each step.
4. **Post-rollout review** (week 5): compare 30-day performance against pre-rollout baseline. Keep or roll back.

### Rollback

- Every deployed model version must be re-deployable within 15 minutes via a single-command rollback.
- Rollback triggers: any Tier-1 alert that persists > 30 minutes during canary or ramp; any Tier-2 red flag during ramp; any regulatory/business escalation.

---

## Connections to the project

This spec exists because data scientists who ship models but don't monitor them cause outages. The dashboard I've built already has the instrumentation scaffolding in place (Phase 2 SQL segment audit, Phase 3 conformal coverage tracking, Phase 5 holdout evaluation); the production version of this would be a live Streamlit or Grafana board fed by Prometheus + a daily batch on the real platform data warehouse. The SentinelBoard pattern (PSI drift, KS secondary validation, Prometheus histograms, WebSocket live feed, per-feature gauges) is the right starting architecture.

---

Built for the ResaleIQ project as the production-deployment companion to the 5-page Streamlit dashboard. See `docs/MODEL_CARD.md` for per-model limitations and `docs/DEPLOYMENT_PLAN.md` for the full rollout sequence.
