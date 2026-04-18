# ResaleIQ: Phase 3 Production Deployment Plan

*One page. How we'd ship `targeted_v1_xgb` to production and keep it healthy.*

## 1. Serving Architecture

Prediction path lives behind a FastAPI service fronting a pickled XGBoost booster plus the sklearn preprocessing pipeline. Target p95 latency: under 50ms per offer (current XGBoost `hist` booster inferences in ~1ms for this feature count; the rest is preprocessing and network). The model artifact is versioned in S3 (or equivalent blob store) by semver plus training-data fingerprint, and loaded once at service startup. Offer data is fetched from the primary Postgres. Cross-brand features (rolling iPhone price medians, ratios) are computed from a materialized feature view refreshed nightly; at request time we look them up by timestamp using the same `merge_asof` pattern as the training code, which guarantees train-serve feature parity.

Two instances minimum behind a load balancer. Cold-start acceptable because model and feature view load in under 2 seconds.

## 2. Rollback Gate

Automatic rollback is triggered on any of three signals, measured on a rolling 24-hour window:

- **Segment MAPE regression**: MAPE on any monitored segment (device_category x condition_grade x month) exceeds that segment's 14-day trailing MAPE by 3 percentage points. Query is effectively Phase 2 query `01_segment_mape_audit.sql` with a different time filter.
- **Conformal coverage drift**: empirical coverage on offers in the last 24 hours drops below 70% (target is 80%, calibrated for 76.4% empirically; a 6pp drop signals distribution shift).
- **Prediction distribution shift**: Population Stability Index (PSI) across prediction deciles exceeds 0.2 vs the canonical reference window.

Rollback is a single S3 symlink flip to the previous model artifact. Inference continues without service interruption. A Slack alert is fired with the triggering signal and the segment(s) involved.

## 3. Monitoring Metrics

Three tiers, all instrumented via Prometheus and alertable:

- **Serving health** (engineering SLOs): request rate, p50/p95/p99 latency, error rate. Alert on 5-minute windows.
- **Model behavior** (ML SLOs, running 24h window): overall MAPE, per-segment MAPE with the Phase 2 query columns, conformal coverage, PSI on features and predictions, prediction distribution quantiles. Dashboarded via the Streamlit page; alerted on 1-hour windows.
- **Business outcome** (joined from downstream tables on a 7-day delay): average clearing-price delta from predictions in dollars, offers-per-lot attributable to this model's recommendations. Slower signal, reviewed weekly.

Drift detection uses the exact features produced by the feature view. A drift on `iphone_price_change_30d` is the signal that the *next* iPhone launch cycle is starting to shift pricing; that is expected behavior, not an alert condition.

## 4. A/B Test Design

Launch behind a 90/10 split: 90% of inbound offers are scored by `baseline_v2_xgb` (control), 10% by `targeted_v1_xgb` (treatment). Assignment by hash of `offer_id` so it's deterministic and replayable. Primary metric: MAPE computed 7 days after each offer clears. Secondary metrics: absolute dollar error, conformal coverage on the treatment arm, segment MAPE on Android Mid during iPhone launch months.

Minimum run time is two weeks or until each segment has at least 500 offers, whichever is later. Ramping plan: 10% to 30% to 50% to 100% with 3 days at each stage, gated on the rollback signals above being clean. Power analysis: at baseline MAPE 11.92% with an expected treatment MAPE of 11.53% and sigma of 8pp within-offer, minimum detectable effect at 80% power and alpha 0.05 is ~0.25pp, requiring roughly 4000 offers per arm. We clear 14,000 sku_offers per 7-month window on test data so the math is comfortable.

## 5. Model Versioning

Semantic version scheme: `{family}_{major}.{minor}.{patch}`, e.g. `targeted_v1.0.0`. Major bump on feature-set change, minor on hyperparameter or training-data-window change, patch on training-data refresh with identical config. Every model artifact is tagged with: feature set name, training-data date range, training-data row count, git SHA of the training code, MAPE on held-out test set, and the calibration fold's conformal half-width. This metadata is stored alongside the pickle and is queryable via an internal `/models` endpoint.

Shadow mode before flip: any new model runs in shadow for 24 hours, scoring the same traffic as production but routing nowhere. Shadow predictions are logged to Postgres with a special `model_version` suffix (`_shadow`). Comparison queries (overall MAPE, segment MAPE) are reviewed before promotion. The Phase 2 segment-MAPE query rerun against shadow predictions is the key go/no-go check.

---

*Written for: ResaleIQ internal. One-page intended, one-page delivered.*
