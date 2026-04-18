# Technical writeup

Deep-dive on architectural decisions, modeling choices, and trade-offs for ResaleIQ. Written for a reader who wants more than the README offers.

## Why synthetic data at all

The goal of this project is to demonstrate the diagnostic and modeling workflow that a wholesale secondhand phone marketplace actually uses. That workflow rests on three things: ability to write non-trivial SQL against a marketplace data model, ability to audit tree-ensemble errors at the segment level, and ability to translate those errors into a targeted feature-engineering fix that ships. The dataset is the vehicle for the workflow, not the point.

Using synthetic data means three concessions: no proprietary data can surface in a public repo, no NDA exposure for any prior employer, and every claim can be reproduced by anyone cloning the repo. The concession in the other direction is that the data does not carry the full messiness of real marketplace inventory (carrier blacklist states, cosmetic sub-grading, relationship-specific pricing floors). Those are flagged honestly in the README limitations section.

## Schema design

Nine tables across four logical layers. The most consequential structural choice is the polymorphic `model_predictions` table with `target_type` in `('sku_offer', 'lot')` and `target_id` pointing at the right row. The alternative, two separate prediction tables, would yield cleaner SQL query plans but would require a UNION across both tables for every segment-MAPE query. Since segment-MAPE queries are the single most common query pattern across the dashboard and the analytics layer, a single table produces the cleaner demo.

Every category field (device_category, carrier, condition_grade, auction_type, lot status, offer outcome, buyer type, region, tier) carries a CHECK constraint in `schema.sql`. This pushes vocabulary invariants into the storage layer and guarantees that downstream queries can rely on the values they expect.

Foreign keys are enforced throughout. The one exception is `lots.winning_buyer_id`, which is nullable because unsold and cancelled lots have no winner. That is an intentional SQL pattern that lets the polymorphic evaluation table stay simple.

The `seller_name` field on `sku_listings` is a string rather than a foreign key to a `sellers` table because sellers carry no modeling signal in this project. Adding a dimension table for them would be schema completeness without payoff.

## Market dynamics design

The fair-value function at the heart of `market_dynamics.py` decomposes clearing price as an anchor value times a sum of fractional factors. Writing the factors additively rather than multiplicatively (that is, `1 + season + shock + noise` rather than `(1 + season) * (1 + shock) * (1 + noise)`) keeps the shocks from compounding in ways that make MAPE analysis unstable. A 7 percent seasonal depression on top of a 6 percent grade adjustment should feel like a 13 percent effect, not the 13.4 percent that multiplicative compounding would produce. This choice has consequences for downstream modeling: a tree-based model does not know the generative structure and has to learn it empirically. The model's job is to recover the effective fraction, not the decomposition.

The planted segment is concentrated on a single device_category and two months per year, rather than smeared broadly. That concentration is the whole point. It makes the segment audit a clean diagnostic story: a baseline model without cross-brand temporal features carries roughly 27 percent MAPE on roughly 3 percent of transactions, pulling the overall average up by about a percentage point. A reader looking only at the aggregate MAPE number would miss it. A reader who groups by `(device_category, month)` would see it immediately.

The mechanism behind the planted effect is a real market dynamic, not an arbitrary injection. When iPhones launch in September, two things happen simultaneously: trade-in volume spikes as consumers upgrade, flooding the supply side of the wholesale market, and potential Android buyers delay their purchases waiting to see how iPhone prices settle. The net effect is downward pressure on Android mid-tier clearing prices specifically, because the substitution effect is strongest where price sensitivity is highest and where the iPhone is the closest comparable purchase. The synthetic shock is calibrated to match: mean depression of 20 percent with 8 percent standard deviation, only in September and October, only on the Android Mid category. That value lies within the 10 to 25 percent range published by Counterpoint Research across 2020 through 2024 iPhone launch windows.

## Noise calibration

Calibration constants live in `config.NoiseConfig` for shocks applied at transaction time, and in `predictions._BASE_SIGMA` and friends for the synthetic baseline predictions. Separating these is deliberate: the generative process is the ground truth, and the prediction generator simulates what an untuned model would produce against that truth. The parameters were chosen empirically to hit realistic target bands (roughly 11 percent overall MAPE for the naive XGBoost baseline, 27 percent on the planted segment, with Grade D elevated by a few percentage points and new-release inventory similar). The end-to-end integration test verifies these bands on every run, so accidental parameter drift will fail CI.

The grade variance multipliers carry a 5x ratio between A+ (0.6x base sigma) and D (3.0x base sigma). That matches the wholesale market's actual dispersion: parts-only inventory really does trade with much wider variance than sealed-in-box inventory. The test `test_grade_d_higher_variance_than_a_plus` verifies this invariant holds in the generated sample.

## Auction mechanics

The auction side uses two mechanics typical of wholesale secondhand phone marketplaces: popcorn-style extensions (bidding auto-extends when activity occurs in the final minutes) and proxy bidding (buyers set a max and the system auto-increments on their behalf). Both are modeled as boolean flags on the bid rows, which means the SQL layer can quantify each mechanism's effect on clearing price independently.

The calibrated popcorn premium is 4.25 percent over reserve, versus 2.27 percent for fixed-end auctions, both averaged across cleared lots. The spec target was 4.2 percent and 1.8 percent, so popcorn is on target and fixed-end is a little hot. The fixed-end premium could be tuned down by adjusting the mean in `build_lots`, but the difference of approximately 2 percentage points is the signal worth preserving (popcorn really does extract more than fixed-end), and the absolute numbers are close enough.

The unsold rate currently lands at 21.6 percent against a target of 18 percent. The overshoot comes from the cumulative effect of the grade-D-share modifier and the high-quantity modifier compounding on lots that hit both conditions. This is realistic: the worst lots really are disproportionately likely to fail to clear. Moving to 18 percent would require dropping the base rate in `NoiseConfig.lot_unsold_rate` to compensate for the modifiers, which is straightforward if the exact number becomes material.

## The synthetic baseline predictions

The predictions in `model_predictions` are not real model outputs. They are samples drawn from a distribution that was tuned to match what a baseline gradient-boosted model would produce if it were trained without the cross-brand temporal features. In production modeling (Phase 4), these will be replaced by actual XGBoost outputs, but the schema and dashboard code will not change.

The prediction formula is `predicted = actual * (1 + N(mu, sigma))` where `(mu, sigma)` depends on segment:
- Base case (non-planted, non-extreme-grade, non-new-release): `N(0, 0.135)`, giving approximately 11 percent MAPE.
- Planted segment (Android Mid, launch months): `N(0.14, 0.28)`, giving approximately 25 percent MAPE. The non-zero mean represents the fact that the baseline model has no way to see the iPhone-launch shock and therefore systematically over-predicts in that slice.
- Grade D: base sigma composed with 0.09 additional via `np.hypot`, giving the wider tail.
- New-release (first 60 days): base sigma composed with 0.07 additional.

The composition-in-quadrature is important: uncorrelated noise sources add in variance space, not standard deviation space. The `np.hypot` call handles this cleanly.

## Reproducibility

A single master seed (`MASTER_SEED = 20260420`) in `config.py` is the root of every random draw in the project. Each generation stage (devices, skus, buyers, sku_listings, sku_offers, lots, predictions) derives its seed from the master via `child_seed(namespace)`, which is a SHA-256 hash of the master seed concatenated with the namespace string. This approach has two properties worth noting: stages can run in any order without affecting their individual output, and adding a new stage does not perturb the seeds of existing stages. Anyone cloning the repo and running `make generate` will get byte-identical parquet output.

## Phase 2: SQL analysis layer

Four CTE-based queries sit in `src/resaleiq/sql/`: a segment-MAPE audit that ranks `(device_category, month, condition_grade)` cells by total absolute error contribution, a lot-performance summary that cross-tabs cleared / unsold / cancelled rates by auction type, a popcorn-premium quantification with a 95 percent confidence band via two-sample normal approximation, and a buyer-tier × auction-type interaction via `GROUPING SETS` that produces a 2×3 crosstab plus margins. A Typer CLI (`resaleiq.sql.runner`) executes each one against Postgres and renders the result with `rich`.

## Phase 3: modeling layer

Five XGBoost variants train on the cleared-offer feature frame with a temporal split (test = Sept 2025 to Mar 2026). The ladder progresses from a 9-feature baseline through three single-feature additions to a 14-feature targeted model. The key design decisions are the ratio target (`clearing_price / baseline_value`), which makes multiplicative shocks like the launch-month depression first-class signals, and fixed-round training (500 rounds, `nthread=1`, no early stopping) for bit-for-bit reproducibility across XGBoost minor versions. Conformal intervals come from split-conformal on ratio residuals scaled multiplicatively. Counterfactual analysis in `scripts/counterfactual_analysis.py` converts MAPE differences into dollar impact per offer and per segment.

## Phase 4: dashboard + lot optimizer

The Streamlit dashboard reads parquet and Postgres with a graceful parquet fallback when Postgres is unreachable. Five pages: Executive Summary, Pricing Model Performance (tabbed diagnostics with interactive segment filters), Auction Lot Optimizer (a partition recommender that scores five strategies against a bag of inventory with 80 percent confidence intervals), Segment Audit (interactive Phase 2 rewrite with richer UX), and Experiment Tracker (shipped work plus three proposed experiments with power analysis).

## Phase 5: hardening

The lot model moved to a time-series holdout (train through 2026-01-31, holdout Feb to Mar 2026, 531 lots strictly future). Conformal half-width switched from Gaussian-assumption MAD×z to empirical-quantile calibration on training residuals, which hits 80 percent on train by construction and exposes real distribution shift on the holdout. Full model card lives at `docs/MODEL_CARD.md` and a production monitoring sketch lives at `docs/MONITORING_SPEC.md`.
