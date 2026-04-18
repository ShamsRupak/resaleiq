-- ============================================================================
-- Query 03: Popcorn auction dynamics
-- ============================================================================
-- Question: does popcorn-style bid extension actually extract more value
-- than fixed-end auctions, and can we quantify the effect with a confidence
-- interval?
--
-- Design philosophy: an "average X is higher than average Y" claim without
-- a confidence interval is not a claim a senior data scientist accepts. This
-- query produces both the point estimate and a normal-approximation 95 CI on
-- the difference in mean clearing premium.
--
-- Variance is estimated using the sample variance within each group. For
-- large-N cases (we expect thousands of cleared lots per auction type), the
-- normal approximation is more than adequate. For tighter small-sample work,
-- we would bootstrap, but that belongs in Python, not in a SQL dashboard
-- query.
-- ============================================================================

WITH
-- ----------------------------------------------------------------------------
-- Step 1: compute clearing premium per cleared lot.
-- Clearing premium is (clearing_price / reserve_price - 1). This is the
-- percentage lift the auction extracted over the seller's floor. It is the
-- metric the business cares about, because it translates directly to GMV.
-- ----------------------------------------------------------------------------
cleared_lot_premiums AS (
    SELECT
        l.lot_id,
        l.auction_type,
        l.reserve_price,
        l.clearing_price,
        -- The premium is always >= 0 given schema constraints (clearing price
        -- must be >= reserve price for cleared lots, enforced by the CHECK
        -- constraint and the generation logic). We compute it here rather
        -- than inline in later aggregations for readability.
        (l.clearing_price / l.reserve_price - 1.0) AS clearing_premium,
        -- Count of bids placed on this lot; thicker markets clear higher.
        -- This is a confounder we will not control for in this simple
        -- comparison, but it would be the next refinement.
        (SELECT COUNT(*) FROM lot_bids lb WHERE lb.lot_id = l.lot_id) AS n_bids,
        -- Flag whether any bid on this lot triggered a popcorn extension.
        -- For fixed-end auctions this is always false; for popcorn auctions
        -- it is true roughly half the time (only close-to-deadline bids
        -- trigger).
        EXISTS (
            SELECT 1 FROM lot_bids lb
            WHERE lb.lot_id = l.lot_id AND lb.popcorn = TRUE
        ) AS had_popcorn_trigger
    FROM lots l
    WHERE l.status = 'cleared'
),

-- ----------------------------------------------------------------------------
-- Step 2: compute per-auction-type mean, variance, and N for the normal
-- approximation CI. Using VAR_SAMP (sample variance, n-1 denominator) is
-- the correct choice for a sample estimate, not VAR_POP.
-- ----------------------------------------------------------------------------
auction_type_premium_stats AS (
    SELECT
        auction_type,
        COUNT(*)                              AS n,
        AVG(clearing_premium)                 AS mean_premium,
        VAR_SAMP(clearing_premium)            AS var_premium,
        STDDEV_SAMP(clearing_premium)         AS sd_premium
    FROM cleared_lot_premiums
    GROUP BY auction_type
),

-- ----------------------------------------------------------------------------
-- Step 3: compute the difference in means between popcorn and fixed_end,
-- plus the standard error of that difference via the two-sample formula:
--   SE_diff = sqrt(var_A / n_A + var_B / n_B)
-- and the 95 percent CI as point_estimate +/- 1.96 * SE_diff.
-- ----------------------------------------------------------------------------
difference_ci AS (
    SELECT
        pop.mean_premium                                  AS popcorn_mean_premium,
        fix.mean_premium                                  AS fixed_mean_premium,
        pop.mean_premium - fix.mean_premium               AS diff_mean_premium,
        SQRT(pop.var_premium / pop.n + fix.var_premium / fix.n) AS se_diff,
        pop.n                                             AS popcorn_n,
        fix.n                                             AS fixed_n
    FROM
        (SELECT * FROM auction_type_premium_stats WHERE auction_type = 'popcorn')    pop
    CROSS JOIN
        (SELECT * FROM auction_type_premium_stats WHERE auction_type = 'fixed_end') fix
),

-- ----------------------------------------------------------------------------
-- Step 4: also compute within-popcorn decomposition. Among popcorn-type
-- auctions, do the ones that actually triggered an extension clear higher
-- than the ones that did not? This is the cleaner causal question, because
-- it conditions on inventory that *could* have triggered.
-- ----------------------------------------------------------------------------
within_popcorn_decomposition AS (
    SELECT
        had_popcorn_trigger,
        COUNT(*)                      AS n,
        AVG(clearing_premium)         AS mean_premium,
        STDDEV_SAMP(clearing_premium) AS sd_premium
    FROM cleared_lot_premiums
    WHERE auction_type = 'popcorn'
    GROUP BY had_popcorn_trigger
)

-- ----------------------------------------------------------------------------
-- Final select: we return a single row with everything, because the
-- dashboard renders this as a formatted stat card rather than a table.
-- ----------------------------------------------------------------------------
SELECT
    ROUND((dci.popcorn_mean_premium * 100)::NUMERIC, 2) AS popcorn_mean_premium_pct,
    ROUND((dci.fixed_mean_premium   * 100)::NUMERIC, 2) AS fixed_end_mean_premium_pct,
    ROUND((dci.diff_mean_premium    * 100)::NUMERIC, 2) AS diff_mean_premium_pp,
    -- 95 percent CI: normal approximation, 1.96 multiplier.
    ROUND(((dci.diff_mean_premium - 1.96 * dci.se_diff) * 100)::NUMERIC, 2)
        AS diff_ci_low_pp,
    ROUND(((dci.diff_mean_premium + 1.96 * dci.se_diff) * 100)::NUMERIC, 2)
        AS diff_ci_high_pp,
    dci.popcorn_n,
    dci.fixed_n,
    -- Within-popcorn decomposition: premium for lots that did vs did not
    -- trigger an extension. Surfacing both rows as scalars keeps the final
    -- output to a single row for dashboard rendering.
    ROUND(
        ((SELECT mean_premium FROM within_popcorn_decomposition
          WHERE had_popcorn_trigger = TRUE) * 100)::NUMERIC, 2
    ) AS premium_when_popcorn_triggered_pct,
    ROUND(
        ((SELECT mean_premium FROM within_popcorn_decomposition
          WHERE had_popcorn_trigger = FALSE) * 100)::NUMERIC, 2
    ) AS premium_when_popcorn_not_triggered_pct
FROM difference_ci dci;
