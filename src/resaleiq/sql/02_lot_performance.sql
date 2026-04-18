-- ============================================================================
-- Query 02: Lot performance by auction type
-- ============================================================================
-- Question: how do lots perform on clearing price, time-to-clear, and unsold
-- rate, split by auction type?
--
-- Design philosophy: a pricing team will be asked "which auction mechanism
-- should we use for X kind of inventory" roughly once a week. The answer
-- needs four numbers per type: clearing rate, avg clearing premium over
-- reserve, time-to-clear distribution, and model accuracy on cleared lots.
--
-- Uses window functions to show per-lot features alongside population-level
-- cohort metrics. This is a common idiom for mixing row-level detail with
-- cohort context in one pass.
-- ============================================================================

WITH
-- ----------------------------------------------------------------------------
-- Step 1: compute lot-level summary features.
-- We precompute total_quantity, n_unique_skus, grade_d_share, etc. because
-- later CTEs want those for cohort slicing. Computing them once here is
-- cheaper than recomputing inside every downstream aggregation.
-- ----------------------------------------------------------------------------
lot_features AS (
    SELECT
        l.lot_id,
        l.auction_type,
        l.status,
        l.reserve_price,
        l.clearing_price,
        l.scheduled_close,
        l.actual_close,
        -- Time between scheduled and actual close. For popcorn auctions this
        -- is the extension delta; for fixed-end it is always zero. NULL for
        -- unsold lots.
        EXTRACT(EPOCH FROM (l.actual_close - l.scheduled_close)) / 60.0
            AS close_delay_minutes,
        -- Lot composition: summed via lot_items. We use SUM and AVG rather
        -- than a subquery so the query planner can push the aggregation down.
        SUM(li.quantity)                                     AS total_quantity,
        COUNT(DISTINCT li.sku_id)                            AS n_unique_skus,
        SUM(li.quantity * li.unit_ref_price)                 AS total_ref_value,
        -- Grade-D share by quantity. This is correlated with unsold rate, so
        -- we carry it through for the cohort slice in the next CTE.
        SUM(li.quantity) FILTER (WHERE s.condition_grade = 'D')::NUMERIC
            / NULLIF(SUM(li.quantity), 0)                    AS grade_d_share
    FROM lots l
    INNER JOIN lot_items li ON li.lot_id = l.lot_id
    INNER JOIN skus       s  ON s.sku_id  = li.sku_id
    GROUP BY l.lot_id
),

-- ----------------------------------------------------------------------------
-- Step 2: enrich cleared lots with prediction context.
-- We left-join model_predictions because some lots might not have a
-- corresponding prediction row (e.g., cancelled lots), and we want to keep
-- the lot in the population even without a prediction.
-- ----------------------------------------------------------------------------
cleared_with_predictions AS (
    SELECT
        lf.*,
        mp.predicted                                              AS predicted_clearing,
        ABS(mp.predicted - lf.clearing_price)                     AS abs_error_usd,
        ABS(mp.predicted - lf.clearing_price) / lf.clearing_price AS ape,
        -- Clearing premium: percentage above reserve. Negative is
        -- mathematically possible if the clearing logic ever permitted
        -- below-reserve closes, but the schema CHECK forbids it.
        (lf.clearing_price / lf.reserve_price - 1)                AS clearing_premium
    FROM lot_features lf
    LEFT JOIN model_predictions mp ON mp.target_id   = lf.lot_id
                                   AND mp.target_type = 'lot'
    WHERE lf.status = 'cleared'
),

-- ----------------------------------------------------------------------------
-- Step 3: aggregate per-auction-type metrics.
-- We compute this in two passes: one for the cleared cohort (premium, MAPE,
-- time-to-clear), and one for the full population (clearing rate, cancel
-- rate). Merging happens in the final SELECT via UNION-like pattern.
-- ----------------------------------------------------------------------------
auction_type_stats AS (
    SELECT
        lf.auction_type,
        COUNT(*)                                                   AS total_lots,
        COUNT(*) FILTER (WHERE lf.status = 'cleared')              AS cleared_lots,
        COUNT(*) FILTER (WHERE lf.status = 'unsold')                AS unsold_lots,
        COUNT(*) FILTER (WHERE lf.status = 'cancelled')             AS cancelled_lots,
        -- Clearing rate excluding cancelled lots (seller-pulled): this is the
        -- metric the auction team actually cares about, since cancellation is
        -- a seller-side decision, not an auction-performance issue.
        COUNT(*) FILTER (WHERE lf.status = 'cleared')::NUMERIC
            / NULLIF(COUNT(*) FILTER (WHERE lf.status IN ('cleared', 'unsold')), 0)
            AS clearing_rate
    FROM lot_features lf
    GROUP BY lf.auction_type
),

cleared_stats AS (
    SELECT
        auction_type,
        AVG(clearing_premium)                  AS avg_premium,
        -- Percentiles give us the distribution shape. P50 is the typical
        -- clearing premium; P10 and P90 tell us how wide the tails are.
        PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY clearing_premium)
            AS p10_premium,
        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY clearing_premium)
            AS p50_premium,
        PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY clearing_premium)
            AS p90_premium,
        AVG(close_delay_minutes)                AS avg_close_delay_min,
        -- MAPE on cleared lots only; unsold lots have no clearing price.
        AVG(ape)                                AS model_mape
    FROM cleared_with_predictions
    GROUP BY auction_type
)

-- ----------------------------------------------------------------------------
-- Final join: stitch population metrics to cleared-cohort metrics.
-- Rounding to 4 decimals on rates, 2 on dollars, 1 on minutes. These are
-- judgment calls about display precision, not numeric precision.
-- ----------------------------------------------------------------------------
SELECT
    ats.auction_type,
    ats.total_lots,
    ats.cleared_lots,
    ats.unsold_lots,
    ats.cancelled_lots,
    ROUND(ats.clearing_rate::NUMERIC,   4) AS clearing_rate,
    ROUND(cs.avg_premium::NUMERIC,      4) AS avg_premium_over_reserve,
    ROUND(cs.p10_premium::NUMERIC,      4) AS p10_premium,
    ROUND(cs.p50_premium::NUMERIC,      4) AS p50_premium,
    ROUND(cs.p90_premium::NUMERIC,      4) AS p90_premium,
    ROUND(cs.avg_close_delay_min::NUMERIC, 1) AS avg_close_delay_min,
    ROUND(cs.model_mape::NUMERIC,       4) AS model_mape
FROM auction_type_stats ats
INNER JOIN cleared_stats cs ON cs.auction_type = ats.auction_type
ORDER BY ats.auction_type;
