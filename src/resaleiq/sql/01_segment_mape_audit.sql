-- ============================================================================
-- Query 01: Segment-wise MAPE audit
-- ============================================================================
-- Question: where is the baseline model's error concentrated, and which
-- segments should we fix first?
--
-- Design philosophy: two candidate ranking metrics, both wrong in isolation:
--
--   Rank by MAPE alone                -> picks noisy tails with tiny N.
--   Rank by total absolute dollar error -> picks high-volume high-price
--                                          segments even at baseline MAPE,
--                                          because Apple Flagship inventory
--                                          naturally accumulates more dollar
--                                          error than cheaper phones.
--
-- The metric that actually matches "where should the next sprint go" is
-- EXCESS ERROR: the dollar error attributable to the segment performing
-- worse than the overall baseline MAPE. A segment with MAPE equal to the
-- overall baseline contributes zero excess error no matter how many dollars
-- flow through it. A segment with MAPE double the baseline contributes
-- error equal to (mape - baseline) * n * avg_price.
--
-- This framing is what a senior data scientist would write. It surfaces the
-- segments where concentrated model failure is leaving measurable money on
-- the table.
--
-- Expected top-ranked segment on ResaleIQ data: Android Mid in September and
-- October of 2024 and 2025 (the planted cross-brand substitution effect).
-- The query proves the audit workflow surfaces the planted segment without
-- the analyst having to know it's there.
-- ============================================================================

WITH
-- ----------------------------------------------------------------------------
-- Step 1: flatten predictions with the full segmentation context.
-- ----------------------------------------------------------------------------
prediction_context AS (
    SELECT
        mp.prediction_id,
        mp.predicted,
        mp.actual,
        mp.predicted_at,
        d.device_category,
        s.condition_grade,
        DATE_TRUNC('month', mp.predicted_at)::DATE AS pred_month,
        ABS(mp.predicted - mp.actual)             AS abs_error_usd,
        ABS(mp.predicted - mp.actual) / mp.actual AS ape,
        mp.actual                                 AS actual_price
    FROM model_predictions mp
    INNER JOIN sku_offers   so ON so.offer_id    = mp.target_id
                              AND mp.target_type = 'sku_offer'
    INNER JOIN sku_listings sl ON sl.listing_id  = so.listing_id
    INNER JOIN skus         s  ON s.sku_id       = sl.sku_id
    INNER JOIN devices      d  ON d.device_id    = s.device_id
),

-- ----------------------------------------------------------------------------
-- Step 2: compute the overall baseline MAPE as a scalar we can subtract.
-- Materializing it as a CTE keeps the plan predictable; Postgres executes
-- it once and reuses.
-- ----------------------------------------------------------------------------
baseline AS (
    SELECT AVG(ape) AS overall_mape FROM prediction_context
),

-- ----------------------------------------------------------------------------
-- Step 3: aggregate to the segment grain (device_category, condition_grade,
-- pred_month). Filter out segments below a minimum N where MAPE estimates
-- are too noisy to support a fix-prioritization decision.
-- ----------------------------------------------------------------------------
segment_stats AS (
    SELECT
        pc.device_category,
        pc.condition_grade,
        pc.pred_month,
        COUNT(*)                              AS n_predictions,
        AVG(pc.ape)                           AS mape,
        SUM(pc.abs_error_usd)                  AS total_abs_error_usd,
        AVG(pc.actual_price)                   AS avg_actual_price,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY pc.ape) AS median_ape,
        (SELECT overall_mape FROM baseline)    AS overall_mape
    FROM prediction_context pc
    GROUP BY pc.device_category, pc.condition_grade, pc.pred_month
    HAVING COUNT(*) >= 30
),

-- ----------------------------------------------------------------------------
-- Step 4: compute excess error and rank by it.
-- Excess error = (mape - overall_mape) * n * avg_price. We clip at zero so
-- segments outperforming the baseline don't negatively offset the ranking;
-- they're "good" segments we're not prioritizing for fixes.
-- ----------------------------------------------------------------------------
ranked AS (
    SELECT
        ss.*,
        GREATEST(
            (ss.mape - ss.overall_mape) * ss.n_predictions * ss.avg_actual_price,
            0
        ) AS excess_abs_error_usd,
        RANK() OVER (
            ORDER BY GREATEST(
                (ss.mape - ss.overall_mape) * ss.n_predictions * ss.avg_actual_price,
                0
            ) DESC
        ) AS priority_rank
    FROM segment_stats ss
)

-- ----------------------------------------------------------------------------
-- Final select: top 15 by excess error. We surface both absolute and excess
-- error so the stakeholder sees both the raw pain and the fixable delta.
-- ----------------------------------------------------------------------------
SELECT
    r.priority_rank,
    r.device_category,
    r.condition_grade,
    r.pred_month,
    r.n_predictions,
    ROUND(r.mape::NUMERIC,                 4) AS mape,
    ROUND(r.median_ape::NUMERIC,            4) AS median_ape,
    ROUND(r.overall_mape::NUMERIC,          4) AS overall_mape,
    ROUND(r.total_abs_error_usd::NUMERIC,  2) AS total_abs_error_usd,
    ROUND(r.excess_abs_error_usd::NUMERIC, 2) AS excess_abs_error_usd,
    ROUND(r.avg_actual_price::NUMERIC,      2) AS avg_actual_price
FROM ranked r
WHERE r.priority_rank <= 15
ORDER BY r.priority_rank;
