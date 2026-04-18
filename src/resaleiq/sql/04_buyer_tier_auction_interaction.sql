-- ============================================================================
-- Query 04: Buyer tier x auction type interaction
-- ============================================================================
-- Question: do Enterprise buyers pay different clearing premiums on popcorn
-- auctions than on fixed-end auctions, compared to Mid-market or SMB?
--
-- Design philosophy: a one-dimensional view (either by tier or by auction
-- type) misses interaction effects. This query builds a 2x3 matrix of
-- (auction_type x buyer_tier) plus marginals, so stakeholders can read both
-- the main effects and the interaction in one place.
--
-- Why this matters commercially: if Enterprise buyers consistently extract
-- higher lift from popcorn, it's a signal that the popcorn mechanism is
-- especially effective at auctions where informed, sophisticated bidders
-- compete. The auction team would use that to decide which lots to route
-- into popcorn versus fixed_end.
--
-- Uses GROUPING SETS to produce the matrix plus marginals in a single pass.
-- GROUPING SETS is standard SQL (not Postgres-specific) and is the idiomatic
-- way to produce this shape. Anyone who's worked with crosstabs will
-- recognize it.
-- ============================================================================

WITH
-- ----------------------------------------------------------------------------
-- Step 1: collect cleared lots with the winning buyer's tier attached.
-- Winning buyer is the buyer_id on the lot row. Unsold and cancelled lots
-- have no winning buyer, so we filter on status = 'cleared'.
-- ----------------------------------------------------------------------------
cleared_with_tier AS (
    SELECT
        l.lot_id,
        l.auction_type,
        l.clearing_price,
        l.reserve_price,
        (l.clearing_price / l.reserve_price - 1.0) AS clearing_premium,
        b.tier        AS buyer_tier,
        b.region      AS buyer_region,
        b.buyer_type  AS buyer_type
    FROM lots l
    INNER JOIN buyers b ON b.buyer_id = l.winning_buyer_id
    WHERE l.status = 'cleared'
),

-- ----------------------------------------------------------------------------
-- Step 2: aggregate using GROUPING SETS.
-- GROUPING SETS lets us request several grouping levels in one query:
--   () means "no grouping" i.e. the grand total
--   (auction_type) means "marginal over tiers"
--   (buyer_tier) means "marginal over auction types"
--   (auction_type, buyer_tier) means "the 2x3 cell values"
--
-- GROUPING() returns 1 for columns that are NULL because of rollup and 0
-- otherwise. This lets us distinguish "marginal" rows from actual NULL
-- values in the data. It is the right idiom for signaling to downstream
-- consumers which rows are aggregations vs detail.
-- ----------------------------------------------------------------------------
grouped_stats AS (
    SELECT
        auction_type,
        buyer_tier,
        GROUPING(auction_type) AS is_auction_marginal,
        GROUPING(buyer_tier)   AS is_tier_marginal,
        COUNT(*)               AS n_lots,
        AVG(clearing_price)    AS avg_clearing_price,
        AVG(clearing_premium)  AS avg_premium,
        STDDEV_SAMP(clearing_premium) AS sd_premium
    FROM cleared_with_tier
    GROUP BY GROUPING SETS (
        (),                             -- grand total
        (auction_type),                 -- marginal by auction type
        (buyer_tier),                   -- marginal by tier
        (auction_type, buyer_tier)      -- the cells
    )
)

-- ----------------------------------------------------------------------------
-- Final select: relabel marginal rows with 'ALL' for readability, and sort
-- so the output reads naturally (cells grouped by auction type, with the
-- auction-type-marginal row at the end of each block, then the grand total).
-- ----------------------------------------------------------------------------
SELECT
    CASE WHEN is_auction_marginal = 1 THEN 'ALL' ELSE auction_type END
        AS auction_type,
    CASE WHEN is_tier_marginal    = 1 THEN 'ALL' ELSE buyer_tier   END
        AS buyer_tier,
    n_lots,
    ROUND(avg_clearing_price::NUMERIC, 2)          AS avg_clearing_price,
    ROUND((avg_premium * 100)::NUMERIC, 2)          AS avg_premium_pct,
    ROUND((sd_premium  * 100)::NUMERIC, 2)          AS sd_premium_pct
FROM grouped_stats
ORDER BY
    -- Cells first (auction first), then tier marginals, then auction
    -- marginals, then grand total. Readers scan top-to-bottom expecting
    -- detail before rollup.
    is_auction_marginal,
    is_tier_marginal,
    auction_type,
    buyer_tier;
