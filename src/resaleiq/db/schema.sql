-- ResaleIQ schema (PostgreSQL 16+).
--
-- Mounted by docker-compose into /docker-entrypoint-initdb.d so a fresh
-- container initializes the database. Safe to re-run: everything is wrapped
-- in IF NOT EXISTS.
--
-- CHECK constraints defend vocabulary invariants at the storage layer. Types
-- are chosen to match the parquet output of the data-generation module:
--   - ids are BIGINT to align with numpy int64
--   - prices are NUMERIC(12,2)
--   - dates/times are TIMESTAMP WITHOUT TIME ZONE (UTC implied)

BEGIN;

CREATE TABLE IF NOT EXISTS devices (
    device_id        BIGINT        PRIMARY KEY,
    manufacturer     VARCHAR(32)   NOT NULL,
    model_family     VARCHAR(64)   NOT NULL,
    release_date     DATE          NOT NULL,
    msrp_new         NUMERIC(10,2) NOT NULL CHECK (msrp_new > 0),
    device_category  VARCHAR(32)   NOT NULL CHECK (
        device_category IN (
            'Apple Flagship', 'Apple Mid',
            'Android Flagship', 'Android Mid', 'Android Budget'
        )
    )
);

CREATE INDEX IF NOT EXISTS idx_devices_category ON devices (device_category);
CREATE INDEX IF NOT EXISTS idx_devices_manufacturer ON devices (manufacturer);

CREATE TABLE IF NOT EXISTS skus (
    sku_id              BIGINT        PRIMARY KEY,
    device_id           BIGINT        NOT NULL REFERENCES devices (device_id),
    storage_gb          INTEGER       NOT NULL CHECK (storage_gb IN (64, 128, 256, 512, 1024)),
    carrier             VARCHAR(16)   NOT NULL CHECK (
        carrier IN ('Unlocked', 'ATT', 'Verizon', 'TMobile', 'Sprint')
    ),
    condition_grade     VARCHAR(4)    NOT NULL CHECK (
        condition_grade IN ('A+', 'A', 'B', 'C', 'D')
    ),
    baseline_value_usd  NUMERIC(12,2) NOT NULL CHECK (baseline_value_usd >= 15.0)
);

CREATE INDEX IF NOT EXISTS idx_skus_device_id ON skus (device_id);
CREATE INDEX IF NOT EXISTS idx_skus_grade ON skus (condition_grade);

CREATE TABLE IF NOT EXISTS buyers (
    buyer_id    BIGINT      PRIMARY KEY,
    buyer_type  VARCHAR(32) NOT NULL CHECK (
        buyer_type IN ('distributor', 'reseller', 'refurbisher', 'carrier')
    ),
    region      VARCHAR(16) NOT NULL CHECK (
        region IN ('US-East', 'US-West', 'EU', 'APAC', 'LATAM')
    ),
    tier        VARCHAR(16) NOT NULL CHECK (
        tier IN ('Enterprise', 'Mid-market', 'SMB')
    )
);

CREATE INDEX IF NOT EXISTS idx_buyers_tier ON buyers (tier);

CREATE TABLE IF NOT EXISTS sku_listings (
    listing_id   BIGINT        PRIMARY KEY,
    sku_id       BIGINT        NOT NULL REFERENCES skus (sku_id),
    list_price   NUMERIC(12,2) NOT NULL CHECK (list_price > 0),
    quantity     INTEGER       NOT NULL CHECK (quantity > 0),
    listed_at    TIMESTAMP     NOT NULL,
    seller_name  VARCHAR(64)   NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sku_listings_sku ON sku_listings (sku_id);
CREATE INDEX IF NOT EXISTS idx_sku_listings_listed_at ON sku_listings (listed_at);

CREATE TABLE IF NOT EXISTS sku_offers (
    offer_id        BIGINT        PRIMARY KEY,
    listing_id      BIGINT        NOT NULL REFERENCES sku_listings (listing_id),
    buyer_id        BIGINT        NOT NULL REFERENCES buyers (buyer_id),
    offer_price     NUMERIC(12,2) NOT NULL CHECK (offer_price > 0),
    outcome         VARCHAR(32)   NOT NULL CHECK (
        outcome IN ('accepted', 'rejected', 'countered_accepted', 'expired')
    ),
    clearing_price  NUMERIC(12,2) CHECK (clearing_price IS NULL OR clearing_price > 0),
    offer_at        TIMESTAMP     NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sku_offers_listing ON sku_offers (listing_id);
CREATE INDEX IF NOT EXISTS idx_sku_offers_buyer ON sku_offers (buyer_id);
CREATE INDEX IF NOT EXISTS idx_sku_offers_outcome ON sku_offers (outcome);
CREATE INDEX IF NOT EXISTS idx_sku_offers_cleared ON sku_offers (offer_at)
    WHERE clearing_price IS NOT NULL;

CREATE TABLE IF NOT EXISTS lots (
    lot_id            BIGINT        PRIMARY KEY,
    winning_buyer_id  BIGINT        REFERENCES buyers (buyer_id),
    reserve_price     NUMERIC(12,2) NOT NULL CHECK (reserve_price > 0),
    clearing_price    NUMERIC(12,2) CHECK (clearing_price IS NULL OR clearing_price > 0),
    scheduled_close   TIMESTAMP     NOT NULL,
    actual_close      TIMESTAMP,
    auction_type      VARCHAR(16)   NOT NULL CHECK (auction_type IN ('popcorn', 'fixed_end')),
    status            VARCHAR(16)   NOT NULL CHECK (status IN ('cleared', 'unsold', 'cancelled'))
);

CREATE INDEX IF NOT EXISTS idx_lots_status ON lots (status);
CREATE INDEX IF NOT EXISTS idx_lots_auction_type ON lots (auction_type);
CREATE INDEX IF NOT EXISTS idx_lots_scheduled_close ON lots (scheduled_close);
CREATE INDEX IF NOT EXISTS idx_lots_cleared ON lots (scheduled_close)
    WHERE status = 'cleared';

CREATE TABLE IF NOT EXISTS lot_items (
    lot_item_id      BIGINT        PRIMARY KEY,
    lot_id           BIGINT        NOT NULL REFERENCES lots (lot_id),
    sku_id           BIGINT        NOT NULL REFERENCES skus (sku_id),
    quantity         INTEGER       NOT NULL CHECK (quantity > 0),
    unit_ref_price   NUMERIC(12,2) NOT NULL CHECK (unit_ref_price > 0)
);

CREATE INDEX IF NOT EXISTS idx_lot_items_lot ON lot_items (lot_id);
CREATE INDEX IF NOT EXISTS idx_lot_items_sku ON lot_items (sku_id);

CREATE TABLE IF NOT EXISTS lot_bids (
    bid_id      BIGINT        PRIMARY KEY,
    lot_id      BIGINT        NOT NULL REFERENCES lots (lot_id),
    buyer_id    BIGINT        NOT NULL REFERENCES buyers (buyer_id),
    bid_amount  NUMERIC(12,2) NOT NULL CHECK (bid_amount > 0),
    bid_at      TIMESTAMP     NOT NULL,
    is_proxy    BOOLEAN       NOT NULL,
    popcorn     BOOLEAN       NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_lot_bids_lot ON lot_bids (lot_id);
CREATE INDEX IF NOT EXISTS idx_lot_bids_buyer ON lot_bids (buyer_id);
CREATE INDEX IF NOT EXISTS idx_lot_bids_popcorn ON lot_bids (popcorn) WHERE popcorn = TRUE;

CREATE TABLE IF NOT EXISTS model_predictions (
    prediction_id   BIGINT        PRIMARY KEY,
    target_type     VARCHAR(16)   NOT NULL CHECK (target_type IN ('sku_offer', 'lot')),
    target_id       BIGINT        NOT NULL,
    predicted       NUMERIC(12,2) NOT NULL CHECK (predicted > 0),
    actual          NUMERIC(12,2) NOT NULL CHECK (actual > 0),
    predicted_at    TIMESTAMP     NOT NULL,
    model_version   VARCHAR(32)   NOT NULL
);

-- Polymorphic pattern: target_type + target_id must be unique so joins are safe.
CREATE UNIQUE INDEX IF NOT EXISTS ux_predictions_target
    ON model_predictions (target_type, target_id, model_version);
CREATE INDEX IF NOT EXISTS idx_predictions_target_type ON model_predictions (target_type);
CREATE INDEX IF NOT EXISTS idx_predictions_predicted_at ON model_predictions (predicted_at);

-- Convenience view: segment-level MAPE for SKU offers joined to device attributes.
-- Used by the dashboard; kept in schema.sql so every environment gets it.
CREATE OR REPLACE VIEW v_sku_offer_mape AS
SELECT
    d.device_category,
    s.condition_grade,
    DATE_TRUNC('month', mp.predicted_at) AS month,
    COUNT(*)                             AS n_predictions,
    AVG(ABS(mp.predicted - mp.actual) / mp.actual) AS mape
FROM model_predictions mp
JOIN sku_offers so   ON mp.target_type = 'sku_offer' AND so.offer_id = mp.target_id
JOIN sku_listings sl ON sl.listing_id = so.listing_id
JOIN skus s          ON s.sku_id = sl.sku_id
JOIN devices d       ON d.device_id = s.device_id
WHERE mp.target_type = 'sku_offer'
GROUP BY d.device_category, s.condition_grade, DATE_TRUNC('month', mp.predicted_at);

COMMIT;
