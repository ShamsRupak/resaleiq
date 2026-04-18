-- Phase 3 migration: add conformal prediction interval columns to model_predictions.
--
-- These columns are populated only for rows where model_version produces
-- conformal intervals (currently targeted_v1_xgb). Other rows leave them NULL.
--
-- Run with: psql -d resaleiq -f migrations/001_add_conformal_intervals.sql

BEGIN;

ALTER TABLE model_predictions
    ADD COLUMN IF NOT EXISTS predicted_low  NUMERIC(12,2),
    ADD COLUMN IF NOT EXISTS predicted_high NUMERIC(12,2);

-- Convenience view: surface interval width and whether actual lies inside.
CREATE OR REPLACE VIEW v_prediction_intervals AS
SELECT
    prediction_id,
    target_type,
    target_id,
    model_version,
    predicted,
    actual,
    predicted_low,
    predicted_high,
    CASE
        WHEN predicted_low IS NULL OR predicted_high IS NULL THEN NULL
        ELSE predicted_high - predicted_low
    END AS interval_width,
    CASE
        WHEN predicted_low IS NULL OR predicted_high IS NULL THEN NULL
        WHEN actual BETWEEN predicted_low AND predicted_high THEN TRUE
        ELSE FALSE
    END AS actual_in_interval
FROM model_predictions;

COMMIT;
