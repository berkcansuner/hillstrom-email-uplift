-- Data quality checks on the cleaned `customers` table (from 01_load_clean.sql).
-- Returns one row. Every column except n_rows counts violating rows and should be 0.

SELECT
    count(*) AS n_rows,
    count(*) FILTER (
        WHERE recency IS NULL OR history_segment IS NULL OR history IS NULL
           OR mens IS NULL OR womens IS NULL OR zip_code IS NULL OR newbie IS NULL
           OR channel IS NULL OR segment IS NULL OR visit IS NULL
           OR conversion IS NULL OR spend IS NULL
    ) AS rows_with_missing,
    count(*) FILTER (WHERE arm IS NULL) AS unmapped_arm,
    count(*) - count(DISTINCT customer_id) AS duplicate_ids,
    count(*) FILTER (WHERE spend < 0 OR history <= 0 OR recency < 1) AS invalid_amounts,
    -- Outcome logic: spend > 0 <=> converted, and converted => visited.
    count(*) FILTER (WHERE spend > 0 AND conversion = 0) AS spend_without_conversion,
    count(*) FILTER (WHERE spend = 0 AND conversion = 1) AS conversion_without_spend,
    count(*) FILTER (WHERE conversion = 1 AND visit = 0) AS conversion_without_visit,
    -- Population is past-year purchasers, so each must have bought mens or womens.
    count(*) FILTER (WHERE mens = 0 AND womens = 0) AS no_purchase_history
FROM customers;
