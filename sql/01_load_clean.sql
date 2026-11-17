-- Load the raw Hillstrom CSV into a typed, cleaned `customers` table.
-- Parameter: $csv_path (path to the raw CSV). The raw file itself is never modified.
--
-- Cleaning steps:
--   * explicit column types instead of type sniffing
--   * customer_id = row position in the raw file (1-based)
--   * zip_code: recode the raw spelling 'Surburban' to 'Suburban'
--   * arm: short treatment labels used throughout the analysis
--   * purchase_group: past-year purchase history (mens only / womens only / both),
--     a pre-treatment variable used for segment analysis

CREATE OR REPLACE TABLE customers AS
WITH raw AS (
    SELECT *
    FROM read_csv(
        $csv_path,
        header = true,
        columns = {
            'recency': 'INTEGER',
            'history_segment': 'VARCHAR',
            'history': 'DOUBLE',
            'mens': 'INTEGER',
            'womens': 'INTEGER',
            'zip_code': 'VARCHAR',
            'newbie': 'INTEGER',
            'channel': 'VARCHAR',
            'segment': 'VARCHAR',
            'visit': 'INTEGER',
            'conversion': 'INTEGER',
            'spend': 'DOUBLE'
        }
    )
)
SELECT
    row_number() OVER () AS customer_id,
    recency,
    history_segment,
    history,
    mens,
    womens,
    CASE
        WHEN mens = 1 AND womens = 1 THEN 'both'
        WHEN mens = 1 THEN 'mens_only'
        WHEN womens = 1 THEN 'womens_only'
    END AS purchase_group,
    CASE WHEN zip_code = 'Surburban' THEN 'Suburban' ELSE zip_code END AS zip_code,
    newbie,
    channel,
    segment,
    CASE segment
        WHEN 'Mens E-Mail' THEN 'mens'
        WHEN 'Womens E-Mail' THEN 'womens'
        WHEN 'No E-Mail' THEN 'control'
    END AS arm,
    visit,
    conversion,
    spend
FROM raw;
