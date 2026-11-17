-- Per-arm outcome summary within each level of the pre-treatment segment variables.
-- Requires the `customers` table from 01_load_clean.sql.
-- Output is long format: one row per (dimension, level, arm).
--
-- UNPIVOT turns the segment columns into (dimension, level) pairs so that a single
-- GROUP BY covers every dimension. Adding a dimension means adding one name to
-- both lists below.

WITH segments AS (
    SELECT
        arm,
        visit,
        conversion,
        spend,
        history_segment,
        channel,
        zip_code,
        CASE newbie WHEN 1 THEN 'new' ELSE 'existing' END AS newbie,
        purchase_group
    FROM customers
),
long AS (
    UNPIVOT segments
    ON history_segment, channel, zip_code, newbie, purchase_group
    INTO NAME dimension VALUE level
)
SELECT
    dimension,
    level,
    arm,
    count(*) AS n,
    avg(visit) AS visit_rate,
    sum(conversion) AS conversions,
    avg(conversion) AS conversion_rate,
    avg(spend) AS spend_per_customer
FROM long
GROUP BY dimension, level, arm
ORDER BY dimension, level, arm;
