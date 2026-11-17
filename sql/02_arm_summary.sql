-- Per-arm summary of the experiment: group sizes and outcome rates.
-- Requires the `customers` table from 01_load_clean.sql.
--
-- spend_per_customer is the main spend metric (all customers, zeros included).
-- spend_per_converter / median_spend_converter are descriptive only: they condition
-- on conversion, which the email itself can change, so differences between arms
-- are not causal effects on basket size.

SELECT
    arm,
    count(*) AS n,
    count(*) / sum(count(*)) OVER () AS share,
    sum(visit) AS visits,
    avg(visit) AS visit_rate,
    sum(conversion) AS conversions,
    avg(conversion) AS conversion_rate,
    sum(spend) AS total_spend,
    avg(spend) AS spend_per_customer,
    stddev_samp(spend) AS spend_sd,
    avg(spend) FILTER (WHERE conversion = 1) AS spend_per_converter,
    median(spend) FILTER (WHERE conversion = 1) AS median_spend_converter
FROM customers
GROUP BY arm
ORDER BY arm;
