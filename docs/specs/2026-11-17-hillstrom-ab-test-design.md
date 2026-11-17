# Hillstrom Email A/B Test: Design

*2026-11-17. Portfolio project for Berk Can Suner. To be moved into the project repo once it exists.*

## Intent

Berk is applying to data scientist roles. Recent postings (Ruby Labs, FunnelFox) asked for A/B testing and hands-on SQL. His CV lists SQL but no work bullet shows it, and he has no A/B testing evidence. This project is public proof of both, at data-scientist depth: not only "did the email work?" but "who should get which email?".

It should also connect to his story:
- Covariate adjustment is his MSc thesis topic (ROC curve length under covariates).
- Deciding whom to target is what he did with the PMT model at the Turkish Red Crescent.

It will sit on the CV next to the CDISC project (which already shows R), as a Python + SQL project.

## Goals

- A recruiter gets the result from the README in about 2 minutes.
- A technical interviewer sees correct, careful experiment analysis: validity checks, effect estimates with uncertainty, multiple comparisons, power, variance reduction.
- Data-scientist depth: a heterogeneous treatment effect / uplift model that turns into a targeting policy with an estimated revenue impact.
- SQL is visibly used for real work (data prep and aggregates), not as decoration.
- Reproducible: someone can clone and re-run end to end.
- Done within 1-2 weeks (by ~2026-12-01).

## Non-goals

- No dashboard or web app (Streamlit etc.).
- No Bayesian A/B framework, sequential testing, or multi-armed bandits.
- No custom causal-ML library; use established packages.

## Anti-goals (would make it a failure even if it runs)

- Looks like a bootcamp notebook: just a t-test and a bar chart.
- Claims that the data can't support (e.g. causal claims about segments without uncertainty, revenue numbers without stating assumptions).
- Berk can't explain a method in an interview. Every method used must be one he can defend; prefer fewer methods done well over many done shallowly.
- Presented on the CV as work experience. It is a personal project.

## Data

Kevin Hillstrom MineThatData E-Mail Analytics Challenge (2008).
- 64,000 customers who purchased in the last 12 months, randomly split into thirds: Mens E-Mail, Womens E-Mail, No E-Mail.
- Outcomes over the following two weeks: `visit`, `conversion`, `spend`.
- Pre-treatment features: `recency`, `history_segment`, `history` (past-year spend), `mens`, `womens`, `zip_code` (Urban/Suburban/Rural), `newbie`, `channel` (Phone/Web/Multichannel).
- Source: the public CSV from minethatdata.com; also available via `sklift.datasets.fetch_hillstrom`. Store the raw CSV under `data/raw/` with its source URL noted; don't modify it.

## Approaches considered

1. **Cookie Cats mobile game retention test (90k players).** Classic product test, but a single table and two metrics. Thin SQL and no room for modeling. Rejected: not enough depth for DS roles.
2. **Hillstrom email test (chosen).** Three arms, skewed spend, meaningful pre-treatment covariates. Supports every analysis below and ties to Berk's thesis and PMT work.
3. **Criteo uplift dataset (13M rows).** Big-data SQL and uplift, but features are anonymized (f0, f1...), so no business story, and heavy for a laptop. Rejected.

Berk picked approach 2.

## How it's built (decided)

- Language: Python. Libraries: pandas, statsmodels, scipy, scikit-learn, plus an uplift package (scikit-uplift or causalml, builder's call).
- SQL: DuckDB, queries kept as `.sql` files and run from Python.
- Delivery: README with headline results and key charts up front; numbered notebooks for each step; shared code in a small `src/` package so notebooks stay readable.
- Separate public GitHub repo under github.com/berkcansuner.

## Analysis steps (all in scope)

1. **SQL prep (DuckDB).** Load raw CSV, type/clean columns, build per-arm and per-segment summary tables (counts, visit/conversion rates, spend stats). Saved queries are part of the deliverable.
2. **Validity.** Sample ratio mismatch check (chi-square vs expected 1/3 split). Covariate balance across arms (standardized mean differences).
3. **Main effects.** Each email vs No E-Mail, and Mens vs Womens, for visit, conversion, spend. Differences with confidence intervals. Multiple-comparison correction across the family of tests (state which method and why). Spend is zero-inflated and skewed: use bootstrap CIs, and report spend per customer and spend among converters separately.
4. **Variance reduction.** CUPED / regression adjustment using pre-treatment `history` (and possibly other covariates). Report variance reduction and what it means for required sample size. Link to the thesis theme in one or two sentences, without overclaiming.
5. **Power analysis.** Minimum detectable effect at the actual sample size for each metric; required sample size for a few effect sizes; how CUPED changes it.
6. **Who to target (uplift).** First, pre-specified segment effects (e.g. by `history_segment`, `channel`, `mens`/`womens` purchase history) with CIs. Then a model of individual uplift for each email vs no email (e.g. T-learner / X-learner or uplift tree). Evaluate with a Qini or uplift curve on held-out data.
7. **Business decision.** A targeting policy: who gets Mens, who gets Womens, who gets nothing. Estimated incremental visits/conversions/spend vs (a) email everyone and (b) email no one, on held-out data, with uncertainty and clearly stated assumptions.
   - *Policy objective (Claude's call, from builder check):* the policy maximizes incremental spend per customer net of email cost. Visits are also reported because spend uplift is noisy (~1% conversion). If the spend-based model does not beat random targeting on the Qini curve, say so plainly and show the visit-based model as the more reliable signal. That honest result is acceptable.
   - *Email cost (Claude's call):* no single assumed number. Show policy value across a range of per-email costs starting at 0, because at cost 0 the "send nothing" group only catches customers with negative estimated uplift, which is mostly noise.

## Constraints

- 1-2 weeks of calendar time.
- Must run on Berk's Windows laptop.
- Berk must be able to explain every step in an interview; the README/notebooks should explain the reasoning in plain language, not only show code.

## Left to the builder

- Exact folder layout, environment manager (pip/venv vs conda), and how notebooks import `src/`.
- Which uplift package and which meta-learner(s); which multiple-comparison method.
- Chart style, provided charts are readable and labeled.
- Testing approach for the `src/` helpers.
