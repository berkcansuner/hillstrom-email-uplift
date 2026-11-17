# Hillstrom Email A/B Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An end-to-end, reproducible analysis of the Hillstrom email experiment: SQL prep, validity checks, main effects, uplift-based targeting policy, CUPED, and power analysis, delivered as a public GitHub repo.

**Architecture:** DuckDB `.sql` files do data prep and aggregates; a small `src/hillstrom/` package holds tested statistical helpers; numbered notebooks call both and explain the reasoning in plain language; the README carries the headline results.

**Tech Stack:** Python 3.12, venv + pip, DuckDB, pandas, numpy, scipy, statsmodels, scikit-learn, scikit-uplift (Qini metrics only), matplotlib, pytest, Jupyter.

**Spec:** `docs/specs/2026-11-17-hillstrom-ab-test-design.md`

## Global Constraints

- Build order (from Berk): steps 1-3 complete first, then 6-7, then 4-5. Target finish: 2026-12-01.
- Notebook numbers follow the spec's step numbers (01-07), not build order. Git history shows the real order.
- Raw CSV is never modified. Source URL recorded in `data/raw/SOURCE.md`.
- Must run on Windows: all paths via `pathlib`, relative to repo root (`src/hillstrom/paths.py`).
- Every random operation takes an explicit `seed`; project default `SEED = 20081`.
- All README, notebook, code and comment text in English.
- No dashboard, no Bayesian/sequential/bandit methods, no custom causal-ML library.
- README states this is a personal project.
- Commits: one small meaningful unit each, imperative sentence-case messages ("Add SRM chi-square check"). No rebase, squash, amend of pushed commits, or date changes.
- Push only after showing Berk the commit list and getting approval.

## Method choices (to defend in interview)

| Decision | Choice | Rejected alternatives and why |
|---|---|---|
| Rate CIs | Wald CI for difference in proportions | Exact/Newcombe: Wald is fine with ~21k per arm and ~200+ conversions; easier to explain |
| Spend CI | Percentile bootstrap (10,000 resamples, seeded) | t-interval: skewed, zero-inflated; bootstrap makes no shape assumption |
| Spend p-value | Welch t-test (CLT holds at n≈21k), cross-checked against bootstrap CI | Mann-Whitney: tests distribution shift, not the mean, which is the business quantity |
| Multiple comparisons | Holm over the 9-test family (3 contrasts × 3 metrics) | Bonferroni: uniformly less powerful. BH/FDR: these are confirmatory tests, so control FWER |
| Spend among converters | Reported as descriptive only | Not causal: conditioning on a post-treatment outcome |
| Balance | Standardized mean differences, threshold 0.1 | Per-covariate t-tests: significance grows with n, SMD measures size |
| Uplift model | T-learner (one gradient-boosting model per arm, sklearn `HistGradientBoosting*`) | X-learner: helps mainly with unbalanced arms; ours are 1/3 each. Uplift trees / causal forests: harder to defend, need extra libraries |
| Uplift evaluation | Qini curve + Qini coefficient on 30% held-out split (sklift) | In-sample evaluation: leaks |
| Policy value | IPW estimate on held-out with known p = 1/3, bootstrap CI | Model-predicted value: optimistic, depends on the model being right |
| Variance reduction | CUPED with `history`; then OLS regression adjustment with all pre-treatment covariates, HC1 SEs | Stratification: CUPED/regression are the industry standard and map to the thesis |
| Power | Closed-form MDE / required n for two-sample means & proportions, α=0.05, power 0.8, cross-checked against statsmodels | Simulation-based power: unnecessary for these designs |

## File Structure

```
README.md
pyproject.toml              # package metadata, makes `hillstrom` importable (pip install -e .)
requirements.txt            # pinned versions (pip freeze subset)
.gitignore                  # .venv, data/raw/*.csv, data/processed/, .ipynb_checkpoints
data/raw/SOURCE.md          # URL, download date, size, SHA-256
data/processed/             # DuckDB file (gitignored)
scripts/download_data.py    # downloads CSV, verifies SHA-256
sql/01_load_clean.sql       # typed `customers` table with customer_id, arm
sql/02_arm_summary.sql      # per-arm counts, rates, spend stats
sql/03_segment_summary.sql  # per-arm × segment summaries
src/hillstrom/paths.py      # REPO_ROOT, RAW_CSV, DB_PATH, SQL_DIR, SEED
src/hillstrom/data.py       # run_sql_file, load_customers
src/hillstrom/validity.py   # srm_test, standardized_mean_diff
src/hillstrom/effects.py    # diff_proportions, diff_means_welch, bootstrap_diff_ci, holm_adjust
src/hillstrom/uplift.py     # TLearner, qini_summary
src/hillstrom/policy.py     # assign_policy, policy_value_ipw, bootstrap_policy_value
src/hillstrom/variance.py   # cuped_adjust, regression_adjusted_effect
src/hillstrom/power.py      # mde_means, mde_proportions, required_n_means
src/hillstrom/plots.py      # shared chart helpers (added when first chart needs it)
tests/test_*.py             # one file per module, synthetic data only
notebooks/01_sql_prep.ipynb ... 07_power_analysis.ipynb
reports/figures/            # PNGs used by README
```

The raw CSV is gitignored and fetched by `scripts/download_data.py` (avoids redistributing third-party data; checksum guarantees everyone analyses the same file).

## Review Focus

- Small segments with very few or zero conversions in an arm: helpers must return a finite CI or a clear warning, not NaN silently or crash. → test in Task 4.
- Bootstrap reproducibility: same seed must give the identical CI. → test in Task 4.
- Policy evaluated on the data it was trained on would be optimistic. → `policy_value_ipw` is only ever called on the held-out split; test in Task 7 checks IPW is unbiased on synthetic data with known value.
- `history_segment` values are strings like `"1) $0 - $100"`; ordering must be preserved in tables and plots. → test in Task 2 checks segment order.
- Wrong arm labels/case in raw CSV (`"Mens E-Mail"`, `"Womens E-Mail"`, `"No E-Mail"`): mapping must cover all rows. → test in Task 2 asserts no null `arm` and 64,000 rows.

---

## Phase A: Setup (first push)

### Task 1: Project skeleton

**Files:** Create `README.md` (skeleton: title, one-line pitch, "personal project" note, planned sections, setup instructions), `pyproject.toml`, `requirements.txt`, `.gitignore`, `src/hillstrom/__init__.py`, `src/hillstrom/paths.py`, `tests/test_paths.py`, empty dirs with `.gitkeep` (`sql/`, `notebooks/`, `reports/figures/`, `data/raw/`, `data/processed/`). Spec and this plan already in `docs/`.

**Interfaces:** Produces `hillstrom.paths.REPO_ROOT: Path`, `RAW_CSV: Path`, `DB_PATH: Path`, `SQL_DIR: Path`, `SEED: int`.

- [ ] Create `.venv`, install packages, verify `import sklift; from sklift.metrics import qini_auc_score` works with current scikit-learn. If not, pin the newest scikit-learn that works and note it. Freeze to `requirements.txt`.
- [ ] Test `test_repo_root_contains_pyproject`: `(REPO_ROOT / "pyproject.toml").exists()`. Run `pytest` → PASS.
- [ ] Commits: "Add design spec and implementation plan", "Add project skeleton and README outline", "Add pinned requirements and package config".
- [ ] **Ask Berk to approve creating public repo `github.com/berkcansuner/hillstrom-email-uplift`.** Create it, add remote, show commit list, get approval, push.

## Phase B: Steps 1-3

### Task 2: Data download and SQL prep (spec step 1)

**Files:** `scripts/download_data.py`, `data/raw/SOURCE.md`, `sql/01_load_clean.sql`, `sql/02_arm_summary.sql`, `sql/03_segment_summary.sql`, `src/hillstrom/data.py`, `tests/test_data.py`, `notebooks/01_sql_prep.ipynb`.

**Interfaces:**
- `run_sql_file(con: duckdb.DuckDBPyConnection, name: str, **params) -> pd.DataFrame | None` (reads `SQL_DIR/name`, runs it).
- `load_customers(con=None) -> pd.DataFrame` with columns: `customer_id, recency, history_segment, history, mens, womens, zip_code, newbie, channel, segment, arm ('mens'|'womens'|'control'), visit, conversion, spend`.

- [ ] Download (URL: `http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv`, 3,964,977 bytes), record SHA-256 in `SOURCE.md` and in the script.
- [ ] Tests (on a 6-row synthetic CSV written to `tmp_path`): `test_load_customers_maps_all_arms` (no null `arm`, values ⊆ {mens, womens, control}); `test_arm_summary_counts_sum_to_total`; `test_history_segment_order_preserved` (segment summary sorted `1)`..`7)`). One integration test marked `@pytest.mark.data` asserts 64,000 rows on the real file.
- [ ] Commit units: download script + SOURCE.md; `01_load_clean.sql` + `data.py` + tests; `02_arm_summary.sql`; `03_segment_summary.sql`; notebook 01.

### Task 3: Validity checks (spec step 2)

**Files:** `src/hillstrom/validity.py`, `tests/test_validity.py`, `notebooks/02_validity.ipynb`.

**Interfaces:**
- `srm_test(counts: Sequence[int], expected_ratios: Sequence[float]) -> SRMResult(chi2: float, p_value: float, counts, expected)`
- `standardized_mean_diff(x_treat: np.ndarray, x_ctrl: np.ndarray) -> float` (pooled SD: `sqrt((var_t + var_c)/2)`); `balance_table(df, covariates: list[str], arm_col="arm", control="control") -> pd.DataFrame`.

- [ ] Tests: `test_srm_equal_counts_p_is_one` (`[100,100,100]`, p == 1.0); `test_srm_detects_imbalance` (`[1000,1000,1200]` → p < 0.001); `test_smd_identical_is_zero`; `test_smd_known_value` (means 0 and 1, SD 1 → 1.0).
- [ ] Commits: SRM check + tests; SMD + balance table + tests; notebook 02 (one-hot categoricals, love plot).

### Task 4: Main effects (spec step 3)

**Files:** `src/hillstrom/effects.py`, `tests/test_effects.py`, `notebooks/03_main_effects.ipynb`, `src/hillstrom/plots.py`.

**Interfaces:**
- `diff_proportions(y_t, y_c, alpha=0.05) -> EffectResult(estimate, ci_low, ci_high, p_value, n_t, n_c)`
- `diff_means_welch(y_t, y_c, alpha=0.05) -> EffectResult`
- `bootstrap_diff_ci(y_t, y_c, n_boot=10_000, alpha=0.05, seed=SEED) -> tuple[float, float]`
- `holm_adjust(p_values: Sequence[float]) -> np.ndarray` (wraps `statsmodels.stats.multitest.multipletests(method="holm")`).

- [ ] Tests: `test_diff_proportions_known` (0.2 vs 0.1, n=1000 each → estimate 0.1, CI contains 0.1, matches hand-computed Wald SE); `test_diff_proportions_zero_events` (all zeros both arms → estimate 0, finite CI, no exception); `test_bootstrap_same_seed_identical`; `test_bootstrap_ci_covers_true_diff` (synthetic zero-inflated lognormal, known mean diff); `test_holm_matches_manual` (`[0.01, 0.04, 0.03]` → `[0.03, 0.06, 0.06]`).
- [ ] Notebook 03: 9-test table (Mens vs control, Womens vs control, Mens vs Womens × visit/conversion/spend) with raw and Holm p; spend bootstrap CIs; spend-among-converters labeled descriptive/non-causal; forest plot.
- [ ] Commits: each helper + its tests separately; notebook sections (rates, spend, multiple comparisons, interpretation).
- [ ] **Push checkpoint: show commits, get approval, push (steps 1-3 done).**

## Phase C: Steps 6-7

### Task 5: Pre-specified segment effects (spec step 6, part 1)

**Files:** `notebooks/06_uplift.ipynb` (first section). Reuses `effects.py`.

- [ ] Effects of each email vs control on visit and spend, within `history_segment`, `channel`, and purchase-history group (mens only / womens only / both). CIs, labeled exploratory (no confirmatory claims; note that segment CIs are wide).
- [ ] Commit: "Add pre-specified segment effects".

### Task 6: T-learner uplift model (spec step 6, part 2)

**Files:** `src/hillstrom/uplift.py`, `tests/test_uplift.py`, `notebooks/06_uplift.ipynb` (continued).

**Interfaces:**
- `split_train_test(df, test_size=0.3, seed=SEED) -> tuple[pd.DataFrame, pd.DataFrame]` (stratified by arm).
- `class TLearner(outcome: Literal["binary","continuous"], seed=SEED)` with `.fit(X, y, treated: np.ndarray[bool]) -> self` and `.predict_uplift(X) -> np.ndarray`. Uses `HistGradientBoostingClassifier` (predict_proba diff) or `HistGradientBoostingRegressor`.
- `qini_summary(y, uplift, treated) -> dict(qini_coef: float, curve: pd.DataFrame)` via `sklift.metrics`.

- [ ] Tests: `test_split_stratified` (arm proportions within 0.01 in both parts); `test_tlearner_recovers_sign` (synthetic: effect +0.2 when x>0, 0 otherwise → mean predicted uplift higher for x>0); `test_qini_random_scores_near_zero` (random uplift on no-effect data → |qini| small).
- [ ] Fit four models on train: {mens, womens} vs control × {visit, spend}. Qini curves on test vs random line. If spend models do not beat random, say so and lean on visit models (spec).
- [ ] Commits: split + test; TLearner + tests; qini helper + test; notebook sections.

### Task 7: Targeting policy (spec step 7)

**Files:** `src/hillstrom/policy.py`, `tests/test_policy.py`, `notebooks/07_targeting_policy.ipynb`.

**Interfaces:**
- `assign_policy(uplift_mens, uplift_womens, cost: float) -> np.ndarray[str]` (argmax of `{control: 0, mens: u_m - cost, womens: u_w - cost}`).
- `policy_value_ipw(policy: np.ndarray, arm: np.ndarray, y: np.ndarray, cost: float, p=1/3) -> float` = `mean(1[arm==policy] * y / p) - cost * mean(policy != "control")`.
- `bootstrap_policy_value(..., n_boot=2000, seed=SEED) -> tuple[float, float, float]`.

- [ ] Tests: `test_assign_policy_high_cost_sends_nothing`; `test_assign_policy_picks_larger_uplift`; `test_ipw_unbiased_synthetic` (randomized synthetic data with known per-arm means; IPW value of "all mens" within 3 SE of truth); `test_ipw_cost_subtracted`.
- [ ] Notebook 07: on held-out only, compare model policy vs email-all-Mens, email-all-Womens, email-no-one, random targeting at same email share; across cost grid 0.00-1.00 $/email; incremental visits/conversions/spend with bootstrap CIs; assumptions box (2-week horizon, cost per email, no long-term/unsubscribe effects, 2008 data).
- [ ] Commits: each helper + tests; notebook sections.
- [ ] **Push checkpoint.**

## Phase D: Steps 4-5

### Task 8: CUPED and regression adjustment (spec step 4)

**Files:** `src/hillstrom/variance.py`, `tests/test_variance.py`, `notebooks/04_cuped.ipynb`.

**Interfaces:**
- `cuped_adjust(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, float]` (θ = cov(y,x)/var(x) pooled across arms; returns `y - θ(x - mean(x))`, θ).
- `regression_adjusted_effect(df, outcome: str, treat_arm: str, covariates: list[str]) -> EffectResult` (statsmodels OLS, HC1, centered covariates).

- [ ] Tests: `test_cuped_theta_known` (y = 2x + noise → θ ≈ 2); `test_cuped_preserves_mean_diff` (adjusted diff ≈ raw diff on randomized data); `test_cuped_reduces_variance` (var ratio ≈ 1 − ρ²).
- [ ] Notebook 04: variance reduction per metric, CI width before/after, implied sample-size saving (factor 1 − ρ²), honest statement if `history` correlates weakly with 2-week outcomes; 1-2 sentence link to thesis (covariate adjustment), no overclaiming.
- [ ] Commits: each helper + tests; notebook sections.

### Task 9: Power analysis (spec step 5)

**Files:** `src/hillstrom/power.py`, `tests/test_power.py`, `notebooks/05_power_analysis.ipynb`.

**Interfaces:**
- `mde_means(sd: float, n_t: int, n_c: int, alpha=0.05, power=0.8) -> float` = `(z_{1-α/2} + z_power) * sd * sqrt(1/n_t + 1/n_c)`.
- `mde_proportions(p_base: float, n_t, n_c, alpha=0.05, power=0.8) -> float` (same, sd = sqrt(p(1-p))).
- `required_n_means(sd, mde, alpha=0.05, power=0.8) -> int` (per arm, ceil).

- [ ] Tests: `test_required_n_matches_statsmodels` (vs `TTestIndPower` with effect size mde/sd, within 1%); `test_mde_and_required_n_inverse`; `test_cuped_shrinks_required_n` (sd·sqrt(1−ρ²) → n × (1−ρ²)).
- [ ] Notebook 05: MDE per metric at actual n; required n for a few effect sizes; with/without CUPED.
- [ ] Commits; **push checkpoint.**

### Task 10: README results and reproducibility check

- [ ] README: headline results up front (≤ 2 min read), 3-4 key figures from `reports/figures/`, "what we can't claim", how to reproduce.
- [ ] Fresh clone → new venv → `pip install -r requirements.txt && pip install -e .` → `python scripts/download_data.py` → `pytest` → execute all notebooks with `jupyter nbconvert --execute`. All must pass.
- [ ] Commits; **final push checkpoint.**

## Rough schedule

| Dates | Work |
|---|---|
| Nov 17-18 | Task 1, first push; Task 2 |
| Nov 19-20 | Tasks 3-4, push |
| Nov 21-25 | Tasks 5-7, push |
| Nov 26-28 | Tasks 8-9, push |
| Nov 29-Dec 1 | Task 10, buffer |
