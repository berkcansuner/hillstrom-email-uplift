# Hillstrom Email Campaign: A/B Test and Uplift Analysis

End-to-end analysis of a randomized email experiment: did the emails work, by how much, and **who should get which email**?

*Personal portfolio project. Work in progress — results will be added as each analysis step is completed.*

## The experiment

Kevin Hillstrom's MineThatData E-Mail Analytics Challenge (2008): 64,000 customers who purchased in the last 12 months were randomly split into three equal groups:

- **Mens E-Mail** — received an email featuring men's merchandise
- **Womens E-Mail** — received an email featuring women's merchandise
- **No E-Mail** — control group

Outcomes were tracked for the following two weeks: website `visit`, `conversion` (purchase), and `spend`.

## Results

*To be added.*

## Analysis steps

| # | Step | Notebook | Status |
|---|------|----------|--------|
| 1 | SQL data prep (DuckDB) | `notebooks/01_sql_prep.ipynb` | done |
| 2 | Validity checks: sample ratio mismatch, covariate balance | `notebooks/02_validity.ipynb` | done |
| 3 | Main effects with confidence intervals and multiple-comparison correction | `notebooks/03_main_effects.ipynb` | planned |
| 4 | Variance reduction (CUPED / regression adjustment) | `notebooks/04_cuped.ipynb` | planned |
| 5 | Power analysis | `notebooks/05_power_analysis.ipynb` | planned |
| 6 | Segment effects and uplift model | `notebooks/06_uplift.ipynb` | planned |
| 7 | Targeting policy and estimated business impact | `notebooks/07_targeting_policy.ipynb` | planned |

Design document: [`docs/specs/2026-11-17-hillstrom-ab-test-design.md`](docs/specs/2026-11-17-hillstrom-ab-test-design.md)
Implementation plan: [`docs/plans/2026-11-17-hillstrom-ab-test-plan.md`](docs/plans/2026-11-17-hillstrom-ab-test-plan.md)

## Repository layout

```
sql/            DuckDB queries for data prep and summary tables
src/hillstrom/  tested helper functions used by the notebooks
tests/          pytest tests for src/
notebooks/      one notebook per analysis step
scripts/        data download
data/raw/       raw CSV (downloaded, not committed)
reports/figures charts used in this README
```

## Reproduce

Requires Python 3.12. On Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
python scripts/download_data.py
pytest
```

On macOS/Linux, activate with `source .venv/bin/activate`.

Note: `scikit-uplift` 0.5.1 emits a deprecation warning with scikit-learn 1.9 and is expected to break with scikit-learn 1.10; versions are pinned in `requirements.txt` for this reason.
