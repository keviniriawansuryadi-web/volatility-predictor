# Simplify the Hypothesis Module — Design

**Date:** 2026-05-29
**Status:** Approved (pending spec review)

## Problem

The hypothesis layer is two modules totalling ~2,792 lines across 23 tests:

- `src/hypothesis_tests.py` — H1–H9 (926 lines)
- `modules/hypothesis_tests_L2.py` — H9–H23 (1,646 lines)

Review feedback: the module is too large and contains hand-rolled statistics
and fabricated inputs. The bulk of the volume is plotting code and long
docstrings, but two functions are genuinely homemade significance tests
(`steiger_dependent_corr`, `scheirer_ray_hare`) with no scipy/statsmodels
equivalent, and two tests (H19, H21) fabricate their inputs (proxy implied
vol, simulated 10-K topic scores) when real data is absent.

## Goal

Cut the hypothesis layer down to a single module with **5 high-value tests**,
each a thin wrapper around a trusted library (`scipy`, `statsmodels`). No
homemade significance tests, no fabricated data, no figure construction inside
the test functions.

Target: **1 module / 5 tests / ~300 lines** (down from 2 modules / 23 tests /
2,792 lines). Every reported number traceable to scipy or statsmodels.

## The 5 hypotheses

All five use real price/VIX data only — no sentiment (which is simulated in
this project) and no fabricated inputs.

| # | Function | Extracted from | Trusted library | Method |
|---|----------|----------------|-----------------|--------|
| 1 | `test_leverage_effect` | `src/hypothesis_tests.py` | `scipy.stats.mannwhitneyu` | Negative-return days → higher forward realized vol than positive-return days. Rank-biserial effect size. |
| 2 | `test_cross_sector_contagion` | `modules/hypothesis_tests_L2.py` | `statsmodels.tsa.stattools.grangercausalitytests` | Directed Granger-causality matrix over sector-average vol series; net-contagion ranking (out − in). |
| 3 | `test_variance_risk_premium` | `src/hypothesis_tests.py` (`analyze_variance_risk_premium`) | `scipy.stats.mannwhitneyu` | Negative VRP (RV > VIX) days → higher forward vol than positive-VRP days. |
| 4 | `test_earnings_vol_premium` | `src/hypothesis_tests.py` (`test_earnings_vol`) | `scipy` permutation test | Earnings-window realized vol vs all other days; permutation p-value. |
| 5 | `test_regime_dependent_leverage` | `modules/hypothesis_tests_L2.py` | `scipy.stats.bootstrap` (replaces manual loop) + `scipy` permutation | Leverage ratio (neg/pos forward vol) by vol regime; tests Extreme > Low. |

## Architecture

### New module: `src/hypotheses.py`

Single module, ~300 lines, containing the 5 `test_*` functions plus:

- `compile_summary(results: dict) -> pd.DataFrame` — collects the result dicts
  into a summary table (one row per hypothesis).
- A small private `_verdict(p)` helper for the significance string.

Dependencies it may import: `numpy`, `pandas`, `scipy.stats`,
`statsmodels.tsa.stattools`, and `src.regime` (`_label`, `REGIME_ORDER`) for
the regime-dependent test. No `matplotlib`, no `plotly`.

### Unified return contract

Every `test_*` function returns a dict with exactly these keys:

```python
{
    "hypothesis": str,    # e.g. "leverage_effect"
    "title":      str,    # human-readable name
    "statistic":  float,  # primary test statistic
    "p_value":    float,  # in [0, 1], or np.nan when available is False
    "effect":     float,  # effect size (rank-biserial, ratio diff, etc.)
    "n":          int,    # sample size used
    "conclusion": str,    # one-line plain-language verdict
    "available":  bool,   # False (+ honest conclusion) when data insufficient
}
```

Tests **degrade gracefully**: when the input data is insufficient they return
`available=False`, `p_value=np.nan`, and a conclusion explaining why — they do
not raise and do not fabricate inputs.

No `figure` key. Charting, if ever wanted, is the notebook's responsibility.

### What is removed

- Both old modules: `src/hypothesis_tests.py`, `modules/hypothesis_tests_L2.py`.
- Homemade significance tests: `steiger_dependent_corr`, `scheirer_ray_hare`.
- Manual bootstrap loop `_bootstrap_ci_mean` → replaced by `scipy.stats.bootstrap`.
- Fabricated-data tests (H19 proxy IV, H21 simulated topic scores) — dropped
  entirely (not in the surviving 5).
- All Plotly/matplotlib figure construction inside test functions.
- The L2 reporting helpers `compile_l2_summary` / `append_findings_report`
  (replaced by the simpler `compile_summary`).
- `modules/__init__.py` reference to the deleted module (cleaned up; the
  `modules/` package is removed if nothing else lives there).

### Effect sizes — kept inline (not "homemade theory")

These are textbook one-line formulas, not invented tests, and stay inline:

- Rank-biserial correlation `r = 1 - 2U / (n1·n2)` for Mann-Whitney tests.
- Leverage ratio `mean(fwd_vol | neg) / mean(fwd_vol | pos)`.

## Data flow

- Tests 1, 3, 4, 5 take a single price `df` (with `log_return`,
  `realized_vol_21d`, and for VRP `vix_level`); test 4 also takes
  `earnings_dates`.
- Test 2 takes `df_dict: {ticker: df}` and a `sectors` map (default
  `DEFAULT_SECTORS`), aggregating each sector to a mean vol series before the
  Granger matrix.
- Test 5 takes a price `df` and derives the regime label via `src.regime`.

## Testing

New `tests/test_hypotheses.py`:

- Build small synthetic price frames (and a `df_dict` of 2–3 synthetic tickers
  for the contagion test) with a fixed seed.
- For each of the 5 tests, assert the return-dict contract: all required keys
  present; `available` is bool; when `available` is True then
  `0 <= p_value <= 1` and `statistic`/`effect` are finite; no exceptions.
- One degraded-input case per test asserting `available=False` rather than a
  raise.

## Knock-on edits

- `notebooks/hypotheses.ipynb` — update imports/calls to use `src.hypotheses`
  and the 5-function API; remove cells for dropped hypotheses.
- `notebooks/07_advanced_hypotheses.ipynb` — retired (deleted); it is the L2
  notebook and the L2 module is gone.
- `README.md` — trim the hypothesis section from H1–H23 to the 5 surviving
  hypotheses.

**Open item for spec review:** confirm notebook handling — retire
`07_advanced_hypotheses.ipynb` outright vs. fold a short version into
`hypotheses.ipynb`. Default: retire it.

## Non-goals

- No changes to the sentiment simulation layer (`src/data_helpers.py`).
- No changes to the forecasting models (GARCH/HAR/ML) or portfolio code.
- No new third-party dependencies (`scipy`/`statsmodels` already in use).

## Expected outcome

- `src/hypotheses.py`: 5 tests, ~300 lines, every statistic from scipy or
  statsmodels.
- ~2,792 lines of hypothesis code deleted.
- Green `tests/test_hypotheses.py`.
