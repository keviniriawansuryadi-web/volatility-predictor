# Design: Revive the Level-2 Advanced Hypothesis Suite (H9–H23)

**Date:** 2026-05-31
**Status:** Approved (pending spec review)
**Branch:** to be created — `revive-l2-hypotheses`

## Background

The `simplify-hypothesis-module` plan (merged `a304b62`) deleted the Level-2
suite `modules/hypothesis_tests_L2.py` (H9–H23, ~1,646 lines) and salvaged only
two of its tests (H15 cross-sector contagion, H16 regime-dependent leverage)
into `src/hypotheses.py` in lean, library-backed form. A copy of the deleted L2
module has since been restored to the working tree as an **untracked** file
(with one already-fixed bug: a `grangercausalitytests` typo at line 748). It is
imported by nothing and runs nowhere.

This project revives the full L2 suite as a working, runnable, tested research
module that coexists with `src/hypotheses.py`.

## Goals

- All 15 `test_*` functions (H9–H23) runnable, including L2's own **rich**
  H15/H16 (directed network graph, fuller per-regime bootstrap CIs).
- Two runnable surfaces: a headless runner script and an interactive notebook.
- Full TDD parity: contract + degraded test per function, plus pinned unit tests
  for the two hand-rolled statistics.
- The module is tracked in git.

## Non-goals (explicit decisions)

- **No rewrites of working internals.** Bugs and a stale docstring reference get
  fixed; logic is otherwise preserved as-is. `_bootstrap_ci_mean` is *not*
  swapped for `scipy.stats.bootstrap`.
- **`src/hypotheses.py` is untouched.** Its lean H15/H16 stay as the production
  path. The two modules coexist intentionally (lean production vs. rich
  research). Two functions sharing a name across two modules is acceptable.
- No new dependencies — `plotly`, `scipy`, `statsmodels`, `scikit_posthocs` are
  already in `requirements.txt`.
- No revival of the deleted L1 module (`src/hypothesis_tests.py`, H1–H8). The L2
  docstrings reference H1–H8 only conceptually.

## Architecture

### Module placement

Keep the module at its historical home `modules/hypothesis_tests_L2.py` (the
`modules/` package with `__init__.py` already exists and is on `sys.path` for
scripts/notebooks). Track it in git. Imported as `modules.hypothesis_tests_L2`.

Its only `src` dependency is `from src.regime import _label, REGIME_ORDER`
(still present). One docstring touch: soften the line referencing the deleted
`src/hypothesis_tests.py` so it no longer points at a non-existent file.

### The 15 tests and their statistics

| H | Function | Inputs | Statistic | Backed by |
|---|----------|--------|-----------|-----------|
| H9 | `test_sentiment_asymmetry` | df, sent | Kruskal-Wallis + Dunn | scipy + scikit_posthocs |
| H10 | `test_sentiment_velocity` | df, sent | Spearman + **Steiger** | scipy + custom |
| H11 | `test_sentiment_model_consensus` | df, sent | Mann-Whitney | scipy |
| H12 | `test_disagreement_persistence` | df, disagreement | AR(1)/half-life | scipy.linregress |
| H13 | `test_disagreement_direction` | df, disagreement | sign test | scipy.binomtest |
| H14 | `test_asymmetric_contagion` | df_dict | interaction OLS | statsmodels |
| H15 | `test_cross_sector_contagion` | df_dict | Granger matrix | statsmodels |
| H16 | `test_regime_dependent_leverage` | df | bootstrap + permutation | custom + numpy |
| H17 | `test_monday_leverage_interaction` | df | **Scheirer-Ray-Hare** | custom |
| H18 | `test_pre_earnings_sentiment_predicts_spike` | df, sent, earnings_dates | Spearman + Mann-Whitney | scipy |
| H19 | `test_earnings_vol_premium_vs_iv` | df, earnings_dates, [iv_series] | Wilcoxon | scipy |
| H20 | `test_10k_language_change_predicts_regime` | df, lm_scores | Fisher exact + OR | scipy |
| H21 | `test_topic_specific_risk_prediction` | df_dict, [topic_scores] | Spearman + Steiger | scipy + custom |
| H22 | `test_triple_signal_interaction` | df, sent, disagreement | Mann-Whitney + OLS interaction | scipy + statsmodels |
| H23 | `test_signal_temporal_precedence` | df, sent, disagreement, [df_dict, filing_dates] | Friedman | scipy |

Every function already degrades gracefully (`available=False` + honest reason,
no raise) when data is insufficient; H19/H21 fall back to a transparent PROXY
input flagged in the result. Results helpers `compile_l2_summary` and
`append_findings_report` aggregate H9–H23 into a summary table / findings report.

### Runnable surfaces (shared data assembly)

A single assembly helper builds the inputs once so the script and notebook don't
duplicate logic. It produces: per-ticker `df`, universe `df_dict`, `sent`
(`src.data_helpers.simulate_sentiment`), `disagreement`
(`compute_disagreement(garch_preds, ml_preds)` from `src.garch_model` /
`src.ml_model`), `earnings_dates`, and `lm_scores`.

- **`scripts/run_advanced_hypotheses.py`** — headless. Assembles inputs, runs
  all 15, prints conclusions, saves figures to disk, writes the
  `compile_l2_summary` CSV. CI-friendly, no notebook execution required.
- **`notebooks/07_advanced_hypotheses.ipynb`** — revived from `e6be0ae~1`,
  repointed to `from modules.hypothesis_tests_L2 import ...`, figures inline for
  interactive research.

## Testing — full TDD parity

New file `tests/test_hypotheses_L2.py`.

- **Contract test per function (15):** run on fixtures; assert returned dict has
  the documented keys, `available=True`, and a sane p-value (NaN or 0≤p≤1).
- **Degraded test per function (15):** feed insufficient/empty data; assert
  `available=False` and that it returns (does not raise).
- **Pinned unit tests for hand-rolled stats:** `steiger_dependent_corr` and
  `scheirer_ray_hare` checked against known textbook / R reference values — these
  have no library to validate against and are the highest-risk math.
- **`compile_l2_summary`** test: mixed available/unavailable results produce the
  expected table shape and Significant column values.

New pytest fixtures (alongside reused `price_df` / `earnings_dates` / `df_dict`
patterns from `tests/test_hypotheses.py`): `sent` (four sentiment columns),
`disagreement` (a Series), `lm_scores` (filing-date-indexed Series). Fixtures use
seeded synthetic data sized so the contract path is exercised and the degraded
path is reachable with a trimmed variant.

## Implementation sequencing

TDD, executed via the executing-plans workflow on branch `revive-l2-hypotheses`:

1. Track the module + cleanup: add `modules/hypothesis_tests_L2.py` to git (typo
   already fixed), soften the stale docstring ref. Commit.
2. Pinned unit tests for `steiger_dependent_corr` + `scheirer_ray_hare` (write
   failing → confirm against reference values → commit).
3. Per hypothesis H9→H23 (~one commit each): write failing contract + degraded
   test → run → commit. ~15 commits.
4. `compile_l2_summary` test + fixture wiring as needed.
5. `scripts/run_advanced_hypotheses.py` — runs headless end-to-end; commit.
6. `notebooks/07_advanced_hypotheses.ipynb` revived + repointed; commit last.

Run `python -m pytest tests/test_hypotheses_L2.py -v` green before each commit.

## Risks / open points

- **Steiger/SRH reference values:** need a trusted source (textbook example or R
  `psych::r.test` / SRH worked example) to pin against. If none is found cleanly,
  fall back to property-based assertions (symmetry, known-zero cases) and flag.
- **PROXY tests (H19, H21):** tests assert the PROXY path runs and sets the
  `proxy`/`proxy_iv` flag; they do not assert exact p-values on simulated data.
- **Notebook data volume:** assembling `df_dict` for the full sector universe +
  GARCH/ML disagreement is slow; the runner script caps the universe via config
  and the notebook documents expected runtime.
