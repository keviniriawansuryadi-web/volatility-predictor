# Design — Stronger L2 Advanced Hypotheses (replace H9, H14, H16)

**Date:** 2026-06-11
**Branch:** `revive-l2-hypotheses`
**Status:** approved design, pending spec review

## Problem

Three of the fifteen Level-2 advanced hypotheses are the weak links of the suite. In the
latest run (`outputs/results/hypothesis_L2_summary.csv`) they are non-findings:

| Slot | Old test | Result | Root cause of weakness |
|------|----------|--------|------------------------|
| **H9** (extends H1) | `test_sentiment_asymmetry` — 3-group Kruskal-Wallis + Dunn | p = 0.44, "No" | Splitting into Neg/Neu/Pos *and* gating on an asymmetry condition burns statistical power; never controls for the return channel. |
| **H14** (extends H4) | `test_asymmetric_contagion` — 1-day-lead sign-interaction OLS | p = 0.22, "No" | A single 1-day lead with a positive/negative sign split is underpowered and ignores the full lead-lag structure. |
| **H16** (extends H5) | `test_regime_dependent_leverage` — Low-vs-Extreme permutation | inconclusive ("--") | Discrete regime buckets leave too few Extreme-regime days in MU's window for the permutation test. |

The robust L2 findings (H12/H15/H17/H22) all lean on **price/vol data**; the fragile ones lean on
simulated sentiment and proxy inputs. The fix is to replace these three with higher-power tests that
stay on price data wherever possible, and — for the sentiment slot — to reframe around the part of the
simulated signal that is genuinely real (its correlation with returns).

## Goal

Replace H9, H14, H16 with stronger hypotheses that:
- keep the same hypothesis IDs and the same Level-1 lineage (H9→H1, H14→H4, H16→H5),
- preserve the existing return-dict contract and graceful degradation,
- keep the suite at exactly 15 `test_*` functions,
- are re-run end-to-end so the summary CSV and the `06_conclusions.ipynb` Section 2B reflect them.

This is **not** a statistics-massage of the old tests — they are retired and replaced.

## The three replacements

### New H9 — Negative sentiment as a vol-spike risk signal (extends H1)
**Function:** `test_negative_sentiment_spike_risk(df, sent, ticker, horizon=5, spike_pct=0.75)`

H1 said negative sentiment precedes vol. The original H9 tested a *symmetric* 3-group mean
difference and lost power. The new H9 tests the **directional, tail** question that H1 actually
implies and that is actionable: *do the most-negative sentiment days precede vol spikes more often
than other days?*

- Define a spike: forward `horizon`-day realized vol in the top `spike_pct` quantile (top quartile).
- Predictor: negative-sentiment magnitude `neg = max(-vader_compound, 0)` (the `min(sentiment,0)`
  feature the original H9 proposed, sign-flipped to a magnitude).
- Test: logistic regression `spike ~ neg`; report the odds ratio for a one-SD increase in `neg`, a
  one-sided Wald/LR p-value (H1: more negativity → more spikes), and McFadden pseudo-R².
- Visual: spike rate by sentiment tercile (Most-negative / Neutral / Most-positive) with Wilson CIs.

**Why stronger:** one focused directional coefficient instead of a 3-way mean test with an asymmetry
gate; uses the full sample; rides the real return-correlation embedded in the simulated sentiment, so
it has a genuine chance at significance while remaining the right question.
**Actionable (unchanged):** `Feature: min(sentiment, 0)` (risk-only signal).
**Degrades** (`available=False`) when fewer than ~30 aligned obs or no spikes in-sample.

### New H14 — NVDA as the directional volatility hub (Diebold-Yilmaz) (extends H4)
**Function:** `test_directional_spillover_hub(df_dict, source="NVDA", members=("MU","NVDA","AMD"), horizon=10)`

H4 showed NVDA vol spills to MU/AMD. The new H14 upgrades to the **Diebold & Yilmaz (2012)**
spillover framework:

- Fit a VAR (lag chosen by AIC, capped) on the members' `realized_vol_21d` (differenced if needed for
  stationarity; guard with a fallback lag of 1).
- Compute the **generalized forecast-error variance decomposition** (Pesaran-Shin, order-invariant) at
  the `horizon`-step ahead, row-normalised to a connectedness matrix.
- Net directional spillover per ticker = (share transmitted TO others) − (share received FROM others).
- Headline test: is the **source's net spillover significantly positive** (it is the hub)? Use a
  moving-block bootstrap (resample VAR residuals in blocks, rebuild series, recompute net spillover) to
  get a CI / one-sided p on `net[source]`.
- Visual: directed connectedness heatmap (or net-spillover bar chart) with the source highlighted.

**Why stronger:** the recognised, order-invariant contagion measure; uses the full multivariate lead-lag
structure instead of one 1-day lead; pure price data → robust.
**Actionable:** `Cross-ticker spillover feature (lead from hub)`.
**Degrades** when `source` absent from `df_dict`, fewer than 2 members with data, or insufficient
overlapping history for a VAR fit.

### New H16 — Continuous leverage amplification with the vol level (extends H5)
**Function:** `test_leverage_amplification(df, ticker, horizon=5)`

H5 confirmed the leverage effect. The original H16 bucketed into discrete regimes and ran out of
Extreme-regime days. The new H16 keeps the *same question* — does leverage amplify under stress? — but
with full-sample power:

- Regression on all aligned days:
  `fwd_vol ~ const + neg_mag + pos_mag + neg_mag:vol_level`
  where `neg_mag = max(-ret, 0)`, `pos_mag = max(ret, 0)`, `vol_level = current realized_vol_21d`
  (standardised).
- Headline test: the **interaction coefficient** `neg_mag:vol_level`. A positive, significant
  coefficient means a negative return adds *more* forward vol when current vol is already high —
  the volatility-feedback effect (Campbell & Hentschel 1992).
- Use HAC (Newey-West) standard errors because overlapping forward windows induce autocorrelation.
- Report the implied leverage slope at low (20th-pct) vs high (80th-pct) vol level.
- Keep a descriptive **`leverage_ratios`** dict (neg/pos forward-vol ratio per regime) so the existing
  H16 contract test (`assert "leverage_ratios" in r`) still passes; show it as the bar chart.

**Why stronger:** one full-sample interaction coefficient with high power instead of a permutation test
on a tiny Extreme subsample; same economic question, robust answer.
**Actionable (unchanged in spirit):** `Regime-conditional leverage param`.
**Degrades** gracefully (never raises) on very short input; returns `available` accordingly.

## Contract & wiring (all preserved)

Each new function returns the standard dict:
`hypothesis, extends, available, p_value, conclusion, actionable` (+ `figure`, `effect`, `statistic`
when available). 1:1 swap keeps the suite at 15 tests.

Files to change:
1. `modules/hypothesis_tests_L2.py` — delete the 3 old `test_*` functions, add the 3 new ones; update
   the section header comments and the module-docstring L1 map lines for H9/H14/H16; update
   `_FINDING_LABEL` for H9/H14/H16.
2. `scripts/run_advanced_hypotheses.py` — update the 3 calls in `run_all` to the new function names
   (H14 now takes `df_dict`; H9/H16 unchanged signatures).
3. `tests/test_hypotheses_L2.py` — point the H9/H14/H16 contract + degraded tests at the new function
   names; update the two references inside the `compile_summary` test; keep
   `test_module_exposes_15_tests` green (still 15). Add one positive-signal characterization test per
   new function where a deterministic fixture makes the expected direction detectable.
4. Re-run `python scripts/run_advanced_hypotheses.py --ticker MU --start 2018-01-01 --end 2024-12-31`
   to regenerate `outputs/results/hypothesis_L2_summary.csv` and the L2 figures.
5. `notebooks/06_conclusions.ipynb` — rewrite Section 2B (cells 17 & 19): move whichever of H9/H14/H16
   now reach significance into the featured-findings narrative, update the "confirmed nulls" /
   "inconclusive" lists, and refresh the bullet descriptions. Report results honestly per the run.

## Testing

Follow the existing "contract + characterization" pattern (deterministic synthetic data, no network):
- Contract test per new function (keys, `available` bool, p in [0,1], `figure` present).
- Degradation test per new function (tiny/insufficient input → `available=False`, no raise).
- One direction-of-effect characterization test per new function using a constructed fixture where the
  effect is real (negative-sentiment→spike; a known lead asset; leverage amplifying with vol level).
- Full suite must stay green: `python -m pytest tests/test_hypotheses_L2.py -q`.

## Out of scope
- The other 12 hypotheses (H10–H13, H15, H17–H23) are untouched.
- `src/hypotheses.py` (the L1 library) is untouched.
- No real sentiment / IV / 10-K feeds are wired in; the H9 reframe deliberately works with the existing
  simulated sentiment by leaning on its real return-correlation.

## Risks
- **H9** still depends on simulated sentiment; the reframe improves its odds of significance but the
  honest result is whatever the run produces. The conclusions narrative will state this plainly.
- **Diebold-Yilmaz** GFEVD + bootstrap is the heaviest new code; mitigated by AIC-capped lag selection,
  a lag-1 fallback, and try/except guards that downgrade to `available=False` rather than raising.
