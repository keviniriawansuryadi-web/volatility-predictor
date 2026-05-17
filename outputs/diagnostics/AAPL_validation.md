# Validation Report — AAPL

**Verdict: REGIME_LIMITED**
Test observations: 230

---

## Check 1 — Leakage Detection

**No leakage detected.** No feature exceeded the 0.95 threshold on the test set.

*Threshold: any feature with |Pearson r| > 0.95 against the test-set target is flagged.*

---

## Check 2 — Vol Regime Coverage (Test Set)

| Regime | Obs | % of Test |
|--------|----:|----------:|
| Low | 36 | 15.7% |
| Elevated | 121 | 52.6% |
| High | 73 | 31.7% |
| Extreme | 0 | 0.0% |

> ⚠ **Extreme-regime coverage < 5%** — model has rarely been tested on true spikes.

*A model evaluated exclusively on Elevated/High regimes will appear strong
but is untested on the spike events that matter most in practice.*

---

## Check 3 — Naive Persistence Baseline

The naive persistence forecast sets tomorrow's vol = today's vol.
For auto-correlated (sticky) vol series, this trivially achieves high Corr.

| Model | Corr | QLIKE |
|-------|-----:|------:|
| Naive Persistence | -0.0532 | -1.6293 |
| StackingEnsemble | 0.8941 | -2.1487 |

> ✓ Persistence Corr=-0.053 — vol is not excessively sticky.

✓ StackingEnsemble **beats** persistence on QLIKE.

---

## Interpretation

- **Verdict: REGIME_LIMITED**
- If VALID: the strong performance is genuine and may be reported without caveats.
- If REGIME_LIMITED: qualify the result — e.g. "StackingEnsemble achieves QLIKE=0.06
  on AAPL, though only 0.0% of test observations are Extreme-regime."
- If PERSISTENCE_DRIVEN: add persistence as a baseline model in the results table and
  report whether the ensemble adds incremental value beyond naive carry-forward.
- If CHECK_LEAKAGE: investigate flagged features before publishing any result.
