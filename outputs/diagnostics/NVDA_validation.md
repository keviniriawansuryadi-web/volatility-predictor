# Validation Report — NVDA

**Verdict: VALID**
Test observations: 230

---

## Check 1 — Leakage Detection

**No leakage detected.** No feature exceeded the 0.95 threshold on the test set.

*Threshold: any feature with |Pearson r| > 0.95 against the test-set target is flagged.*

---

## Check 2 — Vol Regime Coverage (Test Set)

| Regime | Obs | % of Test |
|--------|----:|----------:|
| Low | 0 | 0.0% |
| Elevated | 18 | 7.8% |
| High | 126 | 54.8% |
| Extreme | 86 | 37.4% |

> ✓ Adequate regime coverage.

*A model evaluated exclusively on Elevated/High regimes will appear strong
but is untested on the spike events that matter most in practice.*

---

## Check 3 — Naive Persistence Baseline

The naive persistence forecast sets tomorrow's vol = today's vol.
For auto-correlated (sticky) vol series, this trivially achieves high Corr.

| Model | Corr | QLIKE |
|-------|-----:|------:|
| Naive Persistence | 0.1467 | -1.1252 |
| StackingEnsemble | 0.7906 | -1.1009 |

> ✓ Persistence Corr=0.147 — vol is not excessively sticky.

⚠ StackingEnsemble does **not** beat naive persistence on QLIKE — the model may be exploiting vol stickiness rather than adding genuine forecast skill.

---

## Interpretation

- **Verdict: VALID**
- If VALID: the strong performance is genuine and may be reported without caveats.
- If REGIME_LIMITED: qualify the result — e.g. "StackingEnsemble achieves QLIKE=0.06
  on NVDA, though only 37.4% of test observations are Extreme-regime."
- If PERSISTENCE_DRIVEN: add persistence as a baseline model in the results table and
  report whether the ensemble adds incremental value beyond naive carry-forward.
- If CHECK_LEAKAGE: investigate flagged features before publishing any result.
