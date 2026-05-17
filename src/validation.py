"""
Model performance validation: leakage, regime, and persistence checks.

Addresses suspiciously strong results (QLIKE ~0.06, Corr ~0.93) by
running three sanity checks before any result is claimed in a paper.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd


VAL_DIR = Path(__file__).parent.parent / "outputs" / "diagnostics"
REGIME_BOUNDS = {"Low": 0.15, "Elevated": 0.25, "High": 0.35}


def _regime_label(v: float) -> str:
    if v >= 0.35: return "Extreme"
    if v >= 0.25: return "High"
    if v >= 0.15: return "Elevated"
    return "Low"


def validate_model_performance(
    ticker: str,
    feat_df: pd.DataFrame,
    stacking_preds: pd.Series,
    garch_preds: pd.Series,
    train_size: float = 0.80,
    leakage_corr_threshold: float = 0.95,
    persistence_corr_threshold: float = 0.85,
) -> dict:
    """
    Three sanity checks for suspiciously good model performance.

    Check 1 — Leakage detection
      Computes Pearson correlation of every feature in feat_df with the
      target on the **test set only**.  Any feature correlating > 0.95
      with the target is flagged as a potential future-data leak.
      (A perfectly predictable correlation is fine in training — it only
      constitutes leakage if it persists on held-out data.)

    Check 2 — Regime coverage
      Reports the fraction of test-set observations in each vol regime
      (Low / Elevated / High / Extreme).  A model evaluated exclusively
      on Elevated-regime data will appear strong but is untested on spikes.
      When Extreme-regime test coverage < 5%, the result is flagged as
      "regime-limited".

    Check 3 — Naive persistence baseline
      Computes a persistence forecast: tomorrow's vol = today's vol.
      If Pearson Corr(persistence, target) > 0.85 on the test set, the
      stock's vol is highly auto-correlated and any model that tracks the
      level will look impressive — we must ask whether the complex model
      is adding value BEYOND simple persistence.

    Parameters
    ----------
    ticker         : Ticker symbol (for labelling output).
    feat_df        : Feature DataFrame with 'target' column and all feature cols.
    stacking_preds : pd.Series of StackingEnsemble test-set predictions.
    garch_preds    : pd.Series of EGARCH test-set predictions.
    train_size     : Train fraction (default 0.80).
    leakage_corr_threshold  : Flag threshold for feature–target correlation (0.95).
    persistence_corr_threshold : Flag threshold for persistence Corr (0.85).

    Returns a dict with keys:
      leakage_flags, regime_dist, persistence_corr, persistence_qlike,
      stacking_corr, stacking_qlike, regime_flag, persistence_flag,
      verdict (str — "VALID", "REGIME_LIMITED", "PERSISTENCE_DRIVEN", "CHECK_LEAKAGE")
    """
    split = int(len(feat_df) * train_size)
    test_df  = feat_df.iloc[split:].copy()
    y_test   = test_df["target"]

    feature_cols = [c for c in test_df.columns if c != "target"]

    # ── Check 1: Leakage ─────────────────────────────────────────────────────
    leakage_flags = []
    for col in feature_cols:
        try:
            c = float(y_test.corr(test_df[col]))
            if abs(c) > leakage_corr_threshold:
                leakage_flags.append((col, round(c, 4)))
        except Exception:
            pass

    # ── Check 2: Regime coverage ──────────────────────────────────────────────
    regime_labels = y_test.map(_regime_label)
    regime_counts = regime_labels.value_counts()
    n_test        = len(y_test)
    regime_dist   = {r: int(regime_counts.get(r, 0)) for r in ["Low","Elevated","High","Extreme"]}
    regime_pct    = {r: round(v / n_test * 100, 1) for r, v in regime_dist.items()}
    extreme_pct   = regime_pct["Extreme"]
    regime_flag   = extreme_pct < 5.0

    # ── Check 3: Persistence baseline ────────────────────────────────────────
    # Naive: predict today's target = yesterday's target (1-day lag)
    if "realized_vol_21d" in feat_df.columns:
        rv_col = feat_df["realized_vol_21d"].iloc[split:]
    elif "vol_21d" in feat_df.columns:
        rv_col = feat_df["vol_21d"].iloc[split:]
    else:
        rv_col = y_test.shift(1)

    persist_preds = rv_col.shift(1).reindex(y_test.index).dropna()
    y_align       = y_test.reindex(persist_preds.index)

    persist_corr  = float(persist_preds.corr(y_align))
    # QLIKE for persistence: mean(log(sigma^2) + y^2/sigma^2) with sigma=persist
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        epsilon = 1e-8
        persist_qlike_vals = (
            np.log(persist_preds.clip(lower=epsilon) ** 2)
            + y_align ** 2 / (persist_preds.clip(lower=epsilon) ** 2)
        )
    persist_qlike = float(persist_qlike_vals.mean())

    persistence_flag = persist_corr > persistence_corr_threshold

    # ── Stacking vs persistence ────────────────────────────────────────────────
    common = stacking_preds.index.intersection(y_test.index)
    if len(common) >= 10:
        s_preds = stacking_preds.reindex(common)
        y_s     = y_test.reindex(common)
        stacking_corr  = float(s_preds.corr(y_s))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stacking_qlike_vals = (
                np.log(s_preds.clip(lower=epsilon) ** 2)
                + y_s ** 2 / (s_preds.clip(lower=epsilon) ** 2)
            )
        stacking_qlike = float(stacking_qlike_vals.mean())
    else:
        stacking_corr = stacking_qlike = float("nan")

    beats_persistence = (
        not np.isnan(stacking_qlike) and stacking_qlike < persist_qlike
    ) if not np.isnan(persist_qlike) else None

    # ── Verdict ───────────────────────────────────────────────────────────────
    verdicts = []
    if leakage_flags:
        verdicts.append("CHECK_LEAKAGE")
    if regime_flag:
        verdicts.append("REGIME_LIMITED")
    if persistence_flag and beats_persistence is False:
        verdicts.append("PERSISTENCE_DRIVEN")
    verdict = " | ".join(verdicts) if verdicts else "VALID"

    result = {
        "ticker":            ticker,
        "n_test":            n_test,
        "leakage_flags":     leakage_flags,
        "regime_dist":       regime_dist,
        "regime_pct":        regime_pct,
        "extreme_pct":       extreme_pct,
        "regime_flag":       regime_flag,
        "persistence_corr":  round(persist_corr, 4),
        "persistence_qlike": round(persist_qlike, 4),
        "stacking_corr":     round(stacking_corr, 4) if not np.isnan(stacking_corr) else None,
        "stacking_qlike":    round(stacking_qlike, 4) if not np.isnan(stacking_qlike) else None,
        "beats_persistence": beats_persistence,
        "persistence_flag":  persistence_flag,
        "verdict":           verdict,
    }

    _save_validation_report(result)
    _print_validation(result)
    return result


def _print_validation(r: dict) -> None:
    print(f"\n{'='*60}")
    print(f"  VALIDATION — {r['ticker']}")
    print(f"{'='*60}")
    print(f"  Test observations: {r['n_test']}")

    print(f"\n  [Check 1] Leakage detection (corr > 0.95 with target on test set):")
    if r["leakage_flags"]:
        for feat, c in r["leakage_flags"]:
            print(f"    ⚠ {feat}: corr={c}  ← INVESTIGATE")
    else:
        print(f"    ✓ No leakage detected")

    print(f"\n  [Check 2] Regime coverage (test set):")
    for reg, pct in r["regime_pct"].items():
        flag = "  ← sparse" if reg == "Extreme" and pct < 5 else ""
        print(f"    {reg:10}: {r['regime_dist'][reg]:3d} obs ({pct:.1f}%){flag}")
    if r["regime_flag"]:
        print(f"    ⚠ Extreme-regime coverage < 5% — REGIME_LIMITED")

    print(f"\n  [Check 3] Persistence baseline (y_t = y_{{t-1}}):")
    print(f"    Persistence Corr : {r['persistence_corr']:.4f}")
    print(f"    Persistence QLIKE: {r['persistence_qlike']:.4f}")
    print(f"    StackingEnsemble Corr : {r['stacking_corr']}")
    print(f"    StackingEnsemble QLIKE: {r['stacking_qlike']}")
    beats = r.get("beats_persistence")
    if beats is True:
        print(f"    ✓ StackingEnsemble BEATS persistence on QLIKE")
    elif beats is False:
        print(f"    ⚠ StackingEnsemble does NOT beat persistence — PERSISTENCE_DRIVEN")
    if r["persistence_flag"]:
        print(f"    ⚠ Vol is sticky (persistence Corr > 0.85) — high Corr may be trivial")

    print(f"\n  VERDICT: {r['verdict']}")
    print(f"{'='*60}")


def _save_validation_report(r: dict) -> None:
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    ticker = r["ticker"]

    leakage_block = (
        "**No leakage detected.** No feature exceeded the 0.95 threshold on the test set."
        if not r["leakage_flags"]
        else "**⚠ Potential leakage detected:**\n"
             + "\n".join(f"- `{f}` corr={c}" for f, c in r["leakage_flags"])
    )

    regime_rows = "\n".join(
        f"| {reg} | {r['regime_dist'][reg]} | {r['regime_pct'][reg]:.1f}% |"
        for reg in ["Low", "Elevated", "High", "Extreme"]
    )
    regime_flag_note = (
        "> ⚠ **Extreme-regime coverage < 5%** — model has rarely been tested on true spikes."
        if r["regime_flag"] else
        "> ✓ Adequate regime coverage."
    )

    beats_txt = (
        "✓ StackingEnsemble **beats** persistence on QLIKE."
        if r.get("beats_persistence") is True else
        "⚠ StackingEnsemble does **not** beat naive persistence on QLIKE — "
        "the model may be exploiting vol stickiness rather than adding genuine forecast skill."
        if r.get("beats_persistence") is False else
        "Persistence comparison unavailable."
    )

    persist_flag_note = (
        f"> ⚠ **Vol is sticky** — persistence Corr={r['persistence_corr']:.3f} > 0.85. "
        "High Corr scores may reflect autocorrelation in the target, not model skill."
        if r["persistence_flag"] else
        f"> ✓ Persistence Corr={r['persistence_corr']:.3f} — vol is not excessively sticky."
    )

    md = f"""# Validation Report — {ticker}

**Verdict: {r['verdict']}**
Test observations: {r['n_test']}

---

## Check 1 — Leakage Detection

{leakage_block}

*Threshold: any feature with |Pearson r| > 0.95 against the test-set target is flagged.*

---

## Check 2 — Vol Regime Coverage (Test Set)

| Regime | Obs | % of Test |
|--------|----:|----------:|
{regime_rows}

{regime_flag_note}

*A model evaluated exclusively on Elevated/High regimes will appear strong
but is untested on the spike events that matter most in practice.*

---

## Check 3 — Naive Persistence Baseline

The naive persistence forecast sets tomorrow's vol = today's vol.
For auto-correlated (sticky) vol series, this trivially achieves high Corr.

| Model | Corr | QLIKE |
|-------|-----:|------:|
| Naive Persistence | {r['persistence_corr']:.4f} | {r['persistence_qlike']:.4f} |
| StackingEnsemble | {r['stacking_corr']} | {r['stacking_qlike']} |

{persist_flag_note}

{beats_txt}

---

## Interpretation

- **Verdict: {r['verdict']}**
- If VALID: the strong performance is genuine and may be reported without caveats.
- If REGIME_LIMITED: qualify the result — e.g. "StackingEnsemble achieves QLIKE=0.06
  on {ticker}, though only {r['extreme_pct']:.1f}% of test observations are Extreme-regime."
- If PERSISTENCE_DRIVEN: add persistence as a baseline model in the results table and
  report whether the ensemble adds incremental value beyond naive carry-forward.
- If CHECK_LEAKAGE: investigate flagged features before publishing any result.
"""
    out = VAL_DIR / f"{ticker}_validation.md"
    out.write_text(md, encoding="utf-8")
    print(f"  Validation report saved: {out}")
