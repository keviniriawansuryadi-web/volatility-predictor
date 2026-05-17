"""
Regime persistence analysis for volatility forecasting.

Computes average regime duration, transition probability matrices, and
expected time-to-reversion from the current regime.  Results are used
both for live signal enrichment and for academic reporting.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd


# Regime boundaries (annualised vol, consistent with src/ml_model.py)
REGIME_BOUNDS = {"Low": 0.15, "Elevated": 0.25, "High": 0.35}
REGIME_ORDER  = ["Low", "Elevated", "High", "Extreme"]


def _label(vol: float) -> str:
    if vol >= REGIME_BOUNDS["High"]:     return "Extreme"
    if vol >= REGIME_BOUNDS["Elevated"]: return "High"
    if vol >= REGIME_BOUNDS["Low"]:      return "Elevated"
    return "Low"


def analyze_regime_persistence(
    df: pd.DataFrame,
    ticker: str,
    vol_col: str = "realized_vol_21d",
    forecast_horizons: tuple = (5, 10, 20),
) -> dict:
    """
    Compute vol regime duration statistics and transition probability matrices.

    Methodology:
      1. Label each trading day with a vol regime (Low / Elevated / High / Extreme)
         using fixed annualised thresholds (15% / 25% / 35%).
      2. Identify contiguous regime runs (consecutive days in the same regime)
         and compute the duration of each run in trading days.
      3. Build an empirical regime transition count matrix:
         T[i, j] = number of days in regime i that were followed by regime j.
         Row-normalise to get a probability matrix.
      4. For each forecast horizon h in `forecast_horizons`, compute the
         h-step regime distribution starting from the current (most recent) regime
         by raising the transition matrix to the h-th power.

    Parameters
    ----------
    df             : DataFrame with a `vol_col` column indexed to trading days.
    ticker         : Ticker symbol for labelling output.
    vol_col        : Column name for the vol series (default 'realized_vol_21d').
    forecast_horizons : Tuple of horizons (days) for which to compute reversion
                       probabilities (default (5, 10, 20)).

    Returns a dict with keys:
      'ticker'         : str
      'avg_duration'   : dict {regime: float} — mean run length in trading days
      'current_regime' : str — most recent day's regime
      'current_run_len': int — number of consecutive days in current regime
      'transition_prob': pd.DataFrame — row-normalised Markov matrix
      'horizon_probs'  : dict {h: pd.Series} — regime distribution at each horizon
      'expected_reversion_days': float — expected days until leaving current regime
    """
    if vol_col not in df.columns:
        warnings.warn(f"[regime] '{vol_col}' not in df for {ticker} — returning empty dict.")
        return {}

    vol  = df[vol_col].dropna()
    regs = vol.map(_label)

    # ── 1. Run-length encoding ────────────────────────────────────────────────
    runs: list[dict] = []
    current = regs.iloc[0]
    run_start = 0
    for i in range(1, len(regs)):
        if regs.iloc[i] != current:
            runs.append({"regime": current, "duration": i - run_start})
            run_start = i
            current = regs.iloc[i]
    runs.append({"regime": current, "duration": len(regs) - run_start})

    runs_df = pd.DataFrame(runs)
    avg_duration = runs_df.groupby("regime")["duration"].mean().to_dict()
    for r in REGIME_ORDER:
        avg_duration.setdefault(r, 0.0)

    # Current regime and run length
    current_regime  = regs.iloc[-1]
    current_run_len = 0
    for i in range(len(regs) - 1, -1, -1):
        if regs.iloc[i] == current_regime:
            current_run_len += 1
        else:
            break

    # ── 2. Transition count matrix ────────────────────────────────────────────
    count_mat = pd.DataFrame(0, index=REGIME_ORDER, columns=REGIME_ORDER)
    for i in range(len(regs) - 1):
        count_mat.loc[regs.iloc[i], regs.iloc[i + 1]] += 1

    # Row-normalise to probability matrix (rows with no transitions → uniform)
    prob_mat = count_mat.div(count_mat.sum(axis=1).replace(0, np.nan), axis=0).fillna(
        1.0 / len(REGIME_ORDER)
    )

    # ── 3. h-step horizon distributions ──────────────────────────────────────
    P = prob_mat.values.astype(float)
    start_vec = np.zeros(len(REGIME_ORDER))
    start_vec[REGIME_ORDER.index(current_regime)] = 1.0

    horizon_probs: dict[int, pd.Series] = {}
    for h in forecast_horizons:
        Ph = np.linalg.matrix_power(P, h)
        dist = pd.Series(start_vec @ Ph, index=REGIME_ORDER)
        horizon_probs[h] = dist

    # ── 4. Expected time to leave current regime ──────────────────────────────
    stay_prob = float(prob_mat.loc[current_regime, current_regime])
    if stay_prob < 1.0:
        expected_reversion = 1.0 / (1.0 - stay_prob)
    else:
        expected_reversion = float("inf")

    result = {
        "ticker":                    ticker,
        "avg_duration":              avg_duration,
        "current_regime":            current_regime,
        "current_run_len":           current_run_len,
        "transition_prob":           prob_mat,
        "horizon_probs":             horizon_probs,
        "expected_reversion_days":   expected_reversion,
        "historical_avg_extreme_dur": avg_duration.get("Extreme", 0.0),
    }

    # Print summary
    print(f"\n  [regime] {ticker} — current: {current_regime} "
          f"(run {current_run_len}d, expected reversion in {expected_reversion:.1f}d)")
    print(f"  Avg regime durations: "
          + ", ".join(f"{r}={avg_duration.get(r, 0):.1f}d" for r in REGIME_ORDER))
    for h in forecast_horizons:
        probs = horizon_probs[h]
        print(f"  At t+{h:2d}: " + "  ".join(f"{r}={probs[r]:.0%}" for r in REGIME_ORDER))

    return result


def enrich_live_signal_with_regime(live_signal: dict, regime_result: dict) -> dict:
    """
    Add regime persistence fields to the live signal dict before JSON export.

    Parameters
    ----------
    live_signal   : Existing live signal payload (from _save_live_signal).
    regime_result : Output of analyze_regime_persistence().

    Returns the enriched live_signal dict (in-place modification + return).
    """
    if not regime_result:
        return live_signal

    h5_probs  = regime_result["horizon_probs"].get(5, pd.Series())
    h20_probs = regime_result["horizon_probs"].get(20, pd.Series())

    live_signal["regime_persistence"] = {
        "current_regime":              regime_result["current_regime"],
        "current_run_days":            regime_result["current_run_len"],
        "expected_reversion_days":     round(regime_result["expected_reversion_days"], 1),
        "historical_avg_extreme_dur":  round(regime_result["historical_avg_extreme_dur"], 1),
        "prob_still_extreme_at_t5":    round(float(h5_probs.get("Extreme", 0)), 3),
        "prob_still_extreme_at_t20":   round(float(h20_probs.get("Extreme", 0)), 3),
    }
    return live_signal
