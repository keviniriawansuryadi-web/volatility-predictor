"""
EGARCH-ML disagreement signal backtest.

Treats high EGARCH-ML disagreement as a directional trading signal:
  - Signal fires when normalised |EGARCH - ML| / mean exceeds a threshold
  - Measures hit rate, vol lift, and false positive rate on the test set
  - Adds a percentile interpretation block to the live signal JSON
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd


def backtest_disagreement_signal(
    garch_preds: pd.Series,
    ml_preds: pd.Series,
    realized_vol: pd.Series,
    threshold_pct: float = 0.80,
    vol_spike_pct: float = 0.50,
) -> dict:
    """
    Formalised backtest of the EGARCH-ML disagreement signal.

    The signal fires on day t when the normalised absolute disagreement between
    EGARCH and the ML model exceeds the `threshold_pct` quantile of all
    disagreement values in the test window.  For each signal day, we check
    whether next-day realized vol exceeds the `vol_spike_pct` median.

    Metrics computed:
      hit_rate          — fraction of signal days where next-day vol > median
      vol_lift          — ratio of mean vol on signal days to mean vol on quiet days
      false_positive_rate — fraction of signal days where next-day vol <= median
      signal_count      — total number of signal fires
      no_signal_count   — total number of no-signal days

    Parameters
    ----------
    garch_preds    : pd.Series of EGARCH rolling test-set forecasts (index = dates).
    ml_preds       : pd.Series of ML rolling test-set forecasts (index = dates).
    realized_vol   : pd.Series of realized vol, same index as preds.
    threshold_pct  : Quantile above which disagreement is considered 'high'
                     (default 0.80 = top 20% of disagreement days).
    vol_spike_pct  : Quantile used to define 'elevated vol' for hit-rate calc
                     (default 0.50 = above-median vol counts as a hit).

    Returns
    -------
    dict with keys: available, hit_rate, false_positive_rate, vol_lift,
    signal_count, no_signal_count, vol_median, mean_vol_signal,
    mean_vol_quiet, disagreement_series, threshold, interpretation.
    """
    common = garch_preds.index.intersection(ml_preds.index).intersection(realized_vol.index)
    if len(common) < 30:
        return {"available": False, "reason": f"Only {len(common)} common observations — need >= 30."}

    eg  = garch_preds.reindex(common)
    ml  = ml_preds.reindex(common)
    rv  = realized_vol.reindex(common)

    denom        = (eg.abs() + ml.abs()) / 2 + 1e-8
    disagreement = (eg - ml).abs() / denom

    threshold = float(disagreement.quantile(threshold_pct))
    signal    = disagreement >= threshold

    vol_median = float(rv.quantile(vol_spike_pct))

    rv_signal = rv[signal].dropna()
    rv_quiet  = rv[~signal].dropna()

    if len(rv_signal) < 5 or len(rv_quiet) < 5:
        return {
            "available": False,
            "reason": f"Too few signal ({len(rv_signal)}) or quiet ({len(rv_quiet)}) days.",
        }

    hit_rate  = float((rv_signal > vol_median).mean())
    fpr       = 1.0 - hit_rate
    vol_lift  = float(rv_signal.mean() / (rv_quiet.mean() + 1e-10))

    # Interpretation
    if hit_rate >= 0.65 and vol_lift >= 1.20:
        interp = "STRONG signal — high disagreement reliably precedes elevated vol"
    elif hit_rate >= 0.55 and vol_lift >= 1.10:
        interp = "MODERATE signal — disagreement has modest predictive power"
    else:
        interp = "WEAK signal — disagreement has limited predictive power"

    return {
        "available":           True,
        "hit_rate":            hit_rate,
        "false_positive_rate": fpr,
        "vol_lift":            vol_lift,
        "signal_count":        int(signal.sum()),
        "no_signal_count":     int((~signal).sum()),
        "vol_median":          vol_median,
        "mean_vol_signal":     float(rv_signal.mean()),
        "mean_vol_quiet":      float(rv_quiet.mean()),
        "disagreement_series": disagreement,
        "threshold":           threshold,
        "threshold_pct":       threshold_pct,
        "interpretation":      interp,
    }


def compute_live_disagreement_percentile(
    current_disagreement: float,
    disagreement_series: pd.Series,
) -> dict:
    """
    Compute where today's EGARCH-ML disagreement sits in its historical distribution.

    Parameters
    ----------
    current_disagreement : float — today's raw |EGARCH - ML| / mean(|EGARCH|, |ML|).
    disagreement_series  : pd.Series — full historical disagreement series (test set).

    Returns a dict with percentile, z_score, flag (bool), and interpretation text
    suitable for inclusion in the live signal JSON.
    """
    pct = float((disagreement_series < current_disagreement).mean() * 100)
    mean_d = float(disagreement_series.mean())
    std_d  = float(disagreement_series.std())
    z      = (current_disagreement - mean_d) / (std_d + 1e-10)

    flag = pct >= 80.0  # top-20% = high-disagreement signal

    if pct >= 90:
        text = "EXTREME disagreement — top 10%: strong uncertainty signal"
    elif pct >= 80:
        text = "HIGH disagreement — top 20%: elevated uncertainty"
    elif pct >= 60:
        text = "MODERATE disagreement"
    else:
        text = "LOW disagreement — models broadly agree"

    return {
        "current_value": round(current_disagreement, 4),
        "percentile":    round(pct, 1),
        "z_score":       round(z, 2),
        "flag":          flag,
        "interpretation": text,
    }


def print_backtest_results(ticker: str, result: dict) -> None:
    """Pretty-print the disagreement backtest results."""
    print(f"\n{'='*60}")
    print(f"  EGARCH-ML Disagreement Backtest — {ticker}")
    print(f"{'='*60}")
    if not result.get("available"):
        print(f"  Not available: {result.get('reason', 'unknown')}")
        return

    print(f"  Signal threshold : {result['threshold']:.4f} "
          f"(top {100 - result['threshold_pct']*100:.0f}% of disagreement days)")
    print(f"  Signal fires     : {result['signal_count']} days")
    print(f"  No-signal days   : {result['no_signal_count']} days")
    print(f"")
    print(f"  Hit rate         : {result['hit_rate']:.1%}  "
          f"(signal days where next-day vol > median)")
    print(f"  False pos. rate  : {result['false_positive_rate']:.1%}")
    print(f"  Vol lift         : {result['vol_lift']:.2f}x  "
          f"(signal mean vol / quiet mean vol)")
    print(f"  Mean vol | signal: {result['mean_vol_signal']:.1%}")
    print(f"  Mean vol | quiet : {result['mean_vol_quiet']:.1%}")
    print(f"")
    print(f"  Interpretation   : {result['interpretation']}")
    print(f"{'='*60}")
