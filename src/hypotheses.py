"""
High-value volatility hypothesis tests.

Five tests, each a thin wrapper around a trusted library (scipy / statsmodels).
No homemade significance tests, no fabricated inputs, no plotting.

Every public ``test_*`` returns a dict with these keys::

    hypothesis, title, statistic, p_value, effect, n, conclusion, available

Tests degrade gracefully: when the input data is insufficient they return
``available=False`` with an honest ``conclusion`` rather than raising.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import bootstrap
from statsmodels.tsa.stattools import grangercausalitytests

from src.regime import _label, REGIME_ORDER

warnings.filterwarnings("ignore", category=FutureWarning)

# Sector membership for the cross-sector contagion test.
DEFAULT_SECTORS = {
    "Semiconductor": ["MU", "NVDA", "AMD"],
    "Financial": ["JPM", "BAC"],
    "Energy": ["XOM", "CVX"],
    "Tech": ["AAPL", "MSFT", "AMZN"],
}


def _verdict(p) -> str:
    """Significance verdict string at alpha=0.05."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "inconclusive (insufficient data)"
    return "significant" if p < 0.05 else "not significant"


def _forward_vol(df: pd.DataFrame, horizon: int = 5) -> pd.Series:
    """Forward realized vol over the next ``horizon`` days.

    Reuses ``realized_vol_{horizon}d`` if present, else annualised std of the
    next ``horizon`` log-returns.
    """
    col = f"realized_vol_{horizon}d"
    if col in df.columns:
        return df[col]
    return df["log_return"].shift(-horizon).rolling(horizon).std() * np.sqrt(252)


def _regime_series(df: pd.DataFrame, vol_col: str = "realized_vol_21d") -> pd.Series:
    """Label each trading day with its vol regime (Low/Elevated/High/Extreme)."""
    return df[vol_col].dropna().map(_label)


def _unavailable(hypothesis: str, title: str, why: str) -> dict:
    """Standard 'not enough data' result."""
    return dict(hypothesis=hypothesis, title=title, statistic=np.nan,
                p_value=np.nan, effect=np.nan, n=0, conclusion=why, available=False)


def test_leverage_effect(df: pd.DataFrame, horizon: int = 5) -> dict:
    """Negative-return days are followed by higher realized vol than positive-
    return days (Black 1976 leverage effect). scipy.stats.mannwhitneyu,
    one-sided; rank-biserial effect size.
    """
    title = "Leverage effect (neg returns -> higher forward vol)"
    fwd = _forward_vol(df, horizon).rename("fwd")
    d = pd.DataFrame({"ret": df["log_return"], "fwd": fwd}).dropna()
    neg = d.loc[d["ret"] < 0, "fwd"].values
    pos = d.loc[d["ret"] > 0, "fwd"].values
    if len(neg) < 5 or len(pos) < 5:
        return _unavailable("leverage_effect", title,
                            f"Too few observations (neg={len(neg)}, pos={len(pos)}).")
    u, p = stats.mannwhitneyu(neg, pos, alternative="greater")
    r = 1 - 2 * u / (len(neg) * len(pos))
    conclusion = (f"Negative-return days show {'significantly ' if p < 0.05 else 'not significantly '}"
                  f"higher next-{horizon}d vol (U={u:.0f}, p={p:.4f}, r={r:.3f}).")
    return dict(hypothesis="leverage_effect", title=title, statistic=float(u),
                p_value=float(p), effect=float(r), n=int(len(neg) + len(pos)),
                conclusion=conclusion, available=True)
