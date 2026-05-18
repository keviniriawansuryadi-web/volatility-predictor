"""
Data helpers for hypothesis tests and EDA.

Sentiment note: yfinance only exposes recent news (~1 month). For the full
historical window (2015-2024) we simulate four sentiment scores that are
deliberately correlated with returns at known strength, with added noise.
A production pipeline would replace simulate_sentiment() with a paid
news-API feed (e.g., RavenPack, Bloomberg Terminal, or Tiingo).
"""

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

CACHE_DIR = Path(__file__).parent.parent / "data"

# --------------------------------------------------------------------------- #
# VIX
# --------------------------------------------------------------------------- #

def load_vix(start: str, end: str) -> pd.Series:
    """Return daily VIX closing levels aligned to calendar dates."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache = CACHE_DIR / f"VIX_{start}_{end}.csv"
    if cache.exists():
        s = pd.read_csv(cache, index_col=0, parse_dates=True).squeeze()
        s.index = s.index.tz_localize(None) if s.index.tz is not None else s.index
        return s
    raw = yf.download("^VIX", start=start, end=end, auto_adjust=True, progress=False)
    s = raw["Close"].squeeze()
    s.index = s.index.tz_localize(None) if s.index.tz is not None else s.index
    s.name = "vix"
    s.to_csv(cache)
    return s


# --------------------------------------------------------------------------- #
# Earnings dates
# --------------------------------------------------------------------------- #

def load_earnings_dates(ticker: str) -> pd.DatetimeIndex:
    """
    Return sorted DatetimeIndex of historical earnings announcement dates.
    Falls back to an empty index if yfinance cannot retrieve them.
    """
    try:
        t = yf.Ticker(ticker)
        ed = t.earnings_dates
        if ed is None or ed.empty:
            return pd.DatetimeIndex([])
        idx = pd.DatetimeIndex(ed.index)
        idx = idx.tz_localize(None) if idx.tz is not None else idx
        return idx.sort_values()
    except Exception:
        return pd.DatetimeIndex([])


# --------------------------------------------------------------------------- #
# Sentiment (simulated historical scores)
# --------------------------------------------------------------------------- #

_LM_NEGATIVE = {
    "loss", "losses", "decline", "declined", "risk", "risks", "uncertain",
    "uncertainty", "adverse", "adversely", "impair", "impairment", "weak",
    "weakness", "volatility", "exposure", "litigation", "default", "debt",
    "dilution", "negative", "decrease", "decreased", "fail", "failure",
    "unable", "difficult", "difficulty", "deteriorate", "deterioration",
    "curtail", "curtailed", "shortage", "disruption", "penalty", "penalties",
}


def simulate_sentiment(df: pd.DataFrame, ticker: str, seed: int = 42) -> pd.DataFrame:
    """
    Build a DataFrame of daily sentiment scores for `ticker`.

    Columns returned:
        vader_compound  [-1, 1]  — negative-biased on down days
        finbert         [-1, 1]  — smoother, longer memory
        textblob        [-1, 1]  — noisier, weaker signal
        lm_score        [-1, 1]  — Loughran-McDonald negative proportion (negated)
        news_count      [int]    — synthetic daily article count

    The scores are correlated with same-day log returns at ~0.25, matching
    empirical estimates from Tetlock (2007) and Loughran & McDonald (2011).
    A real implementation would replace this function with a news-API feed.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    ret = df["log_return"].values

    # Common market factor (returns-driven component)
    factor = (ret - ret.mean()) / (ret.std() + 1e-9)

    def _score(loading, noise_scale, ar=0.0):
        base = loading * factor + rng.normal(0, noise_scale, n)
        if ar > 0:
            for i in range(1, n):
                base[i] += ar * base[i - 1]
        return np.clip(base, -1, 1)

    sent = pd.DataFrame(index=df.index)
    sent["vader_compound"] = _score(0.25, 0.35)
    sent["finbert"]        = _score(0.20, 0.25, ar=0.15)
    sent["textblob"]       = _score(0.15, 0.45)
    sent["lm_score"]       = _score(0.22, 0.30, ar=0.10)

    # news_count: correlated weakly with |return| (high vol = more news)
    base_count = 3 + np.abs(factor) * 4 + rng.poisson(2, n)
    sent["news_count"] = base_count.clip(0).astype(int)

    return sent


# --------------------------------------------------------------------------- #
# LM 10-K risk score
# --------------------------------------------------------------------------- #

def compute_lm_risk_score(text: str) -> float:
    """
    Fraction of words in `text` that appear in the Loughran-McDonald
    negative word list. Returns a value in [0, 1].
    """
    words = text.lower().split()
    if not words:
        return 0.0
    neg = sum(1 for w in words if w.strip(".,;:!?\"'()") in _LM_NEGATIVE)
    return neg / len(words)


def simulate_10k_risk_scores(
    ticker: str,
    filing_dates: pd.DatetimeIndex,
    seed: int = 0,
) -> pd.Series:
    """
    Return a Series of simulated LM risk scores indexed by filing date.
    In production, replace with SEC EDGAR full-text search (EFTS API) and
    parse the MD&A section with compute_lm_risk_score().
    """
    rng = np.random.default_rng(seed)
    scores = rng.beta(2, 5, len(filing_dates))  # right-skewed in [0,1]
    return pd.Series(scores, index=filing_dates, name="lm_risk_score")


# --------------------------------------------------------------------------- #
# Volatility regime labels
# --------------------------------------------------------------------------- #

def add_vol_regime(df: pd.DataFrame, vol_col: str = "realized_vol_21d") -> pd.DataFrame:
    """
    Add a categorical 'vol_regime' column: Low / Medium / High,
    cut at the 33rd and 67th percentiles of realized volatility.
    """
    out = df.copy()
    q33, q67 = df[vol_col].quantile([0.33, 0.67])
    out["vol_regime"] = pd.cut(
        df[vol_col],
        bins=[-np.inf, q33, q67, np.inf],
        labels=["Low", "Medium", "High"],
    )
    return out


# --------------------------------------------------------------------------- #
# Forward realized volatility (used by several tests)
# --------------------------------------------------------------------------- #

def add_forward_vol(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """
    Append realized_vol_{horizon}d (forward-looking) to df.
    This is the model target: std of the next `horizon` log-returns × √252.
    """
    out = df.copy()
    col = f"realized_vol_{horizon}d"
    out[col] = (
        out["log_return"]
        .shift(-horizon)
        .rolling(horizon)
        .std()
        * np.sqrt(252)
    )
    return out
