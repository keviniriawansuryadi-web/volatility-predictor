"""Contract tests for the simplified hypothesis module.

Each test runs on deterministic synthetic data (no network) and asserts the
uniform return-dict contract rather than exact statistical values.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import hypotheses as H

REQUIRED_KEYS = {"hypothesis", "title", "statistic", "p_value",
                 "effect", "n", "conclusion", "available"}


def _assert_contract(r: dict):
    assert REQUIRED_KEYS.issubset(r.keys()), f"missing keys: {REQUIRED_KEYS - r.keys()}"
    assert isinstance(r["available"], bool)
    assert isinstance(r["conclusion"], str) and r["conclusion"]
    if r["available"]:
        assert 0.0 <= float(r["p_value"]) <= 1.0


@pytest.fixture
def price_df() -> pd.DataFrame:
    """400-row frame with a low-vol first half and a high-vol second half so
    the regime test sees both Low and Extreme regimes. Includes vix_level."""
    rng = np.random.default_rng(7)
    n = 400
    dates = pd.bdate_range("2022-01-03", periods=n)
    vol_daily = np.concatenate([np.full(n // 2, 0.008), np.full(n - n // 2, 0.03)])
    returns = rng.normal(0.0002, vol_daily)
    close = 100 * np.exp(np.cumsum(returns))
    df = pd.DataFrame({"close": close}, index=dates)
    df["log_return"] = np.log(df["close"]).diff().fillna(0.0)
    df["realized_vol_21d"] = df["log_return"].rolling(21).std() * np.sqrt(252)
    # Synthetic VIX: realized vol plus a positive premium and noise.
    df["vix_level"] = df["realized_vol_21d"].fillna(0.2) + 0.05 + rng.normal(0, 0.02, n)
    return df


@pytest.fixture
def earnings_dates(price_df) -> pd.DatetimeIndex:
    """Four earnings dates spread across the price index."""
    idx = price_df.index
    return pd.DatetimeIndex([idx[60], idx[150], idx[240], idx[330]])


@pytest.fixture
def df_dict() -> dict:
    """Four tickers across three DEFAULT_SECTORS sectors, each with a
    realized_vol_21d series long enough for Granger (>=40 overlapping rows)."""
    rng = np.random.default_rng(11)
    n = 220
    dates = pd.bdate_range("2022-01-03", periods=n)
    base = pd.Series(rng.normal(0.0002, 0.015, n), index=dates)
    out = {}
    for i, t in enumerate(["MU", "NVDA", "JPM", "XOM"]):
        ret = base.shift(i % 3).fillna(0.0) + rng.normal(0, 0.01, n)
        rv = ret.rolling(21).std() * np.sqrt(252)
        out[t] = pd.DataFrame({"log_return": ret, "realized_vol_21d": rv}, index=dates)
    return out


def test_leverage_effect_contract(price_df):
    r = H.test_leverage_effect(price_df)
    _assert_contract(r)
    assert r["hypothesis"] == "leverage_effect"


def test_leverage_effect_degraded():
    tiny = pd.DataFrame({"log_return": [0.01, -0.01, 0.0]},
                        index=pd.bdate_range("2022-01-03", periods=3))
    r = H.test_leverage_effect(tiny)
    assert r["available"] is False
