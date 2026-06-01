"""Contract + characterization tests for the L2 advanced hypothesis suite.

The 15 test_* functions already exist; these tests pin their return-dict
contract, their graceful-degradation behaviour, and the two hand-rolled
statistics that have no library equivalent. All data is deterministic
synthetic (no network).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import modules.hypothesis_tests_L2 as L2

L2_REQUIRED = {"hypothesis", "extends", "available", "p_value", "conclusion", "actionable"}


def _assert_l2_contract(r: dict):
    assert L2_REQUIRED.issubset(r.keys()), f"missing keys: {L2_REQUIRED - r.keys()}"
    assert isinstance(r["available"], bool)
    assert isinstance(r["conclusion"], str) and r["conclusion"]
    if r["available"]:
        p = r["p_value"]
        assert p is None or pd.isna(p) or (0.0 <= float(p) <= 1.0)
        assert "figure" in r


@pytest.fixture
def price_df() -> pd.DataFrame:
    """400 rows: low-vol first half, high-vol second half (both regimes present)."""
    rng = np.random.default_rng(7)
    n = 400
    dates = pd.bdate_range("2022-01-03", periods=n)
    vol_daily = np.concatenate([np.full(n // 2, 0.008), np.full(n - n // 2, 0.03)])
    returns = rng.normal(0.0002, vol_daily)
    close = 100 * np.exp(np.cumsum(returns))
    df = pd.DataFrame({"close": close}, index=dates)
    df["log_return"] = np.log(df["close"]).diff().fillna(0.0)
    df["realized_vol_21d"] = df["log_return"].rolling(21).std() * np.sqrt(252)
    return df


@pytest.fixture
def df_dict() -> dict:
    """MU/NVDA (Semiconductor), JPM (Financial), XOM (Energy); 220 rows each so
    Granger (>=40 overlap) and H21 forward-60d vol (>=80 rows) both have data."""
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


@pytest.fixture
def sent(price_df) -> pd.DataFrame:
    """Four sentiment columns mildly correlated with returns, in [-1, 1]."""
    rng = np.random.default_rng(3)
    n = len(price_df)
    ret = price_df["log_return"].values
    f = (ret - ret.mean()) / (ret.std() + 1e-9)

    def s(load, noise):
        return np.clip(load * f + rng.normal(0, noise, n), -1, 1)

    return pd.DataFrame({
        "vader_compound": s(0.25, 0.35),
        "finbert": s(0.20, 0.25),
        "textblob": s(0.15, 0.45),
        "lm_score": s(0.22, 0.30),
    }, index=price_df.index)


@pytest.fixture
def disagreement(price_df) -> pd.Series:
    """Non-degenerate EGARCH-ML-style disagreement series aligned to price_df."""
    rng = np.random.default_rng(5)
    n = len(price_df)
    base = rng.normal(0, 1, n) + 30 * np.abs(price_df["log_return"].values)
    return pd.Series(base, index=price_df.index, name="disagreement")


@pytest.fixture
def lm_scores(price_df) -> pd.Series:
    """Five filing-date-indexed LM risk scores."""
    idx = price_df.index
    pos = np.linspace(40, len(idx) - 40, 5).astype(int)
    rng = np.random.default_rng(9)
    return pd.Series(rng.beta(2, 5, len(pos)), index=idx[pos], name="lm_risk_score")


@pytest.fixture
def earnings_dates_l2(price_df) -> pd.DatetimeIndex:
    """Eight earnings dates well inside the price index (H19 needs >=5 usable)."""
    idx = price_df.index
    pos = np.linspace(40, len(idx) - 20, 8).astype(int)
    return pd.DatetimeIndex(sorted(set(idx[pos])))


def test_module_exposes_15_tests():
    fns = [f for f in dir(L2) if f.startswith("test_")]
    assert len(fns) == 15
