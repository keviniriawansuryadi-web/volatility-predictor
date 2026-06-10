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


def test_steiger_equal_corrs_is_null():
    z, p = L2.steiger_dependent_corr(0.5, 0.5, 0.3, 100)
    assert abs(z) < 1e-9
    assert p == pytest.approx(1.0, abs=1e-9)


def test_steiger_large_gap_is_significant():
    z, p = L2.steiger_dependent_corr(0.7, 0.1, 0.3, 200)
    assert z > 0
    assert p < 0.05


def test_steiger_antisymmetric_in_first_two_args():
    z1, p1 = L2.steiger_dependent_corr(0.6, 0.2, 0.3, 150)
    z2, p2 = L2.steiger_dependent_corr(0.2, 0.6, 0.3, 150)
    assert z1 == pytest.approx(-z2, abs=1e-9)
    assert p1 == pytest.approx(p2, abs=1e-9)


def test_steiger_small_n_returns_nan():
    z, p = L2.steiger_dependent_corr(0.5, 0.2, 0.3, 3)
    assert np.isnan(z) and np.isnan(p)


def test_srh_structure_and_degrees_of_freedom():
    rng = np.random.default_rng(0)
    n = 200
    a = rng.choice(["lo", "hi"], n)
    b = rng.choice(["x", "y"], n)
    resp = (a == "hi") * 5.0 + rng.normal(0, 1, n)
    d = pd.DataFrame({"resp": resp, "A": a, "B": b})
    out = L2.scheirer_ray_hare(d, "resp", "A", "B")
    assert list(out.index) == ["A", "B", "interaction"]
    assert {"H", "df", "p_value"}.issubset(out.columns)
    assert out.loc["A", "df"] == 1
    assert out.loc["B", "df"] == 1
    assert out.loc["interaction", "df"] == 1
    assert out["p_value"].dropna().between(0.0, 1.0).all()


def test_srh_detects_strong_main_effect_only():
    rng = np.random.default_rng(1)
    n = 300
    a = rng.choice(["lo", "hi"], n)
    b = rng.choice(["x", "y"], n)
    resp = (a == "hi") * 6.0 + rng.normal(0, 1, n)  # driven by A only
    d = pd.DataFrame({"resp": resp, "A": a, "B": b})
    out = L2.scheirer_ray_hare(d, "resp", "A", "B")
    assert out.loc["A", "p_value"] < 0.05
    assert out.loc["B", "p_value"] > 0.05


def _tiny():
    idx = pd.bdate_range("2022-01-03", periods=8)
    df = pd.DataFrame({"log_return": np.linspace(-0.01, 0.01, 8),
                       "realized_vol_21d": np.linspace(0.1, 0.2, 8)}, index=idx)
    sent = pd.DataFrame({c: np.linspace(-0.2, 0.2, 8) for c in
                         ["vader_compound", "finbert", "textblob", "lm_score"]}, index=idx)
    return df, sent


def test_h9_contract(price_df, sent):
    r = L2.test_negative_sentiment_spike_risk(price_df, sent, "TEST")
    _assert_l2_contract(r)
    assert r["hypothesis"] == "H9"


def test_h9_degraded():
    df, sent = _tiny()
    assert L2.test_negative_sentiment_spike_risk(df, sent, "TEST")["available"] is False


def test_h9_detects_negative_sentiment_spike_link():
    # Negative sentiment lifts forward vol, but with overlap (no separation):
    # fwd_vol = base - slope*vader + noise, so the most-negative days are
    # enriched among the top-quartile spikes without being perfectly separable.
    rng = np.random.default_rng(0)
    n = 320
    idx = pd.bdate_range("2021-01-04", periods=n)
    vader = rng.uniform(-1, 1, n)
    fwd = np.clip(0.30 - 0.25 * vader + rng.normal(0, 0.15, n), 0.01, None)
    df = pd.DataFrame({"log_return": rng.normal(0, 0.01, n),
                       "realized_vol_21d": fwd,
                       "realized_vol_5d": fwd}, index=idx)
    sent = pd.DataFrame({"vader_compound": vader}, index=idx)
    r = L2.test_negative_sentiment_spike_risk(df, sent, "TEST")
    assert r["available"] is True
    assert r["statistic"] > 0          # log-odds coefficient positive
    assert r["p_value"] < 0.05


def test_h10_contract(price_df, sent):
    r = L2.test_sentiment_velocity(price_df, sent, "TEST")
    _assert_l2_contract(r)
    assert r["hypothesis"] == "H10"


def test_h10_degraded():
    df, sent = _tiny()
    assert L2.test_sentiment_velocity(df, sent, "TEST")["available"] is False


def test_h11_contract(price_df, sent):
    r = L2.test_sentiment_model_consensus(price_df, sent, "TEST")
    _assert_l2_contract(r)
    assert r["hypothesis"] == "H11"


def test_h11_degraded(price_df):
    # Missing required sentiment columns -> unavailable.
    one_col = pd.DataFrame({"vader_compound": np.zeros(len(price_df))}, index=price_df.index)
    assert L2.test_sentiment_model_consensus(price_df, one_col, "TEST")["available"] is False


def test_h12_contract(price_df, disagreement):
    r = L2.test_disagreement_persistence(price_df, disagreement, "TEST")
    _assert_l2_contract(r)
    assert r["hypothesis"] == "H12"


def test_h12_degraded(price_df):
    short = pd.Series(np.arange(10.0), index=price_df.index[:10])
    assert L2.test_disagreement_persistence(price_df, short, "TEST")["available"] is False


def test_h13_contract(price_df, disagreement):
    r = L2.test_disagreement_direction(price_df, disagreement, "TEST")
    _assert_l2_contract(r)
    assert r["hypothesis"] == "H13"


def test_h13_degraded(price_df):
    short = pd.Series(np.arange(10.0), index=price_df.index[:10])
    assert L2.test_disagreement_direction(price_df, short, "TEST")["available"] is False


def test_h14_contract(df_dict):
    r = L2.test_directional_spillover_hub(df_dict, n_boot=80)
    _assert_l2_contract(r)
    assert r["hypothesis"] == "H14"
    assert "net_spillover" in r


def test_h14_degraded(df_dict):
    # Source NVDA absent -> unavailable.
    no_nvda = {k: v for k, v in df_dict.items() if k != "NVDA"}
    assert L2.test_directional_spillover_hub(no_nvda)["available"] is False


def test_h14_identifies_lead_asset_as_hub():
    # NVDA leads; MU and AMD are lagged copies of NVDA vol + noise.
    rng = np.random.default_rng(1)
    n = 320
    idx = pd.bdate_range("2021-01-04", periods=n)
    nv = pd.Series(np.abs(rng.normal(0, 1, n)).cumsum() % 5 + 1, index=idx)
    nv = nv.rolling(5).mean().bfill()

    def lagged(shift, noise):
        return nv.shift(shift).bfill() + rng.normal(0, noise, n)

    df_dict = {
        "NVDA": pd.DataFrame({"log_return": rng.normal(0, 0.01, n),
                              "realized_vol_21d": nv}, index=idx),
        "MU": pd.DataFrame({"log_return": rng.normal(0, 0.01, n),
                            "realized_vol_21d": lagged(2, 0.2)}, index=idx),
        "AMD": pd.DataFrame({"log_return": rng.normal(0, 0.01, n),
                             "realized_vol_21d": lagged(3, 0.2)}, index=idx),
    }
    r = L2.test_directional_spillover_hub(df_dict, n_boot=80)
    assert r["available"] is True
    net = r["net_spillover"]
    assert net.idxmax() == "NVDA"     # the lead asset is the net source


def test_h15_contract(df_dict):
    r = L2.test_cross_sector_contagion(df_dict)
    _assert_l2_contract(r)
    assert r["hypothesis"] == "H15"


def test_h15_degraded(df_dict):
    one_sector = {"MU": df_dict["MU"]}  # <2 sectors
    assert L2.test_cross_sector_contagion(one_sector)["available"] is False


def test_h16_contract(price_df):
    r = L2.test_leverage_amplification(price_df, "TEST")
    _assert_l2_contract(r)
    assert r["hypothesis"] == "H16"
    assert "leverage_ratios" in r


def test_h16_degraded_does_not_raise(price_df):
    tiny = price_df.iloc[:25]
    r = L2.test_leverage_amplification(tiny, "TEST")
    assert isinstance(r, dict)
    assert "available" in r
    assert "leverage_ratios" in r


def test_h16_detects_amplification():
    # fwd vol = base + slope * neg_mag * vol_level  -> positive interaction.
    rng = np.random.default_rng(2)
    n = 600
    idx = pd.bdate_range("2020-01-02", periods=n)
    ret = rng.normal(0, 0.02, n)
    vol = 0.2 + 0.3 * (np.sin(np.linspace(0, 12, n)) + 1)   # ranges ~0.2..0.8
    neg_mag = np.clip(-ret, 0, None)
    fwd = 0.2 + 5.0 * neg_mag * vol + rng.normal(0, 0.01, n)
    df = pd.DataFrame({"log_return": ret, "realized_vol_21d": vol,
                       "realized_vol_5d": fwd}, index=idx)
    r = L2.test_leverage_amplification(df, "TEST")
    assert r["available"] is True
    assert r["statistic"] > 0          # interaction coefficient positive
    assert r["p_value"] < 0.05


def test_h17_contract(price_df):
    r = L2.test_monday_leverage_interaction(price_df, "TEST")
    _assert_l2_contract(r)
    assert r["hypothesis"] == "H17"


def test_h17_degraded(price_df):
    tiny = price_df.iloc[:8]
    assert L2.test_monday_leverage_interaction(tiny, "TEST")["available"] is False


def test_h18_contract(price_df, sent, earnings_dates_l2):
    r = L2.test_pre_earnings_sentiment_predicts_spike(price_df, sent, earnings_dates_l2, "TEST")
    _assert_l2_contract(r)
    assert r["hypothesis"] == "H18"


def test_h18_degraded(price_df, sent):
    assert L2.test_pre_earnings_sentiment_predicts_spike(
        price_df, sent, pd.DatetimeIndex([]), "TEST")["available"] is False


def test_h19_contract(price_df, earnings_dates_l2):
    r = L2.test_earnings_vol_premium_vs_iv(price_df, earnings_dates_l2, "TEST")
    _assert_l2_contract(r)
    assert r["hypothesis"] == "H19"
    assert r["proxy_iv"] is True  # no iv_series supplied -> proxy path


def test_h19_degraded(price_df):
    two = pd.DatetimeIndex([price_df.index[60], price_df.index[200]])
    assert L2.test_earnings_vol_premium_vs_iv(price_df, two, "TEST")["available"] is False


def test_h20_contract(price_df, lm_scores):
    r = L2.test_10k_language_change_predicts_regime(price_df, lm_scores, "TEST")
    _assert_l2_contract(r)
    assert r["hypothesis"] == "H20"


def test_h20_degraded(price_df):
    two = pd.Series([0.1, 0.2], index=price_df.index[[40, 200]])  # <3 filings
    assert L2.test_10k_language_change_predicts_regime(price_df, two, "TEST")["available"] is False


def test_h21_contract(df_dict):
    r = L2.test_topic_specific_risk_prediction(df_dict)
    _assert_l2_contract(r)
    assert r["hypothesis"] == "H21"
    assert r["proxy"] is True  # no topic_scores supplied -> proxy path


def test_h21_degraded():
    idx = pd.bdate_range("2022-01-03", periods=50)
    short = {"MU": pd.DataFrame({"log_return": np.zeros(50),
                                 "realized_vol_21d": np.linspace(0.1, 0.2, 50)}, index=idx)}
    assert L2.test_topic_specific_risk_prediction(short)["available"] is False


def test_h22_contract(price_df, sent, disagreement):
    r = L2.test_triple_signal_interaction(price_df, sent, disagreement, "TEST")
    _assert_l2_contract(r)
    assert r["hypothesis"] == "H22"


def test_h22_degraded(price_df, sent):
    short_dis = pd.Series(np.arange(10.0), index=price_df.index[:10])
    assert L2.test_triple_signal_interaction(
        price_df.iloc[:10], sent.iloc[:10], short_dis, "TEST")["available"] is False


def test_h23_contract(price_df, sent, disagreement, df_dict):
    r = L2.test_signal_temporal_precedence(price_df, sent, disagreement, "TEST", df_dict=df_dict)
    _assert_l2_contract(r)
    assert r["hypothesis"] == "H23"


def test_h23_degraded_does_not_raise(price_df, sent):
    tiny_dis = pd.Series(np.arange(15.0), index=price_df.index[:15])
    r = L2.test_signal_temporal_precedence(price_df.iloc[:15], sent.iloc[:15], tiny_dis, "TEST")
    assert isinstance(r, dict)
    assert "available" in r


def test_compile_l2_summary_shape(price_df, sent, df_dict, disagreement,
                                  earnings_dates_l2, lm_scores):
    res = {
        "H9": L2.test_sentiment_asymmetry(price_df, sent, "TEST"),
        "H15": L2.test_cross_sector_contagion(df_dict),
        "H16": L2.test_regime_dependent_leverage(price_df, "TEST", n_perm=300),
        "H20": L2.test_10k_language_change_predicts_regime(price_df, lm_scores, "TEST"),
    }
    summary = L2.compile_l2_summary(res)
    # All 15 rows present (missing hypotheses render as blanks), in H9..H23 order.
    assert list(summary.index) == [f"H{i}" for i in range(9, 24)]
    assert {"Extends", "Finding", "p_value", "Effect", "Significant",
            "Actionable"}.issubset(summary.columns)
    assert summary.loc["H9", "Significant"] in {"Yes", "No", "n/a (no data)", "--"}
