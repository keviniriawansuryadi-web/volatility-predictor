# Simplify Hypothesis Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace two large hypothesis modules (2,792 lines, 23 tests) with one module, `src/hypotheses.py`, holding 5 high-value tests that are thin wrappers around scipy/statsmodels — no homemade significance tests, no fabricated data, no plotting.

**Architecture:** A single module exposes 5 `test_*` functions and a `compile_summary` helper. Every test returns a uniform result dict (`hypothesis, title, statistic, p_value, effect, n, conclusion, available`) and degrades gracefully (`available=False`) instead of raising or inventing inputs. Charting is removed entirely. The old modules are deleted and the one Python consumer (`scripts/run_vrp_analysis.py`) is repointed at the new API.

**Tech Stack:** Python 3.11, numpy, pandas, scipy.stats (`mannwhitneyu`, `bootstrap`, permutation), statsmodels (`grangercausalitytests`), pytest.

---

## File Structure

- **Create** `src/hypotheses.py` — the 5 tests + shared helpers + `compile_summary`. Single responsibility: hypothesis testing on price/VIX data.
- **Create** `tests/test_hypotheses.py` — synthetic-data contract tests for all 5 functions.
- **Delete** `src/hypothesis_tests.py` (H1–H9, 926 lines).
- **Delete** `modules/hypothesis_tests_L2.py` (H9–H23, 1,646 lines).
- **Modify** `modules/__init__.py` — drop the reference to the deleted L2 module.
- **Modify** `scripts/run_vrp_analysis.py` — import and call the new `test_variance_risk_premium`.
- **Modify** `README.md` — trim the hypothesis section to the 5 surviving tests.
- **Delete** `notebooks/07_advanced_hypotheses.ipynb` (the L2 notebook).
- **Modify** `notebooks/hypotheses.ipynb` — repoint imports/calls at `src.hypotheses` (best-effort; see Task 8).

**Not in scope / do NOT touch:** `src/hypothesis.py` (singular — the `spike_sentiment_test` module imported by `app.py` and `main.py`), the GARCH/HAR/ML models, `src/data_helpers.py`.

---

### Task 1: Leverage effect + module scaffold + test fixtures

**Files:**
- Create: `src/hypotheses.py`
- Test: `tests/test_hypotheses.py`

- [ ] **Step 1: Write the failing test (with shared fixtures)**

Create `tests/test_hypotheses.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_hypotheses.py::test_leverage_effect_contract -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.hypotheses'` (or `AttributeError`).

- [ ] **Step 3: Create `src/hypotheses.py` with shared helpers + the leverage test**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_hypotheses.py -v`
Expected: PASS — `test_leverage_effect_contract` and `test_leverage_effect_degraded`.

- [ ] **Step 5: Commit**

```bash
git add src/hypotheses.py tests/test_hypotheses.py
git commit -m "feat: add leverage-effect hypothesis test + module scaffold"
```

---

### Task 2: Cross-sector contagion (Granger)

**Files:**
- Modify: `src/hypotheses.py`
- Test: `tests/test_hypotheses.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_hypotheses.py`:

```python
def test_cross_sector_contagion_contract(df_dict):
    r = H.test_cross_sector_contagion(df_dict)
    _assert_contract(r)
    assert r["hypothesis"] == "cross_sector_contagion"


def test_cross_sector_contagion_degraded():
    one = {"MU": pd.DataFrame(
        {"realized_vol_21d": np.linspace(0.1, 0.3, 60)},
        index=pd.bdate_range("2022-01-03", periods=60))}
    r = H.test_cross_sector_contagion(one)
    assert r["available"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_hypotheses.py::test_cross_sector_contagion_contract -v`
Expected: FAIL — `AttributeError: module 'src.hypotheses' has no attribute 'test_cross_sector_contagion'`.

- [ ] **Step 3: Implement `test_cross_sector_contagion`**

Append to `src/hypotheses.py`:

```python
def test_cross_sector_contagion(df_dict: dict, sectors: dict | None = None,
                                lags: tuple = (1, 3, 5)) -> dict:
    """Directed Granger-causality matrix over sector-average vol series.

    Aggregates each sector to a mean realized-vol series, builds the min-p
    Granger matrix over ordered sector pairs (statsmodels grangercausalitytests),
    and ranks sectors by net contagion (significant outgoing - incoming links).
    """
    title = "Cross-sector vol contagion (Granger)"
    sectors = sectors or DEFAULT_SECTORS
    sector_vol = {}
    for name, members in sectors.items():
        cols = [df_dict[t]["realized_vol_21d"].rename(t) for t in members if t in df_dict]
        if cols:
            sector_vol[name] = pd.concat(cols, axis=1).mean(axis=1).rename(name)
    names = list(sector_vol)
    if len(names) < 2:
        return _unavailable("cross_sector_contagion", title,
                            f"Need >=2 sectors with data, got {len(names)}.")

    pmat = pd.DataFrame(np.nan, index=names, columns=names)
    for src in names:
        for tgt in names:
            if src == tgt:
                continue
            data = pd.concat([sector_vol[tgt].rename("y"),
                              sector_vol[src].rename("x")], axis=1).dropna()
            if len(data) < 40:
                continue
            try:
                res = grangercausalitytests(data[["y", "x"]], maxlag=max(lags),
                                            verbose=False)
                pmat.loc[src, tgt] = min(res[lag][0]["ssr_ftest"][1] for lag in lags)
            except Exception as exc:  # noqa: BLE001 - report a null, do not raise
                warnings.warn(f"[contagion] {src}->{tgt} failed: {exc}")

    if not np.isfinite(pmat.values).any():
        return _unavailable("cross_sector_contagion", title,
                            "No sector pair had enough overlapping data.")

    sig = pmat < 0.05
    net = (sig.sum(axis=1) - sig.sum(axis=0)).sort_values(ascending=False)
    p_min = float(np.nanmin(pmat.values))
    source = net.index[0]
    conclusion = (f"Net-contagion ranking (out-in): "
                  + ", ".join(f"{k}={v:+d}" for k, v in net.items())
                  + f". Source sector = {source}. min Granger p={p_min:.4f}.")
    return dict(hypothesis="cross_sector_contagion", title=title, statistic=np.nan,
                p_value=p_min, effect=float(net.iloc[0]), n=int(len(names)),
                conclusion=conclusion, available=True,
                pval_matrix=pmat, net_contagion=net, source_sector=source)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_hypotheses.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/hypotheses.py tests/test_hypotheses.py
git commit -m "feat: add cross-sector Granger contagion test"
```

---

### Task 3: Variance Risk Premium

**Files:**
- Modify: `src/hypotheses.py`
- Test: `tests/test_hypotheses.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_hypotheses.py`:

```python
def test_variance_risk_premium_contract(price_df):
    r = H.test_variance_risk_premium(price_df, ticker="TEST", horizon=10)
    _assert_contract(r)
    assert r["hypothesis"] == "variance_risk_premium"
    # VRP-specific extras consumed by scripts/run_vrp_analysis.py
    for k in ("current_vrp", "current_vix", "current_rv", "current_state",
              "significant", "neg_mean_fwd_vol", "pos_mean_fwd_vol", "n_negative_vrp"):
        assert k in r


def test_variance_risk_premium_degraded(price_df):
    no_vix = price_df.drop(columns=["vix_level"])
    r = H.test_variance_risk_premium(no_vix)
    assert r["available"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_hypotheses.py::test_variance_risk_premium_contract -v`
Expected: FAIL — attribute does not exist.

- [ ] **Step 3: Implement `test_variance_risk_premium`**

Append to `src/hypotheses.py`:

```python
def test_variance_risk_premium(df: pd.DataFrame, ticker: str = "SPY",
                               horizon: int = 10) -> dict:
    """Negative variance risk premium (RV > VIX) predicts higher forward vol.

    VRP = vix_level - realized_vol_21d. Splits forward ``horizon``-day vol by
    VRP sign and compares with scipy.stats.mannwhitneyu (one-sided).
    Returns the standard contract plus VRP state fields used by the VRP script.
    """
    title = "Variance risk premium -> forward vol"
    if "vix_level" not in df.columns or "realized_vol_21d" not in df.columns:
        return _unavailable("variance_risk_premium", title,
                            "Need vix_level and realized_vol_21d columns.")
    d = df.copy()
    d["fwd"] = d["log_return"].shift(-horizon).rolling(horizon).std() * np.sqrt(252)
    d["vrp"] = d["vix_level"] - d["realized_vol_21d"]
    d = d.dropna(subset=["vrp", "fwd"])
    neg = d.loc[d["vrp"] < 0, "fwd"].values
    pos = d.loc[d["vrp"] >= 0, "fwd"].values

    current_vix = float(df["vix_level"].iloc[-1])
    current_rv = float(df["realized_vol_21d"].iloc[-1])
    current_vrp = current_vix - current_rv
    state = "NEGATIVE (RV > VIX)" if current_vrp < 0 else "POSITIVE"

    if len(neg) < 5 or len(pos) < 5:
        r = _unavailable("variance_risk_premium", title,
                         f"Too few VRP episodes (neg={len(neg)}, pos={len(pos)}).")
        r.update(current_vrp=current_vrp, current_vix=current_vix, current_rv=current_rv,
                 current_state=state, significant=False, neg_mean_fwd_vol=np.nan,
                 pos_mean_fwd_vol=np.nan, n_negative_vrp=int(len(neg)))
        return r

    u, p = stats.mannwhitneyu(neg, pos, alternative="greater")
    r_eff = 1 - 2 * u / (len(neg) * len(pos))
    neg_mean, pos_mean = float(neg.mean()), float(pos.mean())
    conclusion = (f"{ticker}: negative-VRP days show {'significantly ' if p < 0.05 else 'not significantly '}"
                  f"higher next-{horizon}d vol ({neg_mean:.1%} vs {pos_mean:.1%}, "
                  f"U={u:.0f}, p={p:.4f}, r={r_eff:.3f}).")
    return dict(hypothesis="variance_risk_premium", title=title, statistic=float(u),
                p_value=float(p), effect=float(r_eff), n=int(len(neg) + len(pos)),
                conclusion=conclusion, available=True,
                current_vrp=current_vrp, current_vix=current_vix, current_rv=current_rv,
                current_state=state, significant=bool(p < 0.05),
                neg_mean_fwd_vol=neg_mean, pos_mean_fwd_vol=pos_mean,
                n_negative_vrp=int(len(neg)))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_hypotheses.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/hypotheses.py tests/test_hypotheses.py
git commit -m "feat: add variance-risk-premium test (with VRP state fields)"
```

---

### Task 4: Earnings-week vol premium (permutation)

**Files:**
- Modify: `src/hypotheses.py`
- Test: `tests/test_hypotheses.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_hypotheses.py`:

```python
def test_earnings_vol_premium_contract(price_df, earnings_dates):
    r = H.test_earnings_vol_premium(price_df, earnings_dates, n_permutations=500)
    _assert_contract(r)
    assert r["hypothesis"] == "earnings_vol_premium"


def test_earnings_vol_premium_degraded(price_df):
    r = H.test_earnings_vol_premium(price_df, pd.DatetimeIndex([]), n_permutations=500)
    assert r["available"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_hypotheses.py::test_earnings_vol_premium_contract -v`
Expected: FAIL — attribute does not exist.

- [ ] **Step 3: Implement `test_earnings_vol_premium`**

Append to `src/hypotheses.py`:

```python
def test_earnings_vol_premium(df: pd.DataFrame, earnings_dates,
                              vol_col: str = "realized_vol_21d", window: int = 2,
                              n_permutations: int = 5000, seed: int = 42) -> dict:
    """Realized vol is higher in the +/-``window`` days around earnings than on
    other days. Permutation test (label shuffling) on the difference in means —
    appropriate because earnings windows are rare events.
    """
    title = "Earnings-week vol premium (permutation)"
    trading_days = df.index
    mask = pd.Series(False, index=trading_days)
    for ed in pd.DatetimeIndex(earnings_dates):
        pos = trading_days.searchsorted(ed)
        lo = max(0, pos - window)
        hi = min(len(trading_days), pos + window + 1)
        mask.iloc[lo:hi] = True

    data = df[[vol_col]].copy()
    data["ew"] = mask.values
    ew = data.loc[data["ew"], vol_col].dropna().values
    non = data.loc[~data["ew"], vol_col].dropna().values
    if len(ew) < 3 or len(non) < 3:
        return _unavailable("earnings_vol_premium", title,
                            f"Too few observations (earnings={len(ew)}, other={len(non)}).")

    obs = float(ew.mean() - non.mean())
    rng = np.random.default_rng(seed)
    allv = np.concatenate([ew, non])
    n_ew = len(ew)
    perm = np.array([rng.permutation(allv)[:n_ew].mean()
                     - rng.permutation(allv)[n_ew:].mean()
                     for _ in range(n_permutations)])
    p = float((perm >= obs).mean())
    conclusion = (f"Earnings-window mean vol={ew.mean():.4f} vs other={non.mean():.4f} "
                  f"(diff={obs:+.4f}); permutation p={p:.4f} ({_verdict(p)}).")
    return dict(hypothesis="earnings_vol_premium", title=title, statistic=obs,
                p_value=p, effect=obs, n=int(len(ew)), conclusion=conclusion,
                available=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_hypotheses.py -v`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add src/hypotheses.py tests/test_hypotheses.py
git commit -m "feat: add earnings-week vol premium permutation test"
```

---

### Task 5: Regime-dependent leverage (scipy.stats.bootstrap)

**Files:**
- Modify: `src/hypotheses.py`
- Test: `tests/test_hypotheses.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_hypotheses.py`:

```python
def test_regime_dependent_leverage_contract(price_df):
    r = H.test_regime_dependent_leverage(price_df, ticker="TEST", n_perm=500)
    _assert_contract(r)
    assert r["hypothesis"] == "regime_dependent_leverage"
    assert "leverage_ratios" in r


def test_regime_dependent_leverage_degraded():
    tiny = pd.DataFrame(
        {"log_return": np.linspace(-0.01, 0.01, 25),
         "realized_vol_21d": np.linspace(0.1, 0.2, 25)},
        index=pd.bdate_range("2022-01-03", periods=25))
    r = H.test_regime_dependent_leverage(tiny, n_perm=200)
    assert r["available"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_hypotheses.py::test_regime_dependent_leverage_contract -v`
Expected: FAIL — attribute does not exist.

- [ ] **Step 3: Implement `test_regime_dependent_leverage`**

Append to `src/hypotheses.py`:

```python
def _leverage_ratio(sub: pd.DataFrame) -> float:
    """neg/pos forward-vol ratio for a regime subset (nan if too few in a side)."""
    neg = sub.loc[sub["ret"] < 0, "fwd"]
    pos = sub.loc[sub["ret"] > 0, "fwd"]
    if len(neg) < 5 or len(pos) < 5:
        return np.nan
    return float(neg.mean() / (pos.mean() + 1e-12))


def test_regime_dependent_leverage(df: pd.DataFrame, ticker: str = "",
                                   horizon: int = 5, n_perm: int = 5000,
                                   seed: int = 42) -> dict:
    """Does the leverage effect amplify in stressed regimes (Campbell & Hentschel
    1992)? Computes the neg/pos forward-vol ratio per vol regime, a
    scipy.stats.bootstrap CI on the Extreme-regime ratio, and a permutation test
    for Extreme ratio > Low ratio.
    """
    title = "Regime-dependent leverage (Extreme vs Low)"
    fwd = _forward_vol(df, horizon)
    d = pd.DataFrame({"ret": df["log_return"], "fwd": fwd,
                      "regime": _regime_series(df)}).dropna()
    if len(d) < 30:
        return _unavailable("regime_dependent_leverage", title,
                            f"Too few observations ({len(d)}).")

    ratios = {r: _leverage_ratio(d[d["regime"] == r]) for r in REGIME_ORDER}
    if np.isnan(ratios.get("Low", np.nan)) or np.isnan(ratios.get("Extreme", np.nan)):
        return _unavailable("regime_dependent_leverage", title,
                            "Low and/or Extreme regime lacks enough neg/pos days.")

    # scipy.stats.bootstrap CI on the Extreme-regime ratio (replaces manual loop).
    ex = d[d["regime"] == "Extreme"]
    neg_ex = ex.loc[ex["ret"] < 0, "fwd"].values
    pos_ex = ex.loc[ex["ret"] > 0, "fwd"].values

    def _ratio_stat(a, b, axis=-1):
        return np.mean(a, axis=axis) / (np.mean(b, axis=axis) + 1e-12)

    boot = bootstrap((neg_ex, pos_ex), _ratio_stat, n_resamples=2000,
                     random_state=seed, vectorized=True, method="percentile")
    extreme_ci = (float(boot.confidence_interval.low),
                  float(boot.confidence_interval.high))

    # Permutation test: Extreme ratio > Low ratio.
    obs_diff = ratios["Extreme"] - ratios["Low"]
    pooled = pd.concat([d[d["regime"] == "Low"], ex])
    n_ex = len(ex)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        perm = pooled.sample(frac=1, replace=False, random_state=int(rng.integers(1e9)))
        ex_p = _leverage_ratio(perm.iloc[:n_ex])
        lo_p = _leverage_ratio(perm.iloc[n_ex:])
        if not np.isnan(ex_p) and not np.isnan(lo_p) and (ex_p - lo_p) >= obs_diff:
            count += 1
    p_perm = count / n_perm

    ratio_str = ", ".join(f"{k}={v:.2f}" for k, v in ratios.items() if not np.isnan(v))
    conclusion = (f"{ticker}: leverage ratios [{ratio_str}]. Extreme-Low diff={obs_diff:+.2f}, "
                  f"permutation p={p_perm:.4f} ({_verdict(p_perm)}). "
                  f"Extreme-ratio 95% CI [{extreme_ci[0]:.2f}, {extreme_ci[1]:.2f}].")
    return dict(hypothesis="regime_dependent_leverage", title=title,
                statistic=float(obs_diff), p_value=float(p_perm), effect=float(obs_diff),
                n=int(len(d)), conclusion=conclusion, available=True,
                leverage_ratios=ratios, extreme_ratio_ci=extreme_ci)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_hypotheses.py -v`
Expected: PASS (10 tests).

- [ ] **Step 5: Commit**

```bash
git add src/hypotheses.py tests/test_hypotheses.py
git commit -m "feat: add regime-dependent leverage test (scipy bootstrap CI)"
```

---

### Task 6: `compile_summary`

**Files:**
- Modify: `src/hypotheses.py`
- Test: `tests/test_hypotheses.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_hypotheses.py`:

```python
def test_compile_summary(price_df, df_dict, earnings_dates):
    results = {
        "leverage_effect": H.test_leverage_effect(price_df),
        "cross_sector_contagion": H.test_cross_sector_contagion(df_dict),
        "variance_risk_premium": H.test_variance_risk_premium(price_df),
        "earnings_vol_premium": H.test_earnings_vol_premium(price_df, earnings_dates, n_permutations=500),
        "regime_dependent_leverage": H.test_regime_dependent_leverage(price_df, n_perm=500),
    }
    summary = H.compile_summary(results)
    assert list(summary.index) == list(results.keys())
    assert {"title", "p_value", "effect", "significant"}.issubset(summary.columns)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_hypotheses.py::test_compile_summary -v`
Expected: FAIL — attribute does not exist.

- [ ] **Step 3: Implement `compile_summary`**

Append to `src/hypotheses.py`:

```python
def compile_summary(results: dict) -> pd.DataFrame:
    """Collect ``test_*`` result dicts into a one-row-per-hypothesis table.

    ``results`` maps a key -> the dict returned by a test. Columns:
    title, p_value, effect, significant. Index = each result's ``hypothesis``.
    """
    rows = []
    for key, r in results.items():
        p = r.get("p_value", np.nan)
        eff = r.get("effect", np.nan)
        p_ok = isinstance(p, (int, float)) and not pd.isna(p)
        eff_ok = isinstance(eff, (int, float)) and not pd.isna(eff)
        if not r.get("available", True):
            sig = "n/a (no data)"
        elif p_ok:
            sig = "Yes" if p < 0.05 else "No"
        else:
            sig = "--"
        rows.append({
            "hypothesis": r.get("hypothesis", key),
            "title": r.get("title", ""),
            "p_value": f"{p:.4f}" if p_ok else "--",
            "effect": f"{eff:.3f}" if eff_ok else "--",
            "significant": sig,
        })
    return pd.DataFrame(rows).set_index("hypothesis")
```

- [ ] **Step 4: Run the full test file to verify everything passes**

Run: `python -m pytest tests/test_hypotheses.py -v`
Expected: PASS (11 tests).

- [ ] **Step 5: Commit**

```bash
git add src/hypotheses.py tests/test_hypotheses.py
git commit -m "feat: add compile_summary for the 5-hypothesis suite"
```

---

### Task 7: Delete old modules and repoint the VRP script

**Files:**
- Delete: `src/hypothesis_tests.py`
- Delete: `modules/hypothesis_tests_L2.py`
- Modify: `modules/__init__.py`
- Modify: `scripts/run_vrp_analysis.py:29` and `:45`

- [ ] **Step 1: Delete the two old modules**

```bash
git rm src/hypothesis_tests.py modules/hypothesis_tests_L2.py
```

- [ ] **Step 2: Clean the L2 reference out of `modules/__init__.py`**

Open `modules/__init__.py` and remove the line referencing
`modules/hypothesis_tests_L2.py` (line 5 in the module docstring). If the file
becomes an empty/near-empty package init, leave a one-line docstring:

```python
"""Project ``modules`` package."""
```

- [ ] **Step 3: Repoint `scripts/run_vrp_analysis.py` at the new API**

In `scripts/run_vrp_analysis.py`, change the import (line 29) from:

```python
from src.hypothesis_tests import analyze_variance_risk_premium
```

to:

```python
from src.hypotheses import test_variance_risk_premium
```

and change the call (line 45) from:

```python
result = analyze_variance_risk_premium(df, ticker="SPY", forward_days=10, plot_dir=PLOT_DIR)
```

to:

```python
result = test_variance_risk_premium(df, ticker="SPY", horizon=10)
```

The downstream `result[...]` accesses (`current_state`, `current_vrp`,
`current_vix`, `current_rv`, `significant`, `p_value`, `neg_mean_fwd_vol`,
`pos_mean_fwd_vol`, `n_negative_vrp`, `available`) are all still provided by
`test_variance_risk_premium`, so no further edits are needed. The now-unused
`PLOT_DIR` assignment (line 33) may be left or removed.

- [ ] **Step 4: Verify nothing else imports the deleted modules**

Run: `python -m pytest tests/ -q`
Expected: PASS — the whole suite, including `tests/test_hypotheses.py`.

Run: `python -c "import ast, pathlib; [None for _ in [0]]"` is not needed; instead grep:
Run: `grep -rn "hypothesis_tests" --include=*.py .`
Expected: no matches outside the plan/docs (the only former consumer,
`scripts/run_vrp_analysis.py`, now imports `src.hypotheses`).

- [ ] **Step 5: Smoke-test the VRP script import path**

Run: `python -c "import ast; ast.parse(open('scripts/run_vrp_analysis.py', encoding='utf-8').read()); print('parse ok')"`
Expected: `parse ok` (syntax valid; full run needs network so is out of scope here).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor: delete legacy hypothesis modules, repoint VRP script"
```

---

### Task 8: Knock-on docs and notebooks

**Files:**
- Delete: `notebooks/07_advanced_hypotheses.ipynb`
- Modify: `notebooks/hypotheses.ipynb`
- Modify: `README.md`

- [ ] **Step 1: Retire the L2 notebook**

```bash
git rm notebooks/07_advanced_hypotheses.ipynb
```

- [ ] **Step 2: Repoint `notebooks/hypotheses.ipynb`**

Open the notebook and update the hypothesis cells:
- Replace any `import src.hypothesis_tests ...` / `import modules.hypothesis_tests_L2 as L2`
  with `from src import hypotheses as H`.
- Replace calls to the old functions with the 5 new ones:
  `H.test_leverage_effect(df)`, `H.test_cross_sector_contagion(df_dict)`,
  `H.test_variance_risk_premium(df, ticker=..., horizon=10)`,
  `H.test_earnings_vol_premium(df, earnings_dates)`,
  `H.test_regime_dependent_leverage(df, ticker=...)`, and
  `H.compile_summary({...})`.
- Delete cells that called now-removed hypotheses or that rendered figures
  from the result dicts (the new results have no `figure` key).

Use `jupyter nbconvert --to notebook --execute notebooks/hypotheses.ipynb --output hypotheses.ipynb`
to confirm it runs end-to-end if a data connection is available; otherwise
verify the edited cells parse by opening the notebook.

- [ ] **Step 3: Trim the README hypothesis section**

In `README.md`, replace the H1–H23 hypothesis listing with the 5 surviving
tests (leverage effect, cross-sector contagion, variance risk premium,
earnings-week vol premium, regime-dependent leverage), each with its one-line
description and the library it uses. Remove references to the deleted L2
notebook and the `findings_report` Level-2 section.

- [ ] **Step 4: Final full-suite run**

Run: `python -m pytest tests/ -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "docs: retire L2 notebook, repoint hypotheses notebook, trim README"
```

---

## Self-Review

**Spec coverage:**
- 5 hypotheses (leverage, contagion, VRP, earnings, regime-leverage) → Tasks 1–5. ✓
- `src/hypotheses.py` single module + `compile_summary` → Tasks 1–6. ✓
- Uniform return contract, graceful degradation → enforced by `_unavailable` and tested in every task. ✓
- Remove homemade tests (Steiger, SRH), manual bootstrap → bootstrap replaced by `scipy.stats.bootstrap` (Task 5); Steiger/SRH simply absent (their hypotheses are not in the 5). ✓
- No fabricated data → H19/H21 excluded entirely. ✓
- No figures → no plotting code in any function; notebook figure cells removed (Task 8). ✓
- Delete both old modules → Task 7. ✓
- Tests → Tasks 1–6 build `tests/test_hypotheses.py`. ✓
- Knock-on (notebooks, README, VRP script) → Tasks 7–8. ✓

**Placeholder scan:** No TBD/TODO; every code step has complete code. ✓

**Type consistency:** Result keys (`hypothesis, title, statistic, p_value, effect, n, conclusion, available`) are identical across `_unavailable` and all 5 tests; `compile_summary` reads only those keys plus optional extras. `test_variance_risk_premium` extra keys match exactly what `scripts/run_vrp_analysis.py` consumes. ✓
