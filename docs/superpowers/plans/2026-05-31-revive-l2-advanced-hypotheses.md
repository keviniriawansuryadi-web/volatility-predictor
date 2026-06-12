# Revive the L2 Advanced Hypothesis Suite (H9–H23) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Revive `modules/hypothesis_tests_L2.py` (H9–H23) as a tracked, fully tested, runnable research suite that coexists with `src/hypotheses.py`.

**Architecture:** The 15 `test_*` functions already exist in the untracked module, so their tests are *characterization tests* — they should pass immediately or surface a real bug to fix (bug fixes allowed; no rewrites of working logic). Net-new code is the test file + fixtures, a headless runner script, and the revived notebook. Inputs (`df`, `df_dict`, `sent`, `disagreement`, `earnings_dates`, `lm_scores`) are assembled from existing `src` helpers.

**Tech Stack:** Python 3.14, pandas/numpy, scipy.stats, statsmodels, scikit_posthocs, plotly, matplotlib, pytest. All already in `requirements.txt`.

**Branch:** Create and work on `revive-l2-hypotheses` (we are on `master`; do not commit the plan's work directly to `master`).

```bash
git checkout -b revive-l2-hypotheses
```

---

## File Structure

| Path | Responsibility | Action |
|------|----------------|--------|
| `modules/hypothesis_tests_L2.py` | The 15 H9–H23 tests + helpers (already exists, untracked) | Track + 1 docstring fix |
| `tests/test_hypotheses_L2.py` | Fixtures, contract+degraded tests, pinned stat unit tests | Create |
| `scripts/run_advanced_hypotheses.py` | Headless runner: assemble inputs, run all 15, save figures + summary CSV | Create |
| `notebooks/07_advanced_hypotheses.ipynb` | Interactive research surface | Restore from git + verify |

**L2 return-dict contract** (differs from `src/hypotheses.py`): every test returns at least `hypothesis, extends, available, p_value, conclusion, actionable`; when `available` is `True` it also returns `figure`. `p_value` may be `NaN` even when available (e.g. H16 permutation, H23 Friedman). **H16 and H23 have no `available=False` guard** — they always return `available=True`, so their degraded tests assert "returns a dict without raising," not `available=False`.

---

## Task 1: Track the module + fix the stale docstring

**Files:**
- Modify: `modules/hypothesis_tests_L2.py` (docstring lines 4–5)

- [ ] **Step 1: Fix the stale reference to the deleted L1 module**

The module docstring points at `src/hypothesis_tests.py`, which was deleted in `8d9aea3`. Replace that dead path reference. Edit:

Old:
```python
Every test in this module **extends a confirmed Level-1 finding** (H1-H8 in
``src/hypothesis_tests.py``) and pushes it toward the kind of question that a
```
New:
```python
Every test in this module **extends a confirmed Level-1 finding** (H1-H8 from
the project's earlier hypothesis work) and pushes it toward the kind of question that a
```

- [ ] **Step 2: Verify the module imports cleanly**

Run: `python -c "import modules.hypothesis_tests_L2 as L2; print(len([f for f in dir(L2) if f.startswith('test_')]))"`
Expected: `15`

- [ ] **Step 3: Commit**

```bash
git add modules/hypothesis_tests_L2.py
git commit -m "feat: track L2 advanced hypothesis suite; fix stale L1 docstring ref"
```

---

## Task 2: Test scaffold — fixtures + contract helper

**Files:**
- Create: `tests/test_hypotheses_L2.py`

- [ ] **Step 1: Write the scaffold with fixtures and a smoke test**

```python
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
```

- [ ] **Step 2: Run the scaffold**

Run: `python -m pytest tests/test_hypotheses_L2.py -v`
Expected: PASS (1 test: `test_module_exposes_15_tests`)

- [ ] **Step 3: Commit**

```bash
git add tests/test_hypotheses_L2.py
git commit -m "test: scaffold L2 test fixtures and contract helper"
```

---

## Task 3: Pin `steiger_dependent_corr`

**Files:**
- Modify: `tests/test_hypotheses_L2.py` (append)

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run them**

Run: `python -m pytest tests/test_hypotheses_L2.py -k steiger -v`
Expected: PASS (the function already exists; these pin its behaviour). If any FAIL, the function has a real bug — fix the math in `modules/hypothesis_tests_L2.py` (do not weaken the test).

- [ ] **Step 3: Commit**

```bash
git add tests/test_hypotheses_L2.py
git commit -m "test: pin steiger_dependent_corr against known properties"
```

---

## Task 4: Pin `scheirer_ray_hare`

**Files:**
- Modify: `tests/test_hypotheses_L2.py` (append)

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run them**

Run: `python -m pytest tests/test_hypotheses_L2.py -k srh -v`
Expected: PASS. If FAIL, fix the SRH implementation (real bug), not the test.

- [ ] **Step 3: Commit**

```bash
git add tests/test_hypotheses_L2.py
git commit -m "test: pin scheirer_ray_hare structure and main-effect detection"
```

---

## Task 5: Sentiment tests (H9, H10, H11)

**Files:**
- Modify: `tests/test_hypotheses_L2.py` (append)

- [ ] **Step 1: Write contract + degraded tests**

```python
def _tiny():
    idx = pd.bdate_range("2022-01-03", periods=8)
    df = pd.DataFrame({"log_return": np.linspace(-0.01, 0.01, 8),
                       "realized_vol_21d": np.linspace(0.1, 0.2, 8)}, index=idx)
    sent = pd.DataFrame({c: np.linspace(-0.2, 0.2, 8) for c in
                         ["vader_compound", "finbert", "textblob", "lm_score"]}, index=idx)
    return df, sent


def test_h9_contract(price_df, sent):
    r = L2.test_sentiment_asymmetry(price_df, sent, "TEST")
    _assert_l2_contract(r)
    assert r["hypothesis"] == "H9"


def test_h9_degraded():
    df, sent = _tiny()
    assert L2.test_sentiment_asymmetry(df, sent, "TEST")["available"] is False


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
```

- [ ] **Step 2: Run**

Run: `python -m pytest tests/test_hypotheses_L2.py -k "h9 or h10 or h11" -v`
Expected: PASS (6 tests). Any failure on a contract test = a real bug to fix in the module.

- [ ] **Step 3: Commit**

```bash
git add tests/test_hypotheses_L2.py
git commit -m "test: contract+degraded for sentiment hypotheses H9-H11"
```

---

## Task 6: Disagreement tests (H12, H13)

**Files:**
- Modify: `tests/test_hypotheses_L2.py` (append)

- [ ] **Step 1: Write contract + degraded tests**

```python
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
```

- [ ] **Step 2: Run**

Run: `python -m pytest tests/test_hypotheses_L2.py -k "h12 or h13" -v`
Expected: PASS (4 tests).

- [ ] **Step 3: Commit**

```bash
git add tests/test_hypotheses_L2.py
git commit -m "test: contract+degraded for disagreement hypotheses H12-H13"
```

---

## Task 7: Contagion tests (H14, H15)

**Files:**
- Modify: `tests/test_hypotheses_L2.py` (append)

- [ ] **Step 1: Write contract + degraded tests**

```python
def test_h14_contract(df_dict):
    r = L2.test_asymmetric_contagion(df_dict)
    _assert_l2_contract(r)
    assert r["hypothesis"] == "H14"


def test_h14_degraded(df_dict):
    # Source NVDA absent -> unavailable.
    no_nvda = {k: v for k, v in df_dict.items() if k != "NVDA"}
    assert L2.test_asymmetric_contagion(no_nvda)["available"] is False


def test_h15_contract(df_dict):
    r = L2.test_cross_sector_contagion(df_dict)
    _assert_l2_contract(r)
    assert r["hypothesis"] == "H15"


def test_h15_degraded(df_dict):
    one_sector = {"MU": df_dict["MU"]}  # <2 sectors
    assert L2.test_cross_sector_contagion(one_sector)["available"] is False
```

- [ ] **Step 2: Run**

Run: `python -m pytest tests/test_hypotheses_L2.py -k "h14 or h15" -v`
Expected: PASS (4 tests). (H15 exercises `grangercausalitytests` — confirms the import typo fix works.)

- [ ] **Step 3: Commit**

```bash
git add tests/test_hypotheses_L2.py
git commit -m "test: contract+degraded for contagion hypotheses H14-H15"
```

---

## Task 8: Leverage tests (H16, H17)

**Files:**
- Modify: `tests/test_hypotheses_L2.py` (append)

- [ ] **Step 1: Write contract + degraded tests**

H16 has no `available=False` guard, so its degraded test only asserts it returns a dict without raising.

```python
def test_h16_contract(price_df):
    r = L2.test_regime_dependent_leverage(price_df, "TEST", n_perm=500)
    _assert_l2_contract(r)
    assert r["hypothesis"] == "H16"
    assert "leverage_ratios" in r


def test_h16_degraded_does_not_raise(price_df):
    tiny = price_df.iloc[:25]
    r = L2.test_regime_dependent_leverage(tiny, "TEST", n_perm=100)
    assert isinstance(r, dict)
    assert "available" in r


def test_h17_contract(price_df):
    r = L2.test_monday_leverage_interaction(price_df, "TEST")
    _assert_l2_contract(r)
    assert r["hypothesis"] == "H17"


def test_h17_degraded(price_df):
    tiny = price_df.iloc[:8]
    assert L2.test_monday_leverage_interaction(tiny, "TEST")["available"] is False
```

- [ ] **Step 2: Run**

Run: `python -m pytest tests/test_hypotheses_L2.py -k "h16 or h17" -v`
Expected: PASS (4 tests).

- [ ] **Step 3: Commit**

```bash
git add tests/test_hypotheses_L2.py
git commit -m "test: contract+degraded for leverage hypotheses H16-H17"
```

---

## Task 9: Earnings tests (H18, H19)

**Files:**
- Modify: `tests/test_hypotheses_L2.py` (append)

- [ ] **Step 1: Write contract + degraded tests**

```python
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
```

- [ ] **Step 2: Run**

Run: `python -m pytest tests/test_hypotheses_L2.py -k "h18 or h19" -v`
Expected: PASS (4 tests).

- [ ] **Step 3: Commit**

```bash
git add tests/test_hypotheses_L2.py
git commit -m "test: contract+degraded for earnings hypotheses H18-H19"
```

---

## Task 10: 10-K text tests (H20, H21)

**Files:**
- Modify: `tests/test_hypotheses_L2.py` (append)

- [ ] **Step 1: Write contract + degraded tests**

```python
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
```

- [ ] **Step 2: Run**

Run: `python -m pytest tests/test_hypotheses_L2.py -k "h20 or h21" -v`
Expected: PASS (4 tests).

- [ ] **Step 3: Commit**

```bash
git add tests/test_hypotheses_L2.py
git commit -m "test: contract+degraded for 10-K text hypotheses H20-H21"
```

---

## Task 11: Cross-hypothesis interaction tests (H22, H23)

**Files:**
- Modify: `tests/test_hypotheses_L2.py` (append)

- [ ] **Step 1: Write contract + degraded tests**

H23 has no `available=False` guard, so its degraded test only asserts a dict is returned without raising.

```python
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
```

- [ ] **Step 2: Run**

Run: `python -m pytest tests/test_hypotheses_L2.py -k "h22 or h23" -v`
Expected: PASS (4 tests).

- [ ] **Step 3: Commit**

```bash
git add tests/test_hypotheses_L2.py
git commit -m "test: contract+degraded for interaction hypotheses H22-H23"
```

---

## Task 12: `compile_l2_summary` test + full-suite green

**Files:**
- Modify: `tests/test_hypotheses_L2.py` (append)

- [ ] **Step 1: Write the summary test**

```python
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
```

- [ ] **Step 2: Run the whole L2 suite**

Run: `python -m pytest tests/test_hypotheses_L2.py -v`
Expected: PASS (all ~35 tests). Also confirm the original suite still passes: `python -m pytest tests/test_hypotheses.py -q` → 11 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/test_hypotheses_L2.py
git commit -m "test: compile_l2_summary table shape; L2 suite green"
```

---

## Task 13: Headless runner script

**Files:**
- Create: `scripts/run_advanced_hypotheses.py`

The assembly mirrors the retired notebook (verified from `e6be0ae~1`): real EGARCH/ML disagreement with a realized-vol proxy fallback, simulated sentiment / 10-K scores, real earnings dates with a synthetic fallback.

- [ ] **Step 1: Write the script**

```python
"""Headless runner for the L2 advanced hypothesis suite (H9-H23).

Assembles inputs from src helpers, runs all 15 tests for a primary ticker
(plus a sector universe for the contagion tests), writes figures to
outputs/figures/L2/ and the summary CSV to outputs/results/.

Usage:
    python scripts/run_advanced_hypotheses.py --ticker MU --start 2018-01-01 --end 2024-12-31
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import load_stock_data
from src.features import build_features
from src.ml_model import train_and_predict
from src.garch_model import rolling_garch_forecast
from src.data_helpers import (add_forward_vol, simulate_sentiment,
                              load_earnings_dates, simulate_10k_risk_scores)
import modules.hypothesis_tests_L2 as L2


def assemble_inputs(ticker: str, start: str, end: str) -> dict:
    """Build every input the L2 suite needs for `ticker` + sector universe."""
    df = add_forward_vol(load_stock_data(ticker, start, end, cache=True), horizon=5)
    sent = simulate_sentiment(df, ticker)

    # EGARCH-ML disagreement (real models; realized-vol proxy if fitting fails).
    try:
        feat_df = build_features(df, forecast_horizon=5)
        ml_preds, _, _ = train_and_predict(feat_df, model_type="xgboost", train_size=0.8)
        garch_preds = rolling_garch_forecast(df["log_return"], train_size=0.8,
                                              forecast_horizon=5, model_type="EGARCH")
        dis = L2.compute_disagreement(garch_preds, ml_preds)
    except Exception as exc:
        warnings.warn(f"EGARCH/ML fit failed ({exc}); using realized-vol proxy disagreement.")
        rng = np.random.default_rng(0)
        split = int(len(df) * 0.4)
        rv = df["realized_vol_21d"].iloc[split:]
        g = (rv * (1 + rng.normal(0, 0.15, len(rv)))).rename("g")
        m = (rv * (1 + rng.normal(0, 0.15, len(rv)))).rename("m")
        dis = L2.compute_disagreement(g, m)

    df_dict = {ticker: df}
    for members in L2.DEFAULT_SECTORS.values():
        for t in members:
            if t in df_dict:
                continue
            try:
                df_dict[t] = load_stock_data(t, start, end, cache=True)
            except Exception as exc:
                warnings.warn(f"Could not load {t}: {exc}")

    earnings = load_earnings_dates(ticker)
    if len(earnings) < 4:
        rng = np.random.default_rng(1)
        earnings = pd.DatetimeIndex(sorted(df.index[rng.integers(30, len(df) - 30, 12)]))
    filings = pd.DatetimeIndex(sorted(set(df.index[np.linspace(60, len(df) - 80, 5).astype(int)])))
    lm_scores = simulate_10k_risk_scores(ticker, filings)

    return dict(df=df, sent=sent, disagree=dis["norm_abs"], signed=dis["signed"],
                df_dict=df_dict, earnings=earnings, filings=filings, lm_scores=lm_scores)


def run_all(in_: dict, ticker: str) -> dict:
    df, sent = in_["df"], in_["sent"]
    res = {}
    res["H9"] = L2.test_sentiment_asymmetry(df, sent, ticker)
    res["H10"] = L2.test_sentiment_velocity(df, sent, ticker)
    res["H11"] = L2.test_sentiment_model_consensus(df, sent, ticker)
    res["H12"] = L2.test_disagreement_persistence(df, in_["disagree"], ticker)
    res["H13"] = L2.test_disagreement_direction(df, in_["signed"], ticker)
    res["H14"] = L2.test_asymmetric_contagion(in_["df_dict"])
    res["H15"] = L2.test_cross_sector_contagion(in_["df_dict"])
    res["H16"] = L2.test_regime_dependent_leverage(df, ticker)
    res["H17"] = L2.test_monday_leverage_interaction(df, ticker)
    res["H18"] = L2.test_pre_earnings_sentiment_predicts_spike(df, sent, in_["earnings"], ticker)
    res["H19"] = L2.test_earnings_vol_premium_vs_iv(df, in_["earnings"], ticker)
    res["H20"] = L2.test_10k_language_change_predicts_regime(df, in_["lm_scores"], ticker)
    res["H21"] = L2.test_topic_specific_risk_prediction(in_["df_dict"])
    res["H22"] = L2.test_triple_signal_interaction(df, sent, in_["disagree"], ticker)
    res["H23"] = L2.test_signal_temporal_precedence(df, sent, in_["disagree"], ticker,
                                                    df_dict=in_["df_dict"], filing_dates=in_["filings"])
    return res


def save_figures(res: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for hyp, r in res.items():
        fig = r.get("figure")
        if fig is None:
            continue
        if isinstance(fig, go.Figure):
            fig.write_image(str(out_dir / f"{hyp}.png"))
        else:
            fig.savefig(out_dir / f"{hyp}.png", dpi=120, bbox_inches="tight")
            plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="MU")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2024-12-31")
    args = ap.parse_args()

    inputs = assemble_inputs(args.ticker, args.start, args.end)
    res = run_all(inputs, args.ticker)

    out_csv = ROOT / "outputs" / "results" / "hypothesis_L2_summary.csv"
    summary = L2.compile_l2_summary(res, out_csv=str(out_csv))
    print(summary.to_string())
    save_figures(res, ROOT / "outputs" / "figures" / "L2")
    print(f"\nFigures: {ROOT / 'outputs' / 'figures' / 'L2'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script end-to-end**

Run: `python scripts/run_advanced_hypotheses.py --ticker MU --start 2020-01-01 --end 2023-12-31`
Expected: prints a 15-row summary table; writes `outputs/results/hypothesis_L2_summary.csv` and `outputs/figures/L2/H*.png`. (Note: `fig.write_image` needs `kaleido`; if it is not installed, the run still completes — wrap the plotly branch in try/except and warn. If you hit a missing-kaleido error, add `try/except Exception` around `fig.write_image` logging a warning and continue.)

- [ ] **Step 3: Commit**

```bash
git add scripts/run_advanced_hypotheses.py
git commit -m "feat: headless runner for L2 advanced hypothesis suite"
```

---

## Task 14: Restore the research notebook

**Files:**
- Restore: `notebooks/07_advanced_hypotheses.ipynb`

The retired notebook already imports `modules.hypothesis_tests_L2` and assembles all inputs, so restoring it verbatim is correct — no repointing needed.

- [ ] **Step 1: Restore from git history**

```bash
git show e6be0ae~1:notebooks/07_advanced_hypotheses.ipynb > notebooks/07_advanced_hypotheses.ipynb
```

- [ ] **Step 2: Verify it imports the tracked module and references no deleted modules**

Run: `python -c "import json; nb=json.load(open('notebooks/07_advanced_hypotheses.ipynb')); src=''.join(s for c in nb['cells'] if c['cell_type']=='code' for s in c['source']); assert 'modules.hypothesis_tests_L2' in src; assert 'hypothesis_tests_L1' not in src and 'src.hypothesis_tests' not in src; print('imports OK')"`
Expected: `imports OK`

- [ ] **Step 3: Execute the notebook headless to confirm it runs**

Run: `python -m jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=1200 --output 07_advanced_hypotheses.ipynb notebooks/07_advanced_hypotheses.ipynb`
Expected: completes without error (downloads data on first run; uses cache after). If `jupyter`/`nbconvert` is not installed, install with `python -m pip install nbconvert ipykernel`, then re-run. If execution is too slow/network-bound in the environment, instead validate by running `scripts/run_advanced_hypotheses.py` (Task 13) and note the notebook is restored but executed manually.

- [ ] **Step 4: Commit**

```bash
git add notebooks/07_advanced_hypotheses.ipynb
git commit -m "docs: restore 07_advanced_hypotheses notebook for the L2 suite"
```

---

## Final verification

- [ ] Run the full test suite: `python -m pytest tests/ -q` → all green (existing 11 + new ~35).
- [ ] Confirm `git status` is clean (no stray untracked files).
- [ ] The branch `revive-l2-hypotheses` is ready to merge/PR via the finishing-a-development-branch skill.

---

## Self-review notes (coverage check)

- **Spec coverage:** all 15 tests (Tasks 5–11), both custom stats pinned (Tasks 3–4), `compile_l2_summary` (Task 12), runner script (Task 13), notebook (Task 14), module tracked + docstring fixed (Task 1). `src/hypotheses.py` untouched (no task modifies it). No new dependencies introduced.
- **H16/H23 no-guard behaviour** handled explicitly in Tasks 8 and 11.
- **PROXY paths** (H19 `proxy_iv`, H21 `proxy`) asserted in Tasks 9–10.
- **Granger typo fix** exercised by Task 7 (H15).
- **Risk — characterization tests may surface real bugs:** each test task says fix the module (bug), never weaken the test. This is the one place "no rewrites" yields to a genuine bug fix.
