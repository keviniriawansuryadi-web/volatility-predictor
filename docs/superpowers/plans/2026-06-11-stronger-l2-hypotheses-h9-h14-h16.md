# Stronger L2 Hypotheses (H9/H14/H16) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the three weak Level-2 hypotheses (H9 sentiment asymmetry, H14 asymmetric contagion, H16 regime-dependent leverage) with higher-power tests, re-run the suite, and update Section 2B of `06_conclusions.ipynb`.

**Architecture:** Edit `modules/hypothesis_tests_L2.py` (swap 3 `test_*` functions 1:1, keep the suite at 15), update the runner call sites and the `_FINDING_LABEL` registry, point the L2 contract/degraded tests at the new names plus add a direction-of-effect test each, regenerate `outputs/results/hypothesis_L2_summary.csv`, then rewrite the notebook's Section 2B markdown from the new CSV.

**Tech Stack:** Python 3.14, pandas, numpy, scipy.stats, statsmodels (Logit, OLS+HAC, tsa.VAR), plotly/matplotlib, pytest.

---

### Task 1: Replace H9 — `test_negative_sentiment_spike_risk`

**Files:**
- Modify: `modules/hypothesis_tests_L2.py` (replace `test_sentiment_asymmetry`, ~lines 214-288; section header ~line 210-212)
- Test: `tests/test_hypotheses_L2.py` (replace `test_h9_contract`/`test_h9_degraded` ~lines 175-183)

- [ ] **Step 1: Replace the H9 test functions in `tests/test_hypotheses_L2.py`**

```python
def test_h9_contract(price_df, sent):
    r = L2.test_negative_sentiment_spike_risk(price_df, sent, "TEST")
    _assert_l2_contract(r)
    assert r["hypothesis"] == "H9"


def test_h9_degraded():
    df, sent = _tiny()
    assert L2.test_negative_sentiment_spike_risk(df, sent, "TEST")["available"] is False


def test_h9_detects_negative_sentiment_spike_link():
    # Build a frame where the most-negative sentiment days precede vol spikes.
    rng = np.random.default_rng(0)
    n = 300
    idx = pd.bdate_range("2021-01-04", periods=n)
    spike_day = rng.random(n) < 0.25
    fwd = np.where(spike_day, 0.6, 0.2) + rng.normal(0, 0.02, n)
    df = pd.DataFrame({"log_return": rng.normal(0, 0.01, n),
                       "realized_vol_21d": fwd,
                       "realized_vol_5d": fwd}, index=idx)
    vader = np.where(spike_day, -0.7, 0.3) + rng.normal(0, 0.05, n)
    sent = pd.DataFrame({"vader_compound": np.clip(vader, -1, 1)}, index=idx)
    r = L2.test_negative_sentiment_spike_risk(df, sent, "TEST")
    assert r["available"] is True
    assert r["statistic"] > 0          # log-odds coefficient positive
    assert r["p_value"] < 0.05
```

- [ ] **Step 2: Run to verify the new tests fail (function missing)**

Run: `python -m pytest tests/test_hypotheses_L2.py::test_h9_contract tests/test_hypotheses_L2.py::test_h9_detects_negative_sentiment_spike_link -q`
Expected: FAIL with `AttributeError: module ... has no attribute 'test_negative_sentiment_spike_risk'`

- [ ] **Step 3: Replace `test_sentiment_asymmetry` and its section header in `modules/hypothesis_tests_L2.py`**

Replace the section header comment block (currently `#  EXTENDING H1 -- SENTIMENT ASYMMETRY (H9, H10, H11)`) with `#  EXTENDING H1 -- SENTIMENT AS A RISK SIGNAL (H9, H10, H11)`, then replace the whole `def test_sentiment_asymmetry(...)` function with:

```python
def test_negative_sentiment_spike_risk(df: pd.DataFrame, sent: pd.DataFrame, ticker: str,
                                       horizon: int = 5, spike_pct: float = 0.75) -> dict:
    """
    H9 (extends H1): Are the most-negative sentiment days a leading indicator of vol spikes?

    H1 confirmed that negative sentiment precedes vol.  The original H9 tested a
    *symmetric* three-group mean difference and lost power.  This version tests the
    directional, tail question H1 actually implies: do the most-negative sentiment
    days precede vol *spikes* more often than other days?

    A spike is a day whose next-``horizon``-day realized vol lands in the top
    ``spike_pct`` quantile.  The predictor is the negative-sentiment magnitude
    ``neg = max(-VADER, 0)`` -- the ``min(sentiment, 0)`` feature the original H9
    proposed, expressed as a magnitude.  A logistic regression ``spike ~ neg``
    gives the odds ratio for a one-SD rise in negativity, a one-sided p-value
    (H1: more negativity -> more spikes), and McFadden pseudo-R-squared.

    Why it matters: if negative sentiment lifts the *probability* of a spike, it is
    a one-sided RISK signal -- useful for detecting danger -- which justifies
    engineering ``min(sentiment, 0)`` as a feature rather than the raw score.

    Returns the logistic odds ratio / p-value and a bar chart of spike rate by
    sentiment tercile with Wilson confidence intervals.
    """
    _print_header("H9", "Negative sentiment -> vol-spike risk (logistic)", "H1")
    import statsmodels.api as sm
    from statsmodels.stats.proportion import proportion_confint

    fwd = _ensure_fwd_vol(df, horizon).rename("fwd_vol")
    d = pd.concat([sent["vader_compound"].rename("vader"), fwd], axis=1).dropna()
    if len(d) < 30:
        return dict(hypothesis="H9", extends="H1", available=False, p_value=np.nan,
                    conclusion=f"{ticker}: only {len(d)} aligned obs.",
                    actionable="Feature: min(sentiment, 0)")

    thr = d["fwd_vol"].quantile(spike_pct)
    d["spike"] = (d["fwd_vol"] >= thr).astype(int)
    d["neg"] = (-d["vader"]).clip(lower=0.0)
    if d["spike"].nunique() < 2 or d["neg"].std() == 0:
        return dict(hypothesis="H9", extends="H1", available=False, p_value=np.nan,
                    conclusion=f"{ticker}: no spike variation or constant sentiment.",
                    actionable="Feature: min(sentiment, 0)")

    neg_z = ((d["neg"] - d["neg"].mean()) / (d["neg"].std() + 1e-12)).rename("neg")
    X = sm.add_constant(neg_z)
    try:
        logit = sm.Logit(d["spike"], X).fit(disp=0)
    except Exception as exc:  # pragma: no cover - separation / singular fits
        return dict(hypothesis="H9", extends="H1", available=False, p_value=np.nan,
                    conclusion=f"{ticker}: logistic fit failed ({exc}).",
                    actionable="Feature: min(sentiment, 0)")

    coef = float(logit.params["neg"])
    p_two = float(logit.pvalues["neg"])
    p_one = p_two / 2 if coef > 0 else 1 - p_two / 2
    odds_ratio = float(np.exp(coef))
    pseudo_r2 = float(logit.prsquared)

    # Spike rate by sentiment tercile (Most-negative / Neutral / Most-positive).
    terc = pd.qcut(d["vader"], 3, labels=["Most-negative", "Neutral", "Most-positive"],
                   duplicates="drop")
    rates, los, his, ns = [], [], [], []
    cats = ["Most-negative", "Neutral", "Most-positive"]
    for c in cats:
        sub = d.loc[terc == c, "spike"]
        k, ntot = int(sub.sum()), int(len(sub))
        rates.append(k / ntot if ntot else np.nan)
        lo, hi = proportion_confint(k, ntot, method="wilson") if ntot else (np.nan, np.nan)
        los.append(lo); his.append(hi); ns.append(ntot)

    conclusion = (
        f"{ticker}: P(spike) ~ negativity. OR per +1SD={odds_ratio:.2f} "
        f"(logit coef={coef:+.3f}, one-sided p={p_one:.4f}, {_verdict(p_one)}), "
        f"pseudo-R2={pseudo_r2:.3f}. Spike rate Most-neg={rates[0]:.0%} vs "
        f"Most-pos={rates[2]:.0%}. "
        f"{'Negative sentiment is a real spike-risk signal.' if (coef > 0 and p_one < 0.05) else 'No significant spike-risk signal.'}"
    )
    print(f"  {conclusion}")

    fig = go.Figure(go.Bar(
        x=cats, y=rates,
        error_y=dict(type="data", symmetric=False,
                     array=[h - r for h, r in zip(his, rates)],
                     arrayminus=[r - l for l, r in zip(los, rates)]),
        marker_color=["#e74c3c", "#95a5a6", "#27ae60"],
        text=[f"n={n}" for n in ns], textposition="outside"))
    fig.add_hline(y=1 - spike_pct, line_dash="dot", line_color="gray",
                  annotation_text="baseline spike rate")
    fig.update_layout(title=f"H9 - {ticker}: Vol-Spike Rate by Sentiment Tercile "
                            f"(OR={odds_ratio:.2f}, p={p_one:.4f})",
                      xaxis_title="Sentiment tercile",
                      yaxis_title=f"P(next-{horizon}d vol in top {int((1-spike_pct)*100)}%)")
    return dict(hypothesis="H9", extends="H1", available=True, statistic=coef,
                p_value=p_one, effect=odds_ratio - 1.0, odds_ratio=odds_ratio,
                pseudo_r2=pseudo_r2, spike_rates=dict(zip(cats, rates)),
                conclusion=conclusion, actionable="Feature: min(sentiment, 0)", figure=fig)
```

- [ ] **Step 4: Run the H9 tests to verify they pass**

Run: `python -m pytest tests/test_hypotheses_L2.py -k h9 -q`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add modules/hypothesis_tests_L2.py tests/test_hypotheses_L2.py
git commit -m "feat(L2): replace H9 with negative-sentiment spike-risk logistic test"
```

---

### Task 2: Replace H14 — `test_directional_spillover_hub` (Diebold-Yilmaz)

**Files:**
- Modify: `modules/hypothesis_tests_L2.py` (replace `test_asymmetric_contagion`, ~lines 631-725; section header ~line 627-629; add a `_net_spillover` helper)
- Test: `tests/test_hypotheses_L2.py` (replace `test_h14_contract`/`test_h14_degraded` ~lines 231-240)

- [ ] **Step 1: Replace the H14 test functions in `tests/test_hypotheses_L2.py`**

```python
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
        return (nv.shift(shift).bfill() + rng.normal(0, noise, n))
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
```

- [ ] **Step 2: Run to verify the new tests fail**

Run: `python -m pytest tests/test_hypotheses_L2.py -k h14 -q`
Expected: FAIL with `AttributeError: ... 'test_directional_spillover_hub'`.

- [ ] **Step 3: Replace `test_asymmetric_contagion` + section header and add `_net_spillover` helper**

Change the section header comment to `#  EXTENDING H4 -- DIRECTIONAL SPILLOVER (H14, H15)`, then replace the whole `def test_asymmetric_contagion(...)` function with the helper + new test:

```python
def _net_spillover(vol: pd.DataFrame, lag: int, horizon: int) -> pd.Series:
    """Diebold-Yilmaz net directional spillover (generalized FEVD, Pesaran-Shin).

    ``vol`` is a (T x k) frame of vol series. Returns a Series of net spillover
    (TO others - FROM others, in connectedness %) indexed by column name.
    """
    from statsmodels.tsa.api import VAR
    res = VAR(vol).fit(lag)
    Sigma = np.asarray(res.sigma_u)
    k = Sigma.shape[0]
    Phi = res.ma_rep(maxn=horizon - 1)          # (horizon, k, k), Phi[0] = I
    sig = np.diag(Sigma)
    theta = np.zeros((k, k))
    for i in range(k):
        denom = sum((Phi[h] @ Sigma @ Phi[h].T)[i, i] for h in range(horizon))
        for j in range(k):
            num = sum((Phi[h] @ Sigma)[i, j] ** 2 for h in range(horizon))
            theta[i, j] = (num / sig[j]) / denom if denom > 0 else 0.0
    row_sum = theta.sum(axis=1, keepdims=True)
    theta_n = np.divide(theta, row_sum, out=np.zeros_like(theta), where=row_sum > 0) * 100.0
    to_others = theta_n.sum(axis=0) - np.diag(theta_n)      # column sums (i -> others)
    from_others = theta_n.sum(axis=1) - np.diag(theta_n)    # row sums (others -> i)
    return pd.Series(to_others - from_others, index=vol.columns)


def test_directional_spillover_hub(df_dict: dict, source: str = "NVDA",
                                   members: tuple = ("MU", "NVDA", "AMD"),
                                   horizon: int = 10, max_lag: int = 5,
                                   n_boot: int = 300, seed: int = 42) -> dict:
    """
    H14 (extends H4): Is the lead asset the directional volatility *hub*?

    H4 confirmed NVDA vol spills over to MU and AMD.  The original H14 tested a
    single one-day-lead sign asymmetry and was underpowered.  This version uses
    the Diebold & Yilmaz (2012) connectedness framework: a VAR on the members'
    realized vol, a generalized (order-invariant) forecast-error variance
    decomposition at ``horizon`` steps, and the *net directional spillover* per
    ticker (share transmitted TO others minus share received FROM others).

    The hub is the ticker with the highest net spillover.  A moving-block
    bootstrap (resampling rows in blocks, recomputing the net spillover) gives a
    one-sided p-value that the ``source``'s net spillover exceeds zero -- i.e. it
    is a *net transmitter*, not a receiver.

    Why it matters: if one ticker is the systemic source, its lagged vol is a
    cross-ticker lead feature for the others' forecasts.

    Returns the net-spillover vector, the total connectedness index, the bootstrap
    p-value for the source, and a net-spillover bar chart with the source flagged.
    """
    _print_header("H14", "Directional vol spillover hub (Diebold-Yilmaz)", "H4")
    if source not in df_dict:
        return dict(hypothesis="H14", extends="H4", available=False, p_value=np.nan,
                    conclusion=f"Source {source} missing from df_dict.",
                    actionable="Cross-ticker spillover feature (lead from hub)")

    cols = [df_dict[t]["realized_vol_21d"].rename(t) for t in members
            if t in df_dict and "realized_vol_21d" in df_dict[t].columns]
    vol = pd.concat(cols, axis=1).dropna() if cols else pd.DataFrame()
    if vol.shape[1] < 2 or source not in vol.columns or len(vol) < 60:
        return dict(hypothesis="H14", extends="H4", available=False, p_value=np.nan,
                    conclusion=f"Insufficient overlapping vol history "
                               f"({vol.shape if len(vol) else 'empty'}).",
                    actionable="Cross-ticker spillover feature (lead from hub)")

    # Lag by AIC (capped), fallback to 1 if selection/fit fails.
    from statsmodels.tsa.api import VAR
    try:
        lag = int(VAR(vol).select_order(maxlags=min(max_lag, len(vol) // 10)).aic) or 1
    except Exception:
        lag = 1
    lag = max(1, min(lag, max_lag))
    try:
        net = _net_spillover(vol, lag, horizon)
    except Exception as exc:
        return dict(hypothesis="H14", extends="H4", available=False, p_value=np.nan,
                    conclusion=f"VAR/GFEVD fit failed ({exc}).",
                    actionable="Cross-ticker spillover feature (lead from hub)")

    # Total connectedness index = mean off-diagonal share (recompute compactly).
    total_ci = float(np.clip((net.abs().sum()) / (2 * len(net)) + 50.0, 0, 100))

    # Moving-block bootstrap on net[source].
    rng = np.random.default_rng(seed)
    blk = max(5, len(vol) // 20)
    n = len(vol)
    boot = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - blk, size=int(np.ceil(n / blk)))
        rows = np.concatenate([np.arange(s, s + blk) for s in starts])[:n]
        try:
            b_net = _net_spillover(vol.iloc[rows].reset_index(drop=True), lag, horizon)
            boot.append(b_net.get(source, np.nan))
        except Exception:
            continue
    boot = np.array([b for b in boot if not np.isnan(b)])
    p_src = float((boot <= 0).mean()) if boot.size else np.nan
    src_net = float(net[source])
    hub = net.idxmax()

    conclusion = (
        f"Net spillover (% pts): "
        + ", ".join(f"{k}={v:+.1f}" for k, v in net.sort_values(ascending=False).items())
        + f". HUB = {hub}; total connectedness={total_ci:.0f}%. "
        f"{source} net={src_net:+.1f}, bootstrap p={p_src:.4f} ({_verdict(p_src)}). "
        f"{f'{source} is the net volatility source.' if (src_net > 0 and isinstance(p_src, float) and p_src < 0.05) else 'No significant net-source role.'}"
    )
    print(f"  {conclusion}")

    order = net.sort_values(ascending=False)
    fig = go.Figure(go.Bar(
        x=order.index.tolist(), y=order.values,
        marker_color=["#c0392b" if t == source else "#7f8c8d" for t in order.index]))
    fig.add_hline(y=0.0, line_dash="dash", line_color="gray")
    fig.update_layout(title=f"H14: Net Directional Vol Spillover (Diebold-Yilmaz, "
                            f"{source} bootstrap p={p_src:.3f})",
                      xaxis_title="Ticker", yaxis_title="Net spillover (TO - FROM, % pts)")
    return dict(hypothesis="H14", extends="H4", available=True, statistic=total_ci,
                p_value=p_src, effect=src_net, net_spillover=net, hub=hub,
                total_connectedness=total_ci, conclusion=conclusion,
                actionable="Cross-ticker spillover feature (lead from hub)", figure=fig)
```

- [ ] **Step 4: Run the H14 tests**

Run: `python -m pytest tests/test_hypotheses_L2.py -k h14 -q`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add modules/hypothesis_tests_L2.py tests/test_hypotheses_L2.py
git commit -m "feat(L2): replace H14 with Diebold-Yilmaz directional spillover hub test"
```

---

### Task 3: Replace H16 — `test_leverage_amplification`

**Files:**
- Modify: `modules/hypothesis_tests_L2.py` (replace `test_regime_dependent_leverage`, ~lines 832-928; section header ~line 828-830)
- Test: `tests/test_hypotheses_L2.py` (replace `test_h16_contract`/`test_h16_degraded_does_not_raise` ~lines 254-265)

- [ ] **Step 1: Replace the H16 test functions in `tests/test_hypotheses_L2.py`**

```python
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
```

- [ ] **Step 2: Run to verify the new tests fail**

Run: `python -m pytest tests/test_hypotheses_L2.py -k h16 -q`
Expected: FAIL with `AttributeError: ... 'test_leverage_amplification'`.

- [ ] **Step 3: Replace `test_regime_dependent_leverage` + section header**

Change the section header to `#  EXTENDING H5 -- LEVERAGE EFFECT DEPTH (H16, H17)` (unchanged text is fine), then replace the whole `def test_regime_dependent_leverage(...)` function with:

```python
def test_leverage_amplification(df: pd.DataFrame, ticker: str, horizon: int = 5) -> dict:
    """
    H16 (extends H5): Does the leverage effect *amplify continuously* with the vol level?

    H5 confirmed the leverage effect (negative returns -> higher future vol).  The
    original H16 bucketed days into discrete regimes and ran out of Extreme-regime
    observations.  This version keeps the same economic question -- does leverage
    strengthen under stress? -- but with full-sample power via one interaction
    regression (HAC standard errors, because overlapping forward windows are
    autocorrelated):

        fwd_vol = b0 + b1*neg_mag + b2*pos_mag + b3*(neg_mag x vol_level)

    where ``neg_mag = max(-ret, 0)``, ``pos_mag = max(ret, 0)`` and ``vol_level``
    is the standardised current 21-day realized vol.  A positive, significant
    ``b3`` means a negative return adds *more* forward vol when current vol is
    already high -- the volatility-feedback effect (Campbell & Hentschel 1992).

    Why it matters: if leverage scales with the vol level, the danger of a negative
    return is state-dependent, which argues for a regime-conditional leverage
    parameter rather than one global coefficient.

    Returns the interaction coefficient and HAC p-value, the implied leverage slope
    at low vs high vol level, the descriptive per-regime neg/pos vol ratios, and a
    bar chart of those ratios annotated with Campbell & Hentschel (1992).
    """
    _print_header("H16", "Leverage amplification with vol level (HAC interaction)", "H5")
    import statsmodels.api as sm

    fwd = _ensure_fwd_vol(df, horizon).rename("fwd")
    d = pd.DataFrame({"ret": df["log_return"], "fwd": fwd,
                      "vol": df["realized_vol_21d"], "regime": _regime_series(df)}).dropna()

    # Descriptive per-regime ratios (kept for the contract + bar chart).
    def _ratio(sub):
        neg = sub.loc[sub["ret"] < 0, "fwd"]
        pos = sub.loc[sub["ret"] > 0, "fwd"]
        if len(neg) < 5 or len(pos) < 5:
            return np.nan
        return float(neg.mean() / (pos.mean() + 1e-12))
    leverage_ratios = {r_name: _ratio(d[d["regime"] == r_name]) for r_name in REGIME_ORDER}

    if len(d) < 50 or d["vol"].std() == 0:
        return dict(hypothesis="H16", extends="H5", available=False, p_value=np.nan,
                    leverage_ratios=leverage_ratios,
                    conclusion=f"{ticker}: only {len(d)} aligned obs (need >=50).",
                    actionable="Regime-conditional leverage param")

    d["neg_mag"] = (-d["ret"]).clip(lower=0)
    d["pos_mag"] = d["ret"].clip(lower=0)
    d["volz"] = (d["vol"] - d["vol"].mean()) / (d["vol"].std() + 1e-12)
    d["neg_x_vol"] = d["neg_mag"] * d["volz"]
    X = sm.add_constant(d[["neg_mag", "pos_mag", "volz", "neg_x_vol"]])
    ols = sm.OLS(d["fwd"], X).fit(cov_type="HAC", cov_kwds={"maxlags": horizon})

    b_int = float(ols.params["neg_x_vol"])
    p_two = float(ols.pvalues["neg_x_vol"])
    p_one = p_two / 2 if b_int > 0 else 1 - p_two / 2
    b_neg = float(ols.params["neg_mag"])
    lo_z, hi_z = float(np.percentile(d["volz"], 20)), float(np.percentile(d["volz"], 80))
    slope_lo, slope_hi = b_neg + b_int * lo_z, b_neg + b_int * hi_z

    ratio_str = ", ".join(f"{k}={v:.2f}" for k, v in leverage_ratios.items()
                          if not np.isnan(v)) or "n/a"
    conclusion = (
        f"{ticker}: leverage x vol-level interaction b={b_int:+.3f} "
        f"(one-sided p={p_one:.4f}, {_verdict(p_one)}). Implied neg-return slope "
        f"low-vol={slope_lo:.2f} -> high-vol={slope_hi:.2f}. "
        f"Per-regime neg/pos ratios [{ratio_str}]. "
        f"{'Leverage amplifies with the vol level (Campbell & Hentschel 1992).' if (b_int > 0 and p_one < 0.05) else 'No significant amplification.'}"
    )
    print(f"  {conclusion}")

    present = [r for r in REGIME_ORDER if not np.isnan(leverage_ratios.get(r, np.nan))]
    fig = go.Figure(go.Bar(
        x=present, y=[leverage_ratios[r] for r in present],
        marker_color=["#3498db", "#f39c12", "#e67e22", "#c0392b"][:len(present)]))
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray",
                  annotation_text="no leverage effect")
    fig.add_annotation(text="Volatility-feedback effect (Campbell & Hentschel 1992)",
                       xref="paper", yref="paper", x=0.5, y=-0.18, showarrow=False,
                       font=dict(size=10, color="gray"))
    fig.update_layout(title=f"H16 - {ticker}: Leverage Ratio by Regime "
                            f"(interaction b={b_int:+.3f}, p={p_one:.4f})",
                      xaxis_title="Vol regime", yaxis_title="neg/pos forward-vol ratio")
    return dict(hypothesis="H16", extends="H5", available=True, statistic=b_int,
                p_value=p_one, effect=b_int, leverage_ratios=leverage_ratios,
                slope_low_vol=slope_lo, slope_high_vol=slope_hi,
                conclusion=conclusion, actionable="Regime-conditional leverage param",
                figure=fig)
```

- [ ] **Step 4: Run the H16 tests**

Run: `python -m pytest tests/test_hypotheses_L2.py -k h16 -q`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add modules/hypothesis_tests_L2.py tests/test_hypotheses_L2.py
git commit -m "feat(L2): replace H16 with full-sample leverage-amplification interaction test"
```

---

### Task 4: Rewire registry, runner, and module docstring

**Files:**
- Modify: `modules/hypothesis_tests_L2.py` (`_FINDING_LABEL` ~lines 1571-1580; module docstring L1 map ~lines 11-18)
- Modify: `scripts/run_advanced_hypotheses.py` (`run_all` ~lines 80, 85, 87)
- Modify: `tests/test_hypotheses_L2.py` (`compile_summary` test references ~lines 355, 357)

- [ ] **Step 1: Update `_FINDING_LABEL` for H9/H14/H16**

```python
    "H9": "Negative sentiment spike risk", "H10": "Velocity vs level",
    ...
    "H13": "EGARCH = early warning", "H14": "Directional spillover hub",
    "H15": "Semi -> Tech contagion", "H16": "Leverage amplifies with vol level",
```

- [ ] **Step 2: Update the three `run_all` call sites in `scripts/run_advanced_hypotheses.py`**

```python
    res["H9"] = L2.test_negative_sentiment_spike_risk(df, sent, ticker)
    ...
    res["H14"] = L2.test_directional_spillover_hub(in_["df_dict"])
    ...
    res["H16"] = L2.test_leverage_amplification(df, ticker)
```

- [ ] **Step 3: Update the `compile_summary` test references in `tests/test_hypotheses_L2.py`**

Find the dict around line 355 and replace the H9/H16 entries:

```python
        "H9": L2.test_negative_sentiment_spike_risk(price_df, sent, "TEST"),
        ...
        "H16": L2.test_leverage_amplification(price_df, "TEST"),
```

(Remove the now-invalid `n_perm=` kwargs.)

- [ ] **Step 4: Update the module-docstring L1 map and any stale references**

In the top docstring, no L1 line changes are needed (H1-H8 unchanged), but verify no remaining references to `test_sentiment_asymmetry`, `test_asymmetric_contagion`, or `test_regime_dependent_leverage` remain anywhere.

Run: `grep -rn "sentiment_asymmetry\|asymmetric_contagion\|regime_dependent_leverage" modules scripts tests`
Expected: no matches.

- [ ] **Step 5: Run the full L2 suite**

Run: `python -m pytest tests/test_hypotheses_L2.py -q`
Expected: all pass (count = previous total, still 15 `test_*` functions in the module so `test_module_exposes_15_tests` passes).

- [ ] **Step 6: Commit**

```bash
git add modules/hypothesis_tests_L2.py scripts/run_advanced_hypotheses.py tests/test_hypotheses_L2.py
git commit -m "refactor(L2): rewire registry+runner+tests for new H9/H14/H16"
```

---

### Task 5: Regenerate the L2 summary and figures

**Files:**
- Output: `outputs/results/hypothesis_L2_summary.csv`, `outputs/figures/L2/*.png`

- [ ] **Step 1: Run the headless runner**

Run: `python scripts/run_advanced_hypotheses.py --ticker MU --start 2018-01-01 --end 2024-12-31`
Expected: prints the H9-H23 summary; H9/H14/H16 rows now show the new Findings; CSV rewritten. Capture the H9/H14/H16 p-values and conclusions from stdout for the notebook.

- [ ] **Step 2: Inspect the new rows**

Run: `python -c "import pandas as pd; print(pd.read_csv('outputs/results/hypothesis_L2_summary.csv').to_string())"`
Expected: H9 = "Negative sentiment spike risk", H14 = "Directional spillover hub", H16 = "Leverage amplifies with vol level" with numeric p-values (H16 no longer "--").

- [ ] **Step 3: Commit**

```bash
git add outputs/results/hypothesis_L2_summary.csv outputs/figures/L2
git commit -m "chore(L2): regenerate summary + figures for new H9/H14/H16"
```

---

### Task 6: Update Section 2B of `06_conclusions.ipynb`

**Files:**
- Modify: `notebooks/06_conclusions.ipynb` (markdown cell 17 — featured-findings table + headline; cell 19 — "what changes" bullets + nulls/inconclusive lists)

- [ ] **Step 1: Edit cell 17** — update the "Five robust findings" table and headline to reflect which of H9/H14/H16 now reach significance (per the Task 5 run). Move any newly-significant slot into the featured table; keep the count/wording consistent with the regenerated CSV. Use `NotebookEdit` (or a JSON-safe Python edit) so the `.ipynb` stays valid UTF-8.

- [ ] **Step 2: Edit cell 19** — rewrite the "Confirmed nulls (do not productionise)" and "Inconclusive" lists: H9 is now the spike-risk logistic, H14 the Diebold-Yilmaz hub, H16 the continuous leverage-amplification interaction. State each new finding honestly (significant → new production signal; null → documented null with the stronger test noted). Remove the old "regime-dependent leverage inconclusive (REGIME_LIMITED)" caveat for H16.

- [ ] **Step 3: Validate the notebook parses**

Run: `python -c "import json; json.load(open('notebooks/06_conclusions.ipynb', encoding='utf-8')); print('valid json')"`
Expected: `valid json`.

- [ ] **Step 4: Commit**

```bash
git add notebooks/06_conclusions.ipynb
git commit -m "docs: refresh conclusions Section 2B for stronger H9/H14/H16"
```

---

---

### Task 7: Short PowerPoint deck for the three new hypotheses

**Files:**
- Create: `outputs/decks/L2_h9_h14_h16.pptx`

- [ ] **Step 1: Build the deck** — use the `pptx` skill. Cover only H9/H14/H16, one section each (plus a title slide and a one-slide summary). Per hypothesis: what it extends (H1/H4/H5), the question, the method (logistic spike-risk / Diebold-Yilmaz spillover / HAC leverage-amplification interaction), the result (p-value + effect from the regenerated CSV), and the figure from `outputs/figures/L2/{H9,H14,H16}.png` if it was exported. Pull the numbers from `outputs/results/hypothesis_L2_summary.csv` and the runner stdout captured in Task 5 — do not invent values.

- [ ] **Step 2: Validate the deck opens** — confirm the file exists and python-pptx can reopen it.

Run: `python -c "from pptx import Presentation; p=Presentation('outputs/decks/L2_h9_h14_h16.pptx'); print(len(p.slides), 'slides')"`
Expected: prints a slide count > 0.

- [ ] **Step 3: Commit**

```bash
git add outputs/decks/L2_h9_h14_h16.pptx
git commit -m "docs: add slide deck for new H9/H14/H16 hypotheses"
```

---

## Self-Review notes
- **Spec coverage:** Task 1=H9, Task 2=H14, Task 3=H16, Task 4=wiring/registry/runner, Task 5=re-run/CSV, Task 6=notebook, Task 7=PPT deck. All spec sections covered (PPT added per user request).
- **Contract:** H16 keeps `leverage_ratios` (returned even on the degraded path); H14 keeps NVDA-required degradation and adds `net_spillover`; all three keep the 6 required keys + `figure` when available.
- **Suite size:** 3-for-3 swap keeps the module at 15 `test_*` functions, so `test_module_exposes_15_tests` stays green.
- **Type consistency:** function names used identically in module, runner, and tests (`test_negative_sentiment_spike_risk`, `test_directional_spillover_hub`, `test_leverage_amplification`); return keys referenced in tests (`statistic`, `net_spillover`, `leverage_ratios`) are all produced by the implementations.
