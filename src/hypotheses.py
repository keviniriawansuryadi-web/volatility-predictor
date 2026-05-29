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
