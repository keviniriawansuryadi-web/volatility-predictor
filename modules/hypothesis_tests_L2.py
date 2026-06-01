"""
Level-2 (advanced) hypothesis tests for the volatility research project.

Every test in this module **extends a confirmed Level-1 finding** (H1-H8 from
the project's earlier hypothesis work) and pushes it toward the kind of question that a
published volatility paper would ask: asymmetry, persistence, regime
conditioning, cross-sector contagion, and multi-signal interaction.

Level-1 findings these tests build on
-------------------------------------
H1  sentiment -> vol spikes              (MU/XOM/JPM/MSFT)
H2  EGARCH-ML disagreement -> vol        (MU)
H3  Granger causality sentiment <-> vol  (bidirectional)
H4  NVDA vol spills over to MU/AMD
H5  leverage effect (all tickers)
H6  Monday effect (semiconductors)
H7  earnings-week vol premium (+38%)
H8  LM 10-K risk score -> forward vol

Data conventions (shared with ``src``)
--------------------------------------
``df``    : price frame with ``log_return`` and ``realized_vol_21d``; a forward
            target ``realized_vol_5d`` is added on demand via ``_ensure_fwd_vol``.
``sent``  : output of ``src.data_helpers.simulate_sentiment`` -- columns
            ``vader_compound``, ``finbert``, ``textblob``, ``lm_score``.
``df_dict``: ``{ticker: df}`` used by the contagion tests.
EGARCH/ML : aligned test-set forecast Series from ``src.garch_model`` /
            ``src.ml_model``; disagreement is derived from them here.

Each public ``test_*`` function returns a dict with at least::

    hypothesis, extends, p_value, effect, conclusion, actionable, available

and (where a chart is produced) a ``figure`` key.  Functions degrade
gracefully -- when the data needed for a test is unavailable they return
``available=False`` with an honest explanation rather than raising, because
a reported null is itself a finding.

References
----------
Black (1976); Campbell & Hentschel (1992); Forbes & Rigobon (2002);
Loughran & McDonald (2011); Steiger (1980); Scheirer, Ray & Hare (1976).
"""

from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots

try:
    import scikit_posthocs as sp
except Exception:  # pragma: no cover - optional at import time
    sp = None

from src.regime import _label as _regime_label, REGIME_ORDER

pio.renderers.default = "notebook"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Stressed = top two regimes; calm = bottom two (used by H16, H22).
_STRESSED = {"High", "Extreme"}

# Default sector map for the contagion tests (H14, H15, H23).
DEFAULT_SECTORS = {
    "Semiconductor": ["MU", "NVDA", "AMD"],
    "Financial": ["JPM", "BAC"],
    "Energy": ["XOM", "CVX"],
    "Tech": ["AAPL", "MSFT", "AMZN"],
}


# ============================================================================ #
#  Shared statistical helpers
# ============================================================================ #

def _ensure_fwd_vol(df: pd.DataFrame, horizon: int = 5) -> pd.Series:
    """Return forward realized vol over the next ``horizon`` days.

    Reuses ``realized_vol_{horizon}d`` if already present (added by
    ``src.data_helpers.add_forward_vol``); otherwise computes it as the
    annualised std of the next ``horizon`` log-returns.
    """
    col = f"realized_vol_{horizon}d"
    if col in df.columns:
        return df[col]
    return df["log_return"].shift(-horizon).rolling(horizon).std() * np.sqrt(252)


def _regime_series(df: pd.DataFrame, vol_col: str = "realized_vol_21d") -> pd.Series:
    """Label every trading day with its vol regime (Low/Elevated/High/Extreme)."""
    return df[vol_col].dropna().map(_regime_label)


def _bootstrap_ci_mean(x: np.ndarray, n_boot: int = 2000, seed: int = 42,
                       alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of ``x``."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(n_boot)])
    return tuple(np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


def steiger_dependent_corr(r_xy: float, r_xz: float, r_yz: float,
                           n: int) -> tuple[float, float]:
    """Steiger's (1980) Z test for two *dependent, overlapping* correlations.

    Tests H0: corr(x,y) == corr(x,z) where y and z share variable x and are
    themselves correlated at ``r_yz``.  Used to ask whether one predictor
    (e.g. sentiment velocity) correlates with forward vol *significantly more*
    than another (e.g. sentiment level) on the same sample.

    Returns ``(Z, two_sided_p)``.
    """
    if n <= 3:
        return (np.nan, np.nan)
    r_xy = np.clip(r_xy, -0.999, 0.999)
    r_xz = np.clip(r_xz, -0.999, 0.999)
    z1, z2 = np.arctanh(r_xy), np.arctanh(r_xz)
    rm2 = (r_xy ** 2 + r_xz ** 2) / 2.0
    f = (1.0 - r_yz) / (2.0 * (1.0 - rm2) + 1e-12)
    f = min(f, 1.0)
    h = (1.0 - f * rm2) / (1.0 - rm2 + 1e-12)
    denom = np.sqrt(2.0 * (1.0 - r_yz) * h / (n - 3))
    if denom == 0:
        return (np.nan, np.nan)
    z = (z1 - z2) / denom
    p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    return (float(z), float(p))


def scheirer_ray_hare(df: pd.DataFrame, response: str, factor_a: str,
                      factor_b: str) -> pd.DataFrame:
    """Non-parametric two-way ANOVA (Scheirer-Ray-Hare 1976) on rank data.

    Returns a table with H statistics, df, and chi-square p-values for the two
    main effects and their interaction.  Used by H17 where the response
    (forward vol) is heavily skewed so a parametric two-way ANOVA is invalid.
    """
    d = df[[response, factor_a, factor_b]].dropna().copy()
    d["R"] = stats.rankdata(d[response].values)
    n = len(d)
    rbar = (n + 1) / 2.0
    ss_total = float(((d["R"] - rbar) ** 2).sum())
    ms_total = ss_total / (n - 1)

    def _ss(group_cols):
        g = d.groupby(group_cols, observed=True)["R"]
        return float((g.count() * (g.mean() - rbar) ** 2).sum())

    ss_a = _ss(factor_a)
    ss_b = _ss(factor_b)
    ss_cells = _ss([factor_a, factor_b])
    ss_ab = ss_cells - ss_a - ss_b

    a = d[factor_a].nunique()
    b = d[factor_b].nunique()
    rows = []
    for name, ss, dof in [
        (factor_a, ss_a, a - 1),
        (factor_b, ss_b, b - 1),
        ("interaction", ss_ab, (a - 1) * (b - 1)),
    ]:
        h = ss / ms_total if ms_total > 0 else np.nan
        p = float(stats.chi2.sf(h, dof)) if dof > 0 else np.nan
        rows.append({"effect": name, "H": h, "df": dof, "p_value": p})
    return pd.DataFrame(rows).set_index("effect")


def compute_disagreement(garch_preds: pd.Series, ml_preds: pd.Series
                         ) -> pd.DataFrame:
    """Align EGARCH and ML forecasts and derive the disagreement signals.

    Returns a frame indexed to the common dates with columns:
      ``signed``    = EGARCH - ML  (positive => EGARCH sees more risk)
      ``norm_abs``  = |EGARCH - ML| / mean(|EGARCH|, |ML|)  (H2/H12 magnitude)
    """
    common = garch_preds.index.intersection(ml_preds.index)
    eg = garch_preds.reindex(common).astype(float)
    ml = ml_preds.reindex(common).astype(float)
    denom = (eg.abs() + ml.abs()) / 2 + 1e-8
    return pd.DataFrame({"signed": eg - ml, "norm_abs": (eg - ml).abs() / denom},
                        index=common).dropna()


def _verdict(p: float) -> str:
    """'** significant **' / 'not significant' verdict string at alpha=0.05."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "inconclusive (insufficient data)"
    return "** significant **" if p < 0.05 else "not significant"


def _print_header(hyp: str, title: str, extends: str) -> None:
    print(f"\n{'='*64}\n  {hyp}  (extends {extends}) -- {title}\n{'='*64}")


# ============================================================================ #
#  EXTENDING H1 -- SENTIMENT ASYMMETRY (H9, H10, H11)
# ============================================================================ #

def test_sentiment_asymmetry(df: pd.DataFrame, sent: pd.DataFrame, ticker: str,
                             horizon: int = 5) -> dict:
    """
    H9 (extends H1): Is the sentiment->vol relationship *asymmetric*?

    H1 confirmed that negative sentiment precedes vol spikes.  This test splits
    days by VADER compound into Negative (< -0.3), Neutral (-0.3..+0.3) and
    Positive (> +0.3) and compares next-``horizon``-day realized vol across the
    three groups with a Kruskal-Wallis test, followed by pairwise Dunn post-hoc
    tests (Negative-Positive, Negative-Neutral, Positive-Neutral).

    Why it matters: if negative sentiment lifts forward vol but positive
    sentiment does *not* depress it below neutral, then sentiment is a one-sided
    RISK signal -- useful for detecting danger, useless as a "calm" signal.
    That asymmetry (markets climb on low vol, crash on high vol -- the leverage
    effect of H5) argues for engineering ``min(sentiment, 0)`` as a feature
    rather than the raw score.

    Returns Kruskal-Wallis p plus all three Dunn pairwise p-values and a
    three-group violin plot of the forward-vol distributions.
    """
    _print_header("H9", "Sentiment asymmetry (Kruskal-Wallis + Dunn)", "H1")
    fwd = _ensure_fwd_vol(df, horizon).rename("fwd_vol")
    d = pd.concat([sent["vader_compound"], fwd], axis=1).dropna()
    d["group"] = np.where(d["vader_compound"] < -0.3, "Negative",
                          np.where(d["vader_compound"] > 0.3, "Positive", "Neutral"))

    groups = {g: d.loc[d["group"] == g, "fwd_vol"].values
              for g in ["Negative", "Neutral", "Positive"]}
    sizes = {g: len(v) for g, v in groups.items()}
    if min(sizes.values()) < 5:
        return dict(hypothesis="H9", extends="H1", available=False, p_value=np.nan,
                    conclusion=f"{ticker}: a sentiment group has <5 obs {sizes}.",
                    actionable="Feature: min(sentiment, 0)")

    h_stat, p_kw = stats.kruskal(*groups.values())
    dunn = (sp.posthoc_dunn(d, val_col="fwd_vol", group_col="group", p_adjust="holm")
            if sp is not None else pd.DataFrame())
    p_neg_pos = float(dunn.loc["Negative", "Positive"]) if not dunn.empty else np.nan
    p_neg_neu = float(dunn.loc["Negative", "Neutral"]) if not dunn.empty else np.nan
    p_pos_neu = float(dunn.loc["Positive", "Neutral"]) if not dunn.empty else np.nan

    means = {g: float(np.mean(v)) for g, v in groups.items()}
    asym = (means["Negative"] > means["Neutral"]) and (p_neg_pos < 0.05) \
        and not (means["Positive"] < means["Neutral"] and p_pos_neu < 0.05)
    conclusion = (
        f"{ticker}: forward vol Negative={means['Negative']:.3f} / "
        f"Neutral={means['Neutral']:.3f} / Positive={means['Positive']:.3f}. "
        f"KW H={h_stat:.2f}, p={p_kw:.4f} ({_verdict(p_kw)}). "
        f"Dunn Neg-Pos p={p_neg_pos:.4f}. "
        f"{'Asymmetric (risk-only) signal confirmed.' if asym else 'No clean asymmetry.'}"
    )
    print(f"  {conclusion}")
    if not dunn.empty:
        print("  Dunn (Holm) pairwise p-values:\n" + dunn.round(4).to_string())

    plot_df = pd.concat([
        pd.DataFrame({"Forward vol": v, "Sentiment group": g}) for g, v in groups.items()
    ])
    fig = px.violin(plot_df, x="Sentiment group", y="Forward vol", color="Sentiment group",
                    box=True, points=False, category_orders={"Sentiment group":
                        ["Negative", "Neutral", "Positive"]},
                    color_discrete_map={"Negative": "#e74c3c", "Neutral": "#95a5a6",
                                        "Positive": "#27ae60"},
                    title=f"H9 - {ticker}: Next-{horizon}d Volatility by Sentiment Sign")
    fig.add_annotation(text=f"KW p={p_kw:.4f} | Dunn Neg-Pos p={p_neg_pos:.4f}",
                       xref="paper", yref="paper", x=0.5, y=1.06, showarrow=False)
    fig.update_layout(showlegend=False,
                      yaxis_title=f"Next-{horizon}d realized vol (annualised)")
    return dict(hypothesis="H9", extends="H1", available=True, statistic=h_stat,
                p_value=p_kw, effect=means["Negative"] - means["Positive"],
                dunn_p_neg_pos=p_neg_pos, dunn_p_neg_neu=p_neg_neu,
                dunn_p_pos_neu=p_pos_neu, group_means=means,
                conclusion=conclusion, actionable="Feature: min(sentiment, 0)",
                figure=fig)


def test_sentiment_velocity(df: pd.DataFrame, sent: pd.DataFrame, ticker: str,
                            horizon: int = 5, vel_lag: int = 3) -> dict:
    """
    H10 (extends H1): Does sentiment *velocity* beat sentiment *level*?

    H1 used the level of sentiment (how negative today).  This test computes
    velocity = VADER_t - VADER_(t-3) and compares how strongly velocity vs level
    correlate (Spearman) with next-``horizon``-day realized vol.  Steiger's test
    for dependent, overlapping correlations decides whether the two correlations
    differ on the same sample.

    Why it matters: if velocity wins, the market is reacting to *news* (a change
    in narrative) more than to *mood* (the current tone) -- a sharp swing from
    +0.5 to -0.3 is more dangerous than a steady -0.3 the market has already
    digested.  That would justify adding ``sentiment_velocity`` as a feature
    alongside (or instead of) the raw VADER level.

    Returns both Spearman correlations, Steiger's Z/p, and side-by-side
    velocity-vs-vol and level-vs-vol scatter plots with LOWESS fits.
    """
    _print_header("H10", "Sentiment velocity vs level (Steiger)", "H1")
    from statsmodels.nonparametric.smoothers_lowess import lowess
    fwd = _ensure_fwd_vol(df, horizon).rename("fwd_vol")
    level = sent["vader_compound"].rename("level")
    velocity = (sent["vader_compound"] - sent["vader_compound"].shift(vel_lag)).rename("velocity")
    d = pd.concat([level, velocity, fwd], axis=1).dropna()
    if len(d) < 30:
        return dict(hypothesis="H10", extends="H1", available=False, p_value=np.nan,
                    conclusion=f"{ticker}: only {len(d)} aligned obs.",
                    actionable="Feature: sentiment_velocity")

    r_level, p_level = stats.spearmanr(d["level"], d["fwd_vol"])
    r_vel, p_vel = stats.spearmanr(d["velocity"], d["fwd_vol"])
    r_lv, _ = stats.spearmanr(d["level"], d["velocity"])
    # Steiger compares corr(vol, velocity) vs corr(vol, level); shared var = vol.
    z, p_diff = steiger_dependent_corr(r_vel, r_level, r_lv, len(d))

    velocity_wins = abs(r_vel) > abs(r_level)
    conclusion = (
        f"{ticker}: |Spearman| velocity={r_vel:+.3f} (p={p_vel:.4f}) vs "
        f"level={r_level:+.3f} (p={p_level:.4f}). Steiger Z={z:.2f}, "
        f"p={p_diff:.4f} ({_verdict(p_diff)}). "
        f"{'Velocity is the stronger predictor.' if velocity_wins else 'Level is at least as strong.'}"
    )
    print(f"  {conclusion}")

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Velocity vs forward vol", "Level vs forward vol"))
    for xcol, color, c in [("velocity", "#8e44ad", 1), ("level", "#2980b9", 2)]:
        xv = d[xcol].values
        yv = d["fwd_vol"].values
        fig.add_trace(go.Scatter(x=xv, y=yv, mode="markers",
                                 marker=dict(size=4, opacity=0.4, color=color),
                                 showlegend=False), row=1, col=c)
        sm = lowess(yv, xv, frac=0.4, return_sorted=True)
        fig.add_trace(go.Scatter(x=sm[:, 0], y=sm[:, 1], mode="lines",
                                 line=dict(color="black", width=2), showlegend=False),
                      row=1, col=c)
    fig.update_xaxes(title_text=f"Sentiment velocity (Δ{vel_lag}d)", row=1, col=1)
    fig.update_xaxes(title_text="Sentiment level (VADER)", row=1, col=2)
    fig.update_yaxes(title_text=f"Next-{horizon}d vol", row=1, col=1)
    fig.update_layout(title_text=f"H10 - {ticker}: Sentiment Velocity vs Level "
                                 f"(Steiger p={p_diff:.3f})")
    return dict(hypothesis="H10", extends="H1", available=True, statistic=z,
                p_value=p_diff, effect=abs(r_vel) - abs(r_level),
                spearman_velocity=r_vel, spearman_level=r_level,
                conclusion=conclusion, actionable="Feature: sentiment_velocity",
                figure=fig)


def test_sentiment_model_consensus(df: pd.DataFrame, sent: pd.DataFrame, ticker: str,
                                   horizon: int = 5) -> dict:
    """
    H11 (extends H1): Is multi-model *agreement* a stronger signal than any one model?

    H1 used average sentiment.  This test counts how many of the four sentiment
    models flag a day as negative -- VADER < 0, FinBERT < 0, TextBlob < 0, and
    LM score below its sample median -- producing a consensus count in 0..4.
    Days where 3-4 models agree (high consensus) are compared with days where
    0-1 agree (mixed signal) on next-``horizon``-day realized vol using a
    Mann-Whitney U test.

    Why it matters: no single sentiment model is reliable, but *agreement across
    models* may be.  If high-consensus negativity precedes higher vol, it
    validates the ensemble approach and justifies a ``sentiment_consensus``
    feature (count of models agreeing on direction) in the ML pipeline.

    Returns the Mann-Whitney result and a grouped bar chart of mean forward vol
    by consensus level (0-4) with 95% bootstrap CIs.
    """
    _print_header("H11", "Multi-model sentiment consensus (Mann-Whitney)", "H1")
    needed = ["vader_compound", "finbert", "textblob", "lm_score"]
    if not all(c in sent.columns for c in needed):
        return dict(hypothesis="H11", extends="H1", available=False, p_value=np.nan,
                    conclusion=f"{ticker}: sentiment frame lacks {needed}.",
                    actionable="Feature: consensus_count")
    fwd = _ensure_fwd_vol(df, horizon).rename("fwd_vol")
    s = sent[needed].copy()
    lm_med = s["lm_score"].median()
    consensus = ((s["vader_compound"] < 0).astype(int)
                 + (s["finbert"] < 0).astype(int)
                 + (s["textblob"] < 0).astype(int)
                 + (s["lm_score"] < lm_med).astype(int)).rename("consensus")
    d = pd.concat([consensus, fwd], axis=1).dropna()

    high = d.loc[d["consensus"] >= 3, "fwd_vol"].values
    low = d.loc[d["consensus"] <= 1, "fwd_vol"].values
    if len(high) < 5 or len(low) < 5:
        return dict(hypothesis="H11", extends="H1", available=False, p_value=np.nan,
                    conclusion=f"{ticker}: high={len(high)}, low={len(low)} consensus days.",
                    actionable="Feature: consensus_count")
    u, p = stats.mannwhitneyu(high, low, alternative="greater")
    r = 1 - 2 * u / (len(high) * len(low))

    by_level = []
    for lvl in range(5):
        vals = d.loc[d["consensus"] == lvl, "fwd_vol"].values
        if len(vals) == 0:
            by_level.append((lvl, np.nan, np.nan, np.nan, 0))
            continue
        lo, hi = _bootstrap_ci_mean(vals)
        by_level.append((lvl, float(vals.mean()), lo, hi, len(vals)))
    bl = pd.DataFrame(by_level, columns=["consensus", "mean", "lo", "hi", "n"])

    conclusion = (
        f"{ticker}: high-consensus (>=3) mean fwd vol={high.mean():.3f} vs "
        f"low (<=1)={low.mean():.3f}. Mann-Whitney U={u:.0f}, p={p:.4f} "
        f"({_verdict(p)}), r={r:.3f}. "
        f"{'Consensus beats any single model.' if p < 0.05 else 'Consensus not decisive.'}"
    )
    print(f"  {conclusion}")

    fig = go.Figure(go.Bar(
        x=bl["consensus"], y=bl["mean"],
        error_y=dict(type="data", symmetric=False,
                     array=(bl["hi"] - bl["mean"]).values,
                     arrayminus=(bl["mean"] - bl["lo"]).values),
        marker_color="#c0392b", text=[f"n={n}" for n in bl["n"]], textposition="outside"))
    fig.update_layout(title=f"H11 - {ticker}: Mean Next-{horizon}d Vol by Model Consensus "
                            f"(p={p:.4f})",
                      xaxis_title="# models calling the day negative (0-4)",
                      yaxis_title=f"Mean next-{horizon}d realized vol (95% bootstrap CI)")
    return dict(hypothesis="H11", extends="H1", available=True, statistic=u, p_value=p,
                effect=r, by_level=bl, conclusion=conclusion,
                actionable="Feature: consensus_count", figure=fig)


# ============================================================================ #
#  EXTENDING H2 -- DISAGREEMENT DYNAMICS (H12, H13)
# ============================================================================ #

def test_disagreement_persistence(df: pd.DataFrame, disagreement: pd.Series,
                                  ticker: str, spike_pct: float = 0.90,
                                  irf_horizon: int = 20) -> dict:
    """
    H12 (extends H2): How persistent is EGARCH-ML disagreement, and does it mean-revert?

    H2 confirmed that high disagreement precedes vol.  A signal is only useful
    over the window it stays informative, so this test characterises the
    *dynamics* of the disagreement series itself:
      1. AR(1) fit -- regress disagreement_t on disagreement_(t-1); the slope phi
         is the persistence and half-life = -ln(2)/ln(phi) days.
      2. Mean-reversion speed = 1 - phi (slope < 1 confirms reversion).
      3. Regime-conditional half-life -- is disagreement more persistent in the
         Extreme regime than the Low regime?

    Why it matters: a 5-10 day half-life means the H2 signal has a short, defined
    window of usefulness (use short-lookback features); a 30+ day half-life means
    it is slow-moving and longer lookbacks are warranted.

    Returns the AR(1) coefficient, half-life (overall and per regime), and an
    impulse-response plot of the average disagreement trajectory for 20 days
    after a spike above the 90th percentile.
    """
    _print_header("H12", "Disagreement persistence / half-life (AR(1))", "H2")
    dgr = disagreement.dropna()
    if len(dgr) < 40:
        return dict(hypothesis="H12", extends="H2", available=False, p_value=np.nan,
                    conclusion=f"{ticker}: only {len(dgr)} disagreement obs.",
                    actionable="Lookback window")

    def _ar1_halflife(series: pd.Series):
        s0 = series.shift(1).dropna()
        s1 = series.loc[s0.index]
        if len(s1) < 10 or s0.std() == 0:
            return np.nan, np.nan, np.nan
        slope, intercept, r, p, _ = stats.linregress(s0.values, s1.values)
        hl = -np.log(2) / np.log(slope) if 0 < slope < 1 else np.inf
        return slope, hl, p

    phi, half_life, p_phi = _ar1_halflife(dgr)

    reg = _regime_series(df).reindex(dgr.index)
    hl_by_regime = {}
    for r_name in REGIME_ORDER:
        sub = dgr[reg == r_name]
        if len(sub) >= 15:
            _, hl_r, _ = _ar1_halflife(sub)
            hl_by_regime[r_name] = hl_r

    # Impulse response: average forward path after a >90th-pct spike.
    thr = dgr.quantile(spike_pct)
    arr = dgr.values
    spike_pos = np.where(arr >= thr)[0]
    paths = []
    for pos in spike_pos:
        end = min(pos + irf_horizon + 1, len(arr))
        seg = arr[pos:end]
        if len(seg) == irf_horizon + 1:
            paths.append(seg)
    irf = np.array(paths).mean(axis=0) if paths else np.array([])

    hl_str = ", ".join(f"{k}={v:.1f}d" for k, v in hl_by_regime.items()) or "n/a"
    conclusion = (
        f"{ticker}: AR(1) phi={phi:.3f} (p={p_phi:.4f}, {_verdict(p_phi)}), "
        f"half-life={half_life:.1f}d, mean-reversion speed={1 - phi:.3f}. "
        f"Regime half-lives: {hl_str}."
    )
    print(f"  {conclusion}")

    fig = go.Figure()
    if irf.size:
        fig.add_trace(go.Scatter(x=list(range(irf_horizon + 1)), y=irf,
                                 mode="lines+markers", line=dict(color="#16a085", width=2),
                                 name="Mean disagreement"))
        fig.add_hline(y=float(dgr.mean()), line_dash="dot", line_color="gray",
                      annotation_text="unconditional mean")
    fig.update_layout(title=f"H12 - {ticker}: Disagreement Impulse Response after "
                            f">{int(spike_pct*100)}th-pct Spike (half-life {half_life:.1f}d)",
                      xaxis_title="Trading days after spike (t+k)",
                      yaxis_title="Mean normalised EGARCH-ML disagreement")
    hl_action = (f"Lookback window ~{half_life:.0f}d"
                 if np.isfinite(half_life) else "Lookback window (slow-moving)")
    return dict(hypothesis="H12", extends="H2", available=True, statistic=phi,
                p_value=p_phi, effect=half_life, half_life=half_life,
                ar1_phi=phi, half_life_by_regime=hl_by_regime,
                conclusion=conclusion, actionable=hl_action, figure=fig)


def test_disagreement_direction(df: pd.DataFrame, disagreement: pd.Series,
                                ticker: str, change_horizon: int = 5) -> dict:
    """
    H13 (extends H2): Does the *sign* of disagreement predict the direction of vol change?

    H2 showed that the *magnitude* of disagreement precedes high vol.  Here we
    use the signed difference (EGARCH - ML):
      EGARCH_HIGH  : EGARCH >> ML  (top quartile of signed disagreement) -- EGARCH
                     sees more risk than the ML model.
      ML_HIGH      : ML >> EGARCH  (bottom quartile) -- ML sees more risk.
    For each group we measure the next-``change_horizon``-day *change* in
    realized vol (RV_{t+h} - RV_t) and run a binomial sign test on its direction.

    Why it matters: if EGARCH_HIGH is followed by vol *increases* and ML_HIGH by
    vol *decreases*, then EGARCH is the better early-warning system while ML is
    better at calling the peak -- a directional, not just magnitude, signal with
    direct portfolio-timing implications.

    Returns the sign-test p-values for each group and an event-study plot of the
    average vol trajectory +/-10 days around EGARCH_HIGH vs ML_HIGH events.
    """
    _print_header("H13", "Signed disagreement -> vol direction (sign test)", "H2")
    dgr = disagreement.dropna()
    rv = df["realized_vol_21d"]
    fwd_change = (rv.shift(-change_horizon) - rv).rename("vol_change")
    d = pd.concat([dgr.rename("signed"), fwd_change], axis=1).dropna()
    if len(d) < 40:
        return dict(hypothesis="H13", extends="H2", available=False, p_value=np.nan,
                    conclusion=f"{ticker}: only {len(d)} aligned obs.",
                    actionable="Directional signal")

    hi_thr = d["signed"].quantile(0.75)
    lo_thr = d["signed"].quantile(0.25)
    eg_high = d.loc[d["signed"] >= hi_thr, "vol_change"].values
    ml_high = d.loc[d["signed"] <= lo_thr, "vol_change"].values

    def _sign_test(x, alt):
        x = x[x != 0]
        if len(x) < 5:
            return np.nan, np.nan
        n_pos = int((x > 0).sum())
        return n_pos / len(x), float(stats.binomtest(n_pos, len(x), 0.5,
                                                      alternative=alt).pvalue)

    eg_share_up, p_eg = _sign_test(eg_high, "greater")   # expect increases
    ml_share_up, p_ml = _sign_test(ml_high, "less")       # expect decreases

    p_combined = float(np.nanmin([p_eg, p_ml]))
    conclusion = (
        f"{ticker}: after EGARCH_HIGH, {eg_share_up:.0%} of windows show vol UP "
        f"(sign p={p_eg:.4f}); after ML_HIGH, {1 - ml_share_up:.0%} show vol DOWN "
        f"(sign p={p_ml:.4f}). "
        f"{'EGARCH leads up-moves, ML flags peaks.' if p_combined < 0.05 else 'No clean directional split.'}"
    )
    print(f"  {conclusion}")

    # Event study +/-10d around each event type.
    rv_arr = rv.values
    idx = {ts: i for i, ts in enumerate(rv.index)}
    win = np.arange(-10, 11)

    def _event_path(dates):
        rows = []
        for ts in dates:
            i = idx.get(ts)
            if i is None:
                continue
            row = [rv_arr[i + k] if 0 <= i + k < len(rv_arr) else np.nan for k in win]
            rows.append(row)
        return np.array(rows, dtype=float) if rows else np.empty((0, len(win)))

    eg_dates = d.index[d["signed"] >= hi_thr]
    ml_dates = d.index[d["signed"] <= lo_thr]
    fig = go.Figure()
    for dates, color, name in [(eg_dates, "#e74c3c", "EGARCH_HIGH"),
                               (ml_dates, "#2980b9", "ML_HIGH")]:
        mat = _event_path(dates)
        if mat.shape[0] == 0:
            continue
        mean = np.nanmean(mat, axis=0)
        se = np.nanstd(mat, axis=0) / np.sqrt(np.maximum((~np.isnan(mat)).sum(axis=0), 1))
        fig.add_trace(go.Scatter(x=win, y=mean, mode="lines",
                                 line=dict(color=color, width=2), name=name))
        fig.add_trace(go.Scatter(x=np.concatenate([win, win[::-1]]),
                                 y=np.concatenate([mean + 1.96 * se, (mean - 1.96 * se)[::-1]]),
                                 fill="toself", fillcolor=color, opacity=0.15,
                                 line=dict(color="rgba(0,0,0,0)"), showlegend=False))
    fig.add_vline(x=0, line_dash="dash", line_color="black", annotation_text="event (t=0)")
    fig.update_layout(title=f"H13 - {ticker}: Vol Trajectory around Signed-Disagreement Events",
                      xaxis_title="Trading days relative to event",
                      yaxis_title="Realized vol (21d)")
    return dict(hypothesis="H13", extends="H2", available=True, statistic=eg_share_up,
                p_value=p_combined, effect=eg_share_up - ml_share_up,
                p_egarch_high=p_eg, p_ml_high=p_ml, conclusion=conclusion,
                actionable="Directional signal (EGARCH=early warning)", figure=fig)


# ============================================================================ #
#  EXTENDING H4 -- SECTOR CONTAGION (H14, H15)
# ============================================================================ #

def test_asymmetric_contagion(df_dict: dict, source: str = "NVDA",
                              targets: tuple = ("MU", "AMD"),
                              lags: tuple = (1, 2, 3, 5)) -> dict:
    """
    H14 (extends H4): Does *negative* vol contagion spread faster than positive?

    H4 confirmed NVDA vol spills over to MU and AMD.  Contagion is rarely
    symmetric -- panic spreads faster than calm (Forbes & Rigobon 2002).  This
    test separates NVDA daily vol changes into positive shocks (> +1 std) and
    negative shocks (< -1 std) and measures the response in each target's vol at
    lags 1/2/3/5.  Asymmetry is tested with the interaction regression

        target_vol_change = a + b1 * NVDA_shock + b2 * NVDA_shock * negative_dummy

    where a significant, sign-consistent ``b2`` means negative shocks transmit
    more strongly than positive ones.

    Why it matters: if contagion is asymmetric, a sector risk model should weight
    *negative* lead-asset shocks more heavily than positive ones.

    Returns b1, b2 and their p-values per target, plus a dual impulse-response
    plot (positive vs negative NVDA shock) of the average target response.
    """
    _print_header("H14", "Asymmetric vol contagion (interaction regression)", "H4")
    import statsmodels.api as sm
    if source not in df_dict:
        return dict(hypothesis="H14", extends="H4", available=False, p_value=np.nan,
                    conclusion=f"Source {source} missing from df_dict.",
                    actionable="Negative-shock weight")

    src_vol = df_dict[source]["realized_vol_21d"].dropna()
    src_chg = src_vol.diff()
    sd = src_chg.std()
    shock = src_chg.copy()
    neg_dummy = (src_chg < 0).astype(int)

    per_target = {}
    irf = {}
    for tgt in targets:
        if tgt not in df_dict:
            continue
        tgt_chg = df_dict[tgt]["realized_vol_21d"].dropna().diff()
        # Use 1-day lead response in the regression; report IRF over all lags.
        reg = pd.DataFrame({
            "y": tgt_chg.shift(-1),
            "shock": shock,
            "shock_neg": shock * neg_dummy,
        }).dropna()
        if len(reg) < 30:
            continue
        X = sm.add_constant(reg[["shock", "shock_neg"]])
        model = sm.OLS(reg["y"], X).fit()
        per_target[tgt] = {
            "b1": float(model.params["shock"]), "p1": float(model.pvalues["shock"]),
            "b2": float(model.params["shock_neg"]), "p2": float(model.pvalues["shock_neg"]),
        }
        pos_evt = src_chg[src_chg > sd].index
        neg_evt = src_chg[src_chg < -sd].index
        tgt_vol = df_dict[tgt]["realized_vol_21d"]
        pos_resp = [tgt_vol.shift(-lag).reindex(pos_evt).mean()
                    - tgt_vol.reindex(pos_evt).mean() for lag in lags]
        neg_resp = [tgt_vol.shift(-lag).reindex(neg_evt).mean()
                    - tgt_vol.reindex(neg_evt).mean() for lag in lags]
        irf[tgt] = {"pos": pos_resp, "neg": neg_resp}

    if not per_target:
        return dict(hypothesis="H14", extends="H4", available=False, p_value=np.nan,
                    conclusion="No targets had enough data.",
                    actionable="Negative-shock weight")

    p_min = min(v["p2"] for v in per_target.values())
    parts = "; ".join(f"{t}: b1={v['b1']:.3f}(p={v['p1']:.3f}), "
                      f"b2={v['b2']:.3f}(p={v['p2']:.3f})" for t, v in per_target.items())
    asym = any(v["b2"] > 0 and v["p2"] < 0.05 for v in per_target.values())
    conclusion = (f"{source}->{list(per_target)}: {parts}. "
                  f"{'Negative contagion is significantly stronger (Forbes & Rigobon 2002).' if asym else 'No significant asymmetry.'}")
    print(f"  {conclusion}")

    fig = make_subplots(rows=1, cols=len(irf),
                        subplot_titles=[f"{source} -> {t}" for t in irf])
    for c, (tgt, paths) in enumerate(irf.items(), start=1):
        fig.add_trace(go.Scatter(x=list(lags), y=paths["pos"], mode="lines+markers",
                                 line=dict(color="#27ae60"), name="Positive shock",
                                 showlegend=(c == 1)), row=1, col=c)
        fig.add_trace(go.Scatter(x=list(lags), y=paths["neg"], mode="lines+markers",
                                 line=dict(color="#c0392b"), name="Negative shock",
                                 showlegend=(c == 1)), row=1, col=c)
        fig.update_xaxes(title_text="Lag (days)", row=1, col=c)
    fig.update_yaxes(title_text="Target vol response", row=1, col=1)
    fig.update_layout(title_text=f"H14: Asymmetric Contagion -- {source} Shocks "
                                 f"(neg interaction p_min={p_min:.4f})")
    return dict(hypothesis="H14", extends="H4", available=True, statistic=np.nan,
                p_value=p_min, effect=np.mean([v["b2"] for v in per_target.values()]),
                per_target=per_target, conclusion=conclusion,
                actionable="Negative-shock weight in contagion model", figure=fig)


def test_cross_sector_contagion(df_dict: dict, sectors: dict | None = None,
                                lags: tuple = (1, 3, 5)) -> dict:
    """
    H15 (extends H4): Does volatility contagion cross *sector* boundaries?

    H4 showed within-sector (semiconductor) spillover.  This test aggregates
    each sector into an average vol series -- Semiconductor mean(MU,NVDA,AMD),
    Financial mean(JPM,BAC), Energy mean(XOM,CVX), Tech mean(AAPL,MSFT,AMZN) --
    and builds the full directed Granger-causality matrix over all ordered
    sector pairs (min p across lags 1/3/5).  A *net contagion index* is computed
    per sector as (count of significant outgoing links) - (incoming): the sector
    with the highest net outgoing score is the systemic risk *source*.

    Why it matters: cross-sector lead-lag structure tells a portfolio which
    sector to watch first and motivates cross-sector lead features.

    Returns the sector p-value matrix, the net-contagion ranking, and a directed
    network graph (arrow thickness ~ Granger F-stat; red = significant).
    """
    _print_header("H15", "Cross-sector contagion (Granger matrix)", "H4")
    from statsmodels.tsa.stattools import grangercausalitytests
    sectors = sectors or DEFAULT_SECTORS

    sector_vol = {}
    for name, members in sectors.items():
        cols = [df_dict[t]["realized_vol_21d"].rename(t) for t in members if t in df_dict]
        if cols:
            sector_vol[name] = pd.concat(cols, axis=1).mean(axis=1).rename(name)
    names = list(sector_vol)
    if len(names) < 2:
        return dict(hypothesis="H15", extends="H4", available=False, p_value=np.nan,
                    conclusion="Fewer than 2 sectors had data.",
                    actionable="Cross-sector feature")

    pmat = pd.DataFrame(np.nan, index=names, columns=names)
    fmat = pd.DataFrame(0.0, index=names, columns=names)
    for src in names:
        for tgt in names:
            if src == tgt:
                continue
            data = pd.concat([sector_vol[tgt].rename("y"), sector_vol[src].rename("x")],
                             axis=1).dropna()
            if len(data) < 40:
                continue
            try:
                res = grangercausalitytests(data[["y", "x"]], maxlag=max(lags), verbose=False)
                ps = {lag: res[lag][0]["ssr_ftest"][1] for lag in lags}
                fs = {lag: res[lag][0]["ssr_ftest"][0] for lag in lags}
                best = min(ps, key=ps.get)
                pmat.loc[src, tgt] = ps[best]
                fmat.loc[src, tgt] = fs[best]
            except Exception as exc:
                warnings.warn(f"[H15] {src}->{tgt} failed: {exc}")

    sig = (pmat < 0.05)
    net = (sig.sum(axis=1) - sig.sum(axis=0)).sort_values(ascending=False)
    p_min = float(np.nanmin(pmat.values)) if np.isfinite(pmat.values).any() else np.nan
    src_sector = net.index[0]
    rcv_sector = net.index[-1]
    conclusion = (
        f"Net-contagion ranking (out-in): "
        + ", ".join(f"{k}={v:+d}" for k, v in net.items())
        + f". SOURCE = {src_sector}; RECEIVER = {rcv_sector}. min p={p_min:.4f}."
    )
    print(f"  {conclusion}")

    # Directed network on a circle (matplotlib; no networkx dependency).
    fig, ax = plt.subplots(figsize=(7, 7))
    ang = np.linspace(0, 2 * np.pi, len(names), endpoint=False)
    pos = {n: (np.cos(a), np.sin(a)) for n, a in zip(names, ang)}
    fmax = fmat.values.max() or 1.0
    for src in names:
        for tgt in names:
            if src == tgt or np.isnan(pmat.loc[src, tgt]):
                continue
            x0, y0 = pos[src]
            x1, y1 = pos[tgt]
            significant = pmat.loc[src, tgt] < 0.05
            ax.annotate("", xy=(x1 * 0.82, y1 * 0.82), xytext=(x0 * 0.82, y0 * 0.82),
                        arrowprops=dict(arrowstyle="-|>",
                                        color="#c0392b" if significant else "#cccccc",
                                        lw=0.5 + 4 * fmat.loc[src, tgt] / fmax,
                                        alpha=0.9 if significant else 0.4,
                                        connectionstyle="arc3,rad=0.15"))
    for n, (x, y) in pos.items():
        ax.scatter([x], [y], s=2600, c="#34495e", zorder=3)
        ax.text(x, y, n, color="white", ha="center", va="center", fontsize=9,
                fontweight="bold", zorder=4)
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.axis("off")
    ax.set_title(f"H15: Cross-Sector Vol Contagion (red = Granger p<0.05)\n"
                 f"Source: {src_sector}  |  Receiver: {rcv_sector}")
    plt.tight_layout()
    return dict(hypothesis="H15", extends="H4", available=True, statistic=np.nan,
                p_value=p_min, effect=int(net.iloc[0]), pval_matrix=pmat,
                net_contagion=net, source_sector=src_sector, conclusion=conclusion,
                actionable="Cross-sector lead feature", figure=fig)


# ============================================================================ #
#  EXTENDING H5 -- LEVERAGE EFFECT DEPTH (H16, H17)
# ============================================================================ #

def test_regime_dependent_leverage(df: pd.DataFrame, ticker: str, horizon: int = 5,
                                   n_perm: int = 5000, seed: int = 42) -> dict:
    """
    H16 (extends H5): Does the leverage effect *amplify* in stressed regimes?

    H5 confirmed the leverage effect (negative returns -> higher future vol) for
    all tickers.  This test asks whether it is constant or state-dependent.
    Within each vol regime (Low/Elevated/High/Extreme) it computes

        leverage_ratio = mean(next-vol | negative return) /
                         mean(next-vol | positive return)

    (1.0 = no effect; 1.5 = negative days bring 50% more forward vol).  It then
    permutation-tests (n=5000) whether the ratio in the Extreme regime exceeds
    the ratio in the Low regime, since regime subsamples are small.

    Why it matters: if leverage strengthens in Extreme regimes, the danger of a
    negative return is *not* constant -- this is the volatility-feedback effect
    (Campbell & Hentschel 1992) and argues for regime-conditional risk
    parameters rather than a single global leverage coefficient.

    Returns the per-regime leverage ratios with bootstrap CIs, the permutation
    p-value for Extreme > Low, and a grouped bar chart annotated with the
    Campbell & Hentschel (1992) reference.
    """
    _print_header("H16", "Regime-dependent leverage (permutation)", "H5")
    fwd = _ensure_fwd_vol(df, horizon)
    d = pd.DataFrame({"ret": df["log_return"], "fwd": fwd,
                      "regime": _regime_series(df)}).dropna()

    def _ratio(sub):
        neg = sub.loc[sub["ret"] < 0, "fwd"]
        pos = sub.loc[sub["ret"] > 0, "fwd"]
        if len(neg) < 5 or len(pos) < 5:
            return np.nan
        return neg.mean() / (pos.mean() + 1e-12)

    ratios, cis = {}, {}
    rng = np.random.default_rng(seed)
    for r_name in REGIME_ORDER:
        sub = d[d["regime"] == r_name]
        ratio = _ratio(sub)
        ratios[r_name] = ratio
        if not np.isnan(ratio) and len(sub) >= 20:
            boot = []
            for _ in range(1000):
                bs = sub.sample(len(sub), replace=True, random_state=rng.integers(1e9))
                boot.append(_ratio(bs))
            boot = np.array([b for b in boot if not np.isnan(b)])
            cis[r_name] = tuple(np.percentile(boot, [2.5, 97.5])) if boot.size else (np.nan, np.nan)
        else:
            cis[r_name] = (np.nan, np.nan)

    # Permutation test: Extreme ratio > Low ratio.
    sub_lo = d[d["regime"] == "Low"]
    sub_ex = d[d["regime"] == "Extreme"]
    p_perm, obs_diff = np.nan, np.nan
    if not np.isnan(ratios.get("Low", np.nan)) and not np.isnan(ratios.get("Extreme", np.nan)):
        obs_diff = ratios["Extreme"] - ratios["Low"]
        pooled = pd.concat([sub_lo.assign(g="Low"), sub_ex.assign(g="Extreme")])
        n_ex = len(sub_ex)
        count = 0
        for _ in range(n_perm):
            perm = pooled.sample(frac=1, replace=False, random_state=rng.integers(1e9))
            ex_p = _ratio(perm.iloc[:n_ex])
            lo_p = _ratio(perm.iloc[n_ex:])
            if not np.isnan(ex_p) and not np.isnan(lo_p) and (ex_p - lo_p) >= obs_diff:
                count += 1
        p_perm = count / n_perm

    ratio_str = ", ".join(f"{k}={v:.2f}" for k, v in ratios.items() if not np.isnan(v))
    conclusion = (
        f"{ticker}: leverage ratios by regime [{ratio_str}]. "
        f"Extreme-vs-Low diff={obs_diff:.2f}, permutation p={p_perm:.4f} ({_verdict(p_perm)}). "
        f"{'Leverage amplifies under stress (Campbell & Hentschel 1992).' if (isinstance(p_perm, float) and p_perm < 0.05) else 'No significant amplification.'}"
    )
    print(f"  {conclusion}")

    present = [r for r in REGIME_ORDER if not np.isnan(ratios.get(r, np.nan))]
    fig = go.Figure(go.Bar(
        x=present, y=[ratios[r] for r in present],
        error_y=dict(type="data", symmetric=False,
                     array=[cis[r][1] - ratios[r] if not np.isnan(cis[r][1]) else 0 for r in present],
                     arrayminus=[ratios[r] - cis[r][0] if not np.isnan(cis[r][0]) else 0 for r in present]),
        marker_color=["#3498db", "#f39c12", "#e67e22", "#c0392b"][:len(present)]))
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray",
                  annotation_text="no leverage effect")
    fig.add_annotation(text="Volatility-feedback effect (Campbell & Hentschel 1992)",
                       xref="paper", yref="paper", x=0.5, y=-0.18, showarrow=False,
                       font=dict(size=10, color="gray"))
    fig.update_layout(title=f"H16 - {ticker}: Leverage Ratio by Vol Regime "
                            f"(Extreme>Low perm p={p_perm:.4f})",
                      xaxis_title="Vol regime", yaxis_title="neg/pos forward-vol ratio")
    return dict(hypothesis="H16", extends="H5", available=True, statistic=obs_diff,
                p_value=p_perm, effect=obs_diff, leverage_ratios=ratios,
                conclusion=conclusion, actionable="Regime-conditional leverage param",
                figure=fig)


def test_monday_leverage_interaction(df: pd.DataFrame, ticker: str, horizon: int = 5) -> dict:
    """
    H17 (extends H5 + H6): Is the leverage effect *stronger on Mondays*?

    H5 gave the leverage effect; H6 gave the Monday vol premium for
    semiconductors.  This test combines them with a non-parametric two-way
    design (Scheirer-Ray-Hare) on next-``horizon``-day vol:
      Factor 1 = weekday (Monday vs Tue-Fri)
      Factor 2 = return sign (Negative vs Positive)
    and reports the two main effects plus the *interaction*.

    Why it matters: a significant interaction means Monday's higher vol is not
    generic -- it is specifically driven by negative weekend news accumulating
    over the close.  That would justify a ``monday_negative`` indicator (Monday
    AND negative return) as its own ML feature.

    Returns the SRH H-statistics/p-values for both main effects and the
    interaction, plus a 2x2 interaction plot (non-parallel lines => interaction).
    """
    _print_header("H17", "Monday x return-sign interaction (Scheirer-Ray-Hare)", "H5+H6")
    fwd = _ensure_fwd_vol(df, horizon)
    d = pd.DataFrame({"fwd": fwd, "ret": df["log_return"]}, index=df.index).dropna()
    d["weekday"] = np.where(d.index.dayofweek == 0, "Monday", "Tue-Fri")
    d["sign"] = np.where(d["ret"] < 0, "Negative", "Positive")
    if d.groupby(["weekday", "sign"]).size().min() < 5:
        return dict(hypothesis="H17", extends="H5+H6", available=False, p_value=np.nan,
                    conclusion=f"{ticker}: a weekday x sign cell has <5 obs.",
                    actionable="Feature: monday_negative")

    srh = scheirer_ray_hare(d, "fwd", "weekday", "sign")
    p_int = float(srh.loc["interaction", "p_value"])
    p_mon = float(srh.loc["weekday", "p_value"])
    p_sign = float(srh.loc["sign", "p_value"])

    cell_means = d.groupby(["weekday", "sign"])["fwd"].mean().unstack()
    conclusion = (
        f"{ticker}: SRH Monday p={p_mon:.4f} (H6), return-sign p={p_sign:.4f} (H5), "
        f"interaction p={p_int:.4f} ({_verdict(p_int)}). "
        f"{'Monday vol is specifically driven by negative returns.' if p_int < 0.05 else 'No significant Monday x sign interaction.'}"
    )
    print(f"  {conclusion}")
    print("  Cell means (next-{}d vol):\n{}".format(horizon, cell_means.round(4).to_string()))

    fig = go.Figure()
    for sign, color in [("Negative", "#c0392b"), ("Positive", "#27ae60")]:
        fig.add_trace(go.Scatter(
            x=["Monday", "Tue-Fri"],
            y=[cell_means.loc["Monday", sign], cell_means.loc["Tue-Fri", sign]],
            mode="lines+markers", name=f"{sign} return",
            line=dict(color=color, width=3), marker=dict(size=10)))
    fig.update_layout(title=f"H17 - {ticker}: Monday x Return-Sign Interaction "
                            f"(interaction p={p_int:.4f})",
                      xaxis_title="Weekday group",
                      yaxis_title=f"Mean next-{horizon}d realized vol")
    return dict(hypothesis="H17", extends="H5+H6", available=True,
                statistic=float(srh.loc["interaction", "H"]), p_value=p_int,
                effect=np.nan, srh_table=srh, conclusion=conclusion,
                actionable="Feature: monday_negative", figure=fig)


# ============================================================================ #
#  EXTENDING H7 -- EARNINGS DEEP DIVE (H18, H19)
# ============================================================================ #

def test_pre_earnings_sentiment_predicts_spike(df: pd.DataFrame, sent: pd.DataFrame,
                                               earnings_dates, ticker: str,
                                               pre: int = 5, post: int = 5) -> dict:
    """
    H18 (extends H7 + H1): Does pre-earnings sentiment predict the size of the spike?

    H7 found a +38% earnings-week vol premium; H1 linked sentiment to spikes.
    Combining them: for each earnings event we measure
      pre_sentiment       = mean VADER over the ``pre`` days before the event
      earnings_vol_spike  = mean realized vol ``post`` days after - ``pre`` days before
    and correlate them (Spearman).  Events are also split into optimistic
    (pre_sentiment > 0) vs anxious (< 0) and their spike magnitudes compared with
    Mann-Whitney U.

    Why it matters: if pre-earnings anxiety (negative news sentiment) precedes
    larger realized spikes, sentiment is a leading indicator of the post-earnings
    reaction -- directly actionable for options traders sizing straddles.

    Returns the Spearman correlation, the Mann-Whitney split, and a labelled
    scatter (pre-sentiment vs spike) with an OLS trend line.
    """
    _print_header("H18", "Pre-earnings sentiment -> spike size (Spearman)", "H7+H1")
    earnings_dates = pd.DatetimeIndex(earnings_dates)
    earnings_dates = earnings_dates[(earnings_dates >= df.index.min())
                                    & (earnings_dates <= df.index.max())]
    rv = df["realized_vol_21d"]
    rows = []
    for ed in earnings_dates:
        pos = df.index.searchsorted(ed)
        if pos < pre or pos + post >= len(df):
            continue
        pre_sent = sent["vader_compound"].iloc[pos - pre:pos].mean()
        pre_vol = rv.iloc[pos - pre:pos].mean()
        post_vol = rv.iloc[pos:pos + post].mean()
        rows.append({"date": ed, "pre_sent": pre_sent, "spike": post_vol - pre_vol})
    ev = pd.DataFrame(rows).dropna()
    if len(ev) < 4:
        return dict(hypothesis="H18", extends="H7+H1", available=False, p_value=np.nan,
                    conclusion=f"{ticker}: only {len(ev)} usable earnings events.",
                    actionable="Options straddle sizing")

    rho, p_rho = stats.spearmanr(ev["pre_sent"], ev["spike"])
    anx = ev.loc[ev["pre_sent"] < 0, "spike"].values
    opt = ev.loc[ev["pre_sent"] >= 0, "spike"].values
    if len(anx) >= 2 and len(opt) >= 2:
        u, p_mw = stats.mannwhitneyu(anx, opt, alternative="greater")
    else:
        u, p_mw = np.nan, np.nan

    conclusion = (
        f"{ticker} ({len(ev)} events): Spearman(pre-sentiment, spike)={rho:+.3f}, "
        f"p={p_rho:.4f} ({_verdict(p_rho)}). Anxious-vs-optimistic spike MW p={p_mw:.4f}. "
        f"{'Pre-earnings anxiety predicts larger spikes.' if (rho < 0 and p_rho < 0.05) else 'No significant predictive link.'}"
    )
    print(f"  {conclusion}")

    fig = px.scatter(ev, x="pre_sent", y="spike", trendline="ols",
                     title=f"H18 - {ticker}: Pre-Earnings Sentiment vs Earnings Vol Spike "
                           f"(rho={rho:+.2f}, p={p_rho:.3f})")
    fig.update_traces(marker=dict(size=10, color="#8e44ad"))
    for _, r in ev.iterrows():
        fig.add_annotation(x=r["pre_sent"], y=r["spike"],
                           text=r["date"].strftime("%y-%m"), showarrow=False,
                           yshift=10, font=dict(size=8, color="gray"))
    fig.update_layout(xaxis_title=f"Mean VADER over {pre}d before earnings",
                      yaxis_title=f"Post-pre vol spike ({post}d - {pre}d)")
    return dict(hypothesis="H18", extends="H7+H1", available=True, statistic=rho,
                p_value=p_rho, effect=rho, n_events=len(ev), mw_p=p_mw,
                conclusion=conclusion, actionable="Options straddle sizing", figure=fig)


def test_earnings_vol_premium_vs_iv(df: pd.DataFrame, earnings_dates, ticker: str,
                                    iv_series: pd.Series | None = None,
                                    pre: int = 5, post: int = 5) -> dict:
    """
    H19 (extends H7): Is the earnings vol premium already priced by options?

    H7 measured the realized earnings-week vol premium.  But the options market
    knows earnings are coming.  For each event this test compares
      IV_premium = ATM implied-vol increase over the ``pre`` days into earnings
      RV_premium = realized post-earnings vol spike vs the pre-earnings baseline
    and forms earnings_surprise = RV_premium - IV_premium.  A one-sample Wilcoxon
    signed-rank test (H0: median surprise = 0) asks whether options *systematically*
    under- or over-price the event.

    Data note: historical ATM implied vol is not stored in this project.  When
    no ``iv_series`` is supplied the test builds a transparent proxy -- a
    forward-looking expectation of realized vol (its own ``pre``-day-ahead
    rolling mean) -- and flags the result as PROXY.  Supply a real ATM IV series
    (e.g. from ``yfinance`` option chains for the most recent 2-3 events) to make
    it exact.

    Returns the Wilcoxon statistic/p-value and a paired bar chart of IV vs RV
    premium per event, coloured by whether straddle buyers or sellers won.
    """
    _print_header("H19", "Earnings RV premium vs IV premium (Wilcoxon)", "H7")
    earnings_dates = pd.DatetimeIndex(earnings_dates)
    earnings_dates = earnings_dates[(earnings_dates >= df.index.min())
                                    & (earnings_dates <= df.index.max())]
    rv = df["realized_vol_21d"]
    proxy = iv_series is None
    if proxy:
        # Proxy ATM IV: market's forward expectation ~ next-pre-day realized vol.
        iv_series = rv.shift(-pre).rolling(pre).mean()

    rows = []
    for ed in earnings_dates:
        pos = df.index.searchsorted(ed)
        if pos < pre or pos + post >= len(df):
            continue
        iv_pre = iv_series.iloc[max(0, pos - pre)]
        iv_at = iv_series.iloc[pos]
        iv_prem = iv_at - iv_pre
        rv_prem = rv.iloc[pos:pos + post].mean() - rv.iloc[pos - pre:pos].mean()
        rows.append({"date": ed, "iv_prem": iv_prem, "rv_prem": rv_prem,
                     "surprise": rv_prem - iv_prem})
    ev = pd.DataFrame(rows).dropna()
    if len(ev) < 5:
        return dict(hypothesis="H19", extends="H7", available=False, p_value=np.nan,
                    conclusion=f"{ticker}: only {len(ev)} usable events"
                               + (" (PROXY IV)" if proxy else "") + ".",
                    actionable="IV/RV arbitrage")

    w, p = stats.wilcoxon(ev["surprise"], alternative="two-sided")
    med = float(ev["surprise"].median())
    tag = " [PROXY IV]" if proxy else ""
    conclusion = (
        f"{ticker} ({len(ev)} events){tag}: median earnings surprise "
        f"(RV-IV premium)={med:+.3f}, Wilcoxon W={w:.1f}, p={p:.4f} ({_verdict(p)}). "
        + ("Options systematically UNDERprice earnings (straddle buyers win)." if (med > 0 and p < 0.05)
           else "Options systematically OVERprice earnings (sellers win)." if (med < 0 and p < 0.05)
           else "No systematic mispricing detected.")
    )
    print(f"  {conclusion}")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=ev["date"].dt.strftime("%y-%m"), y=ev["iv_prem"],
                         name="IV premium" + tag, marker_color="#2980b9"))
    fig.add_trace(go.Bar(x=ev["date"].dt.strftime("%y-%m"), y=ev["rv_prem"],
                         name="RV premium", marker_color="#e67e22"))
    fig.update_layout(barmode="group",
                      title=f"H19 - {ticker}: IV vs RV Earnings Premium "
                            f"(Wilcoxon p={p:.3f}){tag}",
                      xaxis_title="Earnings event", yaxis_title="Vol premium")
    return dict(hypothesis="H19", extends="H7", available=True, statistic=w, p_value=p,
                effect=med, n_events=len(ev), proxy_iv=proxy, conclusion=conclusion,
                actionable="IV/RV arbitrage (options mispricing detector)", figure=fig)


# ============================================================================ #
#  EXTENDING H8 -- 10-K TEXT ANALYTICS (H20, H21)
# ============================================================================ #

def test_10k_language_change_predicts_regime(df: pd.DataFrame, lm_scores: pd.Series,
                                             ticker: str, horizon_days: int = 60) -> dict:
    """
    H20 (extends H8): Does the *change* in 10-K risk language predict regime shifts?

    H8 showed that the *level* of LM negative-word density predicts the vol
    *level*.  This test goes to first differences: for consecutive filings it
    computes lm_delta = LM(current) - LM(previous) and asks whether an increase
    in risk language is associated with an *upward regime transition* in the 60
    trading days after the filing (regime worsened vs not).  A 2x2 Fisher's exact
    test (language increased y/n x regime worsened y/n) plus an odds ratio with
    95% CI quantifies the association.

    Why it matters: if rising risk language leads regime worsening, the 10-K is
    not merely reflecting current conditions -- the firm's own evolving risk
    assessment is a *leading indicator* of vol regime shifts, an information-
    asymmetry result.

    Returns Fisher's exact p, the odds ratio and its CI, and a 2x2 mosaic plot of
    the joint distribution (language increased x regime worsened).
    """
    _print_header("H20", "10-K language change -> regime shift (Fisher exact)", "H8")
    lm = lm_scores.dropna().sort_index()
    if len(lm) < 3:
        return dict(hypothesis="H20", extends="H8", available=False, p_value=np.nan,
                    conclusion=f"{ticker}: only {len(lm)} filings (need 3+).",
                    actionable="Regime-shift detector")
    reg = _regime_series(df)
    rank = {r: i for i, r in enumerate(REGIME_ORDER)}

    rows = []
    prev_val, prev_date = None, None
    for date, val in lm.items():
        if prev_val is not None:
            pos = df.index.searchsorted(prev_date)
            if pos < len(df):
                r0 = reg.iloc[min(pos, len(reg) - 1)]
                end = min(pos + horizon_days, len(reg) - 1)
                r_future = reg.iloc[pos:end + 1]
                worsened = (r_future.map(rank).max() > rank.get(r0, 0))
                rows.append({"lang_up": val - prev_val > 0, "worsened": bool(worsened)})
        prev_val, prev_date = val, date
    tab = pd.DataFrame(rows)
    if len(tab) < 2:
        return dict(hypothesis="H20", extends="H8", available=False, p_value=np.nan,
                    conclusion=f"{ticker}: only {len(tab)} filing-to-filing transitions.",
                    actionable="Regime-shift detector")

    a = int(((tab["lang_up"]) & (tab["worsened"])).sum())     # up & worsened
    b = int(((tab["lang_up"]) & (~tab["worsened"])).sum())
    c = int(((~tab["lang_up"]) & (tab["worsened"])).sum())
    d2 = int(((~tab["lang_up"]) & (~tab["worsened"])).sum())
    table = [[a, b], [c, d2]]
    odds, p = stats.fisher_exact(table, alternative="greater")
    # Haldane-Anscombe corrected OR + log CI.
    aa, bb, cc, dd = a + 0.5, b + 0.5, c + 0.5, d2 + 0.5
    or_c = (aa * dd) / (bb * cc)
    se = np.sqrt(1 / aa + 1 / bb + 1 / cc + 1 / dd)
    ci = (np.exp(np.log(or_c) - 1.96 * se), np.exp(np.log(or_c) + 1.96 * se))

    conclusion = (
        f"{ticker}: {len(tab)} filing transitions. 2x2 [up&worse={a}, up&ok={b}, "
        f"calm&worse={c}, calm&ok={d2}]. Fisher p={p:.4f} ({_verdict(p)}), "
        f"OR={or_c:.2f} (95% CI [{ci[0]:.2f}, {ci[1]:.2f}]). "
        f"{'Rising risk language leads regime worsening.' if p < 0.05 else 'No significant lead relationship.'}"
    )
    print(f"  {conclusion}")

    # Mosaic plot.
    fig, ax = plt.subplots(figsize=(7, 6))
    rows_tot = [a + b, c + d2]
    total = sum(rows_tot) or 1
    y0 = 0.0
    colors = {("up", "worse"): "#c0392b", ("up", "ok"): "#e8a5a0",
              ("calm", "worse"): "#7fb3d5", ("calm", "ok"): "#d4e6f1"}
    for ri, (lang, cnt_row) in enumerate(zip(["up", "calm"], rows_tot)):
        h = cnt_row / total
        worse_cnt = a if lang == "up" else c
        ok_cnt = b if lang == "up" else d2
        row_sum = (worse_cnt + ok_cnt) or 1
        x0 = 0.0
        for outcome, cnt in [("worse", worse_cnt), ("ok", ok_cnt)]:
            w = cnt / row_sum
            ax.add_patch(plt.Rectangle((x0, y0), w, h, facecolor=colors[(lang, outcome)],
                                       edgecolor="white", lw=2))
            if w > 0.04 and h > 0.04:
                ax.text(x0 + w / 2, y0 + h / 2, f"{cnt}", ha="center", va="center",
                        fontsize=12, fontweight="bold")
            x0 += w
        ax.text(-0.02, y0 + h / 2, f"lang {lang}", ha="right", va="center", fontsize=10)
        y0 += h
    ax.text(0.25, 1.03, "regime worsened", ha="center", fontsize=10)
    ax.text(0.75, 1.03, "regime ok", ha="center", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(f"H20 - {ticker}: 10-K Risk-Language Change vs Regime Shift\n"
                 f"Fisher p={p:.3f}, OR={or_c:.2f}")
    plt.tight_layout()
    return dict(hypothesis="H20", extends="H8", available=True, statistic=or_c,
                p_value=p, effect=or_c, odds_ratio=or_c, or_ci=ci, table=table,
                conclusion=conclusion, actionable="Regime-shift detector", figure=fig)


def test_topic_specific_risk_prediction(df_dict: dict, sectors: dict | None = None,
                                        topic_scores: dict | None = None,
                                        horizon_days: int = 60, seed: int = 7) -> dict:
    """
    H21 (extends H8): Are *specific* 10-K risk topics more predictive than the generic count?

    H8 used the total LM negative-word density.  This test decomposes 10-K risk
    language into four topics -- supply_chain, regulatory, geopolitical,
    financial -- and, for each ticker, correlates every topic ratio (and the
    generic ratio) with forward 60-day realized vol (Spearman).  Steiger's test
    checks whether the best topic beats the generic ratio on the same sample.

    Expected sector pattern: supply_chain / geopolitical dominate for
    semiconductors (export bans, HBM shortages); financial dominates for banks
    (credit risk).

    Data note: this project does not store parsed 10-K topic counts.  When
    ``topic_scores`` is not supplied the function simulates per-ticker topic
    ratios with a sector-dependent loading onto realized vol (reproducible via
    ``seed``) so the analysis pipeline runs end-to-end; the result is flagged
    PROXY.  Replace ``topic_scores`` with EDGAR-parsed ratios for exact numbers.

    Returns the per-ticker best topic, the topic-vs-generic Steiger comparison,
    and a tickers x topics heatmap of Spearman correlations with forward vol.
    """
    _print_header("H21", "Topic-specific 10-K risk vs forward vol (heatmap)", "H8")
    sectors = sectors or DEFAULT_SECTORS
    topics = ["supply_chain", "regulatory", "geopolitical", "financial"]
    ticker_sector = {t: s for s, members in sectors.items() for t in members}
    rng = np.random.default_rng(seed)
    proxy = topic_scores is None

    # Sector-specific dominant-topic loadings (used only to build the proxy).
    dominant = {"Semiconductor": ["supply_chain", "geopolitical"],
                "Financial": ["financial"], "Energy": ["geopolitical", "financial"],
                "Tech": ["regulatory"]}

    corr_rows = {}
    best_topic = {}
    steiger = {}
    for t, dft in df_dict.items():
        if "realized_vol_21d" not in dft.columns:
            continue
        fwd60 = dft["log_return"].shift(-horizon_days).rolling(horizon_days).std() * np.sqrt(252)
        fwd60 = fwd60.dropna()
        if len(fwd60) < 80:
            continue
        sector = ticker_sector.get(t, "Tech")
        if proxy:
            base = (fwd60 - fwd60.mean()) / (fwd60.std() + 1e-9)
            ratios = {}
            for tp in topics:
                load = 0.55 if tp in dominant.get(sector, []) else 0.12
                ratios[tp] = pd.Series(load * base.values
                                       + rng.normal(0, 1.0, len(base)), index=fwd60.index)
            generic = pd.Series(0.25 * base.values + rng.normal(0, 1.0, len(base)),
                                index=fwd60.index)
        else:
            ts = topic_scores[t]
            ratios = {tp: ts[tp].reindex(fwd60.index) for tp in topics}
            generic = ts["generic"].reindex(fwd60.index)

        corrs = {tp: stats.spearmanr(ratios[tp], fwd60)[0] for tp in topics}
        corr_rows[t] = corrs
        gen_corr = stats.spearmanr(generic, fwd60)[0]
        bt = max(corrs, key=lambda k: abs(corrs[k]))
        best_topic[t] = bt
        r_bt_gen = stats.spearmanr(ratios[bt], generic)[0]
        _, p_st = steiger_dependent_corr(corrs[bt], gen_corr, r_bt_gen, len(fwd60))
        steiger[t] = {"best": bt, "best_r": corrs[bt], "generic_r": gen_corr, "p": p_st}

    if not corr_rows:
        return dict(hypothesis="H21", extends="H8", available=False, p_value=np.nan,
                    conclusion="No tickers had enough forward-vol history.",
                    actionable="Topic-specific NLP features")

    cmat = pd.DataFrame(corr_rows).T[topics]
    p_min = float(np.nanmin([v["p"] for v in steiger.values()]))
    tag = " [PROXY topics]" if proxy else ""
    summary = "; ".join(f"{t}:{steiger[t]['best']}(r={steiger[t]['best_r']:+.2f})"
                        for t in list(steiger)[:6])
    conclusion = (f"Most-predictive topic per ticker{tag}: {summary}. "
                  f"Best topic-vs-generic Steiger min p={p_min:.4f} ({_verdict(p_min)}).")
    print(f"  {conclusion}")

    fig = px.imshow(cmat, color_continuous_scale="RdYlGn_r", aspect="auto",
                    labels=dict(color="Spearman r"), text_auto=".2f",
                    title=f"H21: 10-K Risk Topic vs Forward {horizon_days}d Vol "
                          f"(Spearman){tag}")
    fig.update_layout(xaxis_title="Risk topic", yaxis_title="Ticker")
    return dict(hypothesis="H21", extends="H8", available=True, statistic=np.nan,
                p_value=p_min, effect=np.nan, corr_matrix=cmat, best_topic=best_topic,
                steiger=steiger, proxy=proxy, conclusion=conclusion,
                actionable="Topic-specific NLP features", figure=fig)


# ============================================================================ #
#  CROSS-HYPOTHESIS INTERACTIONS (H22, H23)
# ============================================================================ #

def test_triple_signal_interaction(df: pd.DataFrame, sent: pd.DataFrame,
                                   disagreement: pd.Series, ticker: str,
                                   horizon: int = 5) -> dict:
    """
    H22 (extends H1 + H2 + regime): Is the combination of three signals stronger than each alone?

    Builds three binary signals and crosses them into 8 groups:
      sentiment  : negative (VADER < -0.1) vs non-negative
      disagree   : high (> 75th pct) vs low
      regime     : stressed (High/Extreme) vs calm (Low/Elevated)
    The key contrast is the "triple-threat" group (negative + high-disagree +
    stressed) vs the "all-clear" group (non-negative + low-disagree + calm) on
    next-``horizon``-day vol (Mann-Whitney U).  A log-linear OLS on log-vol with
    the three-way interaction term tests whether the combination adds power
    beyond the sum of individual effects.

    Why it matters: if triple-threat vol runs 2-3x all-clear, a single composite
    ``danger_flag`` (negative AND high-disagree AND stressed) can replace three
    separate features and improve model parsimony.

    Returns the triple-threat/all-clear vol ratio, the Mann-Whitney p, the
    interaction-term p, and a forest plot of mean vol (95% CI) for all 8 groups.
    """
    _print_header("H22", "Triple signal interaction (sentiment x disagree x regime)", "H1+H2")
    fwd = _ensure_fwd_vol(df, horizon).rename("fwd")
    dgr = disagreement.dropna()
    dgr_hi = dgr.quantile(0.75)
    reg = _regime_series(df)
    d = pd.concat([fwd, sent["vader_compound"].rename("sent"),
                   dgr.rename("dgr"), reg.rename("reg")], axis=1).dropna()
    if len(d) < 40:
        return dict(hypothesis="H22", extends="H1+H2", available=False, p_value=np.nan,
                    conclusion=f"{ticker}: only {len(d)} aligned obs.",
                    actionable="Composite danger_flag")
    d["s_neg"] = d["sent"] < -0.1
    d["d_hi"] = d["dgr"] > dgr_hi
    d["r_stress"] = d["reg"].isin(_STRESSED)

    groups = []
    for s in [True, False]:
        for dd in [True, False]:
            for rr in [True, False]:
                sub = d[(d["s_neg"] == s) & (d["d_hi"] == dd) & (d["r_stress"] == rr)]["fwd"]
                if len(sub) >= 3:
                    lo, hi = _bootstrap_ci_mean(sub.values)
                    groups.append({"label": f"{'neg' if s else 'pos'}|"
                                            f"{'hiD' if dd else 'loD'}|"
                                            f"{'stress' if rr else 'calm'}",
                                   "mean": float(sub.mean()), "lo": lo, "hi": hi,
                                   "n": len(sub), "triple": s and dd and rr,
                                   "clear": (not s) and (not dd) and (not rr)})
    gdf = pd.DataFrame(groups).sort_values("mean")

    tt = d[(d["s_neg"]) & (d["d_hi"]) & (d["r_stress"])]["fwd"].values
    ac = d[(~d["s_neg"]) & (~d["d_hi"]) & (~d["r_stress"])]["fwd"].values
    if len(tt) >= 3 and len(ac) >= 3:
        u, p_mw = stats.mannwhitneyu(tt, ac, alternative="greater")
        ratio = float(np.mean(tt) / (np.mean(ac) + 1e-12))
    else:
        u, p_mw, ratio = np.nan, np.nan, np.nan

    # Log-linear interaction test.
    import statsmodels.api as sm
    reg_df = d.copy()
    reg_df["logv"] = np.log(reg_df["fwd"].clip(lower=1e-4))
    for col in ["s_neg", "d_hi", "r_stress"]:
        reg_df[col] = reg_df[col].astype(float)
    reg_df["inter"] = reg_df["s_neg"] * reg_df["d_hi"] * reg_df["r_stress"]
    X = sm.add_constant(reg_df[["s_neg", "d_hi", "r_stress", "inter"]])
    ols = sm.OLS(reg_df["logv"], X).fit()
    p_inter = float(ols.pvalues["inter"])

    conclusion = (
        f"{ticker}: triple-threat mean vol={np.mean(tt) if len(tt) else np.nan:.3f} vs "
        f"all-clear={np.mean(ac) if len(ac) else np.nan:.3f} (ratio={ratio:.2f}x). "
        f"MW p={p_mw:.4f} ({_verdict(p_mw)}); 3-way interaction p={p_inter:.4f}. "
        f"{'Composite danger flag justified.' if (isinstance(p_mw, float) and p_mw < 0.05 and ratio >= 1.5) else 'Combination not decisively super-additive.'}"
    )
    print(f"  {conclusion}")

    colors = ["#c0392b" if r["triple"] else "#27ae60" if r["clear"] else "#7f8c8d"
              for _, r in gdf.iterrows()]
    fig = go.Figure(go.Scatter(
        x=gdf["mean"], y=gdf["label"], mode="markers",
        marker=dict(size=11, color=colors),
        error_x=dict(type="data", symmetric=False,
                     array=(gdf["hi"] - gdf["mean"]).values,
                     arrayminus=(gdf["mean"] - gdf["lo"]).values)))
    fig.update_layout(title=f"H22 - {ticker}: Forward Vol by Signal Combination "
                            f"(triple/clear={ratio:.2f}x, MW p={p_mw:.3f})",
                      xaxis_title=f"Mean next-{horizon}d vol (95% bootstrap CI)",
                      yaxis_title="sentiment | disagreement | regime")
    return dict(hypothesis="H22", extends="H1+H2", available=True, statistic=ratio,
                p_value=p_mw, effect=ratio, vol_ratio=ratio, p_interaction=p_inter,
                groups=gdf, conclusion=conclusion,
                actionable="Composite danger_flag feature", figure=fig)


def test_signal_temporal_precedence(df: pd.DataFrame, sent: pd.DataFrame,
                                   disagreement: pd.Series, ticker: str,
                                   df_dict: dict | None = None,
                                   filing_dates=None, lead_window: int = 10,
                                   spike_pct: float = 0.90) -> dict:
    """
    H23 (extends H1, H2, H4, H8): When a vol spike hits, which signal fires *first*?

    For each spike day (realized vol > 90th pct) the test looks back up to
    ``lead_window`` days and records the lead time of each signal:
      sentiment   : VADER < -0.2 on 2+ consecutive days
      disagreement: normalised disagreement > 75th pct
      contagion   : a NVDA vol spike (>90th pct) preceding (from ``df_dict``)
      filing      : a 10-K filing with elevated LM score (from ``filing_dates``)
    It reports median lead time and firing frequency per signal, the first-mover
    frequency, and a Friedman test across signals on the spikes where all fired.

    Why it matters: a signal hierarchy (e.g. NVDA vol leads MU sentiment by 3
    days) tells the monitoring strategy which feature to watch first and which
    lag each feature should use.

    Returns per-signal median lead / hit rate / first-mover share and a timeline
    chart of the average position of each signal relative to the spike (day 0).
    """
    _print_header("H23", "Signal temporal precedence (Friedman)", "H1/H2/H4/H8")
    rv = df["realized_vol_21d"].dropna()
    thr = rv.quantile(spike_pct)
    spike_dates = rv.index[rv >= thr]
    idx = {ts: i for i, ts in enumerate(df.index)}

    # Pre-compute signal "active day" boolean series.
    vader = sent["vader_compound"].reindex(df.index)
    sent_active = (vader < -0.2) & (vader.shift(1) < -0.2)
    dgr = disagreement.reindex(df.index)
    dgr_active = dgr > dgr.quantile(0.75)
    nvda_active = pd.Series(False, index=df.index)
    if df_dict and "NVDA" in df_dict:
        nv = df_dict["NVDA"]["realized_vol_21d"].reindex(df.index)
        nvda_active = nv >= nv.quantile(0.90)
    filing_set = set(pd.DatetimeIndex(filing_dates)) if filing_dates is not None else set()

    sig_series = {"sentiment": sent_active.fillna(False),
                  "disagreement": dgr_active.fillna(False),
                  "contagion": nvda_active.fillna(False)}
    leads = {k: [] for k in list(sig_series) + ["filing"]}
    first_mover = []
    all_fired_rows = []

    for sd in spike_dates:
        pos = idx.get(sd)
        if pos is None or pos < lead_window:
            continue
        window_idx = df.index[pos - lead_window:pos]
        row_leads = {}
        for name, ser in sig_series.items():
            fired = ser.loc[window_idx]
            if fired.any():
                first_pos = np.where(fired.values)[0][0]
                lead = lead_window - first_pos
                leads[name].append(lead)
                row_leads[name] = lead
        # filings within window
        fwin = [fd for fd in filing_set if window_idx.min() <= fd <= window_idx.max()]
        if fwin:
            lead = (sd - max(fwin)).days
            leads["filing"].append(lead)
            row_leads["filing"] = lead
        if row_leads:
            first_mover.append(max(row_leads, key=row_leads.get))
        if all(k in row_leads for k in sig_series):
            all_fired_rows.append([row_leads[k] for k in sig_series])

    n_spikes = len([sd for sd in spike_dates if idx.get(sd, 0) >= lead_window])
    med_lead = {k: (float(np.median(v)) if v else np.nan) for k, v in leads.items()}
    hit_rate = {k: (len(v) / n_spikes if n_spikes else np.nan) for k, v in leads.items()}
    fm_counts = pd.Series(first_mover).value_counts(normalize=True).to_dict() if first_mover else {}

    p_fried = np.nan
    if len(all_fired_rows) >= 5:
        arr = np.array(all_fired_rows)
        try:
            _, p_fried = stats.friedmanchisquare(*[arr[:, i] for i in range(arr.shape[1])])
        except Exception:
            p_fried = np.nan

    hierarchy = sorted([k for k in med_lead if not np.isnan(med_lead[k])],
                       key=lambda k: med_lead[k], reverse=True)
    conclusion = (
        f"{ticker} ({n_spikes} spikes): median lead "
        + ", ".join(f"{k}={med_lead[k]:.1f}d" for k in hierarchy)
        + f". First-mover: {max(fm_counts, key=fm_counts.get) if fm_counts else 'n/a'}. "
        f"Friedman p={p_fried:.4f} ({_verdict(p_fried)}). "
        f"Hierarchy (earliest first): {' -> '.join(hierarchy) if hierarchy else 'n/a'}."
    )
    print(f"  {conclusion}")

    sig_names = [k for k in med_lead if not np.isnan(med_lead[k])]
    fig = go.Figure()
    for name in sig_names:
        vals = leads[name]
        fig.add_trace(go.Scatter(
            x=[-med_lead[name]], y=[name], mode="markers",
            marker=dict(size=14, color="#2c3e50"),
            error_x=dict(type="data", array=[np.std(vals) if len(vals) > 1 else 0]),
            name=name, showlegend=False))
    fig.add_vline(x=0, line_dash="dash", line_color="#c0392b",
                  annotation_text="vol spike (t=0)")
    fig.update_layout(title=f"H23 - {ticker}: Average Signal Lead Time before Vol Spike",
                      xaxis_title="Trading days relative to spike (negative = leads)",
                      yaxis_title="Signal")
    return dict(hypothesis="H23", extends="H1/H2/H4/H8", available=True, statistic=np.nan,
                p_value=p_fried, effect=np.nan, median_lead=med_lead, hit_rate=hit_rate,
                first_mover=fm_counts, hierarchy=hierarchy, conclusion=conclusion,
                actionable="Monitoring strategy / feature lags", figure=fig)


# ============================================================================ #
#  RESULTS FRAMEWORK
# ============================================================================ #

_FINDING_LABEL = {
    "H9": "Negative asymmetry", "H10": "Velocity vs level",
    "H11": "Consensus vs single model", "H12": "Disagreement half-life",
    "H13": "EGARCH = early warning", "H14": "Asymmetric contagion",
    "H15": "Semi -> Tech contagion", "H16": "Leverage amplifies in Extreme",
    "H17": "Monday x negative interaction", "H18": "Pre-earnings anxiety -> spike",
    "H19": "Earnings surprise systematic", "H20": "LM delta -> regime shift",
    "H21": "Topic-specific risk ranking", "H22": "Triple threat vol ratio",
    "H23": "Signal hierarchy",
}
_HYP_ORDER = [f"H{i}" for i in range(9, 24)]


def compile_l2_summary(results: dict, out_csv: str | None = None) -> pd.DataFrame:
    """Compile H9-H23 results into the Level-2 summary table (and optional CSV).

    ``results`` maps hypothesis id -> the dict returned by each ``test_*``
    function.  Produces columns: H, Extends, Finding, p-value, Effect,
    Significant, Actionable.  When ``out_csv`` is given the table is written
    there (parent dirs created).
    """
    rows = []
    for hyp in _HYP_ORDER:
        r = results.get(hyp, {})
        p = r.get("p_value", np.nan)
        eff = r.get("effect", r.get("effect_size", np.nan))
        p_ok = isinstance(p, (int, float)) and not pd.isna(p)
        if not r.get("available", True):
            sig = "n/a (no data)"
        elif p_ok:
            sig = "Yes" if p < 0.05 else "No"
        else:
            sig = "--"
        rows.append({
            "H": hyp, "Extends": r.get("extends", ""),
            "Finding": _FINDING_LABEL.get(hyp, ""),
            "p_value": f"{p:.4f}" if p_ok else "--",
            "Effect": f"{eff:.3f}" if isinstance(eff, (int, float)) and not pd.isna(eff) else "--",
            "Significant": sig,
            "Actionable": r.get("actionable", ""),
        })
    summary = pd.DataFrame(rows).set_index("H")
    if out_csv:
        from pathlib import Path
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out_csv)
        print(f"  Level-2 summary written: {out_csv}")
    return summary


def append_findings_report(summary: pd.DataFrame, results: dict,
                           report_path: str, ticker: str = "MU") -> None:
    """Append an 'Advanced Hypothesis Tests' section to ``findings_report.md``.

    Renders the summary table as Markdown and adds the per-hypothesis one-line
    conclusions, so the Level-2 results live alongside the Level-1 report.
    """
    from pathlib import Path
    lines = ["\n\n---\n\n## Advanced Hypothesis Tests (Level 2)\n",
             f"\nExtends the confirmed H1-H8 findings. Primary ticker: {ticker}. "
             "Tests flagged *PROXY* use simulated inputs where historical data "
             "(option-implied vol, parsed 10-K topic counts) is not stored locally.\n",
             "\n| H | Extends | Finding | p-value | Effect | Significant | Actionable |\n",
             "|---|---------|---------|---------|--------|-------------|------------|\n"]
    for hyp, row in summary.iterrows():
        lines.append(f"| {hyp} | {row['Extends']} | {row['Finding']} | {row['p_value']} "
                     f"| {row['Effect']} | {row['Significant']} | {row['Actionable']} |\n")
    lines.append("\n### Per-hypothesis conclusions\n\n")
    for hyp in _HYP_ORDER:
        r = results.get(hyp, {})
        if r.get("conclusion"):
            lines.append(f"- **{hyp}** ({r.get('extends','')}): {r['conclusion']}\n")
    p = Path(report_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"  Appended Level-2 section to {report_path}")
