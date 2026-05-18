"""
Hypothesis test functions for the volatility research project.

Each public function returns a dict with at minimum:
    statistic, p_value, conclusion, effect_size (where applicable)
and prints a human-readable summary.

All tests target Micron Technology (MU) by default but are ticker-agnostic.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import warnings
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
from statsmodels.tsa.stattools import grangercausalitytests
import scikit_posthocs as sp

pio.renderers.default = "notebook"

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================================ #
# H1 -- SPIKE DAYS PRECEDED BY NEGATIVE SENTIMENT
# ============================================================================ #

def test_spike_sentiment(
    df: pd.DataFrame,
    sent: pd.DataFrame,
    spike_pct: float = 0.90,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """
    H1: Spike days (realized vol >= 90th pct) are preceded by more negative
    sentiment than non-spike days.

    Parameters
    ----------
    df   : price DataFrame with 'realized_vol_21d'.
    sent : sentiment DataFrame with 'vader_compound'.
    spike_pct : percentile threshold that defines a spike day.
    n_bootstrap : resamples for bootstrap CI on difference in means.

    Returns dict with t-statistic, p-value, Cohen's d, bootstrap CI,
    and a Plotly figure (side-by-side violin).
    """
    combined = df[["realized_vol_21d"]].join(sent[["vader_compound"]], how="inner").dropna()
    threshold = combined["realized_vol_21d"].quantile(spike_pct)

    spike_mask = combined["realized_vol_21d"] >= threshold
    spike_sent   = combined.loc[spike_mask,   "vader_compound"].values
    nospike_sent = combined.loc[~spike_mask,  "vader_compound"].values

    t_stat, p_val = stats.ttest_ind(spike_sent, nospike_sent)

    # Cohen's d
    pooled_std = np.sqrt(
        (spike_sent.std(ddof=1)**2 + nospike_sent.std(ddof=1)**2) / 2
    )
    cohens_d = (spike_sent.mean() - nospike_sent.mean()) / (pooled_std + 1e-12)

    # Bootstrap CI on difference in means
    rng = np.random.default_rng(seed)
    diffs = np.array([
        rng.choice(spike_sent, len(spike_sent), replace=True).mean()
        - rng.choice(nospike_sent, len(nospike_sent), replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])

    conclusion = (
        f"Spike days show {'more negative' if cohens_d < 0 else 'more positive'} "
        f"prior sentiment (d={cohens_d:.3f}, p={p_val:.4f})."
    )
    _print_result("H1", "Spike-Sentiment", t_stat, p_val, cohens_d, conclusion)

    # Violin plot
    plot_df = pd.DataFrame({
        "Sentiment (VADER)": np.concatenate([spike_sent, nospike_sent]),
        "Day Type": ["Spike"] * len(spike_sent) + ["Non-Spike"] * len(nospike_sent),
    })
    fig = px.violin(
        plot_df, x="Day Type", y="Sentiment (VADER)",
        color="Day Type", box=True, points="outliers",
        title="H1: VADER Sentiment on Spike vs Non-Spike Volatility Days",
        color_discrete_map={"Spike": "#e74c3c", "Non-Spike": "#3498db"},
    )
    fig.add_annotation(
        text=f"Cohen's d = {cohens_d:.3f}  |  95% CI [{ci_lo:.3f}, {ci_hi:.3f}]",
        xref="paper", yref="paper", x=0.5, y=1.04, showarrow=False,
        font=dict(size=12),
    )
    fig.update_layout(showlegend=False)

    return dict(
        hypothesis="H1",
        statistic=t_stat,
        p_value=p_val,
        effect_size=cohens_d,
        bootstrap_ci=(ci_lo, ci_hi),
        conclusion=conclusion,
        figure=fig,
    )


# ============================================================================ #
# H2 -- LEVERAGE EFFECT
# ============================================================================ #

def test_leverage_effect(
    df: pd.DataFrame,
    forward_vol_col: str = "realized_vol_5d",
) -> dict:
    """
    H2: Negative return days produce significantly higher future (5-day)
    realized volatility than positive return days of equivalent magnitude
    (Black 1976 leverage effect).

    Uses Mann-Whitney U test (non-parametric; vol is right-skewed).
    Visualizes with a side-by-side violin plot coloured by return sign.
    """
    data = df[["log_return", forward_vol_col]].dropna()
    neg = data.loc[data["log_return"] < 0, forward_vol_col].values
    pos = data.loc[data["log_return"] > 0, forward_vol_col].values

    u_stat, p_val = stats.mannwhitneyu(neg, pos, alternative="greater")

    # Effect size: rank-biserial correlation r = 1 - 2U/(n1*n2)
    r = 1 - 2 * u_stat / (len(neg) * len(pos))

    conclusion = (
        f"Negative return days show {'significantly' if p_val < 0.05 else 'not significantly'} "
        f"higher future vol (U={u_stat:.0f}, p={p_val:.4f}, r={r:.3f})."
    )
    _print_result("H2", "Leverage Effect (Mann-Whitney U)", u_stat, p_val, r, conclusion)

    plot_df = pd.DataFrame({
        "Future 5d Vol": np.concatenate([neg, pos]),
        "Return Sign": (["Negative Return"] * len(neg) + ["Positive Return"] * len(pos)),
    })
    fig = px.violin(
        plot_df, x="Return Sign", y="Future 5d Vol",
        color="Return Sign", box=True, points="outliers",
        title="H2: Leverage Effect -- Future Volatility by Return Sign",
        color_discrete_map={
            "Negative Return": "#c0392b",
            "Positive Return": "#27ae60",
        },
    )
    fig.add_annotation(
        text=f"Mann-Whitney p = {p_val:.4f}  |  rank-biserial r = {r:.3f}",
        xref="paper", yref="paper", x=0.5, y=1.04, showarrow=False,
        font=dict(size=12),
    )
    fig.update_layout(showlegend=False)

    return dict(
        hypothesis="H2",
        statistic=u_stat,
        p_value=p_val,
        effect_size=r,
        conclusion=conclusion,
        figure=fig,
    )


# ============================================================================ #
# H3 -- GRANGER CAUSALITY
# ============================================================================ #

def test_granger_causality(
    df: pd.DataFrame,
    sent: pd.DataFrame,
    lags: list[int] | None = None,
    vol_col: str = "realized_vol_21d",
    sent_col: str = "vader_compound",
) -> dict:
    """
    H3: Test whether VADER sentiment Granger-causes realized volatility and
    vice versa. Uses statsmodels grangercausalitytests at lags [1,2,3,5].

    Returns a summary DataFrame with F-statistic and p-value per lag for
    both directions, plus a bar chart comparing p-values.
    """
    if lags is None:
        lags = [1, 2, 3, 5]

    combined = (
        df[[vol_col]].join(sent[[sent_col]], how="inner")
        .dropna()
        .sort_index()
    )
    vol  = combined[vol_col].values
    sent_vals = combined[sent_col].values

    rows = []
    for direction, y_data, x_data, label in [
        ("Sentiment -> Vol", vol, sent_vals, f"{sent_col} -> {vol_col}"),
        ("Vol -> Sentiment", sent_vals, vol, f"{vol_col} -> {sent_col}"),
    ]:
        ts = np.column_stack([y_data, x_data])
        try:
            gc_res = grangercausalitytests(ts, maxlag=max(lags), verbose=False)
        except Exception as exc:
            print(f"  Granger test failed ({direction}): {exc}")
            continue
        for lag in lags:
            if lag in gc_res:
                f_stat = gc_res[lag][0]["ssr_ftest"][0]
                p_val  = gc_res[lag][0]["ssr_ftest"][1]
                rows.append(dict(direction=direction, lag=lag,
                                 F=round(f_stat, 4), p_value=round(p_val, 4)))

    result_df = pd.DataFrame(rows)

    # Determine dominant direction
    sent_to_vol = result_df[result_df["direction"] == "Sentiment -> Vol"]["p_value"].min()
    vol_to_sent = result_df[result_df["direction"] == "Vol -> Sentiment"]["p_value"].min()
    dominant = "Sentiment -> Vol" if sent_to_vol < vol_to_sent else "Vol -> Sentiment"

    conclusion = (
        f"Minimum p: Sentiment->Vol={sent_to_vol:.4f}, Vol->Sentiment={vol_to_sent:.4f}. "
        f"Dominant Granger direction: {dominant}."
    )
    print(f"\n{'-'*55}")
    print(f"  H3: Granger Causality")
    print(f"{'-'*55}")
    print(result_df.to_string(index=False))
    print(f"  {conclusion}")

    # Bar chart of p-values by lag and direction
    fig = px.bar(
        result_df, x="lag", y="p_value", color="direction",
        barmode="group",
        title="H3: Granger Causality -- p-value by Lag and Direction",
        labels={"p_value": "p-value", "lag": "Lag (days)"},
        color_discrete_map={
            "Sentiment -> Vol": "#8e44ad",
            "Vol -> Sentiment": "#2980b9",
        },
    )
    fig.add_hline(y=0.05, line_dash="dash", line_color="red",
                  annotation_text="a = 0.05")
    fig.update_layout(legend_title_text="Direction")

    return dict(
        hypothesis="H3",
        result_table=result_df,
        min_p_sent_to_vol=sent_to_vol,
        min_p_vol_to_sent=vol_to_sent,
        conclusion=conclusion,
        figure=fig,
    )


# ============================================================================ #
# H4 -- MONDAY EFFECT
# ============================================================================ #

def test_monday_effect(
    df: pd.DataFrame,
    vol_col: str = "realized_vol_21d",
) -> dict:
    """
    H4: Monday realized volatility is significantly higher than other weekdays.

    Kruskal-Wallis test across all 5 weekdays (non-parametric ANOVA).
    If significant, Dunn post-hoc test identifies which pairs differ.
    Visualized as a boxplot with significance annotation.
    """
    data = df[[vol_col]].dropna().copy()
    data["weekday"]      = data.index.dayofweek          # 0=Mon ... 4=Fri
    data["weekday_name"] = data.index.day_name()

    day_order  = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    groups     = [data.loc[data["weekday_name"] == d, vol_col].values for d in day_order]
    groups     = [g for g in groups if len(g) > 0]

    h_stat, p_val = stats.kruskal(*groups)

    # Dunn post-hoc (Holm-corrected)
    dunn = sp.posthoc_dunn(
        data, val_col=vol_col, group_col="weekday_name", p_adjust="holm"
    )

    effect_size = _eta_squared_kruskal(h_stat, len(data), len(groups))

    conclusion = (
        f"Kruskal-Wallis H={h_stat:.3f}, p={p_val:.4f} "
        f"({'significant' if p_val < 0.05 else 'not significant'} at a=0.05). "
        f"eta2={effect_size:.3f}."
    )
    print(f"\n{'-'*55}")
    print(f"  H4: Monday Effect -- Kruskal-Wallis")
    print(f"  {conclusion}")
    print("\n  Dunn post-hoc p-values (Holm corrected):")
    print(dunn.to_string())

    # Boxplot
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_data = [
        data.loc[data["weekday_name"] == d, vol_col].values
        for d in day_order
        if d in data["weekday_name"].values
    ]
    medians = [np.median(g) for g in plot_data]
    overall_med = np.median(data[vol_col].values)
    colors = ["#e74c3c" if m > overall_med * 1.02 else "#3498db" for m in medians]

    bplot = ax.boxplot(plot_data, patch_artist=True, labels=day_order,
                       medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bplot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_title(f"H4: Realized Volatility by Weekday\n(H={h_stat:.2f}, p={p_val:.4f})")
    ax.set_ylabel("Realized Volatility (21d, annualized)")
    ax.axhline(overall_med, color="gray", linestyle="--", alpha=0.5,
               label="Weekly median")
    ax.legend()
    plt.tight_layout()

    return dict(
        hypothesis="H4",
        statistic=h_stat,
        p_value=p_val,
        effect_size=effect_size,
        dunn_table=dunn,
        conclusion=conclusion,
        figure=fig,
    )


# ============================================================================ #
# H5 -- EARNINGS WEEK VOLATILITY REGIME
# ============================================================================ #

def test_earnings_vol(
    df: pd.DataFrame,
    earnings_dates: pd.DatetimeIndex,
    vol_col: str = "realized_vol_21d",
    window: int = 2,
    n_permutations: int = 5000,
    seed: int = 42,
) -> dict:
    """
    H5: Realized volatility is significantly higher during the 5 trading days
    surrounding earnings announcements vs all other days.

    Uses a permutation test (n=5000) because earnings weeks are rare events
    and the parametric assumptions do not hold on small samples.
    Visualizes the empirical permutation distribution with the observed
    difference marked as a vertical line.
    """
    trading_days = df.index
    earnings_mask = pd.Series(False, index=trading_days)

    for ed in earnings_dates:
        pos = trading_days.searchsorted(ed)
        lo  = max(0, pos - window)
        hi  = min(len(trading_days), pos + window + 1)
        earnings_mask.iloc[lo:hi] = True

    data = df[[vol_col]].copy()
    data["earnings_week"] = earnings_mask.values

    ew   = data.loc[data["earnings_week"],  vol_col].dropna().values
    non  = data.loc[~data["earnings_week"], vol_col].dropna().values

    if len(ew) == 0:
        print("  H5: No earnings dates found -- skipping permutation test.")
        return dict(hypothesis="H5", p_value=np.nan, conclusion="No earnings data.")

    obs_diff = ew.mean() - non.mean()

    rng = np.random.default_rng(seed)
    combined_all = np.concatenate([ew, non])
    n_ew = len(ew)
    perm_diffs = np.array([
        rng.permutation(combined_all)[:n_ew].mean()
        - rng.permutation(combined_all)[n_ew:].mean()
        for _ in range(n_permutations)
    ])
    p_val = (perm_diffs >= obs_diff).mean()

    conclusion = (
        f"Earnings weeks: mean vol = {ew.mean():.4f}, "
        f"non-earnings: {non.mean():.4f}, "
        f"obs diff = {obs_diff:.4f}, permutation p = {p_val:.4f}."
    )
    _print_result("H5", "Earnings Vol Permutation Test", obs_diff, p_val, None, conclusion)

    # Plotly histogram of permutation distribution
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=perm_diffs, nbinsx=80,
        name="Permutation distribution",
        marker_color="#3498db", opacity=0.7,
    ))
    fig.add_vline(x=obs_diff, line_dash="dash", line_color="#e74c3c", line_width=2,
                  annotation_text=f"Observed Delta = {obs_diff:.4f}",
                  annotation_position="top right")
    fig.update_layout(
        title="H5: Permutation Test -- Earnings Week vs Non-Earnings Week Volatility",
        xaxis_title="Difference in Mean Volatility (Earnings - Non-Earnings)",
        yaxis_title="Count",
        showlegend=False,
    )

    return dict(
        hypothesis="H5",
        statistic=obs_diff,
        p_value=p_val,
        earnings_mean=ew.mean(),
        non_earnings_mean=non.mean(),
        conclusion=conclusion,
        figure=fig,
    )


# ============================================================================ #
# H6 -- VIX REGIME AND ML MODEL ACCURACY
# ============================================================================ #

def test_vix_regime_accuracy(
    feat_df: pd.DataFrame,
    xgb_forecast: pd.Series,
    vix: pd.Series,
) -> dict:
    """
    H6: XGBoost RMSE is significantly higher during High VIX periods.

    Splits the test set into Low VIX (below median) and High VIX (above median).
    Uses Levene's test to compare absolute-error variance between regimes.
    Visualizes VIX level vs absolute model error with LOWESS smoothing.
    """
    from sklearn.metrics import mean_squared_error
    from statsmodels.nonparametric.smoothers_lowess import lowess

    aligned = pd.DataFrame({
        "target": feat_df["target"],
        "pred":   xgb_forecast,
        "vix":    vix,
    }).dropna()

    aligned["abs_error"] = np.abs(aligned["target"] - aligned["pred"])
    vix_med = aligned["vix"].median()
    aligned["vix_regime"] = np.where(aligned["vix"] >= vix_med, "High VIX", "Low VIX")

    low_err  = aligned.loc[aligned["vix_regime"] == "Low VIX",  "abs_error"].values
    high_err = aligned.loc[aligned["vix_regime"] == "High VIX", "abs_error"].values

    rmse_low  = np.sqrt(mean_squared_error(
        aligned.loc[aligned["vix_regime"] == "Low VIX",  "target"],
        aligned.loc[aligned["vix_regime"] == "Low VIX",  "pred"],
    ))
    rmse_high = np.sqrt(mean_squared_error(
        aligned.loc[aligned["vix_regime"] == "High VIX", "target"],
        aligned.loc[aligned["vix_regime"] == "High VIX", "pred"],
    ))

    lev_stat, p_val = stats.levene(low_err, high_err)

    conclusion = (
        f"RMSE: Low VIX={rmse_low:.4f}, High VIX={rmse_high:.4f}. "
        f"Levene W={lev_stat:.3f}, p={p_val:.4f} "
        f"({'error variance significantly different' if p_val < 0.05 else 'no significant difference'})."
    )
    _print_result("H6", "VIX Regime -- Levene Test", lev_stat, p_val, None, conclusion)

    # LOWESS scatter
    sorted_idx = np.argsort(aligned["vix"].values)
    vix_s = aligned["vix"].values[sorted_idx]
    err_s = aligned["abs_error"].values[sorted_idx]
    smooth = lowess(err_s, vix_s, frac=0.3, return_sorted=True)

    colors = np.where(aligned["vix"] >= vix_med, "#e74c3c", "#3498db")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=aligned["vix"], y=aligned["abs_error"],
        mode="markers", marker=dict(color=colors, size=5, opacity=0.5),
        name="Absolute Error",
    ))
    fig.add_trace(go.Scatter(
        x=smooth[:, 0], y=smooth[:, 1],
        mode="lines", line=dict(color="black", width=2),
        name="LOWESS",
    ))
    fig.add_vline(x=vix_med, line_dash="dot", line_color="gray",
                  annotation_text=f"VIX median = {vix_med:.1f}")
    fig.update_layout(
        title="H6: VIX Level vs XGBoost Absolute Error (with LOWESS)",
        xaxis_title="VIX", yaxis_title="|Forecast - Realized|",
    )
    # Legend patches via annotation
    fig.add_annotation(
        text="<span style='color:#3498db'># Low VIX</span>  "
             "<span style='color:#e74c3c'># High VIX</span>",
        xref="paper", yref="paper", x=0.01, y=0.97,
        showarrow=False, font=dict(size=11),
    )

    return dict(
        hypothesis="H6",
        statistic=lev_stat,
        p_value=p_val,
        rmse_low_vix=rmse_low,
        rmse_high_vix=rmse_high,
        conclusion=conclusion,
        figure=fig,
    )


# ============================================================================ #
# H7 -- 10-K RISK LANGUAGE AND FUTURE VOLATILITY
# ============================================================================ #

def test_10k_risk_language(
    df: pd.DataFrame,
    filing_dates: pd.DatetimeIndex,
    lm_risk_scores: pd.Series,
    horizon_days: int = 60,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """
    H7: Tickers with high LM negative-word density in their 10-K MD&A section
    show higher realized volatility in the 60 trading days after filing.

    Splits filings into High Risk (top tercile) and Low Risk (bottom tercile)
    by LM risk score, then compares post-filing realized vol using
    Mann-Whitney U + 1000-resample bootstrap CI on the difference in means.
    Visualizes time series of vol for each group with 95% confidence bands.
    """
    tercile_33 = lm_risk_scores.quantile(0.33)
    tercile_67 = lm_risk_scores.quantile(0.67)

    high_filings = lm_risk_scores[lm_risk_scores >= tercile_67].index
    low_filings  = lm_risk_scores[lm_risk_scores <= tercile_33].index

    def _post_vol(filing_group):
        vols = []
        for fd in filing_group:
            pos = df.index.searchsorted(fd)
            end = min(pos + horizon_days, len(df))
            if end - pos >= 5:
                vols.append(df["realized_vol_21d"].iloc[pos:end].mean())
        return np.array(vols)

    high_vols = _post_vol(high_filings)
    low_vols  = _post_vol(low_filings)

    if len(high_vols) == 0 or len(low_vols) == 0:
        conclusion = "Insufficient filing data for H7."
        return dict(hypothesis="H7", p_value=np.nan, conclusion=conclusion)

    u_stat, p_val = stats.mannwhitneyu(high_vols, low_vols, alternative="greater")
    r = 1 - 2 * u_stat / (len(high_vols) * len(low_vols))

    rng = np.random.default_rng(seed)
    boot_diffs = np.array([
        rng.choice(high_vols, len(high_vols), replace=True).mean()
        - rng.choice(low_vols,  len(low_vols),  replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])

    conclusion = (
        f"High LM Risk mean vol={high_vols.mean():.4f}, "
        f"Low LM Risk={low_vols.mean():.4f}. "
        f"Mann-Whitney U={u_stat:.0f}, p={p_val:.4f}, r={r:.3f}. "
        f"Bootstrap 95% CI on Delta: [{ci_lo:.4f}, {ci_hi:.4f}]."
    )
    _print_result("H7", "10-K Risk Language (Mann-Whitney)", u_stat, p_val, r, conclusion)

    # Time-series plot: post-filing vol by group
    def _rolling_post_vol_series(filing_group, label):
        """Average realized vol at each offset (day 0..horizon) across all filings."""
        matrix = []
        for fd in filing_group:
            pos = df.index.searchsorted(fd)
            end = min(pos + horizon_days, len(df))
            if end - pos >= 5:
                matrix.append(df["realized_vol_21d"].iloc[pos:end].values)
        if not matrix:
            return pd.DataFrame()
        min_len = min(len(m) for m in matrix)
        mat = np.array([m[:min_len] for m in matrix])
        return pd.DataFrame({
            "day": np.arange(min_len),
            "mean": mat.mean(axis=0),
            "lo":   np.percentile(mat, 2.5,  axis=0),
            "hi":   np.percentile(mat, 97.5, axis=0),
            "group": label,
        })

    high_ts = _rolling_post_vol_series(high_filings, "High LM Risk")
    low_ts  = _rolling_post_vol_series(low_filings,  "Low LM Risk")

    fig = go.Figure()
    for ts, color in [(high_ts, "#e74c3c"), (low_ts, "#3498db")]:
        if ts.empty:
            continue
        lbl = ts["group"].iloc[0]
        fig.add_trace(go.Scatter(
            x=ts["day"], y=ts["mean"], mode="lines",
            name=lbl, line=dict(color=color, width=2),
        ))
        fig.add_trace(go.Scatter(
            x=pd.concat([ts["day"], ts["day"][::-1]]),
            y=pd.concat([ts["hi"], ts["lo"][::-1]]),
            fill="toself", fillcolor=color.replace(")", ",0.15)").replace("rgb", "rgba")
                if color.startswith("rgb") else color + "30",
            line=dict(color="rgba(255,255,255,0)"),
            showlegend=False, name=f"{lbl} CI",
        ))
    fig.update_layout(
        title="H7: Post-10-K Filing Volatility -- High vs Low LM Risk",
        xaxis_title="Trading Days After Filing",
        yaxis_title="Mean Realized Volatility (21d)",
    )

    return dict(
        hypothesis="H7",
        statistic=u_stat,
        p_value=p_val,
        effect_size=r,
        bootstrap_ci=(ci_lo, ci_hi),
        conclusion=conclusion,
        figure=fig,
    )


# ============================================================================ #
# H8 -- SENTIMENT MEAN REVERSION
# ============================================================================ #

def test_sentiment_mean_reversion(
    sent: pd.DataFrame,
    extreme_threshold: float = -0.5,
    forward_days: int = 3,
    pre_days: int = 5,
    post_days: int = 10,
    sent_col: str = "vader_compound",
) -> dict:
    """
    H8: Extreme negative sentiment days (VADER < -0.5) are followed by
    significant sentiment recovery within 3 trading days.

    Uses paired Wilcoxon signed-rank test on sentiment at day t vs t+3.
    Event study plot shows average sentiment trajectory from t-5 to t+10.
    """
    s = sent[sent_col].values
    idx = sent.index

    extreme_mask = s < extreme_threshold
    extreme_positions = np.where(extreme_mask)[0]

    t_vals, t3_vals = [], []
    for pos in extreme_positions:
        fwd = pos + forward_days
        if fwd < len(s):
            t_vals.append(s[pos])
            t3_vals.append(s[fwd])

    if len(t_vals) < 5:
        conclusion = f"Too few extreme-negative days ({len(t_vals)}) for H8."
        return dict(hypothesis="H8", p_value=np.nan, conclusion=conclusion)

    w_stat, p_val = stats.wilcoxon(t_vals, t3_vals, alternative="less")

    mean_t  = np.mean(t_vals)
    mean_t3 = np.mean(t3_vals)
    conclusion = (
        f"Extreme negative days: mean sentiment = {mean_t:.3f} -> {mean_t3:.3f} "
        f"at t+{forward_days}. Wilcoxon W={w_stat:.1f}, p={p_val:.4f} "
        f"({'significant recovery' if p_val < 0.05 else 'no significant recovery'})."
    )
    _print_result("H8", "Sentiment Mean Reversion (Wilcoxon)", w_stat, p_val, None, conclusion)

    # Event study: average sentiment trajectory t-5 to t+10
    window = np.arange(-pre_days, post_days + 1)
    trajectories = []
    for pos in extreme_positions:
        row = []
        for offset in window:
            p = pos + offset
            row.append(s[p] if 0 <= p < len(s) else np.nan)
        trajectories.append(row)

    mat = np.array(trajectories, dtype=float)
    mean_traj = np.nanmean(mat, axis=0)
    se_traj   = np.nanstd(mat, axis=0) / np.sqrt((~np.isnan(mat)).sum(axis=0).clip(1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=window, y=mean_traj + 1.96 * se_traj,
        fill=None, mode="lines", line=dict(color="rgba(231,76,60,0)"),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=window, y=mean_traj - 1.96 * se_traj,
        fill="tonexty", mode="lines",
        fillcolor="rgba(231,76,60,0.15)",
        line=dict(color="rgba(231,76,60,0)"),
        name="+/-1.96 SE",
    ))
    fig.add_trace(go.Scatter(
        x=window, y=mean_traj, mode="lines+markers",
        name="Mean VADER Sentiment",
        line=dict(color="#e74c3c", width=2),
        marker=dict(size=5),
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="black",
                  annotation_text="Extreme event (t=0)")
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.update_layout(
        title=(
            f"H8: Sentiment Trajectory Around Extreme Negative Days "
            f"(n={len(trajectories)})"
        ),
        xaxis_title="Trading Days Relative to Extreme Event",
        yaxis_title="Mean VADER Compound Score",
    )

    return dict(
        hypothesis="H8",
        statistic=w_stat,
        p_value=p_val,
        mean_sentiment_t=mean_t,
        mean_sentiment_t3=mean_t3,
        conclusion=conclusion,
        figure=fig,
    )


# ============================================================================ #
# Utility helpers
# ============================================================================ #

def _print_result(hyp_id, test_name, stat, p_val, effect, conclusion):
    sig = "** significant **" if p_val < 0.05 else "not significant"
    print(f"\n{'-'*55}")
    print(f"  {hyp_id}: {test_name}")
    print(f"  Statistic : {stat:.4f}")
    print(f"  p-value   : {p_val:.4f}  ({sig} at a=0.05)")
    if effect is not None:
        print(f"  Effect size: {effect:.4f}")
    print(f"  {conclusion}")


def _eta_squared_kruskal(H, N, k):
    """eta2 for Kruskal-Wallis, bounded to [0, 1]."""
    return max(0.0, (H - k + 1) / (N - k))
