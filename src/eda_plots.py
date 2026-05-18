"""
EDA visualization functions for the volatility research project.

Every function has a full docstring explaining:
  - what the chart shows
  - why it matters for the research
  - what to look for

All Plotly figures use pio.renderers.default = 'notebook' so they freeze
correctly when uploaded to a notebook server.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.stats as stats
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import het_arch
from statsmodels.nonparametric.smoothers_lowess import lowess

pio.renderers.default = "notebook"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================================ #
# EDA1 -- VOLATILITY DISTRIBUTION
# ============================================================================ #

def plot_vol_distribution(df: pd.DataFrame, ticker: str = "MU") -> go.Figure:
    """
    Shows the full distribution of realized volatility using a histogram
    with KDE overlay. A vertical line marks the 90th-percentile spike
    threshold and the percentage of days above it is annotated.

    Why it matters: Determining whether realized volatility is normally
    distributed or fat-tailed is critical for choosing the right statistical
    tests downstream. If vol is fat-tailed, non-parametric tests (Mann-Whitney,
    Kruskal-Wallis) are more appropriate than t-tests. A Shapiro-Wilk normality
    test result is printed alongside the chart.

    What to look for: Right-skewed distribution with a long upper tail
    (typical for equity vol). The percentage of spike days tells you how
    rare the extreme regime is -- too rare means small samples for H1.
    """
    vol = df["realized_vol_21d"].dropna()
    threshold = vol.quantile(0.90)
    pct_above = (vol >= threshold).mean() * 100

    stat_sw, p_sw = stats.shapiro(vol.sample(min(5000, len(vol)), random_state=42))
    print(f"  Shapiro-Wilk: W={stat_sw:.4f}, p={p_sw:.4f} "
          f"({'NOT normal' if p_sw < 0.05 else 'normal'} at a=0.05)")

    # KDE
    kde_x = np.linspace(vol.min(), vol.max(), 400)
    kde = stats.gaussian_kde(vol)
    kde_y = kde(kde_x)
    kde_y_scaled = kde_y * len(vol) * (vol.max() - vol.min()) / 50

    # Q-Q data
    osm, osr = stats.probplot(vol, dist="norm")

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f"{ticker} Realized Vol Distribution",
            "Q-Q Plot vs Normal",
        ],
        horizontal_spacing=0.12,
    )

    # Histogram
    fig.add_trace(go.Histogram(
        x=vol, nbinsx=50, name="Realized Vol",
        marker_color="#3498db", opacity=0.6,
    ), row=1, col=1)
    # KDE line
    fig.add_trace(go.Scatter(
        x=kde_x, y=kde_y_scaled, mode="lines",
        line=dict(color="#e74c3c", width=2), name="KDE",
    ), row=1, col=1)
    # Spike threshold
    fig.add_vline(x=threshold, line_dash="dash", line_color="#f39c12",
                  annotation_text=f"90th pct: {threshold:.3f}<br>({pct_above:.1f}% above)",
                  annotation_position="top right", row=1, col=1)

    # Q-Q scatter
    fig.add_trace(go.Scatter(
        x=osm[0], y=osm[1], mode="markers",
        marker=dict(size=3, color="#2ecc71", opacity=0.5), name="Q-Q",
    ), row=1, col=2)
    # Reference line
    qq_x = np.array([osm[0].min(), osm[0].max()])
    qq_y = osr[1] + osr[0] * qq_x
    fig.add_trace(go.Scatter(
        x=qq_x, y=qq_y, mode="lines",
        line=dict(color="#e74c3c", width=1.5), name="Normal ref",
    ), row=1, col=2)

    fig.update_layout(
        title_text=f"EDA1: {ticker} Realized Volatility Distribution "
                   f"(Shapiro-Wilk p={p_sw:.4f})",
        showlegend=False, height=420,
    )
    fig.update_xaxes(title_text="Annualized Realized Vol", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_xaxes(title_text="Theoretical Quantiles", row=1, col=2)
    fig.update_yaxes(title_text="Sample Quantiles", row=1, col=2)

    return fig


# ============================================================================ #
# EDA2 -- ROLLING CORRELATION HEATMAP
# ============================================================================ #

def plot_rolling_correlation(
    df: pd.DataFrame,
    sent: pd.DataFrame,
    vix: pd.Series,
    window: int = 60,
) -> go.Figure:
    """
    Shows how the correlation between each sentiment model and realized
    volatility changes over a rolling 60-day window.

    Why it matters: Correlation between sentiment and volatility is not
    stable -- it spikes during market stress (high VIX) and collapses in
    calm periods. A time-varying correlation motivates using regime-aware
    features in the ML model, rather than a single static correlation.

    What to look for: Periods where all four sentiment models agree
    (correlation spike) correspond to major market events. Divergence
    between models in the same period flags ambiguous news episodes.
    VIX regime background shading reveals whether high-correlation episodes
    coincide with high-fear environments.
    """
    cols = [c for c in ["vader_compound", "finbert", "textblob", "lm_score"]
            if c in sent.columns]
    vol_col = "realized_vol_21d"

    combined = df[[vol_col]].join(sent[cols], how="inner").dropna()
    combined = combined.join(vix.rename("vix"), how="left")

    roll_corrs = pd.DataFrame(index=combined.index)
    for col in cols:
        roll_corrs[col] = (
            combined[col].rolling(window)
            .corr(combined[vol_col])
        )

    # VIX quantiles for background shading
    vix_q33 = combined["vix"].quantile(0.33)
    vix_q67 = combined["vix"].quantile(0.67)
    vix_regime = pd.cut(
        combined["vix"],
        bins=[-np.inf, vix_q33, vix_q67, np.inf],
        labels=["Low", "Medium", "High"],
    )

    colors = {
        "vader_compound": "#e74c3c",
        "finbert":        "#3498db",
        "textblob":       "#2ecc71",
        "lm_score":       "#f39c12",
    }
    labels = {
        "vader_compound": "VADER",
        "finbert":        "FinBERT",
        "textblob":       "TextBlob",
        "lm_score":       "LM Score",
    }

    fig = go.Figure()

    # VIX regime shading
    regime_color = {"Low": "rgba(39,174,96,0.08)",
                    "Medium": "rgba(241,196,15,0.10)",
                    "High": "rgba(231,76,60,0.10)"}
    prev = None
    start_i = combined.index[0]
    for i, (date, reg) in enumerate(zip(combined.index, vix_regime)):
        if reg != prev:
            if prev is not None:
                fig.add_vrect(
                    x0=str(start_i.date()), x1=str(date.date()),
                    fillcolor=regime_color.get(str(prev), "rgba(0,0,0,0)"),
                    layer="below", line_width=0,
                )
            start_i = date
            prev = reg
    if prev:
        fig.add_vrect(
            x0=str(start_i.date()), x1=str(combined.index[-1].date()),
            fillcolor=regime_color.get(str(prev), "rgba(0,0,0,0)"),
            layer="below", line_width=0,
        )

    for col in cols:
        fig.add_trace(go.Scatter(
            x=roll_corrs.index, y=roll_corrs[col],
            mode="lines", name=labels[col],
            line=dict(color=colors[col], width=1.5),
        ))

    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.update_layout(
        title=f"EDA2: {window}-Day Rolling Correlation: Sentiment vs Realized Vol "
              f"(shaded by VIX regime)",
        xaxis_title="Date",
        yaxis_title="Pearson Correlation",
        legend_title="Sentiment Model",
        height=420,
    )
    return fig


# ============================================================================ #
# EDA3 -- ACF / PACF + ARCH TEST
# ============================================================================ #

def plot_volatility_autocorrelation(
    df: pd.DataFrame,
    lags: int = 40,
) -> plt.Figure:
    """
    Plots ACF and PACF of realized volatility up to 40 lags, plus the ACF
    of squared log-returns (ARCH-effects test).

    Why it matters: Volatility clustering -- the tendency for high volatility
    to follow high volatility -- appears as slowly-decaying ACF. This justifies
    the use of GARCH models and lagged features in the ML pipeline. Engle's
    ARCH test on squared returns formally confirms whether conditional
    heteroscedasticity is present.

    What to look for: ACF decaying slowly over many lags (long memory) is
    consistent with GARCH(1,1) behavior. A spike at lag-5 or lag-21 suggests
    weekly and monthly seasonal patterns. Engle's ARCH test p < 0.05 confirms
    it is worth modelling the variance process explicitly.
    """
    vol = df["realized_vol_21d"].dropna()
    ret_sq = df["log_return"].dropna() ** 2

    lm_stat, p_arch, _, _ = het_arch(df["log_return"].dropna(), nlags=5)
    print(f"  Engle ARCH test (5 lags): LM={lm_stat:.3f}, p={p_arch:.4f} "
          f"({'ARCH effects present' if p_arch < 0.05 else 'no ARCH effects'} at a=0.05)")

    fig, axes = plt.subplots(3, 1, figsize=(11, 11))

    plot_acf(vol, lags=lags, ax=axes[0], alpha=0.05,
             title=f"ACF -- Realized Vol 21d (lags 1-{lags})",
             zero=False)
    axes[0].set_xlabel("Lag (days)")

    plot_pacf(vol, lags=lags, ax=axes[1], alpha=0.05, method="ywm",
              title=f"PACF -- Realized Vol 21d (lags 1-{lags})",
              zero=False)
    axes[1].set_xlabel("Lag (days)")

    plot_acf(ret_sq, lags=lags, ax=axes[2], alpha=0.05,
             title=(f"ACF -- Squared Log-Returns "
                    f"(Engle ARCH p={p_arch:.4f})"),
             zero=False)
    axes[2].set_xlabel("Lag (days)")

    plt.tight_layout()
    return fig


# ============================================================================ #
# EDA4 -- SENTIMENT AGREEMENT
# ============================================================================ #

def plot_sentiment_agreement(
    sent: pd.DataFrame,
    df: pd.DataFrame,
) -> tuple[plt.Figure, go.Figure]:
    """
    Scatter matrix comparing all four sentiment scores against each other,
    with hue determined by volatility regime (Low/Medium/High).
    A separate Plotly histogram shows the distribution of sentiment
    disagreement (std of four normalized scores).

    Why it matters: High agreement between models on a given day increases
    confidence in the signal. High disagreement flags ambiguous news -- days
    where models parse the same text in opposite ways. Those days often have
    idiosyncratic price moves that confuse the ML model.

    What to look for: Strong positive correlation between VADER and LM (both
    lexicon-based). Weaker correlation with TextBlob (trained on movie reviews,
    not financial text). Disagreement spikes during earnings, guidance changes,
    and macro surprise days.
    """
    from src.data_helpers import add_vol_regime

    cols = [c for c in ["vader_compound", "finbert", "textblob", "lm_score"]
            if c in sent.columns]
    combined = sent[cols].join(
        add_vol_regime(df)[["vol_regime"]], how="inner"
    ).dropna()

    # Normalize scores to [0,1] for disagreement metric
    norm = (combined[cols] - combined[cols].min()) / (
        combined[cols].max() - combined[cols].min() + 1e-9
    )
    combined["disagreement"] = norm.std(axis=1)

    # Print Pearson correlations
    print("  Pearson correlations between sentiment models:")
    print(combined[cols].corr().round(3).to_string())

    palette = {"Low": "#3498db", "Medium": "#f39c12", "High": "#e74c3c"}
    rename = {
        "vader_compound": "VADER",
        "finbert": "FinBERT",
        "textblob": "TextBlob",
        "lm_score": "LM Score",
    }
    plot_df = combined.rename(columns=rename)
    plot_cols = [rename.get(c, c) for c in cols]

    pairfig = sns.pairplot(
        plot_df[plot_cols + ["vol_regime"]],
        hue="vol_regime",
        palette=palette,
        plot_kws=dict(alpha=0.4, s=12),
        diag_kind="kde",
    )
    pairfig.fig.suptitle(
        "EDA4: Sentiment Score Pairplot (hue = Vol Regime)", y=1.01, fontsize=13
    )

    # Disagreement distribution
    fig2 = go.Figure(go.Histogram(
        x=combined["disagreement"], nbinsx=50,
        marker_color="#8e44ad", opacity=0.75,
        name="Disagreement",
    ))
    fig2.update_layout(
        title="EDA4b: Sentiment Disagreement Distribution (std of 4 normalized scores)",
        xaxis_title="Disagreement Score",
        yaxis_title="Count",
        height=350,
    )

    return pairfig.fig, fig2


# ============================================================================ #
# EDA5 -- 10-K EVENT STUDY
# ============================================================================ #

def plot_10k_event_study(
    df: pd.DataFrame,
    filing_dates: pd.DatetimeIndex,
    lm_risk_scores: pd.Series,
    window: int = 10,
    n_boot: int = 500,
    seed: int = 42,
) -> go.Figure:
    """
    Average realized volatility in the 10 trading days before and 10 days
    after each 10-K filing date, split by High vs Low LM risk score.

    Why it matters: 10-K filings compress months of operational risk into a
    single document. If the market has not fully priced in the language in the
    filing, we expect a volatility response in the post-filing window. Separating
    High vs Low LM risk filings tests whether the textual signal has incremental
    predictive power beyond what GARCH already captures.

    What to look for: A pre-filing vol rise (uncertainty building) followed by
    a post-filing drop (resolution) is the typical pattern. High-LM-Risk filings
    should show a steeper post-filing vol spike and slower decay compared to
    Low-LM-Risk filings, consistent with H7.
    """
    rng = np.random.default_rng(seed)
    tercile_67 = lm_risk_scores.quantile(0.67)
    tercile_33 = lm_risk_scores.quantile(0.33)
    high_dates = lm_risk_scores[lm_risk_scores >= tercile_67].index
    low_dates  = lm_risk_scores[lm_risk_scores <= tercile_33].index

    offsets = np.arange(-window, window + 1)

    def _build_matrix(dates):
        rows = []
        for fd in dates:
            pos = df.index.searchsorted(fd)
            row = []
            for off in offsets:
                p = pos + off
                row.append(df["realized_vol_21d"].iloc[p]
                           if 0 <= p < len(df) else np.nan)
            rows.append(row)
        return np.array(rows, dtype=float)

    def _boot_ci(mat, alpha=0.05):
        means = np.nanmean(mat, axis=0)
        boot = np.array([
            np.nanmean(mat[rng.integers(0, len(mat), len(mat))], axis=0)
            for _ in range(n_boot)
        ])
        lo = np.nanpercentile(boot, 2.5,  axis=0)
        hi = np.nanpercentile(boot, 97.5, axis=0)
        return means, lo, hi

    fig = go.Figure()
    for dates, label, color in [
        (high_dates, "High LM Risk", "#e74c3c"),
        (low_dates,  "Low LM Risk",  "#3498db"),
    ]:
        mat = _build_matrix(dates)
        if mat.shape[0] == 0:
            continue
        mean, lo, hi = _boot_ci(mat)
        fig.add_trace(go.Scatter(
            x=offsets, y=mean, mode="lines+markers",
            name=label, line=dict(color=color, width=2),
        ))
        fig.add_trace(go.Scatter(
            x=np.concatenate([offsets, offsets[::-1]]),
            y=np.concatenate([hi, lo[::-1]]),
            fill="toself",
            fillcolor=color + "30",
            line=dict(color="rgba(255,255,255,0)"),
            showlegend=False, name=f"{label} CI",
        ))

    fig.add_vline(x=0, line_dash="dash", line_color="black",
                  annotation_text="Filing date")
    fig.update_layout(
        title="EDA5: 10-K Event Study -- Realized Vol Around Filing Date",
        xaxis_title="Trading Days Relative to 10-K Filing",
        yaxis_title="Realized Volatility (21d, annualized)",
        height=430,
    )
    return fig


# ============================================================================ #
# EDA6 -- WEEKDAY VOLATILITY
# ============================================================================ #

def plot_weekday_volatility(
    df: pd.DataFrame,
    h4_p_value: float | None = None,
) -> go.Figure:
    """
    Bar chart of median realized volatility by day of week with individual
    data points overlaid as a strip plot. Error bars show bootstrapped 95%
    confidence intervals around the median.

    Why it matters: The Monday Effect -- higher volatility at the start of the
    week due to weekend news accumulation -- is a well-documented equity market
    anomaly. Confirming it for MU validates using day-of-week as a feature and
    supports H4's Kruskal-Wallis test.

    What to look for: A visually higher Monday median with tighter CI that
    does not overlap with mid-week days is strong visual confirmation of the
    effect. Red bars (significantly above the weekly median) mark the anomalous
    days identified by the H4 Kruskal-Wallis test.
    """
    data = df[["realized_vol_21d"]].dropna().copy()
    data["weekday"] = data.index.day_name()
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

    overall_med = data["realized_vol_21d"].median()

    def _boot_med(vals, n=1000, seed=42):
        rng = np.random.default_rng(seed)
        boot = np.array([
            np.median(rng.choice(vals, len(vals), replace=True))
            for _ in range(n)
        ])
        return np.median(vals), np.percentile(boot, 2.5), np.percentile(boot, 97.5)

    medians, los, his, colors_list = [], [], [], []
    for day in day_order:
        vals = data.loc[data["weekday"] == day, "realized_vol_21d"].values
        med, lo, hi = _boot_med(vals)
        medians.append(med)
        los.append(med - lo)
        his.append(hi - med)
        colors_list.append(
            "#e74c3c" if med > overall_med * 1.02 else "#3498db"
        )

    fig = go.Figure()
    # Strip plot
    for i, day in enumerate(day_order):
        vals = data.loc[data["weekday"] == day, "realized_vol_21d"].values
        jitter = np.random.default_rng(i).uniform(-0.2, 0.2, len(vals))
        fig.add_trace(go.Scatter(
            x=[i + j for j in jitter],
            y=vals,
            mode="markers",
            marker=dict(size=3, color="gray", opacity=0.25),
            showlegend=False,
        ))
    # Bars with bootstrap CI
    fig.add_trace(go.Bar(
        x=list(range(len(day_order))),
        y=medians,
        error_y=dict(type="data", symmetric=False,
                     array=his, arrayminus=los),
        marker_color=colors_list,
        opacity=0.8,
        name="Median Vol",
    ))

    title_suffix = (
        f" (Kruskal-Wallis p={h4_p_value:.4f})"
        if h4_p_value is not None else ""
    )
    fig.update_layout(
        title=f"EDA6: Realized Volatility by Weekday{title_suffix}",
        xaxis=dict(tickvals=list(range(5)), ticktext=day_order),
        yaxis_title="Realized Volatility (21d, annualized)",
        height=420,
        showlegend=False,
    )
    fig.add_hline(y=overall_med, line_dash="dot", line_color="gray",
                  annotation_text="Weekly median")
    return fig


# ============================================================================ #
# EDA7 -- FEATURE-TARGET CORRELATION
# ============================================================================ #

def plot_feature_target_correlation(
    feat_df: pd.DataFrame,
    target: str = "target",
) -> go.Figure:
    """
    Horizontal bar chart ranking all features by their Spearman correlation
    with the realized-vol target. A secondary panel compares correlation
    stability between the first and second halves of the data.

    Why it matters: Spearman is used instead of Pearson because volatility and
    many features are non-normally distributed. This chart quickly identifies
    which features carry the strongest monotonic relationship with the target
    before training -- useful for feature selection and sanity-checking the
    pipeline.

    What to look for: Short-window volatility lags (vol_5d, vol_10d) typically
    dominate -- autocorrelation in vol is the strongest predictor. Sentiment and
    technical features (RSI, BB width) should appear lower but still positive.
    Comparing first-half vs second-half correlations shows whether a feature's
    predictive relationship is stable or regime-dependent.
    """
    from src.features import FEATURE_COLS

    feature_cats = {
        "technical": {"rsi_14", "bb_width"},
        "sentiment": {"vader_compound", "finbert", "textblob", "lm_score"},
    }

    def _cat(col):
        if col in feature_cats["technical"]:
            return "Technical"
        if col in feature_cats["sentiment"]:
            return "Sentiment"
        if col.startswith("ret_lag"):
            return "Lagged Return"
        return "Volatility"

    cols = [c for c in FEATURE_COLS if c in feat_df.columns and c != target]
    # Add sentiment cols if present
    for sc in ["vader_compound", "finbert", "textblob", "lm_score"]:
        if sc in feat_df.columns and sc not in cols:
            cols.append(sc)

    y = feat_df[target].dropna()
    X = feat_df[cols].reindex(y.index)

    n = len(y)
    half = n // 2
    rows = []
    for col in cols:
        s = X[col].dropna()
        common = s.index.intersection(y.index)
        rho_all,  _ = stats.spearmanr(s.reindex(common), y.reindex(common))
        rho_h1,   _ = stats.spearmanr(
            s.reindex(common[:half]), y.reindex(common[:half]))
        rho_h2,   _ = stats.spearmanr(
            s.reindex(common[half:]), y.reindex(common[half:]))
        rows.append(dict(
            feature=col, rho=rho_all,
            rho_h1=rho_h1, rho_h2=rho_h2,
            category=_cat(col),
        ))

    df_corr = (
        pd.DataFrame(rows)
        .sort_values("rho", key=abs, ascending=True)
    )

    cat_colors = {
        "Volatility": "#3498db",
        "Lagged Return": "#2ecc71",
        "Technical": "#f39c12",
        "Sentiment": "#e74c3c",
    }
    bar_colors = [cat_colors.get(c, "gray") for c in df_corr["category"]]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Full-sample Spearman rho", "First-half vs Second-half rho"],
        horizontal_spacing=0.12,
    )

    fig.add_trace(go.Bar(
        y=df_corr["feature"], x=df_corr["rho"],
        orientation="h",
        marker_color=bar_colors,
        name="Spearman rho",
    ), row=1, col=1)
    fig.add_vline(x=0, line_color="black", row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df_corr["rho_h1"], y=df_corr["feature"],
        mode="markers", name="First half",
        marker=dict(symbol="circle", size=8, color="#8e44ad"),
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=df_corr["rho_h2"], y=df_corr["feature"],
        mode="markers", name="Second half",
        marker=dict(symbol="diamond", size=8, color="#e67e22"),
    ), row=1, col=2)
    for _, row in df_corr.iterrows():
        fig.add_shape(
            type="line",
            x0=row["rho_h1"], x1=row["rho_h2"],
            y0=row["feature"], y1=row["feature"],
            line=dict(color="gray", width=1), row=1, col=2,
        )

    fig.update_layout(
        title="EDA7: Feature-Target Spearman Correlation (sorted by |rho|)",
        height=max(400, 18 * len(df_corr)),
        showlegend=True,
    )
    fig.update_xaxes(title_text="Spearman rho", row=1, col=1)
    fig.update_xaxes(title_text="Spearman rho", row=1, col=2)

    return fig


# ============================================================================ #
# EDA8 -- JOINT DISTRIBUTION: SENTIMENT vs VOLATILITY
# ============================================================================ #

def plot_sentiment_vol_joint(
    df: pd.DataFrame,
    sent: pd.DataFrame,
    sent_col: str = "vader_compound",
    vol_col: str = "realized_vol_21d",
) -> go.Figure:
    """
    2D density (hexbin) of VADER compound score vs next-day realized
    volatility, with marginal histograms and a LOWESS regression line.

    Why it matters: This chart tests whether the sentiment-volatility
    relationship is linear or U-shaped. A U-shaped pattern -- where both
    extreme positive and extreme negative sentiment predict high vol --
    would suggest using |sentiment| as a feature instead of raw sentiment.
    A strictly monotonic negative relationship confirms the directional
    signal.

    What to look for: If high-vol days cluster at both ends of the sentiment
    axis (extreme positive AND extreme negative), the feature transformation
    should be abs(VADER). If the cluster is only at extreme negative values,
    raw VADER is the right feature.
    """
    combined = df[[vol_col]].join(sent[[sent_col]], how="inner").dropna()
    x = combined[sent_col].values
    y = combined[vol_col].values

    smooth = lowess(y, x, frac=0.3, return_sorted=True)

    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[0.82, 0.18],
        row_heights=[0.18, 0.82],
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.02,
        vertical_spacing=0.02,
    )

    # Hexbin density (simulated with scatter + color)
    fig.add_trace(go.Histogram2dContour(
        x=x, y=y,
        colorscale="Blues",
        contours_coloring="fill",
        line_width=0,
        showscale=False,
        name="Density",
    ), row=2, col=1)

    # LOWESS line
    fig.add_trace(go.Scatter(
        x=smooth[:, 0], y=smooth[:, 1],
        mode="lines", name="LOWESS",
        line=dict(color="#e74c3c", width=2.5),
    ), row=2, col=1)

    # Quadrant annotations
    x_med, y_med = np.median(x), np.median(y)
    for qx, qy, text in [
        (x.min() * 0.7, y.max() * 0.9, "High |Sent| + High Vol"),
        (x.max() * 0.7, y.max() * 0.9, "Pos Sent + High Vol"),
    ]:
        fig.add_annotation(x=qx, y=qy, text=text, showarrow=False,
                           font=dict(size=9, color="gray"), row=2, col=1)

    # Marginal histograms
    fig.add_trace(go.Histogram(
        x=x, nbinsx=40, marker_color="#3498db", opacity=0.7, showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Histogram(
        y=y, nbinsx=40, marker_color="#2ecc71", opacity=0.7, showlegend=False,
    ), row=2, col=2)

    fig.update_layout(
        title=f"EDA8: Joint Distribution -- {sent_col} vs {vol_col}",
        height=520, showlegend=False,
    )
    fig.update_xaxes(title_text=f"VADER Compound Score", row=2, col=1)
    fig.update_yaxes(title_text="Realized Vol (21d)", row=2, col=1)

    return fig


# ============================================================================ #
# EDA9 -- REGIME TRANSITION MATRIX
# ============================================================================ #

def plot_regime_transitions(df: pd.DataFrame) -> tuple[plt.Figure, go.Figure]:
    """
    Heatmap of weekly volatility regime transition probabilities
    (Low -> Medium -> High) plus a bar chart of regime frequency.

    Why it matters: High diagonal values mean regimes are persistent
    (volatility clustering -- the core stylized fact behind GARCH). Off-diagonal
    values quantify how often regimes shift in a single week. A model that
    can only make in-regime predictions will fail precisely when the off-diagonal
    transitions occur -- these are the most important and hardest days.

    What to look for: For MU (a cyclical semiconductor stock), the High-vol
    regime should be more persistent than for a blue-chip like AAPL. The
    Low->High transition probability reflects tail risk. If Low->High probability
    is non-trivial, it means quiet periods can rapidly become volatile -- a key
    risk management insight.
    """
    from src.data_helpers import add_vol_regime

    data = add_vol_regime(df)[["vol_regime"]].dropna()
    data["weekday"] = data.index.dayofweek
    # Use weekly resampled regime (mode of the week)
    weekly = data["vol_regime"].resample("W").agg(
        lambda x: x.mode()[0] if len(x) > 0 else np.nan
    ).dropna()

    regimes = ["Low", "Medium", "High"]
    trans = pd.DataFrame(0, index=regimes, columns=regimes)
    for i in range(len(weekly) - 1):
        fr = str(weekly.iloc[i])
        to = str(weekly.iloc[i + 1])
        if fr in regimes and to in regimes:
            trans.loc[fr, to] += 1

    prob = trans.div(trans.sum(axis=1), axis=0).fillna(0)
    freq = {r: (weekly == r).mean() for r in regimes}

    # Matplotlib heatmap
    mfig, axes = plt.subplots(1, 2, figsize=(12, 4),
                              gridspec_kw={"width_ratios": [2, 1]})
    sns.heatmap(
        prob, annot=True, fmt=".2f", cmap="YlOrRd",
        linewidths=0.5, ax=axes[0],
        cbar_kws={"label": "Transition Probability"},
    )
    axes[0].set_title("EDA9a: Weekly Vol Regime Transition Matrix")
    axes[0].set_xlabel("To Regime")
    axes[0].set_ylabel("From Regime")

    axes[1].bar(regimes, [freq[r] for r in regimes],
                color=["#3498db", "#f39c12", "#e74c3c"], alpha=0.8)
    axes[1].set_title("EDA9b: Regime Frequency")
    axes[1].set_ylabel("Proportion of Weeks")
    for i, r in enumerate(regimes):
        axes[1].text(i, freq[r] + 0.005, f"{freq[r]:.1%}",
                     ha="center", fontsize=9)
    plt.tight_layout()

    # Plotly version
    fig2 = px.imshow(
        prob,
        color_continuous_scale="YlOrRd",
        text_auto=".2f",
        title="EDA9: Vol Regime Transition Probability Matrix",
        labels=dict(x="To Regime", y="From Regime", color="Probability"),
    )
    fig2.update_layout(height=350)

    return mfig, fig2


# ============================================================================ #
# EDA10 -- NEWS VOLUME vs VOLATILITY
# ============================================================================ #

def plot_news_volume_vs_vol(
    df: pd.DataFrame,
    sent: pd.DataFrame,
    earnings_dates: pd.DatetimeIndex | None = None,
) -> go.Figure:
    """
    Dual-axis time series: daily news article count (left) and realized
    volatility (right), both with 5-day rolling averages. Earnings dates
    are highlighted as vertical bands.

    Why it matters: News volume is an attention signal -- high coverage days
    often mark regime transitions because they represent a burst of new
    information that the market must price in. A lead-lag analysis can
    reveal whether news volume rises before or alongside vol spikes, which
    informs whether to include news_count as a leading or contemporaneous
    feature.

    What to look for: Coincident spikes in news count and realized vol
    around known events (earnings, product launches, macro shocks) validate
    the news count signal. Periods where news count is high but vol is low
    may indicate noise (non-market-moving coverage). The 5-day smoothing
    filters out one-off spikes.
    """
    combined = df[["realized_vol_21d"]].join(
        sent[["news_count"]], how="inner"
    ).dropna()

    vol_smooth   = combined["realized_vol_21d"].rolling(5).mean()
    count_smooth = combined["news_count"].rolling(5).mean()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=combined.index, y=combined["news_count"],
        mode="lines", name="News Count (raw)",
        line=dict(color="#bdc3c7", width=1), opacity=0.4,
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=combined.index, y=count_smooth,
        mode="lines", name="News Count (5d avg)",
        line=dict(color="#3498db", width=2),
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=combined.index, y=vol_smooth,
        mode="lines", name="Realized Vol (5d avg)",
        line=dict(color="#e74c3c", width=2),
    ), secondary_y=True)

    if earnings_dates is not None:
        for ed in earnings_dates:
            ed_ts = pd.Timestamp(ed)
            if combined.index.min() <= ed_ts <= combined.index.max():
                fig.add_vrect(
                    x0=str((ed_ts - pd.Timedelta(days=2)).date()),
                    x1=str((ed_ts + pd.Timedelta(days=2)).date()),
                    fillcolor="rgba(241,196,15,0.15)",
                    layer="below", line_width=0,
                )

    fig.update_layout(
        title="EDA10: News Volume vs Realized Volatility (yellow bands = earnings)",
        height=430,
        legend=dict(orientation="h", y=1.05),
    )
    fig.update_yaxes(title_text="Daily News Article Count", secondary_y=False)
    fig.update_yaxes(title_text="Realized Vol (21d, annualized)", secondary_y=True)

    return fig
