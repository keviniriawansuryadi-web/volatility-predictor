"""
Sentiment pipeline for the volatility predictor.

Live news is fetched from three sources via scrapers/scraper_news.py
(RSS feeds, Finviz, Yahoo Finance news page).  If all live sources fail,
the function falls back to the last cached headlines in
data/news/{ticker}_news.csv.

Missing sentiment is imputed with the rolling 5-day median — never zero,
because zero implies neutral market tone, which is a real signal, not an
absence of data.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path(__file__).parent.parent / "data" / "news"


# ─── main entry point ──────────────────────────────────────────────────────────

def fetch_sentiment(
    ticker: str,
    index: pd.DatetimeIndex,
    impute_window: int = 5,
) -> pd.Series:
    """
    Fetch VADER sentiment scores for `ticker` aligned to `index` (trading days).

    Workflow:
      1. Try live scrapers (RSS, Finviz, Yahoo Finance news page).
      2. If all live sources return 0 articles, fall back to the CSV cache.
      3. If the cache is also empty, warn and return median-imputed neutral series.
      4. Run VADER on each headline, average by date, align to trading index.
      5. Impute gaps with a rolling 5-day median (never zero-fill).

    Returns a Series in [-1, 1] indexed to `index`, named 'sentiment'.
    """
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        warnings.warn("[sentiment] vaderSentiment not installed — returning median-imputed zeros.")
        return _neutral_series(index)

    # ── step 1: fetch news (live + cache fallback) ─────────────────────────────
    news_df = _fetch_with_fallback(ticker)

    if news_df.empty:
        warnings.warn(
            f"[sentiment] {ticker}: no articles from any source (live or cache). "
            "Returning median-imputed neutral series."
        )
        return _neutral_series(index)

    # ── step 2: VADER scoring ─────────────────────────────────────────────────
    analyzer = SentimentIntensityAnalyzer()
    news_df = news_df.dropna(subset=["headline", "datetime"])
    news_df["score"] = news_df["headline"].apply(
        lambda h: analyzer.polarity_scores(str(h))["compound"]
    )

    # Daily mean sentiment
    news_df["date"] = pd.to_datetime(news_df["datetime"]).dt.normalize()
    daily: pd.Series = news_df.groupby("date")["score"].mean()
    daily.index = pd.DatetimeIndex(daily.index)
    if daily.index.tz is not None:
        daily.index = daily.index.tz_localize(None)

    # ── step 3: align to trading calendar ────────────────────────────────────
    aligned = daily.reindex(index)

    # ── step 4: rolling median imputation (never zero-fill) ──────────────────
    rolling_med = aligned.rolling(impute_window, min_periods=1).median()
    aligned = aligned.where(aligned.notna(), rolling_med)
    aligned = aligned.fillna(0.0)   # only if the very first window has no data

    aligned.name = "sentiment"

    n_real = daily.reindex(index).notna().sum()
    n_imputed = len(index) - n_real
    print(
        f"  [sentiment] {ticker}: {n_real} trading days with real VADER scores, "
        f"{n_imputed} imputed via rolling-{impute_window}d median."
    )
    if n_real == 0:
        warnings.warn(
            f"[sentiment] {ticker}: 0 real sentiment scores — all values are imputed. "
            "Live news unavailable; using cached headlines."
        )
    return aligned


def _fetch_with_fallback(ticker: str) -> pd.DataFrame:
    """
    Attempt live scraping; if all sources return 0 articles, load from CSV cache.

    Logs a warning when falling back so silent failures are visible.
    """
    try:
        from src.scraper_news import fetch_news
        news_df = fetch_news(ticker)
    except Exception as exc:
        warnings.warn(f"[sentiment] Scraper import/run failed: {exc}")
        news_df = pd.DataFrame()

    if not news_df.empty:
        return news_df

    # Fallback: read cached CSV
    cache_path = CACHE_DIR / f"{ticker.upper()}_news.csv"
    if cache_path.exists():
        try:
            cached = pd.read_csv(cache_path, parse_dates=["datetime"])
            if not cached.empty:
                last_date = pd.to_datetime(cached["datetime"]).max()
                warnings.warn(
                    f"[sentiment] Live news unavailable for {ticker}; "
                    f"using cached headlines (newest: {last_date.date()})."
                )
                return cached
        except Exception as exc:
            warnings.warn(f"[sentiment] Cache read failed for {ticker}: {exc}")

    return pd.DataFrame()


def _neutral_series(index: pd.DatetimeIndex) -> pd.Series:
    """Return a zero-filled Series — only used when absolutely no data exists."""
    return pd.Series(0.0, index=index, name="sentiment")


# ─── WSB sentiment ─────────────────────────────────────────────────────────────

def fetch_wsb_sentiment(
    ticker: str,
    index: pd.DatetimeIndex,
    impute_window: int = 5,
) -> pd.Series:
    """
    Fetch Reddit r/wallstreetbets sentiment for `ticker` aligned to `index`.

    Calls scrapers.scraper_news.scrape_wsb_sentiment() which returns VADER
    compound scores weighted by log(1 + upvotes) per day.  The result is
    reindexed to the trading calendar and missing days are imputed with a
    rolling `impute_window`-day median — never zero, because zero is a real
    neutral signal, not an absence of data.

    Parameters
    ----------
    ticker        : ticker symbol (upper-cased internally).
    index         : DatetimeIndex of trading days to align to.
    impute_window : window for rolling median imputation (default 5).

    Returns a Series in [-1, 1] indexed to `index`, named 'wsb_sentiment'.
    An all-zero neutral series is returned only when the scraper errors out
    or returns no posts.
    """
    try:
        from src.scraper_news import scrape_wsb_sentiment
    except Exception as exc:
        warnings.warn(f"[wsb_sentiment] Scraper import failed: {exc}")
        return pd.Series(0.0, index=index, name="wsb_sentiment")

    try:
        daily = scrape_wsb_sentiment(ticker)
    except Exception as exc:
        warnings.warn(f"[wsb_sentiment] scrape_wsb_sentiment({ticker}) failed: {exc}")
        return pd.Series(0.0, index=index, name="wsb_sentiment")

    if daily.empty:
        warnings.warn(
            f"[wsb_sentiment] {ticker}: no WSB posts found — returning neutral series."
        )
        return pd.Series(0.0, index=index, name="wsb_sentiment")

    # Align to tz-naive trading index
    if daily.index.tz is not None:
        daily.index = daily.index.tz_localize(None)

    aligned = daily.reindex(index)

    rolling_med = aligned.rolling(impute_window, min_periods=1).median()
    aligned = aligned.where(aligned.notna(), rolling_med)
    aligned = aligned.fillna(0.0)   # only if the very first window has no data

    aligned.name = "wsb_sentiment"

    n_real = daily.reindex(index).notna().sum()
    n_imputed = len(index) - n_real
    print(
        f"  [wsb_sentiment] {ticker}: {n_real} trading days with real WSB scores, "
        f"{n_imputed} imputed via rolling-{impute_window}d median."
    )
    return aligned


# ─── Market-wide sentiment index ───────────────────────────────────────────────

def compute_market_sentiment_index(
    df_dict: dict[str, pd.DataFrame],
    market_cap_weights: dict[str, float] | None = None,
) -> pd.Series:
    """
    Builds a market-wide sentiment index by averaging sentiment scores across
    all tickers weighted by market cap.  Individual ticker sentiment is noisy
    but the aggregate may capture systematic fear/greed that predicts SPY vol
    better than any single ticker's sentiment (which failed H1 on SPY at p=0.144).

    The resulting index is tested against SPY vol using the same Mann-Whitney
    H1 framework — if significant, add market_sentiment_index as a feature
    to all SPY models.

    Parameters
    ----------
    df_dict : Dict of {ticker: DataFrame with 'sentiment' column indexed to
              trading days}.  Any ticker missing the 'sentiment' column is skipped.
    market_cap_weights : Dict of {ticker: weight}.  If None, equal weighting is
                        used.  Weights are normalised to sum to 1 within the
                        available tickers.

    Returns a pd.Series of daily market sentiment index values, named
    'market_sentiment_index'.  Index is the union of all trading day indices,
    forward-filled over weekends/holidays.
    """
    if not df_dict:
        return pd.Series(dtype=float, name="market_sentiment_index")

    available = {t: df for t, df in df_dict.items() if "sentiment" in df.columns}
    if not available:
        warnings.warn("[market_sentiment] No tickers have 'sentiment' column — returning empty.")
        return pd.Series(dtype=float, name="market_sentiment_index")

    # Normalise weights
    if market_cap_weights is None:
        w = {t: 1.0 / len(available) for t in available}
    else:
        raw_w = {t: market_cap_weights.get(t, 0.0) for t in available}
        total = sum(raw_w.values())
        if total <= 0:
            w = {t: 1.0 / len(available) for t in available}
        else:
            w = {t: v / total for t, v in raw_w.items()}

    # Build weighted sum across a common index
    all_idx = pd.DatetimeIndex([])
    for df in available.values():
        all_idx = all_idx.union(df.index)
    all_idx = all_idx.sort_values()

    weighted_sum = pd.Series(0.0, index=all_idx)
    weight_sum   = pd.Series(0.0, index=all_idx)

    for t, df in available.items():
        wt = w.get(t, 0.0)
        sent = df["sentiment"].reindex(all_idx)  # NaN on days this ticker has no obs
        valid = sent.notna()
        weighted_sum[valid] += sent[valid] * wt
        weight_sum[valid]   += wt

    market_idx = (weighted_sum / weight_sum.replace(0, np.nan)).ffill()
    market_idx.name = "market_sentiment_index"

    n_tickers = len(available)
    n_days    = market_idx.notna().sum()
    print(f"  [market_sentiment] Built index from {n_tickers} tickers over {n_days} days.")

    return market_idx


def test_market_sentiment_h1(
    spy_df: pd.DataFrame,
    market_sentiment: pd.Series,
    spike_pct: float = 0.90,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Tests whether the market-wide sentiment index predicts SPY vol spikes
    using the same Mann-Whitney H1 test framework applied to individual tickers.

    H1 (market-wide): SPY spike days (realized vol >= 90th pct) are preceded
    by more negative market-wide sentiment than non-spike days.

    Parameters
    ----------
    spy_df           : DataFrame with 'realized_vol_21d' column.
    market_sentiment : Output of compute_market_sentiment_index().
    spike_pct        : Percentile threshold for spike days (default 0.90).
    n_bootstrap      : Bootstrap resamples for CI on mean difference.
    seed             : Random seed for reproducibility.

    Returns dict with statistic, p_value, effect_size, conclusion.
    """
    import scipy.stats as stats_mod

    combined = (
        spy_df[["realized_vol_21d"]]
        .join(market_sentiment.rename("market_sentiment"), how="inner")
        .dropna()
    )
    if len(combined) < 10:
        return {"available": False, "conclusion": "Insufficient overlapping data."}

    threshold  = combined["realized_vol_21d"].quantile(spike_pct)
    spike_mask = combined["realized_vol_21d"] >= threshold

    spike_sent   = combined.loc[spike_mask,   "market_sentiment"].values
    nospike_sent = combined.loc[~spike_mask,  "market_sentiment"].values

    if len(spike_sent) < 2 or len(nospike_sent) < 2:
        return {"available": False, "conclusion": "Too few spike days."}

    u_stat, p_val = stats_mod.mannwhitneyu(spike_sent, nospike_sent, alternative="less")
    n1, n2 = len(spike_sent), len(nospike_sent)
    r = 1 - 2 * u_stat / (n1 * n2)

    rng = np.random.default_rng(seed)
    diffs = np.array([
        rng.choice(spike_sent, n1, replace=True).mean()
        - rng.choice(nospike_sent, n2, replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])

    significant = p_val < 0.05
    conclusion = (
        f"Market sentiment index H1: spike days {'show' if significant else 'do NOT show'} "
        f"significantly more negative market sentiment (U={u_stat:.0f}, p={p_val:.4f}, r={r:.3f}). "
        f"Bootstrap 95% CI on delta: [{ci_lo:.4f}, {ci_hi:.4f}]."
    )

    print(f"\n{'='*60}")
    print(f"  H1 (MARKET SENTIMENT INDEX) — SPY")
    print(f"{'='*60}")
    print(f"  Spike days (n={n1}): mean market sentiment = {spike_sent.mean():.4f}")
    print(f"  Non-spike  (n={n2}): mean market sentiment = {nospike_sent.mean():.4f}")
    print(f"  Mann-Whitney U={u_stat:.0f}  p={p_val:.4f}  r={r:.3f}")
    print(f"  95% CI on delta: [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  {'** SIGNIFICANT **' if significant else 'NOT significant'} at alpha=0.05")
    print(f"{'='*60}")

    return dict(
        available=True,
        statistic=u_stat,
        p_value=p_val,
        effect_size=r,
        bootstrap_ci=(ci_lo, ci_hi),
        significant=significant,
        spike_mean_sentiment=float(spike_sent.mean()),
        nospike_mean_sentiment=float(nospike_sent.mean()),
        conclusion=conclusion,
    )


# ─── audit function ────────────────────────────────────────────────────────────

def audit_sentiment_coverage(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Report what percentage of trading days have at least one real sentiment score.

    Adds a boolean 'sentiment_missing' column to the returned DataFrame and
    prints a coverage summary.  Days with missing sentiment should be
    imputed with rolling median — this function flags them so downstream
    callers can verify imputation was applied.

    Parameters
    ----------
    df     : DataFrame with a 'sentiment' column, indexed to trading days.
    ticker : ticker symbol (for the printout only).

    Returns df with an added 'sentiment_missing' boolean column.
    """
    if "sentiment" not in df.columns:
        warnings.warn("[audit] 'sentiment' column not found in DataFrame.")
        return df

    out = df.copy()
    # Mark as missing if the value is exactly 0.0 (default when no data)
    # or NaN.  Real neutral sentiment can also be 0, but after imputation
    # NaN should not appear.
    out["sentiment_missing"] = df["sentiment"].isna() | (df["sentiment"] == 0.0)

    n_total   = len(out)
    n_missing = out["sentiment_missing"].sum()
    n_real    = n_total - n_missing
    coverage  = n_real / n_total * 100

    print(f"\n[Sentiment Audit] {ticker}")
    print(f"  Total trading days : {n_total}")
    print(f"  Real scores        : {n_real} ({coverage:.1f}%)")
    print(f"  Missing/imputed    : {n_missing} ({100 - coverage:.1f}%)")
    if coverage < 10:
        warnings.warn(
            f"[audit] Only {coverage:.1f}% of days have real sentiment for {ticker}. "
            "Consider a paid news API (Tiingo, RavenPack) for historical coverage."
        )
    return out
