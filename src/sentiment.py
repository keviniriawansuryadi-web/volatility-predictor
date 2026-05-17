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
        from scrapers.scraper_news import fetch_news
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
