"""
Multi-source financial news scraper — no API keys required.

Sources
-------
A  RSS feeds   : CNBC Markets, Reuters Business, Seeking Alpha (ticker-specific)
B  Finviz      : HTML news table scraped with BeautifulSoup
C  Yahoo Finance news page: static HTML (JavaScript-light content)

All sources are merged, deduplicated on normalised headline text, and cached to
data/news/{ticker}_news.csv.  The Reddit WSB scraper lives at the bottom of this
file (see scrape_wsb_sentiment).
"""

from __future__ import annotations

import warnings
import re
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import requests

try:
    import feedparser
    _HAS_FEEDPARSER = True
except ImportError:
    _HAS_FEEDPARSER = False

try:
    from bs4 import BeautifulSoup
    _HAS_BS4 = True
except ImportError:
    _HAS_BS4 = False

CACHE_DIR    = Path(__file__).parent.parent / "data" / "news"
_HEADERS     = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
_WSB_HEADERS = {"User-Agent": "volatility-research/1.0"}
_TIMEOUT     = 12


# ─── internal helpers ──────────────────────────────────────────────────────────

def _to_df(records: list[dict], source: str, ticker: str) -> pd.DataFrame:
    """Normalise a list of {headline, datetime} dicts into the standard schema."""
    if not records:
        return pd.DataFrame(columns=["headline", "source", "datetime", "ticker"])
    df = pd.DataFrame(records)
    df["source"] = source
    df["ticker"] = ticker.upper()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df["datetime"] = df["datetime"].dt.tz_localize(None)
    return df[["headline", "source", "datetime", "ticker"]].dropna(subset=["headline"])


def _norm_key(text: str) -> str:
    """Lowercase + strip punctuation — used for deduplication."""
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


# ─── Source A: RSS feeds ───────────────────────────────────────────────────────

_RSS_GENERAL = {
    "cnbc_markets":     "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "reuters_business": "https://feeds.reuters.com/reuters/businessNews",
}


def scrape_rss_general(ticker: str) -> pd.DataFrame:
    """
    Fetch CNBC and Reuters RSS feeds and keep entries that mention the ticker.

    General market feeds are filtered by ticker symbol in the headline.
    Returns a DataFrame with columns: headline, source, datetime, ticker.
    """
    if not _HAS_FEEDPARSER:
        warnings.warn("feedparser not installed; RSS source disabled. pip install feedparser")
        return pd.DataFrame()

    ticker_upper = ticker.upper()
    records: list[dict] = []

    for name, url in _RSS_GENERAL.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                title = getattr(entry, "title", "") or ""
                if ticker_upper not in title.upper():
                    continue
                parsed = getattr(entry, "published_parsed", None)
                dt = datetime(*parsed[:6]) if parsed else datetime.utcnow()
                records.append({"headline": title, "datetime": dt})
        except Exception as exc:
            warnings.warn(f"[RSS] {name} failed: {exc}")

    return _to_df(records, "rss_general", ticker)


def scrape_rss_seeking_alpha(ticker: str) -> pd.DataFrame:
    """
    Fetch ticker-specific RSS from Seeking Alpha.

    URL pattern: https://seekingalpha.com/api/sa/combined/{TICKER}.xml
    Returns all entries — they are already ticker-filtered by the feed.
    """
    if not _HAS_FEEDPARSER:
        return pd.DataFrame()

    url = f"https://seekingalpha.com/api/sa/combined/{ticker.upper()}.xml"
    records: list[dict] = []
    try:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            title = getattr(entry, "title", "") or ""
            parsed = getattr(entry, "published_parsed", None)
            dt = datetime(*parsed[:6]) if parsed else datetime.utcnow()
            records.append({"headline": title, "datetime": dt})
    except Exception as exc:
        warnings.warn(f"[RSS] Seeking Alpha {ticker} failed: {exc}")

    return _to_df(records, "seeking_alpha", ticker)


# ─── Source B: Finviz ─────────────────────────────────────────────────────────

def scrape_finviz(ticker: str) -> pd.DataFrame:
    """
    Scrape the Finviz news table for a ticker.

    URL: https://finviz.com/quote.ashx?t={ticker}
    Target element: table#news-table tr
    Datetime is partially inlined (date shown once per day, time on every row).
    Returns a DataFrame with columns: headline, source, datetime, ticker.
    """
    if not _HAS_BS4:
        warnings.warn("beautifulsoup4 not installed; Finviz source disabled. pip install beautifulsoup4")
        return pd.DataFrame()

    url = f"https://finviz.com/quote.ashx?t={ticker.upper()}"
    records: list[dict] = []

    try:
        resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table", id="news-table")
        if not table:
            return pd.DataFrame()

        last_date: datetime | None = None
        for row in table.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) < 2:
                continue
            time_text = cells[0].get_text(strip=True)
            headline  = cells[1].get_text(strip=True)

            # Finviz format: "May-18-26 08:30AM" (new date) or "08:30AM" (same day)
            try:
                if len(time_text) > 8:
                    last_date = datetime.strptime(time_text, "%b-%d-%y %I:%M%p")
                elif last_date is not None:
                    t = datetime.strptime(time_text, "%I:%M%p")
                    last_date = last_date.replace(hour=t.hour, minute=t.minute)
            except ValueError:
                pass

            if last_date and headline:
                records.append({"headline": headline, "datetime": last_date})

    except Exception as exc:
        warnings.warn(f"[Finviz] {ticker} failed: {exc}")

    return _to_df(records, "finviz", ticker)


# ─── Source C: Yahoo Finance news page ────────────────────────────────────────

def scrape_yahoo_finance(ticker: str) -> pd.DataFrame:
    """
    Scrape headlines from the Yahoo Finance news page using static HTML.

    URL: https://finance.yahoo.com/quote/{ticker}/news
    Note: Yahoo Finance is JavaScript-heavy; this static approach captures
    server-side-rendered headlines only (typically 10–20 articles).
    Timestamps default to utcnow() because they are not present in static HTML.
    """
    if not _HAS_BS4:
        return pd.DataFrame()

    url = f"https://finance.yahoo.com/quote/{ticker.upper()}/news"
    records: list[dict] = []

    try:
        resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        seen: set[str] = set()
        for tag in soup.find_all(["h3", "h4"], limit=60):
            text = tag.get_text(strip=True)
            key  = _norm_key(text)
            if len(text) > 20 and key not in seen:
                seen.add(key)
                records.append({"headline": text, "datetime": datetime.utcnow()})

    except Exception as exc:
        warnings.warn(f"[Yahoo news page] {ticker} failed: {exc}")

    return _to_df(records, "yahoo_news_page", ticker)


# ─── Merge, deduplicate, persist ──────────────────────────────────────────────

def fetch_news(ticker: str, max_age_days: int = 30) -> pd.DataFrame:
    """
    Collect news from all three sources, merge, deduplicate, and save to cache.

    Pipeline:
      1. Query RSS (CNBC, Reuters, Seeking Alpha), Finviz, and Yahoo Finance.
      2. Concatenate all results.
      3. Deduplicate on normalised headline text.
      4. Drop rows older than max_age_days.
      5. Merge with existing cache, re-deduplicate, and overwrite cache.

    Returns a DataFrame with columns: headline, source, datetime, ticker.
    Empty DataFrame if all sources fail.
    """
    frames = [
        scrape_rss_general(ticker),
        scrape_rss_seeking_alpha(ticker),
        scrape_finviz(ticker),
        scrape_yahoo_finance(ticker),
    ]
    new_df = pd.concat([f for f in frames if not f.empty], ignore_index=True)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{ticker.upper()}_news.csv"

    if new_df.empty:
        warnings.warn(f"[fetch_news] All live sources returned 0 articles for {ticker}.")
        if cache_path.exists():
            cached = pd.read_csv(cache_path, parse_dates=["datetime"])
            return cached
        return new_df

    # Dedup new batch
    new_df["_key"] = new_df["headline"].apply(_norm_key)
    new_df = new_df.drop_duplicates(subset="_key")

    # Filter by age
    cutoff = datetime.utcnow() - timedelta(days=max_age_days)
    new_df = new_df[new_df["datetime"] >= cutoff]

    # Merge with cache
    if cache_path.exists():
        cached = pd.read_csv(cache_path, parse_dates=["datetime"])
        combined = pd.concat([cached, new_df.drop(columns="_key")], ignore_index=True)
        combined["_key"] = combined["headline"].apply(_norm_key)
        combined = combined.drop_duplicates(subset="_key").drop(columns="_key")
    else:
        combined = new_df.drop(columns="_key")

    combined.to_csv(cache_path, index=False)
    return combined


def check_scraper_health(ticker: str) -> dict[str, int]:
    """
    Test all three news sources and report article counts.

    Prints a warning for any source that returns zero articles so that
    silent failures are caught immediately rather than silently defaulting
    to zero sentiment.

    Returns a dict mapping source name → article count.
    """
    results = {
        "rss_general":     len(scrape_rss_general(ticker)),
        "seeking_alpha":   len(scrape_rss_seeking_alpha(ticker)),
        "finviz":          len(scrape_finviz(ticker)),
        "yahoo_news_page": len(scrape_yahoo_finance(ticker)),
    }
    total = sum(results.values())
    print(f"\n[Scraper Health] {ticker}")
    print(f"{'─'*40}")
    for source, count in results.items():
        flag = "OK" if count > 0 else "WARNING: 0 articles"
        print(f"  {source:<20} {count:3d} articles  [{flag}]")
    print(f"  {'Total':<20} {total:3d} articles")
    if total == 0:
        warnings.warn(f"[check_scraper_health] All sources returned 0 for {ticker}.")
    return results


# ─── Reddit WSB sentiment ─────────────────────────────────────────────────────

def scrape_wsb_sentiment(ticker: str, limit: int = 100, time_filter: str = "week") -> pd.Series:
    """
    Fetch recent r/wallstreetbets posts mentioning the ticker and compute a
    VADER sentiment score weighted by log(1 + upvotes).

    URL: https://www.reddit.com/r/wallstreetbets/search.json
         ?q={ticker}&sort=new&limit={limit}&t={time_filter}

    Each post's VADER compound score is multiplied by log(1 + score) to
    give more weight to popular posts.  Scores are grouped by UTC date and
    the daily weighted mean is returned as a pd.Series indexed by date.

    Returns an empty Series if the request fails or no posts are found.
    """
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        warnings.warn("[WSB] vaderSentiment not installed — skipping WSB sentiment.")
        return pd.Series(dtype=float, name="wsb_sentiment")

    url = (
        f"https://www.reddit.com/r/wallstreetbets/search.json"
        f"?q={ticker.upper()}&sort=new&limit={limit}&t={time_filter}&restrict_sr=1"
    )
    try:
        resp = requests.get(url, headers=_WSB_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        warnings.warn(f"[WSB] Reddit fetch for {ticker} failed: {exc}")
        return pd.Series(dtype=float, name="wsb_sentiment")

    posts = data.get("data", {}).get("children", [])
    if not posts:
        return pd.Series(dtype=float, name="wsb_sentiment")

    analyzer = SentimentIntensityAnalyzer()
    records: list[dict] = []

    for post in posts:
        d = post.get("data", {})
        title    = d.get("title", "")
        upvotes  = max(int(d.get("score", 0)), 0)
        created  = d.get("created_utc", 0)
        date     = pd.Timestamp(created, unit="s").normalize()

        if not title:
            continue

        compound = analyzer.polarity_scores(title)["compound"]
        weight   = np.log1p(upvotes)        # log(1 + upvotes)
        records.append({"date": date, "sentiment": compound, "weight": weight})

    if not records:
        return pd.Series(dtype=float, name="wsb_sentiment")

    df = pd.DataFrame(records)

    # Weighted average per day:  sum(sentiment * weight) / sum(weight)
    daily = (
        df.groupby("date")
        .apply(lambda g: (
            np.average(g["sentiment"], weights=g["weight"])
            if g["weight"].sum() > 0
            else g["sentiment"].mean()
        ))
        .rename("wsb_sentiment")
    )
    daily.index = pd.DatetimeIndex(daily.index)
    n_posts = len(records)
    n_days  = len(daily)
    print(f"  [WSB] {ticker}: {n_posts} posts over {n_days} days fetched from r/wallstreetbets.")
    return daily
