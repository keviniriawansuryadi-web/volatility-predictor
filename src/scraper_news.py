"""
Multi-source financial news scraper — no API keys required.

Sources
-------
A  RSS feeds (general)  : CNBC (5 feeds), Reuters, MarketWatch
B  RSS feeds (ticker)   : Yahoo Finance RSS, Google News RSS, Seeking Alpha
C  HTML scrapers        : Finviz, Yahoo Finance page, Benzinga
D  Macro events         : Forex Factory economic calendar (faireconomy.media JSON)

All sources are merged, deduplicated on normalised headline text, and cached to
data/news/{ticker}_news.csv.  The cache is purely additive — old articles are
never deleted, so the historical window grows with every run.
"""

from __future__ import annotations

import json
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
_HEADERS     = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
_WSB_HEADERS = {"User-Agent": "volatility-research/1.0"}
_TIMEOUT     = 15


# ─── internal helpers ──────────────────────────────────────────────────────────

def _to_df(records: list[dict], source: str, ticker: str) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=["headline", "source", "datetime", "ticker"])
    df = pd.DataFrame(records)
    df["source"] = source
    df["ticker"] = ticker.upper()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df["datetime"] = df["datetime"].dt.tz_localize(None)
    return df[["headline", "source", "datetime", "ticker"]].dropna(subset=["headline"])


def _norm_key(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def _parse_feed(url: str, name: str) -> list[dict]:
    """Fetch and parse a single RSS/Atom feed. Returns list of {headline, datetime}."""
    records: list[dict] = []
    try:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            title = getattr(entry, "title", "") or ""
            parsed = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
            dt = datetime(*parsed[:6]) if parsed else datetime.utcnow()
            if title:
                records.append({"headline": title, "datetime": dt})
    except Exception as exc:
        warnings.warn(f"[RSS] {name} failed: {exc}")
    return records


# ─── Source A: General RSS feeds (filter by ticker mention) ───────────────────

_RSS_GENERAL = {
    "cnbc_markets":    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "cnbc_finance":    "https://www.cnbc.com/id/10001147/device/rss/rss.html",
    "cnbc_tech":       "https://www.cnbc.com/id/19854910/device/rss/rss.html",
    "cnbc_business":   "https://www.cnbc.com/id/10001109/device/rss/rss.html",
    "cnbc_economy":    "https://www.cnbc.com/id/20910258/device/rss/rss.html",
    "reuters_business":"https://feeds.reuters.com/reuters/businessNews",
    "marketwatch_top": "https://www.marketwatch.com/rss/topstories",
    "marketwatch_rt":  "https://www.marketwatch.com/rss/realtimeheadlines",
}


_TICKER_ALIASES: dict[str, list[str]] = {
    "SPY":  ["SPY", "S&P 500", "S&P500", "SPDR", "SP500"],
    "QQQ":  ["QQQ", "NASDAQ", "Nasdaq 100"],
    "AAPL": ["AAPL", "Apple"],
    "MSFT": ["MSFT", "Microsoft"],
    "AMZN": ["AMZN", "Amazon"],
    "NVDA": ["NVDA", "Nvidia", "NVIDIA"],
    "TSLA": ["TSLA", "Tesla"],
    "GOOGL":["GOOGL", "Alphabet", "Google"],
    "META": ["META", "Meta", "Facebook"],
    "JPM":  ["JPM", "JPMorgan", "JP Morgan"],
    "BAC":  ["BAC", "Bank of America"],
    "GS":   ["GS", "Goldman Sachs", "Goldman"],
    "XOM":  ["XOM", "ExxonMobil", "Exxon"],
    "CVX":  ["CVX", "Chevron"],
    "AMD":  ["AMD"],
    "MU":   ["MU", "Micron"],
    "PFE":  ["PFE", "Pfizer"],
    "GLD":  ["GLD", "Gold"],
    "TLT":  ["TLT", "Treasury"],
    "XLF":  ["XLF", "financials ETF"],
    "XLE":  ["XLE", "energy ETF"],
    "XLK":  ["XLK", "technology ETF"],
}


# Broad-market tickers: all market news is relevant, not just articles that
# name the ticker symbol.  Individual stocks still require alias matching.
_BROAD_MARKET_TICKERS = {"SPY", "QQQ", "GLD", "TLT", "XLF", "XLE", "XLK", "XLV", "^VIX"}


def scrape_rss_general(ticker: str) -> pd.DataFrame:
    """
    Fetch all general RSS feeds filtered by ticker relevance.

    For broad-market ETFs (SPY, QQQ, GLD, etc.) all articles are included
    because any market news affects the index.  For individual stocks, only
    articles that mention the ticker symbol or a known alias are kept.
    Covers CNBC (5 topic feeds), Reuters, and MarketWatch.
    """
    if not _HAS_FEEDPARSER:
        warnings.warn("feedparser not installed — RSS source disabled. pip install feedparser")
        return pd.DataFrame()

    t = ticker.upper()
    broad = t in _BROAD_MARKET_TICKERS
    terms = [s.upper() for s in _TICKER_ALIASES.get(t, [t])]
    records: list[dict] = []

    for name, url in _RSS_GENERAL.items():
        for item in _parse_feed(url, name):
            if broad or any(term in item["headline"].upper() for term in terms):
                records.append(item)

    return _to_df(records, "rss_general", ticker)


# ─── Source B: Yahoo Finance RSS (ticker-specific, proper endpoint) ────────────

def scrape_yahoo_rss(ticker: str) -> pd.DataFrame:
    """
    Fetch the official Yahoo Finance RSS feed for a specific ticker.

    Endpoint: feeds.finance.yahoo.com/rss/2.0/headline
    This is the structured RSS endpoint — much more reliable than scraping
    the Yahoo Finance HTML page.  Returns ~20 recent headlines with correct
    publish timestamps.
    """
    if not _HAS_FEEDPARSER:
        return pd.DataFrame()

    url = (
        f"https://feeds.finance.yahoo.com/rss/2.0/headline"
        f"?s={ticker.upper()}&region=US&lang=en-US"
    )
    records = _parse_feed(url, f"yahoo_rss_{ticker}")
    return _to_df(records, "yahoo_rss", ticker)


# ─── Source C: Google News RSS (search-based, ~100 articles) ──────────────────

def scrape_google_news(ticker: str) -> pd.DataFrame:
    """
    Query Google News RSS for articles about the ticker.

    Endpoint: news.google.com/rss/search
    Returns up to ~100 recent articles with proper timestamps.  This is the
    highest-volume free source — it pulls from hundreds of publications.
    Query includes the company name when available via a static lookup so
    broader searches catch articles that don't use the ticker symbol.
    """
    if not _HAS_FEEDPARSER:
        return pd.DataFrame()

    _name_map = {
        "SPY": "S&P 500 ETF", "QQQ": "Nasdaq ETF", "AAPL": "Apple",
        "MSFT": "Microsoft", "AMZN": "Amazon", "NVDA": "Nvidia",
        "TSLA": "Tesla", "GOOGL": "Alphabet Google", "META": "Meta Facebook",
        "JPM": "JPMorgan", "BAC": "Bank of America", "GS": "Goldman Sachs",
        "XOM": "ExxonMobil", "CVX": "Chevron", "AMD": "AMD semiconductor",
        "MU": "Micron Technology", "PFE": "Pfizer", "GLD": "Gold ETF",
        "TLT": "Treasury Bond ETF",
    }
    t = ticker.upper()
    name_query = _name_map.get(t, t)
    query = f"{t} {name_query} stock"
    encoded = query.replace(" ", "+").replace("&", "%26")
    url = f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"

    records = _parse_feed(url, f"google_news_{ticker}")
    return _to_df(records, "google_news", ticker)


# ─── Source D: Seeking Alpha RSS (ticker-specific) ────────────────────────────

def scrape_rss_seeking_alpha(ticker: str) -> pd.DataFrame:
    """Fetch ticker-specific RSS from Seeking Alpha."""
    if not _HAS_FEEDPARSER:
        return pd.DataFrame()

    url = f"https://seekingalpha.com/api/sa/combined/{ticker.upper()}.xml"
    records = _parse_feed(url, f"seekingalpha_{ticker}")
    return _to_df(records, "seeking_alpha", ticker)


# ─── Source E: Finviz HTML scraper ────────────────────────────────────────────

def scrape_finviz(ticker: str) -> pd.DataFrame:
    """
    Scrape the Finviz news table for a ticker.

    URL: finviz.com/quote.ashx?t={ticker}
    Target: table#news-table — contains ~50 recent articles with dates and times.
    Datetime format: "May-18-26 08:30AM" for the first row of a new day,
    then "08:30AM" for subsequent rows on the same day.
    """
    if not _HAS_BS4:
        warnings.warn("beautifulsoup4 not installed — Finviz source disabled. pip install beautifulsoup4")
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


# ─── Source F: Yahoo Finance news page (fallback HTML scrape) ─────────────────

def scrape_yahoo_finance(ticker: str) -> pd.DataFrame:
    """
    Pull recent headlines via the yfinance library (.news property).
    Handles both the legacy format {title, providerPublishTime} and the
    current nested format {content: {title, pubDate}}.
    """
    records: list[dict] = []
    try:
        import yfinance as yf
        info = yf.Ticker(ticker.upper()).news or []
        for item in info:
            # New format: item = {id, content: {title, pubDate, ...}}
            content = item.get("content") or {}
            title = (
                content.get("title")
                or item.get("title")
                or item.get("headline")
                or ""
            )
            if not title:
                continue
            # New format uses ISO string; legacy uses UNIX timestamp
            pub_date = content.get("pubDate") or ""
            ts = item.get("providerPublishTime") or 0
            if pub_date:
                try:
                    dt = datetime.fromisoformat(pub_date.replace("Z", "+00:00"))
                except ValueError:
                    dt = datetime.utcnow()
            elif ts:
                dt = datetime.utcfromtimestamp(ts)
            else:
                dt = datetime.utcnow()
            records.append({"headline": title, "datetime": dt})
    except Exception as exc:
        warnings.warn(f"[yfinance news] {ticker} failed: {exc}")

    return _to_df(records, "yfinance_news", ticker)


# ─── Source G: Benzinga (ticker-specific news page) ───────────────────────────

_RSS_INVESTING = {
    "investing_stocks": "https://www.investing.com/rss/news_25.rss",
    "investing_economy":"https://www.investing.com/rss/news_14.rss",
    "investing_etfs":   "https://www.investing.com/rss/news_95.rss",
}


def scrape_investing_com(ticker: str) -> pd.DataFrame:
    """
    Fetch Investing.com RSS feeds (stocks, economy, ETFs) filtered by ticker/alias.
    Broad-market ETFs (SPY, QQQ, etc.) get all articles; individual stocks are
    filtered to only articles that mention the ticker or its known aliases.
    """
    if not _HAS_FEEDPARSER:
        return pd.DataFrame()

    t = ticker.upper()
    broad = t in _BROAD_MARKET_TICKERS
    terms = [s.upper() for s in _TICKER_ALIASES.get(t, [t])]
    records: list[dict] = []

    for name, url in _RSS_INVESTING.items():
        for item in _parse_feed(url, name):
            if broad or any(term in item["headline"].upper() for term in terms):
                records.append(item)

    return _to_df(records, "investing_com", ticker)


# ─── Source H: Forex Factory economic calendar (macro events) ─────────────────

_FF_CALENDAR_URLS = [
    "https://nfs.faireconomy.media/ff_calendar_thisweek.json?utc_offset=0",
]
# USD events at or above this impact level are included as news items.
_FF_MIN_IMPACT = {"Medium", "High"}


def scrape_forex_factory(ticker: str = "SPY") -> pd.DataFrame:
    """
    Fetch the Forex Factory economic calendar via the public faireconomy.media
    JSON endpoint (used by MetaTrader EA plugins — no API key required).

    High/Medium-impact USD events (CPI, NFP, FOMC, GDP, etc.) are converted
    into synthetic news headlines: "Economic event: {title} (Impact: {impact})".
    VADER scores these as slightly negative/uncertain, which aligns with the
    elevated volatility typically surrounding macro announcements.

    Covers this week, next week, and last week.  Ticker argument is kept for
    API consistency but the events are market-wide (useful for SPY/QQQ/ETFs).
    """
    records: list[dict] = []

    for url in _FF_CALENDAR_URLS:
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
            resp.raise_for_status()
            events = resp.json()
        except Exception as exc:
            warnings.warn(f"[ForexFactory] {url} failed: {exc}")
            continue

        for event in events:
            country = event.get("country", "")
            impact  = event.get("impact", "")
            title   = event.get("title", "")
            date_str = event.get("date", "")

            if country != "USD" or impact not in _FF_MIN_IMPACT or not title:
                continue

            try:
                dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                dt = datetime.utcnow()

            headline = f"Economic event: {title} (Impact: {impact})"
            records.append({"headline": headline, "datetime": dt})

    return _to_df(records, "forex_factory", ticker)


# ─── Merge, deduplicate, persist ──────────────────────────────────────────────

def fetch_news(ticker: str, fresh_days: int = 90) -> pd.DataFrame:
    """
    Collect news from all sources, merge, deduplicate, and persist to cache.

    The cache is ADDITIVE — historical articles are never deleted.  Only freshly
    scraped articles older than `fresh_days` are excluded from the current batch
    (to avoid re-importing stale data from slow feeds), but everything already in
    the cache is preserved.

    Sources queried every call:
      - RSS general (CNBC x5, Reuters, MarketWatch x2) — filtered by ticker
      - Yahoo Finance RSS (ticker-specific)
      - Google News RSS (ticker + company name search)
      - Seeking Alpha RSS (ticker-specific)
      - Finviz HTML news table
      - Yahoo Finance news page (fallback)
      - Benzinga news page
      - Forex Factory economic calendar (USD high/medium-impact events)

    Returns combined DataFrame with columns: headline, source, datetime, ticker.
    """
    frames = [
        scrape_rss_general(ticker),
        scrape_yahoo_rss(ticker),
        scrape_google_news(ticker),
        scrape_rss_seeking_alpha(ticker),
        scrape_finviz(ticker),
        scrape_yahoo_finance(ticker),
        scrape_investing_com(ticker),
        scrape_forex_factory(ticker),
    ]
    new_df = pd.concat([f for f in frames if not f.empty], ignore_index=True)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{ticker.upper()}_news.csv"

    if new_df.empty:
        warnings.warn(f"[fetch_news] All live sources returned 0 articles for {ticker}.")
        if cache_path.exists():
            return pd.read_csv(cache_path, parse_dates=["datetime"])
        return new_df

    # Dedup and optionally drop very stale fresh articles
    new_df["_key"] = new_df["headline"].apply(_norm_key)
    new_df = new_df.drop_duplicates(subset="_key")

    cutoff = datetime.utcnow() - timedelta(days=fresh_days)
    # Keep articles that have real timestamps OR are very recent
    # (articles with utcnow() timestamps pass through regardless)
    new_df = new_df[
        new_df["datetime"].isna() |
        (new_df["datetime"] >= cutoff)
    ]

    # Merge with cache — cache is NEVER filtered, only appended to
    if cache_path.exists():
        try:
            cached = pd.read_csv(cache_path, parse_dates=["datetime"])
            combined = pd.concat([cached, new_df.drop(columns="_key")], ignore_index=True)
        except Exception:
            combined = new_df.drop(columns="_key")
    else:
        combined = new_df.drop(columns="_key")

    combined["_key"] = combined["headline"].apply(_norm_key)
    combined = combined.drop_duplicates(subset="_key").drop(columns="_key")
    combined = combined.sort_values("datetime", ascending=False)

    combined.to_csv(cache_path, index=False)
    return combined


def check_scraper_health(ticker: str) -> dict[str, int]:
    """
    Test all news sources and report article counts per source.
    Prints a warning for any source that returns zero articles.
    """
    results = {
        "rss_general":      len(scrape_rss_general(ticker)),
        "yahoo_rss":        len(scrape_yahoo_rss(ticker)),
        "google_news":      len(scrape_google_news(ticker)),
        "seeking_alpha":    len(scrape_rss_seeking_alpha(ticker)),
        "finviz":           len(scrape_finviz(ticker)),
        "yahoo_news_page":  len(scrape_yahoo_finance(ticker)),
        "investing_com":    len(scrape_investing_com(ticker)),
        "forex_factory":    len(scrape_forex_factory(ticker)),
    }
    total = sum(results.values())
    print(f"\n[Scraper Health] {ticker}")
    print(f"{'-'*44}")
    for source, count in results.items():
        flag = "OK" if count > 0 else "WARNING: 0 articles"
        print(f"  {source:<22} {count:3d} articles  [{flag}]")
    print(f"  {'Total':<22} {total:3d} articles")
    if total == 0:
        warnings.warn(f"[check_scraper_health] All sources returned 0 for {ticker}.")
    return results


# ─── Reddit WSB sentiment ─────────────────────────────────────────────────────

def scrape_wsb_sentiment(ticker: str, limit: int = 100, time_filter: str = "week") -> pd.Series:
    """
    Fetch recent r/wallstreetbets posts mentioning the ticker and compute a
    VADER sentiment score weighted by log(1 + upvotes).

    Each post's VADER compound score is multiplied by log(1 + score) to give
    more weight to popular posts.  Scores are grouped by UTC date and the daily
    weighted mean is returned as a pd.Series indexed by date.
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
        title   = d.get("title", "")
        upvotes = max(int(d.get("score", 0)), 0)
        created = d.get("created_utc", 0)
        date    = pd.Timestamp(created, unit="s").normalize()

        if not title:
            continue

        compound = analyzer.polarity_scores(title)["compound"]
        weight   = np.log1p(upvotes)
        records.append({"date": date, "sentiment": compound, "weight": weight})

    if not records:
        return pd.Series(dtype=float, name="wsb_sentiment")

    df = pd.DataFrame(records)
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
    print(f"  [WSB] {ticker}: {len(records)} posts over {len(daily)} days from r/wallstreetbets.")
    return daily
