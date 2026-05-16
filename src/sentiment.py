import pandas as pd
import numpy as np


def fetch_sentiment(ticker: str, index: pd.DatetimeIndex) -> pd.Series:
    """
    Fetch VADER sentiment from yfinance news headlines, aligned to trading days.
    yfinance typically returns only recent articles (~30-60 days).
    Historical dates fill with 0 (neutral). Returns scores in [-1, +1].
    """
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        import yfinance as yf

        analyzer = SentimentIntensityAnalyzer()
        t = yf.Ticker(ticker)
        news_items = t.news or []

        records = []
        for item in news_items:
            # Handle both old format (flat dict) and new format (nested 'content')
            if "content" in item:
                content = item["content"]
                pub = content.get("pubDate", "")
                try:
                    ts = pd.Timestamp(pub).normalize()
                except Exception:
                    continue
                title = content.get("title", "")
                summary = content.get("summary", "")
            else:
                epoch = item.get("providerPublishTime", 0)
                if not epoch:
                    continue
                ts = pd.Timestamp(epoch, unit="s").normalize()
                title = item.get("title", "")
                summary = item.get("summary", "")

            text = f"{title} {summary}".strip()
            if text:
                score = analyzer.polarity_scores(text)["compound"]
                records.append({"date": ts, "score": score})

        if records:
            news_df = pd.DataFrame(records)
            daily = news_df.groupby("date")["score"].mean()
            daily.index = pd.DatetimeIndex(daily.index)
            # Strip tz if present
            if daily.index.tz is not None:
                daily.index = daily.index.tz_localize(None)
        else:
            daily = pd.Series(dtype=float)

        aligned = daily.reindex(index).ffill(limit=3).fillna(0.0)
        aligned.name = "sentiment"
        n_real = aligned[aligned != 0].shape[0]
        print(f"  [sentiment] {n_real} trading days with real VADER scores (rest neutral).")
        return aligned

    except ImportError:
        print("  [sentiment] vaderSentiment not installed — using neutral sentiment.")
        return pd.Series(0.0, index=index, name="sentiment")
    except Exception as e:
        print(f"  [sentiment] Failed: {e} — using neutral sentiment.")
        return pd.Series(0.0, index=index, name="sentiment")
