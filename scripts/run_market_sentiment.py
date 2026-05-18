"""
Section 9 — Market sentiment index for SPY.

Builds a market-wide sentiment index from all 10 tickers weighted by
approximate S&P 500 market cap, then re-tests H1 on SPY using this
aggregate signal.

H1 failed on SPY (p=0.144) using individual ticker sentiment.
The hypothesis: individual ticker sentiment is noisy but the cap-weighted
aggregate may capture systematic fear/greed that individual tickers miss.

Usage:
    python scripts/run_market_sentiment.py
"""

import sys
import json
import warnings
from pathlib import Path
from datetime import date, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.data_loader import load_stock_data, load_vix_data
from src.sentiment import fetch_sentiment, compute_market_sentiment_index, test_market_sentiment_h1
from config import TICKERS

TODAY = date.today().isoformat()
START = (date.today() - timedelta(days=5 * 365)).isoformat()

MARKET_CAP_WEIGHTS = {
    "AAPL":  0.22,
    "MSFT":  0.20,
    "AMZN":  0.12,
    "NVDA":  0.11,
    "JPM":   0.08,
    "XOM":   0.06,
    "BAC":   0.05,
    "AMD":   0.04,
    "CVX":   0.04,
    "MU":    0.03,
}

print(f"\n{'='*65}")
print(f"  SECTION 9 — MARKET SENTIMENT INDEX")
print(f"{'='*65}")

# ── Build sentiment for all 10 tickers ───────────────────────────────────────
print(f"\nFetching sentiment for all {len(TICKERS)} tickers...")
df_dict = {}
for ticker in TICKERS:
    try:
        df = load_stock_data(ticker, START, TODAY, cache=True)
        df["sentiment"] = fetch_sentiment(ticker, df.index)
        df_dict[ticker] = df
    except Exception as exc:
        print(f"  [{ticker}] Failed: {exc}")

# ── Compute market-wide sentiment index ──────────────────────────────────────
print(f"\nBuilding market-cap-weighted sentiment index...")
market_sent = compute_market_sentiment_index(df_dict, market_cap_weights=MARKET_CAP_WEIGHTS)

# ── Test on SPY (H1 with market-wide sentiment) ──────────────────────────────
print(f"\nLoading SPY price data for H1 test...")
df_spy = load_stock_data("SPY", START, TODAY, cache=True)
vix_df = load_vix_data(START, TODAY)
if not vix_df.empty:
    df_spy = df_spy.join(vix_df, how="left")
    df_spy[["vix_level","vix_change"]] = df_spy[["vix_level","vix_change"]].ffill()

h1_result = test_market_sentiment_h1(df_spy, market_sent, spike_pct=0.90)

# ── Also test with equal weights ─────────────────────────────────────────────
print(f"\nBuilding equal-weight sentiment index for comparison...")
market_sent_eq = compute_market_sentiment_index(df_dict, market_cap_weights=None)
h1_eq = test_market_sentiment_h1(df_spy, market_sent_eq, spike_pct=0.90)

# ── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  H1 TEST COMPARISON — INDIVIDUAL vs MARKET-WIDE SENTIMENT")
print(f"{'='*65}")
print(f"  Individual SPY sentiment H1  : p=0.1440 (from prior run — NOT significant)")
print(f"  Market-cap weighted index H1 : p={h1_result.get('p_value', float('nan')):.4f}  "
      f"{'** significant **' if h1_result.get('significant') else 'not significant'}")
print(f"  Equal-weight index H1        : p={h1_eq.get('p_value', float('nan')):.4f}  "
      f"{'** significant **' if h1_eq.get('significant') else 'not significant'}")

if h1_result.get("significant") or h1_eq.get("significant"):
    print(f"\n  FINDING: Market-wide sentiment improves H1 test on SPY.")
    print(f"  RECOMMENDATION: Add market_sentiment_index as a feature to SPY models.")
else:
    print(f"\n  Market-wide sentiment does NOT improve H1 test on SPY.")
    print(f"  Sentiment signal remains stock-specific, not market-wide.")

# Save results
out = {
    "individual_spy_p": 0.1440,
    "market_cap_weighted": {
        "p_value": float(h1_result.get("p_value", float("nan"))),
        "significant": bool(h1_result.get("significant", False)),
        "effect_size": float(h1_result.get("effect_size", float("nan"))),
    },
    "equal_weighted": {
        "p_value": float(h1_eq.get("p_value", float("nan"))),
        "significant": bool(h1_eq.get("significant", False)),
        "effect_size": float(h1_eq.get("effect_size", float("nan"))),
    },
}
out_path = Path("outputs/results/market_sentiment_h1.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
print(f"\n  Saved: {out_path}")
print(f"\n  Section 9 COMPLETE.")
