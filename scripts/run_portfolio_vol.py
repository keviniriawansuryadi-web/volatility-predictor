"""
Section 6 — Portfolio-level vol forecast aggregation.

Loads live signal JSONs for all 10 original tickers + SPY,
builds a correlation matrix from the last 60 days of returns,
and computes portfolio vol under two weighting schemes:
  (a) Equal weights (1/N)
  (b) S&P 500 approximate market-cap weights

Reports whether portfolio vol < SPY vol (diversification sanity check).

Usage:
    python scripts/run_portfolio_vol.py
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

from src.data_loader import load_stock_data
from src.portfolio import compute_portfolio_vol_forecast, build_correlation_matrix
from config import TICKERS

TODAY    = date.today().isoformat()
START    = (date.today() - timedelta(days=5 * 365)).isoformat()
OUT_DIR  = Path(__file__).parent.parent / "outputs" / "results"
LIVE_DIR = Path(__file__).parent.parent / "outputs"

# S&P 500 approximate market-cap weights for the 10 covered tickers
# (proportional to 2026 approximate market caps, normalised to sum < 1;
#  remainder goes to "other" — we simply re-normalise within the 10)
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

EQUAL_WEIGHTS = {t: 1.0 / len(TICKERS) for t in TICKERS}

# ── Load forecasts from live signal JSONs ────────────────────────────────────
print(f"\n{'='*65}")
print(f"  PORTFOLIO VOL FORECAST")
print(f"  Loading live signals from outputs/live_signal_*_2026-05-15.json")
print(f"{'='*65}")

forecasts_ensemble: dict[str, float] = {}
for ticker in TICKERS:
    pattern = f"live_signal_{ticker}_*.json"
    candidates = sorted(LIVE_DIR.glob(pattern))
    if not candidates:
        print(f"  [{ticker}] No live signal JSON found — skipping.")
        continue
    latest = candidates[-1]
    try:
        data = json.loads(latest.read_text(encoding="utf-8"))
        ens = data.get("forecasts", {}).get("ensemble")
        if ens is not None:
            forecasts_ensemble[ticker] = float(ens)
            print(f"  [{ticker}] ensemble={ens:.1%}  (from {latest.name})")
    except Exception as exc:
        print(f"  [{ticker}] Failed to load {latest}: {exc}")

# SPY forecast
spy_json = sorted(LIVE_DIR.glob("live_signal_SPY_*.json"))
spy_forecast = None
if spy_json:
    d = json.loads(spy_json[-1].read_text(encoding="utf-8"))
    spy_forecast = float(d.get("forecasts", {}).get("ensemble", float("nan")))
    print(f"\n  [SPY] ensemble={spy_forecast:.1%}  (from {spy_json[-1].name})")

# ── Build correlation matrix (last 60d returns) ──────────────────────────────
print(f"\n  Building 60-day return correlation matrix...")
returns_dict = {}
for ticker in forecasts_ensemble:
    try:
        df = load_stock_data(ticker, START, TODAY, cache=True)
        returns_dict[ticker] = df["log_return"].tail(60)
    except Exception:
        pass

corr_matrix = build_correlation_matrix(returns_dict, window=60)
print(f"  Correlation matrix built for {len(returns_dict)} tickers.")

# ── Run portfolio forecasts ───────────────────────────────────────────────────
for scheme, weights in [("Equal weights (1/N)", EQUAL_WEIGHTS), ("Market-cap weights", MARKET_CAP_WEIGHTS)]:
    print(f"\n  ── {scheme} ──")
    result = compute_portfolio_vol_forecast(
        tickers=list(forecasts_ensemble.keys()),
        weights=weights,
        forecasts=forecasts_ensemble,
        corr_matrix=corr_matrix,
        spy_forecast=spy_forecast,
    )

    # Save result
    out_key = scheme.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
    out_path = OUT_DIR / f"portfolio_vol_{out_key}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        k: v for k, v in result.items()
        if not isinstance(v, pd.DataFrame)
    }
    out_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    print(f"  Saved: {out_path}")

print(f"\n  Section 6 COMPLETE.")
