"""
Section 11 — Walk-forward results visualization.

Loads the walk-forward results CSV (already computed from the previous
batch run) and generates QLIKE-by-fold line charts for SPY and MU.

SPY vs MU is the most interesting contrast:
  - SPY: Low regime, StackingEnsemble dominates
  - MU:  Mix of Extreme regimes, EGARCH leads on raw QLIKE but
         StackingEnsemble shows better stability

Saves to:
  outputs/plots/SPY_walk_forward.png
  outputs/plots/MU_walk_forward.png

Usage:
    python scripts/run_walkforward_viz.py
"""

import sys
import warnings
from pathlib import Path
from datetime import date, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import pandas as pd

from src.walk_forward import plot_walk_forward_results
from src.data_loader import load_stock_data, load_vix_data
from src.features import build_features
from src.garch_model import garch_in_sample_vol
from src.sentiment import fetch_sentiment, fetch_wsb_sentiment

TODAY = date.today().isoformat()
START = (date.today() - timedelta(days=5 * 365)).isoformat()

WF_CSV = Path("outputs/results/walk_forward_results.csv")

print(f"\n{'='*65}")
print(f"  SECTION 11 — WALK-FORWARD VISUALIZATION")
print(f"{'='*65}")

if not WF_CSV.exists():
    print(f"  Walk-forward CSV not found: {WF_CSV}")
    print(f"  Run scripts/run_walk_forward.py first.")
    sys.exit(1)

wf_df = pd.read_csv(WF_CSV)
print(f"  Loaded walk-forward results: {wf_df.shape}")
print(f"  Tickers available: {wf_df['ticker'].unique().tolist()}")


def _load_feat(ticker):
    df = load_stock_data(ticker, START, TODAY, cache=True)
    vix = load_vix_data(START, TODAY)
    if not vix.empty:
        df = df.join(vix, how="left")
        df[["vix_level","vix_change"]] = df[["vix_level","vix_change"]].ffill()
    df["sentiment"]     = fetch_sentiment(ticker, df.index)
    df["wsb_sentiment"] = fetch_wsb_sentiment(ticker, df.index)
    df["garch_vol"]     = garch_in_sample_vol(df["log_return"])
    return build_features(df, forecast_horizon=5)


for ticker in ["SPY", "MU"]:
    ticker_df = wf_df[wf_df["ticker"] == ticker].copy() if "ticker" in wf_df.columns else wf_df.copy()
    if ticker_df.empty:
        print(f"  [{ticker}] No walk-forward data in CSV — skipping.")
        continue

    print(f"\n  Loading {ticker} features for regime shading...")
    try:
        feat_df = _load_feat(ticker)
    except Exception as exc:
        print(f"  [{ticker}] Feature load failed: {exc} — plotting without regime shading.")
        feat_df = None

    path = plot_walk_forward_results(
        ticker_df,
        ticker=ticker,
        feat_df=feat_df,
    )
    print(f"  [{ticker}] Chart: {path}")

print(f"\n  Section 11 COMPLETE.")
