"""
Section 7 — Jump-augmented EGARCH comparison.

Compares the standard EGARCH(1,1) with a Jump-EGARCH(1,1) where the
jump_flag indicator enters the variance equation as an external regressor.

Runs on SPY (where jump_flag is #2 XGBoost feature at importance=0.191)
and reports whether the augmentation improves QLIKE > 5%.

If improvement > 5% on QLIKE, add to all tickers (flagged in output).

Usage:
    python scripts/run_jump_egarch.py
"""

import sys
import warnings
from pathlib import Path
from datetime import date, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.data_loader import load_stock_data, load_vix_data
from src.features import build_features
from src.garch_model import rolling_garch_forecast, fit_jump_egarch, garch_in_sample_vol
from src.sentiment import fetch_sentiment, fetch_wsb_sentiment
from src.evaluate import _metrics as eval_metrics
from config import DEFAULT_TRAIN_SIZE

TODAY = date.today().isoformat()
START = (date.today() - timedelta(days=5 * 365)).isoformat()

THRESHOLD_PCT = 0.05  # 5% QLIKE improvement triggers "add to all tickers" flag


def _load(ticker: str):
    df = load_stock_data(ticker, START, TODAY, cache=True)
    vix_df = load_vix_data(START, TODAY)
    if not vix_df.empty:
        df = df.join(vix_df, how="left")
        df[["vix_level", "vix_change"]] = df[["vix_level", "vix_change"]].ffill()
    df["sentiment"]     = fetch_sentiment(ticker, df.index)
    df["wsb_sentiment"] = fetch_wsb_sentiment(ticker, df.index)
    df["garch_vol"]     = garch_in_sample_vol(df["log_return"])
    feat_df = build_features(df, forecast_horizon=5)
    return df, feat_df


for ticker in ["SPY"]:
    print(f"\n{'='*65}")
    print(f"  JUMP-EGARCH vs EGARCH — {ticker}")
    print(f"{'='*65}")

    df, feat_df = _load(ticker)
    split = int(len(feat_df) * DEFAULT_TRAIN_SIZE)
    y_test = feat_df["target"].values[split:]
    test_index = feat_df.index[split:]
    spike_thresh = float(np.percentile(y_test, 90))

    # Build jump_flag series aligned to full dataframe
    jump_col = feat_df["jump_flag"] if "jump_flag" in feat_df.columns else pd.Series(0.0, index=feat_df.index)

    print(f"\nRunning standard EGARCH...")
    std_preds = rolling_garch_forecast(
        df["log_return"], train_size=DEFAULT_TRAIN_SIZE, forecast_horizon=5
    )

    print(f"\nRunning Jump-EGARCH (exogenous jump_flag regressor)...")
    jump_preds = fit_jump_egarch(
        returns=df["log_return"],
        jump_flags=jump_col.reindex(df.index).fillna(0),
        train_size=DEFAULT_TRAIN_SIZE,
        forecast_horizon=5,
    )

    def _eval(preds_series, name):
        yp = preds_series.reindex(test_index).values
        return eval_metrics(y_test, yp, name, spike_thresh)

    std_m  = _eval(std_preds, "EGARCH")
    jump_m = _eval(jump_preds, "Jump-EGARCH")

    print(f"\n{'='*55}")
    print(f"  MODEL COMPARISON — {ticker}")
    print(f"{'='*55}")
    print(f"  {'Model':>15}  {'QLIKE':>7}  {'Corr':>7}  {'Spike_Acc':>10}")
    print(f"  {'-'*50}")
    for m in [std_m, jump_m]:
        sa = f"{m['Spike_Acc']:.1%}" if not np.isnan(m.get("Spike_Acc", float("nan"))) else "n/a"
        print(f"  {m['model']:>15}  {m['QLIKE']:>7.4f}  {m['Corr']:>7.4f}  {sa:>10}")

    qlike_improvement = (std_m["QLIKE"] - jump_m["QLIKE"]) / (std_m["QLIKE"] + 1e-10)
    corr_improvement  = jump_m["Corr"] - std_m["Corr"]

    print(f"\n  QLIKE improvement : {qlike_improvement:+.1%}")
    print(f"  Corr  improvement : {corr_improvement:+.4f}")

    if qlike_improvement > THRESHOLD_PCT:
        print(f"\n  ✓ Jump-EGARCH improves QLIKE by > 5% on {ticker}.")
        print(f"    RECOMMENDATION: Add Jump-EGARCH to all tickers pipeline.")
    else:
        print(f"\n  Jump-EGARCH improvement below 5% threshold on {ticker}.")
        print(f"    Standard EGARCH is sufficient for now.")

print(f"\n  Section 7 COMPLETE.")
