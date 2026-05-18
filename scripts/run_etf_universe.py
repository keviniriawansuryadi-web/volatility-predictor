"""
Section 4 — Sector ETF universe run.

Runs the full volatility-forecasting pipeline on sector ETFs:
  QQQ  (Nasdaq-100 tech), XLF (financials), XLE (energy),
  XLK  (tech broad),      XLV (healthcare), GLD (gold),
  TLT  (20-year bonds),   ^VIX (VIX itself)

Key research questions:
  - Does QQQ behave like SPY or like NVDA/AMD?
  - Does GLD vol respond to sentiment differently?
  - Does TLT (bonds) show opposite sentiment pattern?
  - Does the EGARCH-wins-semiconductors pattern hold for sector ETFs?

Results are saved to outputs/results/{ticker}/ and a master comparison
table to outputs/results/etf_comparison.csv.

Usage:
    python scripts/run_etf_universe.py
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

from main import run_ticker
from config import TICKERS_ETF, DEFAULT_START, DEFAULT_END

TODAY = date.today().isoformat()
START = (date.today() - timedelta(days=5 * 365)).isoformat()
PLOT_DIR = str(Path(__file__).parent.parent)
OUT_DIR  = Path(__file__).parent.parent / "outputs" / "results"


# Predicted model winners based on SPY findings:
#   QQQ  — expect StackingEnsemble (broad-market, VIX-sensitive like SPY)
#   XLF  — expect EGARCH competitive (financial, systematic like JPM)
#   XLE  — expect XGBoost (energy, like XOM/CVX)
#   XLK  — expect StackingEnsemble (tech ETF, similar to SPY/QQQ)
#   XLV  — unknown (healthcare binary vol)
#   GLD  — unknown (commodity, inflation-driven)
#   TLT  — unknown (bond, interest-rate-driven, possibly negative VIX correlation)
#   ^VIX — interesting — VIX's own vol (vol-of-vol)

PREDICTIONS = {
    "QQQ":  "StackingEnsemble",
    "XLF":  "EGARCH",
    "XLE":  "XGBoost",
    "XLK":  "StackingEnsemble",
    "XLV":  "unknown",
    "GLD":  "unknown",
    "TLT":  "unknown",
    "^VIX": "unknown",
}

print(f"\n{'='*70}")
print(f"  ETF UNIVERSE RUN  ({len(TICKERS_ETF)} tickers)")
print(f"  Period: {START} to {TODAY}")
print(f"{'='*70}")

etf_rows = []
for ticker in TICKERS_ETF:
    pred = PREDICTIONS.get(ticker, "unknown")
    print(f"\n  Running {ticker}  [predicted winner: {pred}]")
    try:
        result = run_ticker(
            ticker=ticker,
            start=START,
            end=TODAY,
            horizon=5,
            train_size=0.8,
            garch_type="EGARCH",
            use_cache=True,
            plot_dir=PLOT_DIR,
        )
        metrics = result["metrics_df"]
        best_row = metrics.sort_values("QLIKE").iloc[0]
        actual_winner = best_row.name
        etf_rows.append({
            "ticker": ticker,
            "predicted": pred,
            "actual_winner": actual_winner,
            "best_qlike": round(float(best_row["QLIKE"]), 4),
            "best_corr":  round(float(best_row["Corr"]),  4),
            "best_spike_acc": round(float(best_row["Spike_Acc"]), 4) if not pd.isna(best_row["Spike_Acc"]) else None,
            "prediction_correct": (pred == actual_winner) or pred == "unknown",
        })
        print(f"    [{ticker}] Winner: {actual_winner}  QLIKE={best_row['QLIKE']:.4f}  Corr={best_row['Corr']:.4f}")
    except Exception as exc:
        print(f"    [{ticker}] FAILED: {exc}")
        etf_rows.append({"ticker": ticker, "predicted": pred, "error": str(exc)})

# ── Master comparison table ───────────────────────────────────────────────────
if etf_rows:
    df_etf = pd.DataFrame(etf_rows)
    out_path = OUT_DIR / "etf_comparison.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_etf.to_csv(out_path, index=False)
    print(f"\n  ETF comparison saved: {out_path}")

    print(f"\n{'='*70}")
    print("  ETF UNIVERSE SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Ticker':>6}  {'Predicted':>20}  {'Actual':>20}  {'QLIKE':>7}  {'Corr':>7}")
    print(f"  {'-'*65}")
    for r in etf_rows:
        if "error" in r:
            print(f"  {r['ticker']:>6}  ERROR: {r['error'][:40]}")
            continue
        match = "✓" if r.get("prediction_correct") else "✗"
        print(f"  {r['ticker']:>6}  {r['predicted']:>20}  {r['actual_winner']:>20}"
              f"  {r['best_qlike']:>7.4f}  {r['best_corr']:>7.4f}  {match}")

print(f"\n  Section 4 COMPLETE.")
