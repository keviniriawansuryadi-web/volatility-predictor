"""
Section 5 — Out-of-sample generalization test: TSLA, GS, PFE.

These tickers were never seen during model development. Hyperparameters
are NOT retuned — models are used exactly as trained on the original
10-ticker universe.

Pre-run predictions (informed by SPY/sector findings):
  TSLA: StackingEnsemble likely wins (high vol, VIX-sensitive, jump-prone — like SPY)
  GS:   EGARCH competitive (financial, idiosyncratic — like JPM, not BAC)
  PFE:  Unknown — pharma has binary vol from FDA decisions; neither model
        was trained on this pattern

Results are saved to outputs/results/oos_comparison.csv.

Usage:
    python scripts/run_oos_tickers.py
"""

import sys
import warnings
from pathlib import Path
from datetime import date, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import pandas as pd

from main import run_ticker
from config import TICKERS_OOS

TODAY    = date.today().isoformat()
START    = (date.today() - timedelta(days=5 * 365)).isoformat()
PLOT_DIR = str(Path(__file__).parent.parent)
OUT_DIR  = Path(__file__).parent.parent / "outputs" / "results"

# Predictions informed by the sector/ETF pattern:
#   - TSLA: VIX-sensitive, large-cap, frequent jumps → like SPY/QQQ → StackingEnsemble
#   - GS:   Financial, idiosyncratic (M&A news, proprietary trading) → like JPM → EGARCH
#   - PFE:  Healthcare binary vol → unknown; StackingEnsemble shown to win on XLV
PREDICTIONS = {
    "TSLA": "StackingEnsemble",
    "GS":   "EGARCH",
    "PFE":  "StackingEnsemble (via XLV finding)",
}

print(f"\n{'='*70}")
print(f"  OUT-OF-SAMPLE TICKERS: TSLA, GS, PFE")
print(f"  Period: {START} to {TODAY}")
print(f"  NOTE: No retuning — models run as-is from original 10-ticker training")
print(f"{'='*70}")

rows = []
for ticker in TICKERS_OOS:
    pred = PREDICTIONS.get(ticker, "unknown")
    print(f"\n  [{ticker}]  Predicted: {pred}")
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
        rows.append({
            "ticker":         ticker,
            "predicted":      pred,
            "actual_winner":  actual_winner,
            "best_qlike":     round(float(best_row["QLIKE"]), 4),
            "best_corr":      round(float(best_row["Corr"]),  4),
            "best_spike_acc": round(float(best_row["Spike_Acc"]), 4) if not pd.isna(best_row["Spike_Acc"]) else None,
            "egarch_qlike":   round(float(metrics.loc["EGARCH", "QLIKE"]), 4) if "EGARCH" in metrics.index else None,
            "stack_qlike":    round(float(metrics.loc["StackingEnsemble", "QLIKE"]), 4)
                              if "StackingEnsemble" in metrics.index else None,
        })
    except Exception as exc:
        print(f"  [{ticker}] FAILED: {exc}")
        rows.append({"ticker": ticker, "predicted": pred, "error": str(exc)})

# ── Summary table ─────────────────────────────────────────────────────────────
if rows:
    df_oos = pd.DataFrame(rows)
    out_path = OUT_DIR / "oos_comparison.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_oos.to_csv(out_path, index=False)
    print(f"\n  OOS comparison saved: {out_path}")

    print(f"\n{'='*75}")
    print("  OUT-OF-SAMPLE RESULTS")
    print(f"{'='*75}")
    print(f"  {'Ticker':>6}  {'Predicted':>28}  {'Actual':>20}  {'QLIKE':>7}  {'Corr':>7}")
    print(f"  {'-'*72}")
    for r in rows:
        if "error" in r:
            print(f"  {r['ticker']:>6}  ERROR: {r['error'][:45]}")
            continue
        pred_short = r["predicted"].split("(")[0].strip()
        match = "✓" if pred_short == r["actual_winner"] else "✗"
        print(f"  {r['ticker']:>6}  {pred_short:>28}  {r['actual_winner']:>20}"
              f"  {r['best_qlike']:>7.4f}  {r['best_corr']:>7.4f}  {match}")

    # Pattern check: does sector pattern hold?
    print(f"\n  Pattern check:")
    for r in rows:
        if "error" in r:
            continue
        stk = r.get("stack_qlike")
        egr = r.get("egarch_qlike")
        if stk and egr:
            winner_note = f"StackingEnsemble wins by {(egr - stk):.4f} QLIKE" if stk < egr else f"EGARCH wins by {(stk - egr):.4f} QLIKE"
            print(f"    {r['ticker']}: {winner_note}")

print(f"\n  Section 5 COMPLETE.")
