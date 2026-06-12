"""
Run the full volatility pipeline over every sector constituent + SPY.

Takes the 33 large-caps in run_hypotheses_by_sector.SECTORS (3 per GICS sector,
11 sectors) and adds SPY as the broad-market benchmark, then calls main.run_ticker
on each over a fixed window. This populates outputs/results/{ticker}/metrics.csv
for every name so scripts/model_leaderboard.py can rank models across the whole
sector universe, and writes a winner-per-ticker comparison table tagged with each
name's sector.

The SECTORS dict is imported (not copied) from run_hypotheses_by_sector so the two
scripts share one source of truth for the sector universe.

Design notes (mirrors run_sp500.py):
  - Each ticker is wrapped in try/except so one delisted/throttled symbol logs
    FAILED and the run continues.
  - run_ticker's verbose stdout/stderr is redirected to a sink; only concise
    "[i/N] TICKER OK|FAILED" lines go to outputs/sector_pipeline_status.log.
  - plt.close('all') after each ticker prevents matplotlib figure accumulation.
  - Data + VIX are cached by the loaders, so re-runs are far faster.

Run (in background — model fitting per ticker takes a while):
    python scripts/run_sector_pipeline.py
"""
import io
import sys
import warnings
from contextlib import redirect_stderr, redirect_stdout
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from main import run_ticker
from run_hypotheses_by_sector import SECTORS

HORIZON, TRAIN, GARCH = 5, 0.8, "EGARCH"
END = date.today().isoformat()
START = (date.today() - timedelta(days=5 * 365)).isoformat()
PLOT_DIR = str(ROOT)

LOG = ROOT / "outputs" / "sector_pipeline_status.log"
OUT_CSV = ROOT / "outputs" / "results" / "sector_pipeline_comparison.csv"


def logline(msg: str) -> None:
    """Append one line to the status log (separate handle from redirected stdout)."""
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def build_universe() -> list[tuple[str, str]]:
    """(ticker, sector) pairs for every sector constituent + SPY, deduped."""
    pairs, seen = [], set()
    for sector, names in SECTORS.items():
        for t in names:
            if t not in seen:
                pairs.append((t, sector))
                seen.add(t)
    if "SPY" not in seen:
        pairs.append(("SPY", "Benchmark"))
    return pairs


def main() -> None:
    universe = build_universe()
    n = len(universe)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text("", encoding="utf-8")
    logline(f"START {datetime.now().isoformat(timespec='seconds')} — {n} tickers "
            f"({START} -> {END})")

    rows = []
    ok = fail = 0
    sink = io.StringIO()
    for i, (ticker, sector) in enumerate(universe, 1):
        try:
            with redirect_stdout(sink), redirect_stderr(sink):
                result = run_ticker(
                    ticker=ticker, start=START, end=END, horizon=HORIZON,
                    train_size=TRAIN, garch_type=GARCH, use_cache=True,
                    plot_dir=PLOT_DIR,
                )
            metrics = result["metrics_df"]
            best = metrics.sort_values("QLIKE").iloc[0]
            rows.append({
                "ticker": ticker,
                "sector": sector,
                "winner": best.name,
                "best_qlike": round(float(best["QLIKE"]), 4),
                "best_corr": round(float(best["Corr"]), 4),
                "best_spike_acc": round(float(best["Spike_Acc"]), 4)
                if not pd.isna(best["Spike_Acc"]) else None,
            })
            ok += 1
            logline(f"[{i}/{n}] {ticker} ({sector}) OK — winner={best.name} "
                    f"QLIKE={best['QLIKE']:.4f}")
        except Exception as exc:  # noqa: BLE001 — isolate per-ticker failures
            fail += 1
            rows.append({"ticker": ticker, "sector": sector, "error": str(exc)[:120]})
            logline(f"[{i}/{n}] {ticker} ({sector}) FAILED: "
                    f"{type(exc).__name__}: {str(exc)[:120]}")
        finally:
            plt.close("all")
            sink.truncate(0)
            sink.seek(0)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    logline(f"DONE {datetime.now().isoformat(timespec='seconds')} — ok={ok} fail={fail}")

    print(f"\n{'='*72}")
    print(f"  SECTOR PIPELINE — {n} tickers (sector constituents + SPY)")
    print(f"  Period: {START} -> {END}   |   ok={ok} fail={fail}")
    print(f"{'='*72}")
    print(df.to_string(index=False))
    print(f"\nSaved: {OUT_CSV}")
    print("Next: python scripts/model_leaderboard.py  (aggregates the per-ticker metrics)")


if __name__ == "__main__":
    main()
