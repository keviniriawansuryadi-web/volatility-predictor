"""
Run the full volatility pipeline across every S&P 500 constituent.

Loads the cached ticker list (data/sp500_tickers.json, scraped from Wikipedia),
then calls main.run_ticker for each name over the fixed 2019-2024 window. Runs
in-process so the heavy imports (xgboost, arch, sklearn) are paid once rather
than 503 times.

Design notes:
  - Each ticker is wrapped in try/except: a single delisted/throttled symbol
    logs FAILED and the run continues.
  - run_ticker's verbose stdout/stderr is redirected to devnull; only concise
    "[i/N] TICKER OK|FAILED" lines go to outputs/sp500_status.log so progress is
    readable and the log stays small.
  - plt.close('all') after each ticker prevents matplotlib figure accumulation
    from leaking memory across 503 iterations.
  - Data + sentiment are cached by the loaders, so re-runs are far faster.

Run (in background — this takes hours):
    python scripts/run_sp500.py
"""
import io
import json
import os
import sys
import warnings
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from main import run_ticker

START, END = "2019-01-01", "2024-12-31"
HORIZON, TRAIN, GARCH = 5, 0.8, "EGARCH"
PLOT_DIR = "outputs"

TICKERS_PATH = Path("data/sp500_tickers.json")
LOG = Path("outputs/sp500_status.log")


def logline(msg: str) -> None:
    """Append one line to the status log (separate handle from redirected stdout)."""
    with LOG.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def main() -> None:
    tickers = json.loads(TICKERS_PATH.read_text())
    n = len(tickers)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text("", encoding="utf-8")  # reset
    logline(f"START {datetime.now().isoformat(timespec='seconds')} — {n} tickers")

    ok = fail = 0
    sink = io.StringIO()
    for i, ticker in enumerate(tickers, 1):
        try:
            with redirect_stdout(sink), redirect_stderr(sink):
                run_ticker(
                    ticker=ticker, start=START, end=END, horizon=HORIZON,
                    train_size=TRAIN, garch_type=GARCH, use_cache=True,
                    plot_dir=PLOT_DIR,
                )
            ok += 1
            logline(f"[{i}/{n}] {ticker} OK")
        except Exception as exc:  # noqa: BLE001 — isolate per-ticker failures
            fail += 1
            logline(f"[{i}/{n}] {ticker} FAILED: {type(exc).__name__}: {str(exc)[:120]}")
        finally:
            plt.close("all")
            sink.truncate(0)
            sink.seek(0)

    logline(f"DONE {datetime.now().isoformat(timespec='seconds')} — ok={ok} fail={fail}")


if __name__ == "__main__":
    main()
