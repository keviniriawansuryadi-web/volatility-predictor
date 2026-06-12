"""
Honest cross-ticker model leaderboard.

Aggregates the per-ticker model-comparison metrics (outputs/.../results/{ticker}/
metrics.csv) into a single ranking so we can answer the question the per-ticker
runs can't: *which model actually wins out-of-sample across the whole universe?*

For each model it reports the median RMSE / QLIKE / Corr across all tickers and
two win-counts — how many tickers the model wins on QLIKE and on RMSE (lower is
better for both). This is the evidence that should drive sector-model routing in
config.py, and it immediately exposes any single model whose metrics look too
good to be true.

Run:
    python scripts/model_leaderboard.py
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

# UTF-8 console so the em dashes / arrows render on Windows cp1252 stdout.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

RESULTS_DIR = Path("outputs/outputs/results")
OUT_CSV = Path("outputs/model_leaderboard.csv")
LOWER_IS_BETTER = ("RMSE", "MAE", "QLIKE")


def load_all_metrics(results_dir: Path = RESULTS_DIR,
                     allowed: set[str] | None = None) -> pd.DataFrame:
    """Concatenate every ticker's metrics.csv into one long frame with a ticker col.

    If `allowed` is given, only those tickers are aggregated — used to score a
    specific run (e.g. the sector + SPY universe) when results_dir also holds
    unrelated ad-hoc runs.
    """
    frames = []
    for metrics_path in sorted(results_dir.glob("*/metrics.csv")):
        ticker = metrics_path.parent.name
        if allowed is not None and ticker not in allowed:
            continue
        df = pd.read_csv(metrics_path)
        df.insert(0, "ticker", ticker)
        frames.append(df)
    if not frames:
        raise SystemExit(f"No metrics.csv files found under {results_dir}"
                         + (f" for tickers {sorted(allowed)}" if allowed else ""))
    return pd.concat(frames, ignore_index=True)


def win_counts(long_df: pd.DataFrame, metric: str) -> pd.Series:
    """Count, per model, how many tickers it wins on `metric` (lowest value)."""
    winners = long_df.loc[long_df.groupby("ticker")[metric].idxmin(), ["model"]]
    return winners["model"].value_counts()


def build_leaderboard(long_df: pd.DataFrame) -> pd.DataFrame:
    """Per-model median metrics + QLIKE/RMSE win counts across all tickers."""
    agg = (
        long_df.groupby("model")
        .agg(
            n=("ticker", "nunique"),
            med_RMSE=("RMSE", "median"),
            med_QLIKE=("QLIKE", "median"),
            med_Corr=("Corr", "median"),
            med_SpikeAcc=("Spike_Acc", "median"),
        )
    )
    agg["QLIKE_wins"] = win_counts(long_df, "QLIKE")
    agg["RMSE_wins"] = win_counts(long_df, "RMSE")
    agg[["QLIKE_wins", "RMSE_wins"]] = agg[["QLIKE_wins", "RMSE_wins"]].fillna(0).astype(int)
    # rank by median QLIKE (the project's headline metric), lower = better
    return agg.sort_values("med_QLIKE").reset_index()


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                    help="dir holding {ticker}/metrics.csv (default: full S&P 500 run)")
    ap.add_argument("--tickers-from", type=Path, default=None,
                    help="CSV with a 'ticker' column; restrict the leaderboard to it")
    ap.add_argument("--out", type=Path, default=OUT_CSV, help="output CSV path")
    ap.add_argument("--label", default="", help="extra text for the header line")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    allowed = None
    if args.tickers_from is not None:
        allowed = set(pd.read_csv(args.tickers_from)["ticker"].astype(str))
    long_df = load_all_metrics(args.results_dir, allowed)
    n_tickers = long_df["ticker"].nunique()
    board = build_leaderboard(long_df)

    disp = board.copy()
    for col in ("med_RMSE", "med_QLIKE", "med_Corr", "med_SpikeAcc"):
        disp[col] = disp[col].map(lambda v: f"{v:.3f}")

    print("\n" + "=" * 84)
    print(f"  MODEL LEADERBOARD — median metrics across {n_tickers} tickers (lower QLIKE/RMSE = better)")
    print("=" * 84)
    print(disp.to_string(index=False))
    print("-" * 84)
    best = board.iloc[0]
    print(f"  Best median QLIKE : {best['model']} ({best['med_QLIKE']:.3f})")
    print(f"  Most QLIKE wins   : {board.loc[board['QLIKE_wins'].idxmax(), 'model']} "
          f"({board['QLIKE_wins'].max()}/{n_tickers} tickers)")
    print(f"  Most RMSE wins    : {board.loc[board['RMSE_wins'].idxmax(), 'model']} "
          f"({board['RMSE_wins'].max()}/{n_tickers} tickers)")
    print("=" * 84 + "\n")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    board.to_csv(OUT_CSV, index=False)
    print(f"Leaderboard saved: {OUT_CSV}")


if __name__ == "__main__":
    main()
