"""
Cross-sector comparison: sector ETF vs. flagship single stock.

Reads the live-signal JSONs produced by main.py for the 2024-12-30 signal
date (the multi-sector smoke run) and builds:

  1. A comparison table (CSV + console) pairing each major S&P sector's SPDR
     ETF with a flagship large-cap from the same sector, showing the 5-day
     ensemble vol forecast, the regime call, and the single-stock vs. ETF
     vol gap (the idiosyncratic-risk premium diversification removes).
  2. A grouped bar chart of ETF vs. company vol per sector, saved as PNG.

Run:
    python scripts/cross_sector_comparison.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Major S&P sector → (SPDR sector ETF, flagship large-cap company)
SECTOR_PAIRS = [
    ("Technology",          "XLK",  "AAPL"),
    ("Financials",          "XLF",  "JPM"),
    ("Energy",              "XLE",  "XOM"),
    ("Health Care",         "XLV",  "JNJ"),
    ("Industrials",         "XLI",  "CAT"),
    ("Consumer Disc.",      "XLY",  "AMZN"),
    ("Consumer Staples",    "XLP",  "PG"),
    ("Utilities",           "XLU",  "NEE"),
    ("Materials",           "XLB",  "LIN"),
    ("Real Estate",         "XLRE", "AMT"),
    ("Communication Svcs",  "XLC",  "GOOGL"),
]

SIGNAL_DATE = "2024-12-30"
SIGNAL_DIR = Path("outputs/outputs")   # main.py --plot-dir outputs nests an outputs/ subdir
OUT_CSV = Path("outputs/cross_sector_comparison.csv")
OUT_PNG = Path("outputs/cross_sector_comparison.png")


def load_signal(ticker: str) -> dict:
    """Load one live-signal JSON for the fixed signal date."""
    path = SIGNAL_DIR / f"live_signal_{ticker}_{SIGNAL_DATE}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing signal file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def regime_label(regime: str) -> str:
    """Strip the scalping note, keep the short regime word (ELEVATED, HIGH...)."""
    return regime.split("--")[0].strip()


def build_table() -> pd.DataFrame:
    """Assemble the per-sector ETF-vs-company comparison frame."""
    rows = []
    for sector, etf, company in SECTOR_PAIRS:
        e = load_signal(etf)
        c = load_signal(company)
        etf_vol = e["forecasts"]["ensemble"]
        co_vol = c["forecasts"]["ensemble"]
        rows.append({
            "Sector": sector,
            "ETF": etf,
            "ETF Vol": etf_vol,
            "ETF Regime": regime_label(e["regime"]),
            "Company": company,
            "Company Vol": co_vol,
            "Company Regime": regime_label(c["regime"]),
            "Stock-ETF Gap": co_vol - etf_vol,
        })
    df = pd.DataFrame(rows)
    return df.sort_values("Stock-ETF Gap", ascending=False).reset_index(drop=True)


def fmt_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with vol columns formatted as percentages for display/CSV."""
    out = df.copy()
    for col in ("ETF Vol", "Company Vol", "Stock-ETF Gap"):
        out[col] = out[col].map(lambda v: f"{v:+.1%}" if col == "Stock-ETF Gap" else f"{v:.1%}")
    return out


def plot(df: pd.DataFrame) -> None:
    """Grouped bar chart: ETF vs. company 5-day ensemble vol per sector."""
    # plot in a readable order — by company vol descending
    d = df.sort_values("Company Vol", ascending=False).reset_index(drop=True)
    x = np.arange(len(d))
    w = 0.38

    fig, ax = plt.subplots(figsize=(13, 6.5))
    etf_bars = ax.bar(x - w / 2, d["ETF Vol"] * 100, w,
                      label="Sector ETF", color="#4C72B0", edgecolor="white")
    co_bars = ax.bar(x + w / 2, d["Company Vol"] * 100, w,
                     label="Flagship company", color="#DD8452", edgecolor="white")

    for bars in (etf_bars, co_bars):
        ax.bar_label(bars, fmt="%.0f", padding=2, fontsize=8, color="#333")

    labels = [f"{r.Sector}\n{r.ETF} / {r.Company}" for r in d.itertuples()]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("5-day ensemble volatility forecast (annualised %)")
    ax.set_title("Cross-Sector Volatility: Single Stock vs. Sector ETF\n"
                 f"5-day ensemble forecast as of {SIGNAL_DATE} (2019–2024 train window)",
                 fontsize=12, fontweight="bold")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150)
    print(f"Plot saved: {OUT_PNG}")


def main() -> None:
    df = build_table()
    disp = fmt_table(df)

    print("\n" + "=" * 78)
    print("  CROSS-SECTOR COMPARISON — single stock vs. sector ETF (5-day ensemble vol)")
    print("=" * 78)
    print(disp.to_string(index=False))

    avg_gap = df["Stock-ETF Gap"].mean()
    n_higher = int((df["Stock-ETF Gap"] > 0).sum())
    print("-" * 78)
    print(f"  Mean stock-minus-ETF vol gap : {avg_gap:+.1%}")
    print(f"  Stocks more volatile than ETF: {n_higher}/{len(df)} sectors")
    print("=" * 78 + "\n")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    disp.to_csv(OUT_CSV, index=False)
    print(f"Table saved: {OUT_CSV}")

    plot(df)


if __name__ == "__main__":
    main()
