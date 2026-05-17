"""
Volatility spillover analysis using pairwise Granger causality.

The primary function test_volatility_spillover() builds a directional
causality matrix (p-values) for a group of tickers, testing whether
realized vol in one ticker Granger-causes vol in another.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def test_volatility_spillover(
    vol_dict: dict[str, pd.Series],
    max_lag: int = 5,
    significance: float = 0.05,
) -> pd.DataFrame:
    """
    Build a pairwise Granger causality matrix for realized volatility spillover.

    For each ordered pair (source, target), tests whether the lagged values of
    source Granger-cause target at lags 1, 3, and 5 days using the statsmodels
    Granger causality F-test.  The reported p-value is the minimum across the
    tested lags (i.e., the strongest causal signal at any lag).

    Interpretation:
      - p < 0.05: reject null that source does NOT Granger-cause target
        → source vol likely leads/predicts target vol
      - The matrix is NOT symmetric; MU→NVDA may be significant while NVDA→MU
        is not, revealing directional vol transmission.

    Parameters
    ----------
    vol_dict    : Dict of {ticker: pd.Series} of daily realized vol, all
                  indexed to the same DatetimeIndex.
    max_lag     : Maximum lag to test (default 5 days).
    significance: Alpha level for printing significance markers (default 0.05).

    Returns a DataFrame of shape (n_tickers, n_tickers) where entry [i, j] is
    the min p-value for the test "ticker[i] Granger-causes ticker[j]".
    Diagonal entries are NaN (self-causality is undefined).
    """
    from statsmodels.tsa.stattools import grangercausalitytests

    tickers = list(vol_dict.keys())
    n = len(tickers)
    pval_matrix = pd.DataFrame(np.nan, index=tickers, columns=tickers)

    for src in tickers:
        for tgt in tickers:
            if src == tgt:
                continue
            s = vol_dict[src].dropna()
            t = vol_dict[tgt].dropna()
            common = s.index.intersection(t.index)
            if len(common) < max_lag * 3 + 10:
                warnings.warn(f"[spillover] {src}→{tgt}: too few common obs ({len(common)})")
                continue
            data = pd.DataFrame({"target": t.loc[common], "source": s.loc[common]}).dropna()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    results = grangercausalitytests(data[["target", "source"]], maxlag=max_lag,
                                                   verbose=False)
                # Minimum p-value across lags 1..max_lag (strongest signal)
                min_p = min(
                    results[lag][0]["ssr_ftest"][1]
                    for lag in range(1, max_lag + 1)
                )
                pval_matrix.loc[src, tgt] = min_p
            except Exception as exc:
                warnings.warn(f"[spillover] {src}→{tgt} Granger test failed: {exc}")

    return pval_matrix


def plot_spillover_heatmap(
    pval_matrix: pd.DataFrame,
    title: str,
    out_path: Path,
    significance: float = 0.05,
) -> None:
    """
    Save a heatmap of Granger causality p-values with significance annotations.

    Cells are colored on a green-white-red scale where green = low p-value
    (strong causality) and white = high p-value (no evidence of causality).
    Diagonal cells are masked grey (undefined).  Each cell shows the p-value
    rounded to 3 decimal places, and cells below `significance` are bold-starred.

    Parameters
    ----------
    pval_matrix  : Square DataFrame of p-values (from test_volatility_spillover).
    title        : Chart title string.
    out_path     : Full Path for the saved PNG file.
    significance : Cells with p < significance are annotated with '*'.
    """
    tickers = list(pval_matrix.index)
    n = len(tickers)
    data = pval_matrix.values.astype(float)

    fig, ax = plt.subplots(figsize=(max(5, n * 1.4), max(4, n * 1.2)))

    # Color: 0 = dark green, 0.05 = yellow, 1 = light grey
    cmap = plt.cm.RdYlGn_r
    cmap.set_bad(color="#cccccc")

    masked = np.ma.array(data, mask=np.isnan(data))
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tickers, fontsize=10)
    ax.set_yticklabels(tickers, fontsize=10)
    ax.set_xlabel("Target (caused)", fontsize=11)
    ax.set_ylabel("Source (causer)", fontsize=11)
    ax.set_title(title, fontsize=12, pad=10)

    for i in range(n):
        for j in range(n):
            if np.isnan(data[i, j]):
                ax.text(j, i, "—", ha="center", va="center", fontsize=9, color="#888")
                continue
            pval = data[i, j]
            star = "*" if pval < significance else ""
            color = "white" if pval < 0.3 else "black"
            ax.text(j, i, f"{pval:.3f}{star}", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold" if star else "normal")

    plt.colorbar(im, ax=ax, label="p-value (lower = stronger causality)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Spillover heatmap saved: {out_path}")


def run_spillover_analysis(
    df_dict: dict[str, pd.DataFrame],
    sector_groups: dict[str, list[str]],
    plot_dir: str = ".",
    max_lag: int = 5,
) -> dict[str, pd.DataFrame]:
    """
    Run Granger causality spillover analysis for each sector group and save heatmaps.

    For each sector in sector_groups, extracts the realized_vol_21d series for
    each ticker, builds the pairwise p-value matrix via test_volatility_spillover(),
    and saves a heatmap PNG.

    Parameters
    ----------
    df_dict       : Dict of {ticker: DataFrame} where each DataFrame has a
                    'realized_vol_21d' column indexed to trading days.
    sector_groups : Dict of {sector_name: [ticker, ...]} defining which tickers
                    to test together (e.g. {'Semiconductors': ['MU','NVDA','AMD']}).
    plot_dir      : Root output directory for plots.
    max_lag       : Maximum Granger lag to test (default 5).

    Returns a dict of {sector_name: pval_matrix_DataFrame}.
    """
    out_dir = Path(plot_dir) / "outputs" / "plots"
    results: dict[str, pd.DataFrame] = {}

    for sector, tickers in sector_groups.items():
        available = [t for t in tickers if t in df_dict]
        if len(available) < 2:
            warnings.warn(f"[spillover] {sector}: need >= 2 tickers, got {len(available)}")
            continue

        vol_dict = {t: df_dict[t]["realized_vol_21d"] for t in available}
        print(f"\n  [spillover] Testing {sector}: {available} ...")
        pval_matrix = test_volatility_spillover(vol_dict, max_lag=max_lag)
        results[sector] = pval_matrix

        out_path = out_dir / f"spillover_{sector.lower().replace(' ', '_')}.png"
        plot_spillover_heatmap(
            pval_matrix,
            title=f"Granger Causality — {sector} Vol Spillover (min p across lags 1-{max_lag})",
            out_path=out_path,
        )

        # Print summary
        print(f"\n  {sector} spillover matrix (p-values, * = significant at 5%):")
        for src in pval_matrix.index:
            for tgt in pval_matrix.columns:
                if src == tgt:
                    continue
                p = pval_matrix.loc[src, tgt]
                if not np.isnan(p):
                    sig = " *" if p < 0.05 else ""
                    print(f"    {src} → {tgt}: p={p:.4f}{sig}")

    return results
