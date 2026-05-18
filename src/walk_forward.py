"""
Walk-forward cross-validation for volatility forecasting models.

Replaces the single 80/20 train/test split with 5 expanding-window folds.
Each fold trains on all data up to a cutoff and tests on the next block.
Reports mean ± std QLIKE across folds.
"""

from __future__ import annotations

import warnings
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from typing import Callable


def walk_forward_validate(
    feat_df: pd.DataFrame,
    df_raw: pd.DataFrame,
    model_fns: dict[str, Callable],
    n_splits: int = 5,
    min_train_frac: float = 0.50,
) -> pd.DataFrame:
    """
    Time-series walk-forward (expanding window) cross-validation.

    Splits the time series into `n_splits` folds with expanding training windows.
    Fold k trains on the first (min_train_frac + k * step)% of data and tests on
    the subsequent block. The last fold's test set ends at the final observation.

    Each model_fn in model_fns receives (feat_df, df_raw, train_idx, test_idx)
    and returns a pd.Series of test-set predictions.

    Reports mean and std of QLIKE across folds.  High std relative to mean
    indicates the model is regime-sensitive — it performs well in some market
    environments but poorly in others.

    Parameters
    ----------
    feat_df      : Feature DataFrame with 'target' column.
    df_raw       : Raw price/vol DataFrame (needed for GARCH refitting).
    model_fns    : Dict of {label: callable(feat_df, df_raw, train_idx, test_idx)
                   -> pd.Series}.
    n_splits     : Number of folds (default 5).
    min_train_frac : Minimum fraction of data in the first training window (0.50).

    Returns a DataFrame with columns [model, fold, n_train, n_test, QLIKE]
    and a summary row per model with mean_QLIKE and std_QLIKE.
    """
    n = len(feat_df)
    step = (1.0 - min_train_frac) / n_splits
    cutoffs = [int(n * (min_train_frac + k * step)) for k in range(1, n_splits + 1)]
    cutoffs[-1] = n  # ensure last fold covers everything

    records = []

    for k, cut in enumerate(cutoffs):
        train_start = 0
        train_end   = int(n * (min_train_frac + (k - 0) * step)) if k > 0 else int(n * min_train_frac)
        if k == 0:
            train_end = int(n * min_train_frac)
        else:
            train_end = int(n * (min_train_frac + (k) * step))
        test_end = cut if k < n_splits - 1 else n

        # Expanding window: train on [0, train_end), test on [train_end, test_end)
        train_end_k = int(n * (min_train_frac + k * step)) if k < n_splits else n
        if k == 0:
            train_end_k = int(n * min_train_frac)

        test_start = train_end_k
        test_end_k = int(n * (min_train_frac + (k + 1) * step)) if k < n_splits - 1 else n

        train_idx = slice(0, test_start)
        test_idx  = slice(test_start, test_end_k)

        y_test = feat_df["target"].iloc[test_idx]
        if len(y_test) < 5:
            continue

        spike_thresh = float(y_test.quantile(0.90))

        for label, fn in model_fns.items():
            try:
                preds = fn(feat_df, df_raw, train_idx, test_idx)
                common = y_test.index.intersection(preds.index)
                if len(common) < 5:
                    continue
                yt = y_test.reindex(common).values
                yp = preds.reindex(common).values
                mask = ~(np.isnan(yt) | np.isnan(yp))
                yt, yp = yt[mask], yp[mask]
                if len(yt) < 3:
                    continue
                h = np.maximum(yp, 1e-8) ** 2
                s2 = yt ** 2
                qlike = float(np.mean(s2 / h - np.log(s2 / h) - 1))
                records.append({
                    "model": label, "fold": k + 1,
                    "n_train": test_start, "n_test": len(yt),
                    "QLIKE": qlike,
                })
            except Exception as exc:
                warnings.warn(f"[walk_forward] {label} fold {k+1}: {exc}")

    detail_df = pd.DataFrame(records)

    print("\n  Walk-forward CV — mean ± std QLIKE across folds:")
    print(f"  {'Model':<22} {'Mean QLIKE':>12} {'Std QLIKE':>12} {'CV Stable?':>12}")
    for model, grp in detail_df.groupby("model", sort=False):
        mu  = grp["QLIKE"].mean()
        std = grp["QLIKE"].std()
        stable = "YES" if std < mu else "⚠ HIGH STD"
        print(f"  {model:<22} {mu:>12.4f} {std:>12.4f} {stable:>12}")

    return detail_df


def plot_walk_forward_results(
    wf_results: pd.DataFrame,
    ticker: str,
    save_path: str | None = None,
    feat_df: pd.DataFrame | None = None,
) -> str:
    """
    Visualizes walk-forward CV results.  Supports two input formats:

    Per-fold format (preferred): columns [model, fold, QLIKE, ticker?]
      → Draws a line chart of QLIKE per fold, with regime shading.

    Summary format (fallback): columns [model, mean, std, ticker?]
      → Draws a bar chart of mean QLIKE with ±std error bars.
        Shading is replaced by a stability annotation (std/mean ratio).

    The shaded background (per-fold mode) encodes regime: green = calm
    (median test-set RV < 25%), red = stressed (median RV ≥ 25%).

    Parameters
    ----------
    wf_results : DataFrame from walk_forward_validate() OR the summary CSV.
    ticker     : Ticker symbol for title.
    save_path  : Full path to save PNG.  Defaults to
                 outputs/plots/{ticker}_walk_forward.png.
    feat_df    : Feature DataFrame for regime shading (per-fold mode only).

    Returns the path of the saved PNG.
    """
    if wf_results.empty:
        return ""

    colors = {
        "EGARCH":          "#e74c3c",
        "HAR-RV":          "#e67e22",
        "XGBoost":         "#3498db",
        "XGB-Asymmetric":  "#2980b9",
        "RandomForest":    "#27ae60",
        "StackingEnsemble":"#8e44ad",
        "Persistence":     "#95a5a6",
    }

    out_dir = Path(__file__).parent.parent / "outputs" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    if save_path is None:
        save_path = str(out_dir / f"{ticker}_walk_forward.png")

    # ── Per-fold format ───────────────────────────────────────────────────────
    if "fold" in wf_results.columns and "QLIKE" in wf_results.columns:
        models  = wf_results["model"].unique().tolist()
        n_folds = int(wf_results["fold"].max())
        folds   = list(range(1, n_folds + 1))

        fold_stressed = {}
        if feat_df is not None:
            n = len(feat_df)
            step = 0.5 / n_folds
            for k in range(n_folds):
                ts = int(n * (0.50 + k * step))
                te = int(n * (0.50 + (k + 1) * step)) if k < n_folds - 1 else n
                median_rv = float(feat_df["target"].iloc[ts:te].median()) if te > ts else 0
                fold_stressed[k + 1] = median_rv >= 0.25

        fig, ax = plt.subplots(figsize=(12, 6))
        for fold_num in folds:
            color = "#fadbd8" if fold_stressed.get(fold_num, False) else "#d5f5e3"
            ax.axvspan(fold_num - 0.5, fold_num + 0.5, alpha=0.35, color=color, linewidth=0)

        for model in models:
            grp = wf_results[wf_results["model"] == model].sort_values("fold")
            ql = [float(grp.loc[grp["fold"] == f, "QLIKE"].values[0])
                  if len(grp.loc[grp["fold"] == f]) > 0 else float("nan")
                  for f in folds]
            ax.plot(folds, ql, marker="o", linewidth=2, markersize=6,
                    color=colors.get(model, "#7f8c8d"), label=model)

        ax.set_xticks(folds)
        ax.set_xticklabels([f"Fold {f}" for f in folds])
        ax.set_xlabel("Walk-Forward Fold")

        calm_patch    = mpatches.Patch(facecolor="#d5f5e3", alpha=0.7, label="Calm (RV < 25%)")
        stressed_patch = mpatches.Patch(facecolor="#fadbd8", alpha=0.7, label="Stressed (RV ≥ 25%)")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles + [calm_patch, stressed_patch],
                  labels=labels + ["Calm (RV < 25%)", "Stressed (RV ≥ 25%)"],
                  loc="upper right", fontsize=8)

    # ── Summary format (mean/std) ─────────────────────────────────────────────
    elif "mean" in wf_results.columns:
        df_t   = wf_results.sort_values("mean")
        models = df_t["model"].tolist()
        means  = df_t["mean"].values
        stds   = df_t["std"].fillna(0).values
        bar_colors = [colors.get(m, "#7f8c8d") for m in models]

        fig, ax = plt.subplots(figsize=(10, 6))
        x = range(len(models))
        bars = ax.bar(x, means, color=bar_colors, alpha=0.75, width=0.6)
        ax.errorbar(x, means, yerr=stds, fmt="none", color="black",
                    capsize=4, linewidth=1.5)

        for i, (m, mu, sd) in enumerate(zip(models, means, stds)):
            cv = sd / (mu + 1e-10)
            label = f"cv={cv:.2f}" if sd > 0 else "single"
            ax.text(i, mu + sd + 0.005, label, ha="center", va="bottom",
                    fontsize=7, color="gray")

        ax.set_xticks(list(x))
        ax.set_xticklabels(models, rotation=20, ha="right")
        ax.set_xlabel("Model")

    else:
        print(f"  [walk_forward] Unrecognised format: {wf_results.columns.tolist()}")
        return ""

    ax.set_ylabel("QLIKE (lower is better)")
    ax.set_title(f"Walk-Forward CV — Mean QLIKE ± Std — {ticker}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Walk-forward chart saved: {save_path}")
    return save_path
