"""
Walk-forward cross-validation for volatility forecasting models.

Replaces the single 80/20 train/test split with 5 expanding-window folds.
Each fold trains on all data up to a cutoff and tests on the next block.
Reports mean ± std QLIKE across folds.
"""

from __future__ import annotations

import warnings
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
