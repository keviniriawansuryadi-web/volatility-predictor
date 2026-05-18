"""
Regime-conditional model performance breakdown.

Splits the test set by volatility regime and computes QLIKE separately for
Low / Elevated / High / Extreme periods.  A model with good overall QLIKE
but poor Extreme-regime QLIKE is dangerous in practice — it looks good on
average but fails exactly when forecasting matters most.

Regime thresholds (annualised vol, consistent with src/regime.py):
  Low      < 15%
  Elevated 15-25%
  High     25-35%
  Extreme  ≥ 35%
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from typing import Callable


REGIME_BOUNDS = {"Low": 0.15, "Elevated": 0.25, "High": 0.35}
REGIME_ORDER  = ["Low", "Elevated", "High", "Extreme"]


def _label_regime(vol: float) -> str:
    if vol >= REGIME_BOUNDS["High"]:     return "Extreme"
    if vol >= REGIME_BOUNDS["Elevated"]: return "High"
    if vol >= REGIME_BOUNDS["Low"]:      return "Elevated"
    return "Low"


def _qlike(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    h  = np.maximum(y_pred, 1e-8) ** 2
    s2 = y_true ** 2
    return float(np.mean(s2 / h - np.log(s2 / h) - 1))


def regime_conditional_performance(
    ticker: str,
    feat_df: pd.DataFrame,
    forecasts: dict[str, pd.Series],
    train_size: float = 0.80,
    vol_col: str = "realized_vol_21d",
    save_dir: str | None = "outputs/results",
) -> pd.DataFrame:
    """
    Compute QLIKE separately for each volatility regime on the test split.

    A model's aggregate QLIKE can look strong while hiding catastrophic failure
    during Extreme-volatility periods (≥ 35% annualised), which are precisely
    the episodes where accurate forecasting has the highest economic value.
    This function exposes that regime-level breakdown so regime-sensitive models
    can be identified and flagged.

    Parameters
    ----------
    ticker     : Ticker symbol for labelling output.
    feat_df    : Feature DataFrame with 'target' and `vol_col` columns.  Must
                 be the same frame used to generate the forecasts so that the
                 index aligns.
    forecasts  : Dict of {model_label: pd.Series of volatility predictions}.
                 Series must share the same DatetimeIndex as feat_df.
    train_size : Train fraction used to determine the test split (default 0.80).
    vol_col    : Column in feat_df used to assign regime labels.  Must be the
                 realised volatility series, not the target.  Defaults to
                 'realized_vol_21d'.
    save_dir   : Directory to write the CSV output.  Pass None to skip saving.

    Returns
    -------
    pd.DataFrame  with columns [model, regime, n_obs, QLIKE].
    Rows are sorted by (model, regime) using REGIME_ORDER.
    """
    split   = int(len(feat_df) * train_size)
    test_df = feat_df.iloc[split:].copy()

    if vol_col not in test_df.columns:
        raise ValueError(f"'{vol_col}' not found in feat_df columns: {list(test_df.columns)}")
    if "target" not in test_df.columns:
        raise ValueError("'target' column not found in feat_df")

    test_df["_regime"] = test_df[vol_col].apply(_label_regime)
    y_test = test_df["target"]

    records = []

    for label, preds in forecasts.items():
        common = y_test.index.intersection(preds.index)
        if len(common) < 5:
            warnings.warn(f"[regime_perf] {label}: fewer than 5 common test observations — skipping")
            continue
        yt_all  = y_test.reindex(common)
        yp_all  = preds.reindex(common)
        reg_all = test_df["_regime"].reindex(common)

        for regime in REGIME_ORDER:
            mask = reg_all == regime
            n    = mask.sum()
            if n < 3:
                qlike = np.nan
            else:
                yt = yt_all[mask].values
                yp = yp_all[mask].values
                valid = ~(np.isnan(yt) | np.isnan(yp))
                if valid.sum() < 3:
                    qlike = np.nan
                else:
                    qlike = _qlike(yt[valid], yp[valid])
            records.append({"model": label, "regime": regime, "n_obs": int(n), "QLIKE": qlike})

    df_out = pd.DataFrame(records)
    if df_out.empty:
        warnings.warn(f"[regime_perf] {ticker}: no valid records produced")
        return df_out

    # Preserve regime order
    df_out["_order"] = df_out["regime"].map({r: i for i, r in enumerate(REGIME_ORDER)})
    df_out = df_out.sort_values(["model", "_order"]).drop(columns="_order").reset_index(drop=True)

    print(f"\n  {ticker} — regime-conditional QLIKE (test set):")
    print(f"  {'Model':<22} {'Low':>8} {'Elevated':>10} {'High':>8} {'Extreme':>10}  (n per regime)")

    for model, grp in df_out.groupby("model", sort=False):
        row = grp.set_index("regime")
        parts = []
        for r in REGIME_ORDER:
            if r in row.index:
                q = row.loc[r, "QLIKE"]
                n = int(row.loc[r, "n_obs"])
                parts.append(f"{q:8.4f}(n={n})" if not np.isnan(q) else f"{'NaN':>8}(n={n})")
            else:
                parts.append(f"{'---':>8}      ")
        print(f"  {model:<22} {'  '.join(parts)}")

    if save_dir is not None:
        out_dir = Path(save_dir) / ticker
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{ticker}_regime_perf.csv"
        df_out.to_csv(path, index=False)
        print(f"  Saved: {path}")

    return df_out
