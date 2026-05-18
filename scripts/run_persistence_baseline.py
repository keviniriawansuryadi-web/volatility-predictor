"""
Compute and compare the naive persistence baseline against all models for all 10 tickers.

Persistence forecast: tomorrow's vol = today's vol (1-day carry-forward of realized_vol_21d).
Reports whether StackingEnsemble beats persistence on QLIKE for each ticker.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding="utf-8")
import warnings
warnings.filterwarnings("ignore")

from datetime import date, timedelta
import numpy as np
import pandas as pd
from pathlib import Path

from src.data_loader import load_stock_data, load_vix_data
from src.sentiment import fetch_sentiment, fetch_wsb_sentiment
from src.features import build_features
from src.garch_model import rolling_garch_forecast, garch_in_sample_vol
from src.ml_model import train_and_predict, train_stacking_ensemble
from src.har_model import har_rv_forecast
from src.evaluate import _qlike, _metrics
from config import TICKERS, DEFAULT_TRAIN_SIZE, DEFAULT_GARCH_TYPE

TODAY = date.today().isoformat()
START = (date.today() - timedelta(days=5 * 365)).isoformat()
HORIZON = 21


def persistence_forecast(feat_df: pd.DataFrame, train_size: float = 0.8) -> pd.Series:
    """
    Simplest possible volatility baseline: tomorrow's forecast = today's realized vol.

    If a sophisticated model cannot beat this baseline, it is not adding genuine
    forecast value — it is merely tracking the autocorrelation already present
    in the target series.

    Parameters
    ----------
    feat_df    : Feature DataFrame with 'realized_vol_21d' column.
    train_size : Train fraction (same as other models, for fair comparison).

    Returns a pd.Series of persistence forecasts aligned to the test-set index.
    """
    split = int(len(feat_df) * train_size)
    # Use the lagged realized vol (the most recent observation available at prediction time)
    if "realized_vol_21d" in feat_df.columns:
        rv = feat_df["realized_vol_21d"]
    else:
        rv = feat_df["target"]
    # At test time t, we know rv[t-1]; shift(1) produces that lag
    persist = rv.shift(1).iloc[split:]
    return persist.rename("Persistence")


rows = []

for ticker in TICKERS:
    print(f"  {ticker}...", end=" ", flush=True)
    try:
        df = load_stock_data(ticker, START, TODAY, cache=True)
        vix_df = load_vix_data(START, TODAY)
        if not vix_df.empty:
            df = df.join(vix_df, how="left")
            df[["vix_level","vix_change"]] = df[["vix_level","vix_change"]].ffill()
        df["sentiment"]     = fetch_sentiment(ticker, df.index)
        df["wsb_sentiment"] = fetch_wsb_sentiment(ticker, df.index)
        df["garch_vol"]     = garch_in_sample_vol(df["log_return"], model_type=DEFAULT_GARCH_TYPE)

        feat_df = build_features(df, forecast_horizon=HORIZON)
        split   = int(len(feat_df) * DEFAULT_TRAIN_SIZE)
        y_test  = feat_df["target"].iloc[split:]

        # Models
        garch_preds = rolling_garch_forecast(df["log_return"], DEFAULT_TRAIN_SIZE, HORIZON, DEFAULT_GARCH_TYPE)
        har_preds   = har_rv_forecast(df["realized_vol_21d"], train_size=DEFAULT_TRAIN_SIZE, forecast_horizon=HORIZON)
        xgb_preds, _, _ = train_and_predict(feat_df, model_type="xgboost",            train_size=DEFAULT_TRAIN_SIZE)
        xgb_asym, _, _  = train_and_predict(feat_df, model_type="xgboost_asymmetric", train_size=DEFAULT_TRAIN_SIZE)
        rf_preds, _, _   = train_and_predict(feat_df, model_type="random_forest",      train_size=DEFAULT_TRAIN_SIZE)

        vix_col = df["vix_level"] if "vix_level" in df.columns else None
        stack_preds = train_stacking_ensemble(
            feat_df,
            base_forecasts={DEFAULT_GARCH_TYPE: garch_preds, "XGBoost": xgb_preds,
                            "XGB-Asymmetric": xgb_asym, "RandomForest": rf_preds},
            train_size=DEFAULT_TRAIN_SIZE, vix_series=vix_col,
        )

        persist_preds = persistence_forecast(feat_df, DEFAULT_TRAIN_SIZE)

        spike_thresh = float(y_test.quantile(0.90))
        all_forecasts = {
            "Persistence":      persist_preds,
            DEFAULT_GARCH_TYPE: garch_preds,
            "HAR-RV":           har_preds,
            "XGBoost":          xgb_preds,
            "RandomForest":     rf_preds,
            "StackingEnsemble": stack_preds,
        }

        ticker_rows = []
        for label, preds in all_forecasts.items():
            common = y_test.index.intersection(preds.index)
            if len(common) < 10:
                continue
            m = _metrics(y_test.reindex(common).values, preds.reindex(common).values,
                         label, spike_thresh)
            m["ticker"] = ticker
            ticker_rows.append(m)
        rows.extend(ticker_rows)
        print("done")
    except Exception as e:
        print(f"FAILED: {e}")

df_all = pd.DataFrame(rows)[["ticker","model","QLIKE","RMSE","Corr"]]
df_pivot = df_all.pivot(index="ticker", columns="model", values="QLIKE").round(4)

print("\n=== QLIKE vs Persistence Baseline (all tickers) ===")
print(df_pivot.to_string())

print("\n=== Does StackingEnsemble beat Persistence? ===")
for ticker in TICKERS:
    t = df_all[df_all["ticker"] == ticker]
    p_q = t[t["model"]=="Persistence"]["QLIKE"].values
    s_q = t[t["model"]=="StackingEnsemble"]["QLIKE"].values
    if len(p_q) and len(s_q):
        beats = "✓" if s_q[0] < p_q[0] else "✗"
        print(f"  {ticker}: Stacking={s_q[0]:.4f}  Persistence={p_q[0]:.4f}  {beats}")

# Save combined table
out_dir = Path("outputs/results")
out_dir.mkdir(parents=True, exist_ok=True)
df_all.to_csv(out_dir / "all_tickers_with_persistence.csv", index=False)
print(f"\nSaved: {out_dir}/all_tickers_with_persistence.csv")
