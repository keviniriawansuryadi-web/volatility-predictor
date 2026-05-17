"""
Walk-forward CV for MU, NVDA, AMD, JPM (key tickers).
5 expanding-window folds, reports mean±std QLIKE per model.
"""
import sys
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
from src.walk_forward import walk_forward_validate
from config import DEFAULT_GARCH_TYPE

TODAY   = date.today().isoformat()
START   = (date.today() - timedelta(days=5 * 365)).isoformat()
HORIZON = 21
TICKERS = ["MU", "NVDA", "AMD", "JPM", "BAC", "XOM", "AAPL", "MSFT"]


def _build(ticker):
    df = load_stock_data(ticker, START, TODAY, cache=True)
    vix_df = load_vix_data(START, TODAY)
    if not vix_df.empty:
        df = df.join(vix_df, how="left")
        df[["vix_level","vix_change"]] = df[["vix_level","vix_change"]].ffill()
    df["sentiment"]     = fetch_sentiment(ticker, df.index)
    df["wsb_sentiment"] = fetch_wsb_sentiment(ticker, df.index)
    df["garch_vol"]     = garch_in_sample_vol(df["log_return"], model_type=DEFAULT_GARCH_TYPE)
    feat_df = build_features(df, forecast_horizon=HORIZON)
    return df, feat_df


def make_model_fns(ticker):
    """Return dict of model_fn(feat_df, df_raw, train_idx, test_idx) -> pd.Series."""

    def egarch_fn(feat_df, df_raw, train_idx, test_idx):
        train_size = test_idx.start / len(feat_df)
        return rolling_garch_forecast(df_raw["log_return"], train_size, HORIZON, DEFAULT_GARCH_TYPE)

    def har_fn(feat_df, df_raw, train_idx, test_idx):
        train_size = test_idx.start / len(feat_df)
        return har_rv_forecast(df_raw["realized_vol_21d"], train_size=train_size, forecast_horizon=HORIZON)

    def xgb_fn(feat_df, df_raw, train_idx, test_idx):
        train_size = test_idx.start / len(feat_df)
        preds, _, _ = train_and_predict(feat_df, model_type="xgboost", train_size=train_size)
        return preds

    def rf_fn(feat_df, df_raw, train_idx, test_idx):
        train_size = test_idx.start / len(feat_df)
        preds, _, _ = train_and_predict(feat_df, model_type="random_forest", train_size=train_size)
        return preds

    def persist_fn(feat_df, df_raw, train_idx, test_idx):
        rv = feat_df["realized_vol_21d"] if "realized_vol_21d" in feat_df.columns else feat_df["target"]
        return rv.shift(1).iloc[test_idx]

    def stack_fn(feat_df, df_raw, train_idx, test_idx):
        train_size = test_idx.start / len(feat_df)
        eg = rolling_garch_forecast(df_raw["log_return"], train_size, HORIZON, DEFAULT_GARCH_TYPE)
        xg, _, _ = train_and_predict(feat_df, model_type="xgboost",            train_size=train_size)
        xa, _, _ = train_and_predict(feat_df, model_type="xgboost_asymmetric",  train_size=train_size)
        rf, _, _ = train_and_predict(feat_df, model_type="random_forest",       train_size=train_size)
        vix_col  = df_raw["vix_level"] if "vix_level" in df_raw.columns else None
        return train_stacking_ensemble(feat_df,
               base_forecasts={DEFAULT_GARCH_TYPE: eg, "XGBoost": xg, "XGB-Asymmetric": xa, "RandomForest": rf},
               train_size=train_size, vix_series=vix_col)

    return {
        "Persistence":      persist_fn,
        DEFAULT_GARCH_TYPE: egarch_fn,
        "HAR-RV":           har_fn,
        "XGBoost":          xgb_fn,
        "RandomForest":     rf_fn,
        "StackingEnsemble": stack_fn,
    }


all_summary = []

for ticker in TICKERS:
    print(f"\n{'='*55}\n  {ticker} — walk-forward CV (5 folds)\n{'='*55}")
    try:
        df, feat_df = _build(ticker)
        model_fns   = make_model_fns(ticker)
        detail = walk_forward_validate(feat_df, df, model_fns, n_splits=5)
        summary = detail.groupby("model")["QLIKE"].agg(["mean","std"]).reset_index()
        summary["ticker"] = ticker
        all_summary.append(summary)
    except Exception as e:
        print(f"  FAILED: {e}")

combined = pd.concat(all_summary, ignore_index=True)

# Pivot: ticker x model, show mean QLIKE
pivot = combined.pivot(index="ticker", columns="model", values="mean").round(4)
print("\n\n=== Mean QLIKE across 5 walk-forward folds ===")
print(pivot.to_string())

std_pivot = combined.pivot(index="ticker", columns="model", values="std").round(4)
print("\n=== Std QLIKE across folds (stability) ===")
print(std_pivot.to_string())

# Save
out = Path("outputs/results/walk_forward_results.csv")
out.parent.mkdir(parents=True, exist_ok=True)
combined.to_csv(out, index=False)
print(f"\nSaved: {out}")
