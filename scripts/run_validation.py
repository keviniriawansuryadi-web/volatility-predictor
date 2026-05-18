"""Run validate_model_performance() for NVDA, BAC, AAPL (suspiciously strong results)."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import warnings
warnings.filterwarnings("ignore")

from datetime import date, timedelta
from src.data_loader import load_stock_data, load_vix_data
from src.sentiment import fetch_sentiment, fetch_wsb_sentiment
from src.features import build_features
from src.garch_model import rolling_garch_forecast, garch_in_sample_vol
from src.ml_model import train_and_predict, train_stacking_ensemble
from src.validation import validate_model_performance
from config import DEFAULT_TRAIN_SIZE, DEFAULT_GARCH_TYPE

TICKERS   = ["NVDA", "BAC", "AAPL"]
TODAY     = date.today().isoformat()
START     = (date.today() - __import__("datetime").timedelta(days=5 * 365)).isoformat()

results = {}
for ticker in TICKERS:
    print(f"\n{'='*50}\n  {ticker}\n{'='*50}")
    df = load_stock_data(ticker, TODAY.__class__.__new__(str) or START, TODAY, cache=True)
    # simpler: use date imports already available
    from datetime import date, timedelta
    start_str = (date.today() - timedelta(days=5*365)).isoformat()
    df = load_stock_data(ticker, start_str, TODAY, cache=True)

    vix_df = load_vix_data(start_str, TODAY)
    if not vix_df.empty:
        df = df.join(vix_df, how="left")
        df[["vix_level","vix_change"]] = df[["vix_level","vix_change"]].ffill()

    df["sentiment"]     = fetch_sentiment(ticker, df.index)
    df["wsb_sentiment"] = fetch_wsb_sentiment(ticker, df.index)
    df["garch_vol"]     = garch_in_sample_vol(df["log_return"], model_type=DEFAULT_GARCH_TYPE)

    feat_df = build_features(df, forecast_horizon=21)

    garch_preds = rolling_garch_forecast(df["log_return"], DEFAULT_TRAIN_SIZE, 21, DEFAULT_GARCH_TYPE)
    xgb_preds, xgb_model, xgb_feats = train_and_predict(feat_df, model_type="xgboost", train_size=DEFAULT_TRAIN_SIZE)
    xgb_asym, _, _ = train_and_predict(feat_df, model_type="xgboost_asymmetric", train_size=DEFAULT_TRAIN_SIZE)
    rf_preds, _, _ = train_and_predict(feat_df, model_type="random_forest", train_size=DEFAULT_TRAIN_SIZE)

    stack_base = {DEFAULT_GARCH_TYPE: garch_preds, "XGBoost": xgb_preds,
                  "XGB-Asymmetric": xgb_asym, "RandomForest": rf_preds}
    vix_col = df["vix_level"] if "vix_level" in df.columns else None
    stack_preds = train_stacking_ensemble(feat_df, base_forecasts=stack_base,
                                          train_size=DEFAULT_TRAIN_SIZE, vix_series=vix_col)

    result = validate_model_performance(
        ticker, feat_df, stack_preds, garch_preds, train_size=DEFAULT_TRAIN_SIZE
    )
    results[ticker] = result

print("\n\n=== SUMMARY ===")
for t, r in results.items():
    print(f"  {t}: verdict={r['verdict']}  persist_corr={r['persistence_corr']}  "
          f"beats_persistence={r['beats_persistence']}  extreme%={r['extreme_pct']}")
