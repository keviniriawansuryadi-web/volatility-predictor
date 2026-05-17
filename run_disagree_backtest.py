"""
Run EGARCH-ML disagreement signal backtest for MU and JPM.
Uses cached price data — does NOT re-download or retrain full pipeline.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import warnings
warnings.filterwarnings("ignore")

from datetime import date, timedelta
import numpy as np

from src.data_loader import load_stock_data, load_vix_data
from src.sentiment import fetch_sentiment, fetch_wsb_sentiment
from src.features import build_features, FEATURE_COLS
from src.garch_model import rolling_garch_forecast, garch_in_sample_vol
from src.ml_model import train_and_predict
from src.disagree_backtest import backtest_disagreement_signal, print_backtest_results
from config import DEFAULT_TRAIN_SIZE, DEFAULT_GARCH_TYPE

TICKERS     = ["MU", "JPM"]
TRAIN_SIZE  = DEFAULT_TRAIN_SIZE
HORIZON     = 21
GARCH_TYPE  = DEFAULT_GARCH_TYPE

today = date.today().isoformat()
start = (date.today() - timedelta(days=5 * 365)).isoformat()

backtest_results = {}

for ticker in TICKERS:
    print(f"\n{'='*50}\n  {ticker} — loading cached data...\n{'='*50}")

    df = load_stock_data(ticker, start, today, cache=True)
    vix_df = load_vix_data(start, today)
    if not vix_df.empty:
        df = df.join(vix_df, how="left")
        df[["vix_level", "vix_change"]] = df[["vix_level", "vix_change"]].ffill()

    df["sentiment"]     = fetch_sentiment(ticker, df.index)
    df["wsb_sentiment"] = fetch_wsb_sentiment(ticker, df.index)
    df["garch_vol"]     = garch_in_sample_vol(df["log_return"], model_type=GARCH_TYPE)

    feat_df = build_features(df, forecast_horizon=HORIZON)

    print(f"  Running {GARCH_TYPE} rolling forecast...")
    garch_preds = rolling_garch_forecast(
        df["log_return"], train_size=TRAIN_SIZE,
        forecast_horizon=HORIZON, model_type=GARCH_TYPE,
    )

    print(f"  Training XGBoost...")
    xgb_preds, _, _ = train_and_predict(feat_df, model_type="xgboost", train_size=TRAIN_SIZE)

    split = int(len(feat_df) * TRAIN_SIZE)
    rv_test = feat_df["target"].iloc[split:]

    result = backtest_disagreement_signal(
        garch_preds=garch_preds,
        ml_preds=xgb_preds,
        realized_vol=rv_test,
    )
    backtest_results[ticker] = result
    print_backtest_results(ticker, result)

print(f"\n{'='*60}")
print("  BACKTEST SUMMARY")
print(f"{'='*60}")
for t, r in backtest_results.items():
    if r.get("available"):
        print(f"  {t}: hit_rate={r['hit_rate']:.1%}  "
              f"vol_lift={r['vol_lift']:.2f}x  "
              f"FPR={r['false_positive_rate']:.1%}  "
              f"→ {r['interpretation']}")
    else:
        print(f"  {t}: not available — {r.get('reason')}")
