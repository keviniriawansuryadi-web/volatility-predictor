"""
Regime-conditional performance breakdown for AMD and MU.

AMD: 155-day Extreme-regime run in test set.
MU:  200-day Extreme-regime run in test set.

Uses cached data and the same hyperparameters as the main pipeline (no retuning).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding="utf-8")
import warnings
warnings.filterwarnings("ignore")

from datetime import date, timedelta
import pandas as pd
from pathlib import Path

from src.data_loader import load_stock_data, load_vix_data
from src.sentiment import fetch_sentiment, fetch_wsb_sentiment
from src.features import build_features
from src.garch_model import rolling_garch_forecast, garch_in_sample_vol
from src.ml_model import train_and_predict, train_stacking_ensemble
from src.har_model import har_rv_forecast
from src.regime_perf import regime_conditional_performance
from config import DEFAULT_TRAIN_SIZE, DEFAULT_GARCH_TYPE

TODAY   = date.today().isoformat()
START   = (date.today() - timedelta(days=5 * 365)).isoformat()
HORIZON = 21
TICKERS = ["AMD", "MU"]

all_results = {}

for ticker in TICKERS:
    print(f"\n{'='*55}\n  {ticker} — regime-conditional breakdown\n{'='*55}")
    df = load_stock_data(ticker, START, TODAY, cache=True)

    vix_df = load_vix_data(START, TODAY)
    if not vix_df.empty:
        df = df.join(vix_df, how="left")
        df[["vix_level", "vix_change"]] = df[["vix_level", "vix_change"]].ffill()

    df["sentiment"]     = fetch_sentiment(ticker, df.index)
    df["wsb_sentiment"] = fetch_wsb_sentiment(ticker, df.index)
    df["garch_vol"]     = garch_in_sample_vol(df["log_return"], model_type=DEFAULT_GARCH_TYPE)

    feat_df = build_features(df, forecast_horizon=HORIZON)

    garch_preds          = rolling_garch_forecast(df["log_return"], DEFAULT_TRAIN_SIZE, HORIZON, DEFAULT_GARCH_TYPE)
    har_preds            = har_rv_forecast(df["realized_vol_21d"], train_size=DEFAULT_TRAIN_SIZE, forecast_horizon=HORIZON)
    xgb_preds,  _, _     = train_and_predict(feat_df, model_type="xgboost",            train_size=DEFAULT_TRAIN_SIZE)
    xgb_asym,   _, _     = train_and_predict(feat_df, model_type="xgboost_asymmetric", train_size=DEFAULT_TRAIN_SIZE)
    rf_preds,   _, _     = train_and_predict(feat_df, model_type="random_forest",       train_size=DEFAULT_TRAIN_SIZE)

    vix_col = df["vix_level"] if "vix_level" in df.columns else None
    stack_preds = train_stacking_ensemble(
        feat_df,
        base_forecasts={DEFAULT_GARCH_TYPE: garch_preds, "XGBoost": xgb_preds,
                        "XGB-Asymmetric": xgb_asym, "RandomForest": rf_preds},
        train_size=DEFAULT_TRAIN_SIZE, vix_series=vix_col,
    )

    split = int(len(feat_df) * DEFAULT_TRAIN_SIZE)
    rv    = feat_df["realized_vol_21d"] if "realized_vol_21d" in feat_df.columns else feat_df["target"]
    persist_preds = rv.shift(1).iloc[split:]

    forecasts = {
        "Persistence":      persist_preds,
        DEFAULT_GARCH_TYPE: garch_preds,
        "HAR-RV":           har_preds,
        "XGBoost":          xgb_preds,
        "RandomForest":     rf_preds,
        "StackingEnsemble": stack_preds,
    }

    result = regime_conditional_performance(
        ticker=ticker,
        feat_df=feat_df,
        forecasts=forecasts,
        train_size=DEFAULT_TRAIN_SIZE,
    )
    all_results[ticker] = result

print("\n\nDone. Results saved to outputs/results/{ticker}/{ticker}_regime_perf.csv")
