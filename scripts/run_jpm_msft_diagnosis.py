"""Run diagnose_poor_performer() on JPM and MSFT."""
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
from src.diagnostics import diagnose_poor_performer
from config import DEFAULT_TRAIN_SIZE, DEFAULT_GARCH_TYPE

TICKERS = ["JPM", "MSFT"]
TODAY   = date.today().isoformat()
START   = (date.today() - timedelta(days=5 * 365)).isoformat()

for ticker in TICKERS:
    print(f"\n{'='*50}\n  {ticker}\n{'='*50}")
    df = load_stock_data(ticker, START, TODAY, cache=True)

    vix_df = load_vix_data(START, TODAY)
    if not vix_df.empty:
        df = df.join(vix_df, how="left")
        df[["vix_level","vix_change"]] = df[["vix_level","vix_change"]].ffill()

    df["sentiment"]     = fetch_sentiment(ticker, df.index)
    df["wsb_sentiment"] = fetch_wsb_sentiment(ticker, df.index)
    df["garch_vol"]     = garch_in_sample_vol(df["log_return"], model_type=DEFAULT_GARCH_TYPE)

    feat_df = build_features(df, forecast_horizon=21)

    from src.evaluate import compare_models
    from src.har_model import har_rv_forecast

    garch_preds = rolling_garch_forecast(df["log_return"], DEFAULT_TRAIN_SIZE, 21, DEFAULT_GARCH_TYPE)
    har_preds   = har_rv_forecast(df["realized_vol_21d"], train_size=DEFAULT_TRAIN_SIZE, forecast_horizon=21)
    xgb_preds, _, _ = train_and_predict(feat_df, model_type="xgboost", train_size=DEFAULT_TRAIN_SIZE)
    xgb_asym, _, _  = train_and_predict(feat_df, model_type="xgboost_asymmetric", train_size=DEFAULT_TRAIN_SIZE)
    rf_preds, _, _  = train_and_predict(feat_df, model_type="random_forest", train_size=DEFAULT_TRAIN_SIZE)

    forecasts  = {DEFAULT_GARCH_TYPE: garch_preds, "HAR-RV": har_preds,
                  "XGBoost": xgb_preds, "XGB-Asymmetric": xgb_asym, "RandomForest": rf_preds}
    results_df = compare_models(feat_df, forecasts=forecasts, train_size=DEFAULT_TRAIN_SIZE,
                                ticker=ticker, plot_dir=".").reset_index()

    diagnose_poor_performer(
        ticker=ticker,
        feat_df=feat_df,
        results_df=results_df,
        qlike_threshold=0.3,
        plot_dir=".",
    )
