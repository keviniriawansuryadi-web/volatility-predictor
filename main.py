import argparse
import numpy as np
from datetime import date, timedelta

from src.data_loader import load_stock_data, load_vix_data
from src.sentiment import fetch_sentiment, fetch_wsb_sentiment
from src.features import build_features, latest_feature_row, FEATURE_COLS
from src.garch_model import rolling_garch_forecast, garch_latest_forecast, garch_in_sample_vol
from src.har_model import har_rv_forecast
from src.ml_model import train_and_predict, predict_latest, feature_importance
from src.evaluate import compare_models, plot_shap
from src.hypothesis import spike_sentiment_test, print_hypothesis_results
from config import TICKERS, DEFAULT_START, DEFAULT_END, DEFAULT_HORIZON, DEFAULT_TRAIN_SIZE, DEFAULT_GARCH_TYPE


def parse_args():
    today          = date.today().isoformat()
    five_years_ago = (date.today() - timedelta(days=5 * 365)).isoformat()

    p = argparse.ArgumentParser(description="Stock volatility predictor: GARCH vs. ML")
    p.add_argument("--ticker",      default="SPY",  help="Single ticker to analyse")
    p.add_argument("--all-tickers", action="store_true",
                   help="Run on all tickers defined in config.TICKERS")
    p.add_argument("--start",       default=five_years_ago,
                   help="Start date (default: 5 years ago)")
    p.add_argument("--end",         default=today,
                   help="End date (default: today)")
    p.add_argument("--horizon",     type=int,   default=DEFAULT_HORIZON)
    p.add_argument("--train-size",  type=float, default=DEFAULT_TRAIN_SIZE)
    p.add_argument("--garch-type",  default=DEFAULT_GARCH_TYPE, choices=["GARCH", "EGARCH"])
    p.add_argument("--no-cache",    action="store_true")
    p.add_argument("--plot-dir",    default=".")
    return p.parse_args()


def run_ticker(
    ticker: str,
    start: str,
    end: str,
    horizon: int,
    train_size: float,
    garch_type: str,
    use_cache: bool,
    plot_dir: str,
) -> dict:
    """
    Run the full volatility-forecasting pipeline for a single ticker.

    Downloads price and VIX data, engineers features, trains EGARCH / HAR-RV /
    XGBoost (standard + asymmetric) / Random Forest, evaluates on a held-out
    test set, saves a forecast chart and a metrics CSV, then prints a live
    forward signal.

    Returns a dict with keys: ticker, metrics_df, hyp_result.
    """
    print(f"\n{'='*60}")
    print(f"  Volatility Predictor -- {ticker}")
    print(f"  Period : {start} -> {end}")
    print(f"  Horizon: {horizon} days")
    print(f"{'='*60}\n")

    # 1. Price data
    df = load_stock_data(ticker, start, end, cache=use_cache)
    print(f"Loaded {len(df)} trading days.\n")

    # 2. VIX, sentiment, GARCH in-sample vol
    print("Loading VIX data...")
    vix_df = load_vix_data(start, end)
    if not vix_df.empty:
        df = df.join(vix_df, how="left")
        df[["vix_level", "vix_change"]] = df[["vix_level", "vix_change"]].ffill()
        print(f"  VIX loaded ({vix_df.shape[0]} rows).")
    else:
        print("  VIX unavailable — vix features will be skipped.")

    print("\nFetching VADER sentiment from news...")
    df["sentiment"] = fetch_sentiment(ticker, df.index)

    print("\nFetching Reddit WSB sentiment...")
    df["wsb_sentiment"] = fetch_wsb_sentiment(ticker, df.index)

    print(f"\nFitting {garch_type} in-sample for hybrid feature...")
    df["garch_vol"] = garch_in_sample_vol(df["log_return"], model_type=garch_type)

    # 3. Feature matrix
    feat_df = build_features(df, forecast_horizon=horizon)
    active = [c for c in FEATURE_COLS if c in feat_df.columns]
    print(f"\nFeature matrix: {feat_df.shape}  ({len(active)} features active)\n")

    # 4. Models
    print(f"Running {garch_type} rolling forecast...")
    garch_preds = rolling_garch_forecast(
        df["log_return"], train_size=train_size,
        forecast_horizon=horizon, model_type=garch_type,
    )

    print("\nFitting HAR-RV model...")
    har_preds = har_rv_forecast(
        df["realized_vol_21d"], train_size=train_size, forecast_horizon=horizon)

    print("\nTraining XGBoost (standard)...")
    xgb_preds, xgb_model, xgb_features = train_and_predict(
        feat_df, model_type="xgboost", train_size=train_size)

    print("Training XGBoost (asymmetric spike loss)...")
    xgb_asym_preds, xgb_asym_model, _ = train_and_predict(
        feat_df, model_type="xgboost_asymmetric", train_size=train_size)

    print("Training Random Forest...")
    rf_preds, rf_model, _ = train_and_predict(
        feat_df, model_type="random_forest", train_size=train_size)

    # 5. Feature importance
    print("\n--- XGBoost Feature Importance (top 8) ---")
    print(feature_importance(xgb_model, xgb_features).head(8).to_string(index=False))

    # 6. Evaluation + save metrics CSV + save plot
    forecasts = {
        garch_type:       garch_preds,
        "HAR-RV":         har_preds,
        "XGBoost":        xgb_preds,
        "XGB-Asymmetric": xgb_asym_preds,
        "RandomForest":   rf_preds,
    }
    metrics_df = compare_models(
        feat_df, forecasts=forecasts,
        train_size=train_size, ticker=ticker, plot_dir=plot_dir,
    )

    # 7. SHAP plot
    split = int(len(feat_df) * train_size)
    X_test = feat_df[xgb_features].values[split:]
    plot_shap(xgb_model, X_test, xgb_features, ticker, plot_dir)

    # 8. Hypothesis test
    hyp_result = spike_sentiment_test(feat_df)
    print_hypothesis_results(hyp_result)

    # 9. Live forward signal
    print(f"\n{'='*60}")
    print(f"  LIVE SIGNAL -- {ticker}  ({horizon}-day forward vol)")
    print(f"{'='*60}")

    latest_row   = latest_feature_row(df)
    latest_date  = latest_row.index[-1].strftime("%Y-%m-%d")
    current_price = df["close"].iloc[-1]
    realized_vol  = df["realized_vol_21d"].iloc[-1]

    xgb_now      = predict_latest(xgb_model, latest_row)
    xgb_asym_now = predict_latest(xgb_asym_model, latest_row)
    rf_now       = predict_latest(rf_model, latest_row)
    print(f"  Fitting {garch_type} on full history for forward forecast...")
    garch_now    = garch_latest_forecast(df["log_return"], horizon, garch_type)
    ensemble     = float(np.nanmean([xgb_now, xgb_asym_now, rf_now, garch_now]))

    def _regime(v: float) -> str:
        if v > 0.35: return "EXTREME  -- Very high risk/reward, tight stops essential"
        if v > 0.25: return "HIGH     -- Strong scalping conditions"
        if v > 0.15: return "ELEVATED -- Decent intraday movement expected"
        if v > 0.10: return "MODERATE -- Selective scalps only"
        return              "LOW      -- Avoid scalping, insufficient movement"

    print(f"  As of      : {latest_date}")
    print(f"  Price      : ${current_price:.2f}")
    print(f"  21d Real.Vol: {realized_vol:.1%}")
    print(f"")
    print(f"  XGBoost    : {xgb_now:.1%}")
    print(f"  XGB-Asym   : {xgb_asym_now:.1%}")
    print(f"  Rand.Forest: {rf_now:.1%}")
    print(f"  {garch_type:6}     : {garch_now:.1%}")
    print(f"  -- Ensemble: {ensemble:.1%}")
    print(f"")
    print(f"  REGIME: {_regime(ensemble)}")
    print(f"{'='*60}\n")

    return {"ticker": ticker, "metrics_df": metrics_df, "hyp_result": hyp_result}


def main():
    args = parse_args()
    use_cache = not args.no_cache

    if args.all_tickers:
        all_results = []
        for ticker in TICKERS:
            try:
                result = run_ticker(
                    ticker=ticker,
                    start=args.start,
                    end=args.end,
                    horizon=args.horizon,
                    train_size=args.train_size,
                    garch_type=args.garch_type,
                    use_cache=use_cache,
                    plot_dir=args.plot_dir,
                )
                all_results.append(result)
            except Exception as e:
                print(f"\n[ERROR] {ticker} failed: {e}\n")

        print(f"\n{'='*60}")
        print(f"  Batch complete: {len(all_results)}/{len(TICKERS)} tickers processed.")
        print(f"  Metrics saved to outputs/results/{{ticker}}/metrics.csv")
        print(f"  Plots saved to outputs/plots/")
        print(f"{'='*60}\n")
    else:
        run_ticker(
            ticker=args.ticker,
            start=args.start,
            end=args.end,
            horizon=args.horizon,
            train_size=args.train_size,
            garch_type=args.garch_type,
            use_cache=use_cache,
            plot_dir=args.plot_dir,
        )


if __name__ == "__main__":
    main()
