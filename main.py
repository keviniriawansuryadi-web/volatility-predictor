import argparse
import numpy as np
from datetime import date, timedelta

from src.data_loader import load_stock_data, load_vix_data
from src.sentiment import fetch_sentiment
from src.features import build_features, latest_feature_row, FEATURE_COLS
from src.garch_model import rolling_garch_forecast, garch_latest_forecast, garch_in_sample_vol
from src.har_model import har_rv_forecast
from src.ml_model import train_and_predict, predict_latest, feature_importance
from src.evaluate import compare_models, plot_shap
from src.hypothesis import spike_sentiment_test, print_hypothesis_results


def parse_args():
    today = date.today().isoformat()
    five_years_ago = (date.today() - timedelta(days=5 * 365)).isoformat()

    p = argparse.ArgumentParser(description="Stock volatility predictor: GARCH vs. ML")
    p.add_argument("--ticker", default="SPY")
    p.add_argument("--start", default=five_years_ago)
    p.add_argument("--end", default=today)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--train-size", type=float, default=0.8)
    p.add_argument("--garch-type", default="EGARCH", choices=["GARCH", "EGARCH"])
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--plot-dir", default=".")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"  Volatility Predictor -- {args.ticker}")
    print(f"  Period : {args.start} -> {args.end}")
    print(f"  Horizon: {args.horizon} days")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # 1. Load price data
    # ------------------------------------------------------------------
    df = load_stock_data(args.ticker, args.start, args.end, cache=not args.no_cache)
    print(f"Loaded {len(df)} trading days.\n")

    # ------------------------------------------------------------------
    # 2. Enrich df with VIX, sentiment, and GARCH in-sample vol
    # ------------------------------------------------------------------
    print("Loading VIX data...")
    vix_df = load_vix_data(args.start, args.end)
    if not vix_df.empty:
        df = df.join(vix_df, how="left")
        df[["vix_level", "vix_change"]] = df[["vix_level", "vix_change"]].ffill()
        print(f"  VIX loaded ({vix_df.shape[0]} rows).")
    else:
        print("  VIX unavailable — vix features will be skipped.")

    print("\nFetching VADER sentiment from news...")
    sentiment = fetch_sentiment(args.ticker, df.index)
    df["sentiment"] = sentiment

    print(f"\nFitting {args.garch_type} in-sample for hybrid feature...")
    garch_vol_series = garch_in_sample_vol(df["log_return"], model_type=args.garch_type)
    df["garch_vol"] = garch_vol_series

    # ------------------------------------------------------------------
    # 3. Build feature matrix
    # ------------------------------------------------------------------
    feat_df = build_features(df, forecast_horizon=args.horizon)
    print(f"\nFeature matrix: {feat_df.shape}  ({len([c for c in FEATURE_COLS if c in feat_df.columns])} features active)\n")

    # ------------------------------------------------------------------
    # 4. Model training
    # ------------------------------------------------------------------
    print(f"Running {args.garch_type} rolling forecast...")
    garch_preds = rolling_garch_forecast(
        df["log_return"], train_size=args.train_size,
        forecast_horizon=args.horizon, model_type=args.garch_type,
    )

    print("\nFitting HAR-RV model...")
    har_preds = har_rv_forecast(
        df["realized_vol_21d"], train_size=args.train_size,
        forecast_horizon=args.horizon,
    )

    print("\nTraining XGBoost (standard)...")
    xgb_preds, xgb_model, xgb_features = train_and_predict(
        feat_df, model_type="xgboost", train_size=args.train_size)

    print("Training XGBoost (asymmetric spike loss)...")
    xgb_asym_preds, xgb_asym_model, _ = train_and_predict(
        feat_df, model_type="xgboost_asymmetric", train_size=args.train_size)

    print("Training Random Forest...")
    rf_preds, rf_model, _ = train_and_predict(
        feat_df, model_type="random_forest", train_size=args.train_size)

    # ------------------------------------------------------------------
    # 5. Feature importance
    # ------------------------------------------------------------------
    print("\n--- XGBoost Feature Importance (top 8) ---")
    print(feature_importance(xgb_model, xgb_features).head(8).to_string(index=False))

    # ------------------------------------------------------------------
    # 6. Evaluation
    # ------------------------------------------------------------------
    forecasts = {
        args.garch_type: garch_preds,
        "HAR-RV": har_preds,
        "XGBoost": xgb_preds,
        "XGB-Asymmetric": xgb_asym_preds,
        "RandomForest": rf_preds,
    }
    compare_models(
        feat_df,
        forecasts=forecasts,
        train_size=args.train_size,
        ticker=args.ticker,
        plot_dir=args.plot_dir,
    )

    # ------------------------------------------------------------------
    # 7. SHAP plot (standard XGBoost)
    # ------------------------------------------------------------------
    split = int(len(feat_df) * args.train_size)
    available = [c for c in FEATURE_COLS if c in feat_df.columns]
    X_test = feat_df[available].values[split:]
    plot_shap(xgb_model, X_test, available, args.ticker, args.plot_dir)

    # ------------------------------------------------------------------
    # 8. Hypothesis test
    # ------------------------------------------------------------------
    hyp_result = spike_sentiment_test(feat_df)
    print_hypothesis_results(hyp_result)

    # ------------------------------------------------------------------
    # 9. Live forward signal on latest data
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  LIVE SIGNAL -- {args.ticker}  ({args.horizon}-day forward vol)")
    print(f"{'='*60}")

    latest_row = latest_feature_row(df)
    latest_date = latest_row.index[-1].strftime("%Y-%m-%d")
    current_price = df["close"].iloc[-1]
    realized_vol = df["realized_vol_21d"].iloc[-1]

    xgb_now = predict_latest(xgb_model, latest_row)
    xgb_asym_now = predict_latest(xgb_asym_model, latest_row)
    rf_now = predict_latest(rf_model, latest_row)
    print(f"  Fitting {args.garch_type} on full history for forward forecast...")
    garch_now = garch_latest_forecast(df["log_return"], args.horizon, args.garch_type)

    ensemble = float(np.nanmean([xgb_now, xgb_asym_now, rf_now, garch_now]))

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
    print(f"  {args.garch_type:6}     : {garch_now:.1%}")
    print(f"  -- Ensemble: {ensemble:.1%}")
    print(f"")
    print(f"  REGIME: {_regime(ensemble)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
