"""Run AMD diagnostic and save report — uses cached results, no refit needed."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding="utf-8")
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from datetime import date, timedelta
from src.data_loader import load_stock_data, load_vix_data
from src.sentiment import fetch_sentiment, fetch_wsb_sentiment
from src.features import build_features
from src.garch_model import garch_in_sample_vol
from src.diagnostics import diagnose_poor_performer

today = date.today().isoformat()
start = (date.today() - timedelta(days=5 * 365)).isoformat()

for ticker in ["AMD"]:
    print(f"\nBuilding feature matrix for {ticker}...")
    df = load_stock_data(ticker, start, today)
    vix_df = load_vix_data(start, today)
    if not vix_df.empty:
        df = df.join(vix_df, how="left")
        df[["vix_level", "vix_change"]] = df[["vix_level", "vix_change"]].ffill()
    df["sentiment"] = fetch_sentiment(ticker, df.index)
    df["wsb_sentiment"] = fetch_wsb_sentiment(ticker, df.index)
    df["garch_vol"] = garch_in_sample_vol(df["log_return"], model_type="EGARCH")
    feat_df = build_features(df, forecast_horizon=5)
    results_df = pd.read_csv(
        f"outputs/results/{ticker}/{ticker}_model_comparison.csv"
    )
    diagnose_poor_performer(ticker, feat_df, results_df)
