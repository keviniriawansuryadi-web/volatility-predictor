"""Run cross-ticker sentiment decomposition using cached price data."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import warnings
warnings.filterwarnings("ignore")
from datetime import date, timedelta
from src.data_loader import load_stock_data
from src.sentiment import fetch_sentiment, fetch_wsb_sentiment
from src.sentiment_decomp import decompose_sentiment
from config import TICKERS

today = date.today().isoformat()
start = (date.today() - timedelta(days=5 * 365)).isoformat()

print("Loading price + sentiment data for all tickers...")
df_dict = {}
for t in TICKERS:
    try:
        df = load_stock_data(t, start, today, cache=True)
        df["sentiment"]     = fetch_sentiment(t, df.index)
        df["wsb_sentiment"] = fetch_wsb_sentiment(t, df.index)
        df_dict[t] = df
        print(f"  {t}: {len(df)} rows loaded")
    except Exception as e:
        print(f"  {t}: FAILED — {e}")

print(f"\nRunning sentiment decomposition across {len(df_dict)} tickers...")
result = decompose_sentiment(df_dict, list(df_dict.keys()))

if result:
    print("\n  === SUMMARY ===")
    print(f"  Systematic-dominant tickers (>60%): {result['flag_systematic']}")
    print(f"\n  Variance decomposition:")
    for t in result["systematic_var_pct"].index:
        sp = result["systematic_var_pct"][t]
        ip = result["idio_var_pct"][t]
        gs = result["granger_systematic"].get(t, float("nan"))
        gi = result["granger_idio"].get(t, float("nan"))
        import numpy as np
        flag = "  *** SYSTEMATIC DOMINANT ***" if sp > 60 else ""
        gs_str = f"{gs:.4f}" if not np.isnan(gs) else "n/a"
        gi_str = f"{gi:.4f}" if not np.isnan(gi) else "n/a"
        print(f"    {t:<6}: sys={sp:.1f}%  idio={ip:.1f}%  "
              f"granger_sys={gs_str}  granger_idio={gi_str}{flag}")
