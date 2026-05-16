import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date as _date

CACHE_DIR = Path(__file__).parent.parent / "data"


def load_stock_data(ticker: str, start: str, end: str, cache: bool = True) -> pd.DataFrame:
    CACHE_DIR.mkdir(exist_ok=True)
    cache_path = CACHE_DIR / f"{ticker}_{start}_{end}.csv"

    # Always re-download when end == today so we get the latest bar
    today = _date.today().isoformat()
    use_cache = cache and end != today

    if use_cache and cache_path.exists():
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        print(f"Loaded {ticker} from cache.")
        return df

    print(f"Downloading {ticker} from {start} to {end}...")
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if raw.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'.")

    df = raw[["Close", "Volume"]].copy()
    df.columns = ["close", "volume"]
    df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
    df.index.name = "date"

    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["realized_vol_21d"] = df["log_return"].rolling(21).std() * np.sqrt(252)

    df.dropna(inplace=True)

    if use_cache:
        df.to_csv(cache_path)

    return df


def load_vix_data(start: str, end: str) -> pd.DataFrame:
    """Fetch ^VIX and return vix_level (fractional annual vol) and vix_change columns."""
    try:
        raw = yf.download("^VIX", start=start, end=end, auto_adjust=True, progress=False)
        if raw.empty:
            return pd.DataFrame()
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        vix = raw[["Close"]].copy()
        vix.columns = ["vix_close"]
        vix.index = vix.index.tz_localize(None) if vix.index.tz is not None else vix.index
        vix.index.name = "date"
        vix["vix_level"] = vix["vix_close"] / 100.0
        vix["vix_change"] = vix["vix_level"].diff()
        return vix[["vix_level", "vix_change"]]
    except Exception as e:
        print(f"  [VIX] Failed to load ^VIX: {e}")
        return pd.DataFrame()
