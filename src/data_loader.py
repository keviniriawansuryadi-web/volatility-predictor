import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date as _date

CACHE_DIR = Path(__file__).parent.parent / "data"


def load_stock_data(ticker: str, start: str, end: str, cache: bool = True) -> pd.DataFrame:
    """
    Download OHLCV data for `ticker` from Yahoo Finance and return a cleaned DataFrame.

    Columns returned: close, volume, log_return, realized_vol_21d.
    Results are CSV-cached under data/ to avoid redundant downloads.
    The cache is bypassed when end == today so the latest bar is always fresh.
    """
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


def load_vix_term_structure(start: str, end: str) -> pd.DataFrame:
    """
    Fetches VIX (^VIX, 30-day implied), VIX9D (9-day), and VIX3M (3-month)
    from yfinance and computes three term-structure signals:

    vix_term_slope   : VIX3M - VIX  (contango vs backwardation — negative means
                       inverted curve, historically associated with stress regimes)
    vix_short_squeeze: VIX9D - VIX  (near-term fear relative to 30d baseline)
    vix_rv_premium   : VIX - realized_vol_21d  (variance risk premium — when
                       negative, realized vol has exceeded implied; unusual and
                       often a leading indicator of continued stress or reversion)

    Note: realized_vol_21d is not available in this function — it must be merged
    in from the price DataFrame after calling load_stock_data().  This function
    returns vix_term_slope and vix_short_squeeze only; vix_rv_premium is computed
    in build_features() after the price data is available.

    Returns a DataFrame with columns [vix_level, vix_change, vix9d_level,
    vix3m_level, vix_term_slope, vix_short_squeeze].  Returns an empty
    DataFrame if any series fails to download.
    """
    # Base VIX (30-day)
    vix_base = load_vix_data(start, end)
    if vix_base.empty:
        return pd.DataFrame()

    result = vix_base.copy()

    for symbol, col in [("^VIX9D", "vix9d_level"), ("^VIX3M", "vix3m_level")]:
        try:
            raw = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
            if raw.empty:
                print(f"  [VIX term] {symbol} returned empty — skipping.")
                result[col] = np.nan
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            s = raw["Close"].copy()
            s.index = s.index.tz_localize(None) if s.index.tz is not None else s.index
            s.index.name = "date"
            s_frac = s / 100.0
            s_frac.name = col
            result = result.join(s_frac, how="left")
        except Exception as exc:
            print(f"  [VIX term] Failed to load {symbol}: {exc}")
            result[col] = np.nan

    # Compute term-structure signals
    if "vix3m_level" in result.columns:
        result["vix_term_slope"] = result["vix3m_level"] - result["vix_level"]
    else:
        result["vix_term_slope"] = np.nan

    if "vix9d_level" in result.columns:
        result["vix_short_squeeze"] = result["vix9d_level"] - result["vix_level"]
    else:
        result["vix_short_squeeze"] = np.nan

    cols = ["vix_level", "vix_change"]
    for c in ["vix9d_level", "vix3m_level", "vix_term_slope", "vix_short_squeeze"]:
        if c in result.columns:
            cols.append(c)

    print(f"  [VIX term] Loaded {len(result)} rows; columns: {cols}")
    return result[cols]
