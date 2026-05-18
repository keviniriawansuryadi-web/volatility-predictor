import pandas as pd
import numpy as np

FEATURE_COLS = [
    # Returns
    "ret_lag1", "ret_lag2", "ret_lag3", "ret_lag5", "ret_lag10",
    # Realized vol at multiple lookbacks
    "vol_5d", "vol_10d", "vol_20d", "vol_21d", "vol_60d", "vol_63d",
    # Vol-of-vol
    "vol_of_vol",
    # Jump indicator
    "jump_flag",
    # Volume
    "volume_ratio",
    # Statistical
    "ret_skew_21d", "ret_kurt_21d",
    # Technical
    "rsi_14", "bb_width",
    # VIX (populated when ^VIX data is available)
    "vix_level", "vix_change",
    # VIX term structure (populated when load_vix_term_structure() is used)
    "vix_term_slope", "vix_short_squeeze", "vix_rv_premium",
    # Sentiment (populated when news data is available)
    "sentiment_3d", "sentiment_lag1", "sentiment_lag2", "sentiment_lag3",
    # Reddit WSB sentiment (populated when scraper returns data)
    "wsb_sentiment_3d", "wsb_sentiment_lag1",
    # Vol lags
    "vol_lag1", "vol_lag2", "vol_lag3",
    # GARCH fitted vol (hybrid feature)
    "garch_vol",
]


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the full set of predictive features from a price DataFrame.

    Expects columns: close, volume, log_return, realized_vol_21d.
    Optionally enriches with vix_level/vix_change and sentiment if present.
    Returns a copy of df with all FEATURE_COLS appended (NaNs for missing data).
    """
    feat = df.copy()

    # Lagged returns
    for lag in [1, 2, 3, 5, 10]:
        feat[f"ret_lag{lag}"] = feat["log_return"].shift(lag)

    # Realized vol at multiple lookbacks
    for window in [5, 10, 20, 21, 60, 63]:
        feat[f"vol_{window}d"] = feat["log_return"].rolling(window).std() * np.sqrt(252)

    # Vol-of-vol (rolling std of 21d vol)
    feat["vol_of_vol"] = feat["vol_21d"].rolling(10).std()

    # Jump indicator: |return| > 2.5 * 21d rolling std
    rolling_std = feat["log_return"].rolling(21).std()
    feat["jump_flag"] = (feat["log_return"].abs() > 2.5 * rolling_std).astype(float)

    # Volume features
    feat["volume_ma10"] = feat["volume"].rolling(10).mean()
    feat["volume_ratio"] = feat["volume"] / feat["volume_ma10"]

    # Return skew / kurtosis (rolling 21d)
    feat["ret_skew_21d"] = feat["log_return"].rolling(21).skew()
    feat["ret_kurt_21d"] = feat["log_return"].rolling(21).kurt()

    # RSI (14-day)
    delta = feat["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    feat["rsi_14"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    # Bollinger band width
    sma20 = feat["close"].rolling(20).mean()
    std20 = feat["close"].rolling(20).std()
    feat["bb_width"] = (2 * std20) / sma20

    # Vol lags (lagged realized_vol_21d)
    if "realized_vol_21d" in feat.columns:
        for lag in [1, 2, 3]:
            feat[f"vol_lag{lag}"] = feat["realized_vol_21d"].shift(lag)

    # Sentiment-derived features (when sentiment column exists in df)
    if "sentiment" in feat.columns:
        feat["sentiment_3d"] = feat["sentiment"].rolling(3).mean()
        for lag in [1, 2, 3]:
            feat[f"sentiment_lag{lag}"] = feat["sentiment"].shift(lag)

    # WSB sentiment features (when wsb_sentiment column exists in df)
    if "wsb_sentiment" in feat.columns:
        feat["wsb_sentiment_3d"]   = feat["wsb_sentiment"].rolling(3).mean()
        feat["wsb_sentiment_lag1"] = feat["wsb_sentiment"].shift(1)

    # VIX term structure features (populated when load_vix_term_structure() is used)
    # vix_term_slope and vix_short_squeeze are passed through directly from the DF.
    # vix_rv_premium = VIX - realized_vol_21d (variance risk premium)
    if "vix_level" in feat.columns and "realized_vol_21d" in feat.columns:
        feat["vix_rv_premium"] = feat["vix_level"] - feat["realized_vol_21d"]

    return feat


def build_features(df: pd.DataFrame, forecast_horizon: int = 5) -> pd.DataFrame:
    """
    Build the supervised feature matrix used for model training.

    Calls _add_features, then appends a 'target' column equal to the
    annualised realized vol over the next `forecast_horizon` trading days.
    Rows with any NaN (early lookback period) are dropped.
    """
    feat = _add_features(df)
    feat["target"] = (
        feat["log_return"].shift(-forecast_horizon).rolling(forecast_horizon).std() * np.sqrt(252)
    )
    feat.dropna(inplace=True)
    return feat


def latest_feature_row(df: pd.DataFrame) -> pd.DataFrame:
    """Return the last row with all base features computed (no target required)."""
    feat = _add_features(df)
    available = [c for c in FEATURE_COLS if c in feat.columns]
    return feat.dropna(subset=available).iloc[[-1]]
