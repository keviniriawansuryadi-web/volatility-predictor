import numpy as np
import pandas as pd
from arch import arch_model


def rolling_garch_forecast(
    log_returns: pd.Series,
    train_size: float = 0.8,
    forecast_horizon: int = 5,
    model_type: str = "EGARCH",
) -> pd.Series:
    """
    Produce out-of-sample vol forecasts using an expanding-window GARCH fit.

    For each step i in the test set the model is re-fitted on all returns
    up to i, then a `forecast_horizon`-step-ahead variance forecast is taken.
    Returns an annualised vol series indexed to the test dates.
    EGARCH uses simulation-based multi-step forecasting (20 paths by default).
    """
    n = len(log_returns)
    split = int(n * train_size)
    returns_scaled = log_returns * 100  # arch library works better with scaled returns

    forecasts = {}

    print(f"  Fitting {model_type} on {split} obs, forecasting {n - split} steps...")
    for i in range(split, n):
        window = returns_scaled.iloc[:i]
        try:
            if model_type == "EGARCH":
                am = arch_model(window, vol="EGARCH", p=1, q=1, dist="normal")
            else:
                am = arch_model(window, vol="GARCH", p=1, q=1, dist="normal")

            res = am.fit(disp="off", show_warning=False)
            fc_kwargs = {"horizon": forecast_horizon, "reindex": False}
            if model_type == "EGARCH":
                fc_kwargs["method"] = "simulation"
                fc_kwargs["simulations"] = 20
            fc = res.forecast(**fc_kwargs)
            var_h = fc.variance.values[-1, -1]
            forecasts[log_returns.index[i]] = np.sqrt(var_h) / 100 * np.sqrt(252)
        except Exception:
            forecasts[log_returns.index[i]] = np.nan

    return pd.Series(forecasts, name="garch_forecast")


def garch_in_sample_vol(
    log_returns: pd.Series,
    model_type: str = "EGARCH",
) -> pd.Series:
    """Fit one GARCH on all returns and return the conditional volatility series (annualized)."""
    returns_scaled = log_returns * 100
    try:
        if model_type == "EGARCH":
            am = arch_model(returns_scaled, vol="EGARCH", p=1, q=1, dist="normal")
        else:
            am = arch_model(returns_scaled, vol="GARCH", p=1, q=1, dist="normal")
        res = am.fit(disp="off", show_warning=False)
        cond_vol = (res.conditional_volatility / 100) * np.sqrt(252)
        return pd.Series(cond_vol.values, index=log_returns.index, name="garch_vol")
    except Exception as e:
        print(f"  [GARCH in-sample] Failed: {e}")
        return pd.Series(np.nan, index=log_returns.index, name="garch_vol")


def garch_latest_forecast(
    log_returns: pd.Series,
    forecast_horizon: int = 5,
    model_type: str = "EGARCH",
) -> float:
    """Fit on all available returns and return a single forward vol estimate."""
    returns_scaled = log_returns * 100
    try:
        if model_type == "EGARCH":
            am = arch_model(returns_scaled, vol="EGARCH", p=1, q=1, dist="normal")
        else:
            am = arch_model(returns_scaled, vol="GARCH", p=1, q=1, dist="normal")
        res = am.fit(disp="off", show_warning=False)
        fc_kwargs = {"horizon": forecast_horizon, "reindex": False}
        if model_type == "EGARCH":
            fc_kwargs["method"] = "simulation"
            fc_kwargs["simulations"] = 100
        fc = res.forecast(**fc_kwargs)
        var_h = fc.variance.values[-1, -1]
        return float(np.sqrt(var_h) / 100 * np.sqrt(252))
    except Exception:
        return float("nan")
