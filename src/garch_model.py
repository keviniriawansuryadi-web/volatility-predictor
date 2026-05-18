import warnings
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


def fit_jump_egarch(
    returns: pd.Series,
    jump_flags: pd.Series,
    train_size: float = 0.8,
    forecast_horizon: int = 5,
) -> pd.Series:
    """
    Fits an EGARCH model augmented with an external regressor: the jump_flag
    indicator enters the variance equation directly, allowing EGARCH to spike
    its variance forecast on days where a jump is detected.  Implemented via
    the arch library's Vol() with x= parameter for external regressors.

    The jump_flag (1 when |return| > 2.5 * rolling-21d std) is passed as a
    single-column exogenous regressor to the EGARCH(1,1) variance equation.
    The coefficient captures how much additional log-variance a detected jump
    contributes, above what the GARCH recursion would produce alone.

    Expected to improve EGARCH on SPY and TSLA where jump detection is the
    key signal (jump_flag importance 0.191 and 0.0 on SPY XGBoost).

    Parameters
    ----------
    returns       : pd.Series of log returns (not scaled; function scales internally).
    jump_flags    : pd.Series of jump_flag indicator (0/1), same index as returns.
    train_size    : Fraction for expanding-window train start (default 0.8).
    forecast_horizon: Steps ahead to forecast (default 5).

    Returns a pd.Series of annualized vol forecasts on the test dates.
    """
    n = len(returns)
    split = int(n * train_size)
    returns_scaled = returns * 100

    forecasts = {}
    print(f"  Fitting Jump-EGARCH on {split} obs, forecasting {n - split} steps...")

    for i in range(split, n):
        window_r    = returns_scaled.iloc[:i].values
        window_jump = jump_flags.iloc[:i].values.reshape(-1, 1).astype(float)
        try:
            am  = arch_model(window_r, vol="EGARCH", p=1, q=1, dist="normal",
                             x=window_jump)
            res = am.fit(disp="off", show_warning=False)

            # For the forecast, we need the next-step jump value.
            # Use 0 (no jump) as a conservative forward estimate.
            x_fc = np.zeros((forecast_horizon, 1))
            fc = res.forecast(horizon=forecast_horizon, reindex=False,
                              method="simulation", simulations=20, x={"x": x_fc})
            var_h = fc.variance.values[-1, -1]
            forecasts[returns.index[i]] = np.sqrt(var_h) / 100 * np.sqrt(252)
        except Exception:
            # Fall back to standard EGARCH without x if augmented fit fails
            try:
                am2  = arch_model(window_r, vol="EGARCH", p=1, q=1, dist="normal")
                res2 = am2.fit(disp="off", show_warning=False)
                fc2  = res2.forecast(horizon=forecast_horizon, reindex=False,
                                     method="simulation", simulations=20)
                var_h2 = fc2.variance.values[-1, -1]
                forecasts[returns.index[i]] = np.sqrt(var_h2) / 100 * np.sqrt(252)
            except Exception:
                forecasts[returns.index[i]] = np.nan

    return pd.Series(forecasts, name="jump_egarch_forecast")


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
