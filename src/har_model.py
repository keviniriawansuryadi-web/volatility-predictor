import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


def har_rv_forecast(
    realized_vol: pd.Series,
    train_size: float = 0.8,
    forecast_horizon: int = 5,
) -> pd.Series:
    """
    HAR-RV (Heterogeneous Autoregressive Realized Volatility) model.
    Regresses forward vol on daily, weekly (5d), and monthly (22d) vol averages.
    Consistently beats GARCH on equity data (Corsi 2009).
    """
    rv = realized_vol.copy()
    df = pd.DataFrame({
        "RV_d": rv,
        "RV_w": rv.rolling(5).mean(),
        "RV_m": rv.rolling(22).mean(),
        "target": rv.shift(-forecast_horizon).rolling(forecast_horizon).mean(),
    }).dropna()

    split = int(len(df) * train_size)
    X_train = df[["RV_d", "RV_w", "RV_m"]].iloc[:split].values
    y_train = df["target"].iloc[:split].values
    X_test = df[["RV_d", "RV_w", "RV_m"]].iloc[split:].values

    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = np.maximum(model.predict(X_test), 0.0)  # vol >= 0

    print(f"  HAR-RV coefficients: daily={model.coef_[0]:.3f}, "
          f"weekly={model.coef_[1]:.3f}, monthly={model.coef_[2]:.3f}, "
          f"intercept={model.intercept_:.4f}")

    return pd.Series(preds, index=df.index[split:], name="har_forecast")
