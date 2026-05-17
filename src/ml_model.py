import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from src.features import FEATURE_COLS


def asymmetric_vol_loss(pred, dtrain):
    """
    Asymmetric XGBoost objective that penalises underestimation more than overestimation.

    Gradient and Hessian follow the standard squared-error derivation, but the
    coefficient is tripled (6 vs 2) when the model underestimates realised vol
    (residual = pred − y_true < 0).  This is mathematically correct:

        L(r) = k * r^2 / 2   where k = 6 if r < 0 else 2
        dL/dr  = k * r         ← gradient returned
        d²L/dr² = k            ← hessian returned

    The old `_spike_weighted_obj` only applied the penalty for y_true > 0.5
    (annualised vol > 50%), which almost never fires on normal tickers.
    This version penalises *any* underestimation, which is the correct
    asymmetry for a risk-aware volatility forecast.
    """
    y_true = dtrain.get_label()
    residual = pred - y_true
    grad = np.where(residual < 0, 6.0 * residual, 2.0 * residual)
    hess = np.where(residual < 0, 6.0, 2.0)
    return grad, hess


class _BoosterWrapper:
    """Wraps xgb.Booster to provide sklearn-compatible predict() and feature_importances_."""

    def __init__(self, booster: xgb.Booster, n_features: int):
        self.booster = booster
        scores = booster.get_score(importance_type="gain")
        raw = np.array([scores.get(f"f{i}", 0.0) for i in range(n_features)])
        total = raw.sum()
        self.feature_importances_ = raw / total if total > 0 else raw

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.booster.predict(xgb.DMatrix(X))


def train_and_predict(
    feat_df: pd.DataFrame,
    model_type: str = "xgboost",
    train_size: float = 0.8,
):
    """
    Train a volatility model on the training split and return test-set predictions.

    Supported model_type values: 'xgboost', 'xgboost_asymmetric', 'random_forest'.
    The asymmetric variant uses a custom XGBoost objective (asymmetric_vol_loss)
    that applies a 3× gradient penalty on any underestimation of realised vol.

    Returns (pred_series, fitted_model, feature_list).
    """
    available = [c for c in FEATURE_COLS if c in feat_df.columns]
    X = feat_df[available].values
    y = feat_df["target"].values

    split = int(len(X) * train_size)
    X_train, X_test = X[:split], X[split:]
    y_train = y[:split]

    if model_type == "xgboost":
        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )
        model.fit(X_train, y_train)

    elif model_type == "xgboost_asymmetric":
        params = {
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "seed": 42,
            "verbosity": 0,
        }
        dtrain = xgb.DMatrix(X_train, label=y_train)
        booster = xgb.train(params, dtrain, num_boost_round=300, obj=asymmetric_vol_loss)
        model = _BoosterWrapper(booster, X_train.shape[1])

    else:  # random_forest
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

    preds = model.predict(X_test)
    pred_series = pd.Series(preds, index=feat_df.index[split:], name=f"{model_type}_forecast")
    return pred_series, model, available  # also return feature list for SHAP


def predict_latest(model, latest_row: pd.DataFrame) -> float:
    """Return a scalar vol forecast for the most recent trading day."""
    available = [c for c in FEATURE_COLS if c in latest_row.columns]
    X = latest_row[available].values
    if isinstance(model, _BoosterWrapper):
        return float(model.predict(X)[0])
    return float(model.predict(X)[0])


def feature_importance(model, feature_names: list) -> pd.DataFrame:
    scores = model.feature_importances_
    n = min(len(scores), len(feature_names))
    return (
        pd.DataFrame({"feature": feature_names[:n], "importance": scores[:n]})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
