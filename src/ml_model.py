import pandas as pd
import numpy as np
import xgboost as xgb
import warnings
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
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


def train_quantile_models(
    feat_df: pd.DataFrame,
    train_size: float = 0.8,
    quantiles: tuple = (0.10, 0.50, 0.90),
    n_estimators: int = 200,
) -> dict[str, pd.Series]:
    """
    Train a GradientBoostingRegressor for each quantile and return test-set forecasts.

    Quantile regression provides a probabilistic band around the point forecast:
      - q10 is the lower bound (model expects vol to stay above this ~90% of the time)
      - q50 is the median forecast (robust to outliers, a better central estimate)
      - q90 is the upper bound (early warning for spike days)

    Skips gracefully on any error and returns an empty dict entry for that quantile
    rather than crashing the full pipeline.

    Parameters
    ----------
    feat_df      : Feature DataFrame with a 'target' column (output of build_features).
    train_size   : Fraction of data used for training (default 0.8).
    quantiles    : Tuple of quantiles to fit (default (0.10, 0.50, 0.90)).
    n_estimators : Number of boosting rounds per quantile (default 200).

    Returns a dict mapping 'q10'/'q50'/'q90' → pd.Series of test-set predictions.
    """
    available = [c for c in FEATURE_COLS if c in feat_df.columns]
    X = feat_df[available].values
    y = feat_df["target"].values
    split = int(len(X) * train_size)
    X_train, X_test = X[:split], X[split:]
    y_train = y[:split]
    test_index = feat_df.index[split:]

    result: dict[str, pd.Series] = {}
    for q in quantiles:
        key = f"q{int(q * 100):02d}"
        try:
            model = GradientBoostingRegressor(
                loss="quantile",
                alpha=q,
                n_estimators=n_estimators,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            result[key] = pd.Series(preds, index=test_index, name=f"quantile_{key}")
        except Exception as exc:
            warnings.warn(f"[quantile] {key} model failed: {exc} — skipping.")
    return result


def train_stacking_ensemble(
    feat_df: pd.DataFrame,
    base_forecasts: dict[str, pd.Series],
    train_size: float = 0.8,
    vix_series: pd.Series | None = None,
) -> pd.Series:
    """
    Train a Ridge regression meta-learner that stacks base model forecasts.

    The stacking protocol is:
      - The test set is split in half: the first half is used to fit the
        meta-learner (so it sees OOF-style predictions), and the second half
        is the true evaluation window.
      - Base model forecasts are used as features; if a 'vix_level' column
        exists in feat_df, a normalised VIX regime indicator is added as an
        extra meta-feature so the ensemble can learn to trust different models
        in high-vol vs low-vol regimes.
      - The meta-learner is constrained to non-negative weights so it behaves
        like a proper weighted average (no short-selling of model outputs).

    Skips gracefully and returns a constant-NaN series if fewer than 2 base
    forecasts are available or if the Ridge fit fails.

    Parameters
    ----------
    feat_df        : Full feature DataFrame (must contain a 'target' column).
    base_forecasts : Dict of {label: pd.Series} test-set predictions from base models.
    train_size     : Fraction used for the base-model training split (default 0.8).
    vix_series     : Optional pd.Series of VIX levels indexed to feat_df.index;
                     used as a regime feature in the meta-learner.

    Returns a pd.Series of stacked ensemble predictions on the second half of the
    test set, named 'stacking_ensemble'.
    """
    from sklearn.linear_model import Ridge

    split = int(len(feat_df) * train_size)
    test_df = feat_df.iloc[split:]
    test_index = test_df.index
    y_test = test_df["target"].values

    # Align all base forecasts to the test index
    aligned: dict[str, np.ndarray] = {}
    for name, series in base_forecasts.items():
        arr = series.reindex(test_index).values
        if not np.all(np.isnan(arr)):
            aligned[name] = arr

    if len(aligned) < 2:
        warnings.warn("[stacking] Fewer than 2 base forecasts available — skipping ensemble.")
        return pd.Series(np.nan, index=test_index, name="stacking_ensemble")

    X_meta = np.column_stack(list(aligned.values()))

    # Add VIX regime feature: normalised to [0, 1] over the test window
    if vix_series is not None:
        vix_aligned = vix_series.reindex(test_index).values.astype(float)
        vix_min, vix_max = np.nanmin(vix_aligned), np.nanmax(vix_aligned)
        if vix_max > vix_min:
            vix_norm = (vix_aligned - vix_min) / (vix_max - vix_min)
        else:
            vix_norm = np.zeros_like(vix_aligned)
        # Fill residual NaN with 0.5 (neutral regime)
        vix_norm = np.where(np.isnan(vix_norm), 0.5, vix_norm)
        X_meta = np.column_stack([X_meta, vix_norm])

    # Replace NaN in base forecasts with column medians before fitting
    col_medians = np.nanmedian(X_meta, axis=0)
    nan_mask = np.isnan(X_meta)
    for j in range(X_meta.shape[1]):
        X_meta[nan_mask[:, j], j] = col_medians[j]

    # Meta-train on first half of test, evaluate on second half
    half = len(test_index) // 2
    if half < 5:
        warnings.warn("[stacking] Test set too small for meta-train split — skipping ensemble.")
        return pd.Series(np.nan, index=test_index, name="stacking_ensemble")

    X_meta_train, X_meta_eval = X_meta[:half], X_meta[half:]
    y_meta_train = y_test[:half]
    eval_index   = test_index[half:]

    try:
        ridge = Ridge(alpha=1.0, positive=True)
        ridge.fit(X_meta_train, y_meta_train)
        preds_eval = ridge.predict(X_meta_eval)
    except Exception as exc:
        warnings.warn(f"[stacking] Ridge fit failed: {exc} — skipping ensemble.")
        return pd.Series(np.nan, index=test_index, name="stacking_ensemble")

    model_names = list(aligned.keys())
    n_base = len(model_names)
    coef_labels = model_names + (["vix_regime"] if vix_series is not None else [])
    coef_str = ", ".join(f"{lbl}={c:.3f}" for lbl, c in zip(coef_labels, ridge.coef_))
    print(f"  [stacking] Ridge meta-learner coefficients: {coef_str}")

    return pd.Series(preds_eval, index=eval_index, name="stacking_ensemble")


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
