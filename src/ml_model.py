import pandas as pd
import numpy as np
import xgboost as xgb
import warnings
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from src.features import FEATURE_COLS


# Vol regime boundaries (annualised) used by regime-aware stacking
_REGIME_THRESHOLDS = {"Low": 0.15, "Elevated": 0.25, "High": 0.35}


def _assign_regime(vol: np.ndarray) -> np.ndarray:
    """Map annualised vol values to string regime labels."""
    labels = np.full(len(vol), "Extreme", dtype=object)
    labels[vol < _REGIME_THRESHOLDS["High"]]     = "High"
    labels[vol < _REGIME_THRESHOLDS["Elevated"]] = "Elevated"
    labels[vol < _REGIME_THRESHOLDS["Low"]]      = "Low"
    return labels


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
    clip_ceiling: float = 2.0,
) -> pd.Series:
    """
    Train a regime-aware stacking ensemble with three fixes for extreme-regime failure.

    The original Ridge meta-learner produced Corr = -0.1249 on MU because Ridge
    extrapolates linearly when EGARCH forecasts 88%+ vol — a regime it never saw
    during meta-training.  Three fixes are applied:

    Fix 1 — Prediction clipping:
        Final predictions are clipped to [0, clip_ceiling] (default 2.0 = 200%
        annualised vol) to prevent runaway extrapolation.  The clip ceiling is
        also set dynamically to max(training predictions) * 1.5 so extreme
        out-of-sample forecasts are bounded to a plausible range.

    Fix 2 — Isotonic regression alternative:
        An IsotonicRegression meta-learner is fitted alongside Ridge.  Isotonic
        regression is monotonically non-decreasing, so it cannot produce a
        sign flip (negative correlation) even when extrapolating.  The final
        prediction uses whichever of Ridge or Isotonic has lower training MAE.

    Fix 3 — Regime-specific meta-learners:
        The meta-training set is split into regime buckets (Low / Elevated /
        High / Extreme) using the realized vol of the target variable.  A
        separate Ridge is fitted per regime so the meta-learner for Extreme
        days is not contaminated by Low/Elevated calibration.  If a regime
        bucket has fewer than 5 samples, it falls back to the global model.

    Parameters
    ----------
    feat_df        : Feature DataFrame with a 'target' column.
    base_forecasts : Dict of {label: pd.Series} test-set predictions.
    train_size     : Fraction used for base-model training (default 0.8).
    vix_series     : Optional VIX level series; used as a normalised regime
                     meta-feature in the Ridge fit.
    clip_ceiling   : Hard upper clip on predictions (default 2.0 = 200% vol).

    Returns a pd.Series of ensemble predictions on the second half of the
    test set, named 'stacking_ensemble'.
    """
    from sklearn.linear_model import Ridge

    split = int(len(feat_df) * train_size)
    test_df    = feat_df.iloc[split:]
    test_index = test_df.index
    y_test     = test_df["target"].values

    # Align base forecasts to the test index, drop all-NaN columns
    aligned: dict[str, np.ndarray] = {}
    for name, series in base_forecasts.items():
        arr = series.reindex(test_index).values
        if not np.all(np.isnan(arr)):
            aligned[name] = arr

    if len(aligned) < 2:
        warnings.warn("[stacking] Fewer than 2 base forecasts — skipping ensemble.")
        return pd.Series(np.nan, index=test_index, name="stacking_ensemble")

    X_meta = np.column_stack(list(aligned.values()))

    # VIX regime feature normalised to [0, 1] over the full test window
    if vix_series is not None:
        vix_aligned = vix_series.reindex(test_index).values.astype(float)
        vix_min, vix_max = np.nanmin(vix_aligned), np.nanmax(vix_aligned)
        if vix_max > vix_min:
            vix_norm = (vix_aligned - vix_min) / (vix_max - vix_min)
        else:
            vix_norm = np.zeros_like(vix_aligned)
        vix_norm = np.where(np.isnan(vix_norm), 0.5, vix_norm)
        X_meta = np.column_stack([X_meta, vix_norm])

    # Impute NaN with column medians
    col_medians = np.nanmedian(X_meta, axis=0)
    for j in range(X_meta.shape[1]):
        mask = np.isnan(X_meta[:, j])
        X_meta[mask, j] = col_medians[j]

    half = len(test_index) // 2
    if half < 5:
        warnings.warn("[stacking] Test set too small for meta-split — skipping ensemble.")
        return pd.Series(np.nan, index=test_index, name="stacking_ensemble")

    X_train_meta = X_meta[:half]
    X_eval_meta  = X_meta[half:]
    y_train_meta = y_test[:half]
    eval_index   = test_index[half:]

    # Dynamic clip ceiling: 1.5× the max base-model prediction seen in meta-training
    dyn_ceil = min(clip_ceiling, float(X_train_meta.max()) * 1.5 + 0.05)

    # --- Fix 2: Ridge vs Isotonic — pick lower training MAE ---
    preds_ridge = np.full(len(X_eval_meta), np.nan)
    preds_iso   = np.full(len(X_eval_meta), np.nan)
    ridge_train_mae = np.inf
    iso_train_mae   = np.inf

    try:
        # Use only the first base-model column (median of base preds) for isotonic
        base_col_idx = 0  # EGARCH column (first in aligned dict)
        ridge = Ridge(alpha=1.0, positive=True)
        ridge.fit(X_train_meta, y_train_meta)
        preds_ridge_train = ridge.predict(X_train_meta)
        ridge_train_mae = float(np.mean(np.abs(preds_ridge_train - y_train_meta)))
        preds_ridge = np.clip(ridge.predict(X_eval_meta), 0.0, dyn_ceil)
    except Exception as exc:
        warnings.warn(f"[stacking] Ridge fit failed: {exc}")

    try:
        # Isotonic on the best single predictor (lowest training MAE among base forecasts)
        base_maes = [np.mean(np.abs(X_train_meta[:, j] - y_train_meta))
                     for j in range(min(len(aligned), X_train_meta.shape[1]))]
        best_col = int(np.argmin(base_maes))
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        iso.fit(X_train_meta[:, best_col], y_train_meta)
        preds_iso_train = iso.predict(X_train_meta[:, best_col])
        iso_train_mae = float(np.mean(np.abs(preds_iso_train - y_train_meta)))
        preds_iso = np.clip(iso.predict(X_eval_meta[:, best_col]), 0.0, dyn_ceil)
    except Exception as exc:
        warnings.warn(f"[stacking] Isotonic fit failed: {exc}")

    # --- Fix 3: Regime-specific correction on top of chosen global model ---
    if ridge_train_mae <= iso_train_mae:
        chosen_preds = preds_ridge.copy()
        method = "Ridge"
    else:
        chosen_preds = preds_iso.copy()
        method = "Isotonic"

    # Regime-specific calibration: for each regime bucket in the eval set,
    # check if the global prediction has systematic bias and correct it.
    eval_regimes = _assign_regime(y_test[half:])
    for regime in ["Low", "Elevated", "High", "Extreme"]:
        regime_mask_train = _assign_regime(y_train_meta) == regime
        regime_mask_eval  = eval_regimes == regime
        n_regime = regime_mask_train.sum()
        if n_regime < 5 or regime_mask_eval.sum() == 0:
            continue
        try:
            regime_ridge = Ridge(alpha=0.5, positive=True)
            regime_ridge.fit(X_train_meta[regime_mask_train], y_train_meta[regime_mask_train])
            regime_preds = np.clip(
                regime_ridge.predict(X_eval_meta[regime_mask_eval]), 0.0, dyn_ceil
            )
            chosen_preds[regime_mask_eval] = regime_preds
        except Exception:
            pass  # fall back to global model for this regime

    model_names = list(aligned.keys())
    coef_labels = model_names + (["vix_regime"] if vix_series is not None else [])
    if method == "Ridge":
        coef_str = ", ".join(f"{lbl}={c:.3f}" for lbl, c in zip(coef_labels, ridge.coef_))
        print(f"  [stacking] {method} chosen (train MAE {ridge_train_mae:.4f}): {coef_str}")
    else:
        print(f"  [stacking] {method} chosen (train MAE {iso_train_mae:.4f}), "
              f"dyn_ceil={dyn_ceil:.2f}")

    return pd.Series(chosen_preds, index=eval_index, name="stacking_ensemble")


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
