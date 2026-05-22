"""Smoke test for the core modelling path (no network access).

Exercises feature engineering -> model training -> evaluation -> regime
analysis on synthetic data to verify the pieces wire together and nothing
crashes. This intentionally avoids yfinance/news downloads so it can run in CI.
"""

import numpy as np

from src.features import build_features, FEATURE_COLS
from src.ml_model import train_and_predict, feature_importance
from src.evaluate import _qlike
from src.regime import analyze_regime_persistence


def test_pipeline_smoke_end_to_end(synthetic_prices):
    """Build features, train XGBoost, score it, and analyse regimes."""
    feat = build_features(synthetic_prices, forecast_horizon=5)
    assert len(feat) > 50, "not enough rows survived feature engineering"

    preds, model, used_features = train_and_predict(
        feat, model_type="xgboost", train_size=0.8
    )
    # Predictions cover the test split and are finite & positive.
    assert len(preds) > 0
    assert np.isfinite(preds.values).all()
    assert (preds.values > 0).all()

    # Every used feature is a known feature column.
    assert set(used_features).issubset(set(FEATURE_COLS))

    # Importance table is well-formed.
    imp = feature_importance(model, used_features)
    assert list(imp.columns) == ["feature", "importance"]
    assert len(imp) == len(used_features)

    # QLIKE against the realized test target is a finite, non-negative number.
    split = int(len(feat) * 0.8)
    y_test = feat["target"].values[split:]
    aligned = preds.reindex(feat.index[split:]).values
    mask = ~np.isnan(aligned)
    score = _qlike(y_test[mask], aligned[mask])
    assert np.isfinite(score)
    assert score >= 0


def test_regime_analysis_returns_valid_label(synthetic_prices):
    """Regime persistence analysis returns a valid current regime label."""
    result = analyze_regime_persistence(synthetic_prices, ticker="TEST")
    assert result["current_regime"] in {"Low", "Elevated", "High", "Extreme"}
    assert result["expected_reversion_days"] > 0
