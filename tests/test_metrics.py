"""Tests for evaluation metrics and regime labelling."""

import numpy as np

from src.evaluate import _qlike
from src.regime import _label
from src.regime import REGIME_BOUNDS


def test_qlike_zero_for_perfect_forecast():
    """QLIKE is ~0 when predictions exactly match realized vol."""
    y = np.array([0.1, 0.2, 0.3, 0.25])
    assert np.isclose(_qlike(y, y), 0.0, atol=1e-9)


def test_qlike_positive_for_imperfect_forecast():
    """QLIKE is strictly positive when predictions deviate from truth."""
    y_true = np.array([0.1, 0.2, 0.3, 0.25])
    y_pred = np.array([0.2, 0.1, 0.4, 0.15])
    assert _qlike(y_true, y_pred) > 0


def test_qlike_penalizes_underprediction_more():
    """Under-predicting vol incurs a larger QLIKE penalty than over-predicting
    by the same amount (the metric is asymmetric)."""
    y_true = np.array([0.30])
    under = _qlike(y_true, np.array([0.20]))
    over = _qlike(y_true, np.array([0.40]))
    assert under > over


def test_regime_label_boundaries():
    """_label maps vol levels to the correct regime at and around bounds."""
    assert _label(0.05) == "Low"
    assert _label(REGIME_BOUNDS["Low"]) == "Elevated"
    assert _label(REGIME_BOUNDS["Elevated"]) == "High"
    assert _label(REGIME_BOUNDS["High"]) == "Extreme"
    assert _label(0.99) == "Extreme"
