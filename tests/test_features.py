"""Tests for the feature-engineering pipeline in src/features.py."""

import numpy as np

from src.features import _add_features, build_features, latest_feature_row


def test_jump_flag_detects_injected_jump(synthetic_prices):
    """The +15% return injected at row 150 is flagged as a jump."""
    feat = _add_features(synthetic_prices)
    assert feat["jump_flag"].iloc[150] == 1.0
    # jump_flag is strictly binary.
    assert set(feat["jump_flag"].dropna().unique()).issubset({0.0, 1.0})


def test_realized_vol_columns_are_positive(synthetic_prices):
    """All rolling realized-vol columns are non-negative where defined."""
    feat = _add_features(synthetic_prices)
    for col in ["vol_5d", "vol_10d", "vol_21d", "vol_63d"]:
        vals = feat[col].dropna()
        assert (vals >= 0).all(), f"{col} has negative values"


def test_build_features_adds_target_and_drops_nans(synthetic_prices):
    """build_features appends a 'target' column and leaves no NaNs."""
    feat = build_features(synthetic_prices, forecast_horizon=5)
    assert "target" in feat.columns
    assert not feat.isna().any().any()
    assert len(feat) > 0


def test_build_features_target_is_annualized_vol(synthetic_prices):
    """The target (forward realized vol) is strictly positive."""
    feat = build_features(synthetic_prices, forecast_horizon=5)
    assert (feat["target"] > 0).all()


def test_latest_feature_row_returns_single_row(synthetic_prices):
    """latest_feature_row returns exactly one fully-populated row."""
    row = latest_feature_row(synthetic_prices)
    assert len(row) == 1
    assert not np.isnan(row["vol_21d"].iloc[0])
