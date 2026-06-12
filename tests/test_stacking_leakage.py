"""
Regression test: the stacking ensemble must not leak the evaluation-window
target into its own predictions.

A forecast is only legitimate if it depends solely on information available
*before* the value it predicts — i.e. the base forecasts and the meta-training
data. It must NOT depend on the true realized vol of the days it is scoring.

This test runs train_stacking_ensemble twice with identical base forecasts and
identical meta-training targets, changing ONLY the eval-window target values.
A leak-free ensemble returns identical predictions both times. The buggy
version bucketed eval days by their *true* regime and fitted a per-regime
meta-learner, so changing the eval targets changed the predictions — that is
the leak this test pins.
"""

import numpy as np
import pandas as pd

from src.ml_model import train_stacking_ensemble


def _make_feat_df(eval_target_value: float) -> pd.DataFrame:
    """
    Build a 500-row feature frame whose only relevant content is the 'target'
    column over the test window (last 20%).

    The meta-training half of the test window spans two regimes (Low=0.10 and
    Extreme=0.45, 25 days each). The eval half is set entirely to
    `eval_target_value` so we can vary it between runs.
    """
    n = 500
    idx = pd.bdate_range("2021-01-04", periods=n)
    target = np.full(n, 0.20)  # rows before the split are irrelevant to stacking

    # test window = last 100 rows (train_size=0.8 → split at 400)
    # meta-training half = rows 400-449: 25 Low + 25 Extreme
    target[400:425] = 0.10   # Low
    target[425:450] = 0.45   # Extreme
    # eval half = rows 450-499: the value under test
    target[450:500] = eval_target_value

    return pd.DataFrame({"target": target}, index=idx)


def _base_forecasts(feat_df: pd.DataFrame, train_size: float = 0.8) -> dict:
    """Two constant, uninformative base forecasts aligned to the test window."""
    split = int(len(feat_df) * train_size)
    test_index = feat_df.index[split:]
    return {
        "EGARCH": pd.Series(0.25, index=test_index, name="egarch"),
        "XGBoost": pd.Series(0.25, index=test_index, name="xgb"),
    }


def test_stacking_predictions_independent_of_eval_target():
    """
    Changing only the eval-window true target must not change the ensemble's
    predictions. If it does, the ensemble is peeking at the answer.
    """
    df_low = _make_feat_df(eval_target_value=0.10)    # eval days are all Low
    df_extreme = _make_feat_df(eval_target_value=0.45)  # eval days are all Extreme

    preds_low = train_stacking_ensemble(df_low, _base_forecasts(df_low))
    preds_extreme = train_stacking_ensemble(df_extreme, _base_forecasts(df_extreme))

    assert np.allclose(preds_low.values, preds_extreme.values), (
        "Stacking predictions changed when only the eval-window target changed — "
        "the ensemble is leaking the true label it is supposed to forecast."
    )
