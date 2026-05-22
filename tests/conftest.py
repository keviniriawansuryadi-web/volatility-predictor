"""Shared pytest fixtures.

Provides a deterministic synthetic OHLCV price frame so the test suite can
exercise the feature/model pipeline without any network access.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make the project root importable (so `import src...` / `import config` work).
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    """Return a 300-row synthetic daily price frame with the columns the
    feature pipeline expects: close, volume, log_return, realized_vol_21d.

    A fixed seed makes the data reproducible; one deliberate +15% jump is
    injected at row 150 so jump-detection tests have a known positive case.
    """
    rng = np.random.default_rng(42)
    n = 300
    dates = pd.bdate_range("2022-01-03", periods=n)

    returns = rng.normal(0.0003, 0.012, n)
    returns[150] = 0.15  # injected jump for jump_flag tests
    close = 100 * np.exp(np.cumsum(returns))

    df = pd.DataFrame(
        {
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        },
        index=dates,
    )
    df["log_return"] = np.log(df["close"]).diff().fillna(0.0)
    df["realized_vol_21d"] = df["log_return"].rolling(21).std() * np.sqrt(252)
    return df
