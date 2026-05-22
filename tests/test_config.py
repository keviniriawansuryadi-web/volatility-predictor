"""Tests for config.py constants and sector-based model routing."""

import config
from config import select_model_by_sector


def test_tickers_nonempty_and_unique():
    """TICKERS is a non-empty list with no duplicates."""
    assert isinstance(config.TICKERS, list)
    assert len(config.TICKERS) > 0
    assert len(config.TICKERS) == len(set(config.TICKERS))


def test_every_ticker_has_a_sector():
    """Each ticker in TICKERS is mapped in TICKER_SECTORS."""
    for ticker in config.TICKERS:
        assert ticker in config.TICKER_SECTORS, f"{ticker} missing from TICKER_SECTORS"


def test_sector_best_model_keys_are_valid():
    """Every sector routed to by TICKER_SECTORS has an entry in SECTOR_BEST_MODEL."""
    for sector in set(config.TICKER_SECTORS.values()):
        assert sector in config.SECTOR_BEST_MODEL


def test_select_model_by_sector_returns_known_model():
    """Routing returns a non-empty model label for known and unknown tickers."""
    known = {"EGARCH", "RandomForest", "XGBoost", "StackingEnsemble"}
    for ticker in config.TICKERS:
        assert select_model_by_sector(ticker) in known
    # Unknown ticker falls back to EGARCH.
    assert select_model_by_sector("ZZZZ") == "EGARCH"


def test_select_model_is_case_insensitive():
    """Ticker case does not change the routed model."""
    assert select_model_by_sector("nvda") == select_model_by_sector("NVDA")
