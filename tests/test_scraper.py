"""Tests for pure helpers in src/scraper_news.py (no network access)."""

from datetime import datetime

from src.scraper_news import _norm_key, _to_df


def test_norm_key_lowercases_and_strips_punctuation():
    """_norm_key normalises case and removes punctuation for dedup."""
    assert _norm_key("Apple's Q3 EARNINGS!!!") == "apples q3 earnings"
    assert _norm_key("S&P 500 hits record") == "sp 500 hits record"


def test_norm_key_collapses_equivalent_headlines():
    """Headlines differing only in punctuation/case share a dedup key."""
    assert _norm_key("Fed Raises Rates.") == _norm_key("fed raises rates")


def test_to_df_schema_and_ticker_uppercased():
    """_to_df returns the standard schema with an upper-cased ticker column."""
    records = [{"headline": "Stock surges", "datetime": datetime(2024, 1, 2)}]
    df = _to_df(records, source="unit_test", ticker="aapl")
    assert list(df.columns) == ["headline", "source", "datetime", "ticker"]
    assert df["ticker"].iloc[0] == "AAPL"
    assert df["source"].iloc[0] == "unit_test"


def test_to_df_empty_records_returns_empty_frame():
    """_to_df returns an empty, correctly-typed frame for no records."""
    df = _to_df([], source="unit_test", ticker="MU")
    assert df.empty
    assert list(df.columns) == ["headline", "source", "datetime", "ticker"]


def test_to_df_drops_null_headlines():
    """Rows with a missing headline are dropped."""
    records = [
        {"headline": "Valid headline", "datetime": datetime(2024, 1, 2)},
        {"headline": None, "datetime": datetime(2024, 1, 3)},
    ]
    df = _to_df(records, source="unit_test", ticker="MU")
    assert len(df) == 1
