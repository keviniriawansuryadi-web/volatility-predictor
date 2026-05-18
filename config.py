"""
Central configuration for the volatility predictor.
All scripts and notebooks should import from here rather than hard-coding values.
"""

# Tickers covered by the batch pipeline, grouped by sector
TICKERS = [
    "MU",   "NVDA", "AMD",   # Semiconductors
    "JPM",  "BAC",            # Financials
    "XOM",  "CVX",            # Energy
    "AAPL", "MSFT",           # Tech (large-cap)
    "AMZN",                   # Consumer / Tech
]

# Sector ETFs for generalisation analysis (Section 4).
# ^VIX is not tradeable but its own vol is interesting.
TICKERS_ETF = ["QQQ", "XLF", "XLE", "XLK", "XLV", "GLD", "TLT", "^VIX"]

# Out-of-sample tickers (Section 5) — never tuned, pure generalisation test.
TICKERS_OOS = ["TSLA", "GS", "PFE"]

SECTOR_MAP = {
    "MU":   "Semiconductors",
    "NVDA": "Semiconductors",
    "AMD":  "Semiconductors",
    "JPM":  "Financials",
    "BAC":  "Financials",
    "XOM":  "Energy",
    "CVX":  "Energy",
    "AAPL": "Tech",
    "MSFT": "Tech",
    "AMZN": "Consumer/Tech",
}

ETF_SECTOR_MAP = {
    "QQQ":  "Tech-ETF",
    "XLF":  "Financial-ETF",
    "XLE":  "Energy-ETF",
    "XLK":  "Tech-ETF",
    "XLV":  "Healthcare-ETF",
    "GLD":  "Commodity-ETF",
    "TLT":  "Bond-ETF",
    "^VIX": "VIX",
}

SECTOR_COLORS = {
    "Semiconductors": "#e74c3c",
    "Financials":     "#3498db",
    "Energy":         "#2ecc71",
    "Tech":           "#9b59b6",
    "Consumer/Tech":  "#f39c12",
}

# ── Sector-aware model routing ──────────────────────────────────────────────────
# Maps each ticker to its sector key (lower-case, matches SECTOR_BEST_MODEL).
TICKER_SECTORS = {
    "MU":   "semiconductor",
    "NVDA": "semiconductor",
    "AMD":  "semiconductor",
    "JPM":  "financial",
    "BAC":  "financial",
    "XOM":  "energy",
    "CVX":  "energy",
    "AAPL": "tech",
    "MSFT": "tech",
    "AMZN": "tech",
}

# Empirically best model per sector based on median QLIKE across the 10-ticker universe.
# Derived from the full multi-ticker run (2021-2026):
#   Semiconductors : EGARCH wins on MU, AMD (idiosyncratic spike vol)
#   Financials     : RandomForest wins on BAC; HAR-RV on JPM — RF chosen as tie-break
#   Energy         : XGBoost wins on XOM; RF on CVX — XGBoost chosen (lower QLIKE)
#   Tech           : StackingEnsemble wins on AAPL, AMZN; EGARCH on MSFT — Stacking chosen
SECTOR_BEST_MODEL = {
    "semiconductor": "EGARCH",
    "financial":     "RandomForest",
    "energy":        "XGBoost",
    "tech":          "StackingEnsemble",
}

# Per-ticker overrides when a ticker's best model diverges from its sector default.
# MSFT: StackingEnsemble QLIKE=1.97 (all-zero Ridge coefficients in batch run);
#       EGARCH is clearly superior (QLIKE=0.59) and is the correct choice.
# JPM:  RandomForest QLIKE=0.52 is worse than EGARCH QLIKE=0.42;
#       EGARCH wins in the financial sector for JPM specifically.
TICKER_MODEL_OVERRIDE = {
    "MSFT": "EGARCH",
    "JPM":  "EGARCH",
}


def select_model_by_sector(ticker: str) -> str:
    """
    Return the empirically best model for a ticker based on sector QLIKE results.

    Model routing is determined by SECTOR_BEST_MODEL, which was derived from
    the median QLIKE across all tickers in each sector over the 2021-2026
    out-of-sample evaluation.  Falls back to 'EGARCH' if the ticker's sector
    is not in the lookup table.

    Rationale:
      - Semiconductors (MU, NVDA, AMD): EGARCH dominates due to asymmetric vol
        clustering driven by supply-demand cycles and binary earnings events.
        ML models underperform because spike drivers are not in lagging features.
      - Financials (JPM, BAC): RandomForest captures non-linear interactions
        between VIX regime and credit-spread dynamics. Stable vol structure
        gives ML models a training-data advantage over pure GARCH.
      - Energy (XOM, CVX): XGBoost's VIX and jump features capture oil-price
        shock responses that EGARCH misses during structural breaks.
      - Tech (AAPL, MSFT, AMZN): StackingEnsemble balances EGARCH's vol
        persistence with ML's macro feature sensitivity. MSFT is an exception
        (EGARCH wins), but the ensemble still improves on AAPL and AMZN.

    Parameters
    ----------
    ticker : Ticker symbol (case-insensitive).

    Returns the model label matching the keys used in compare_models() forecasts dict.
    """
    upper = ticker.upper()
    if upper in TICKER_MODEL_OVERRIDE:
        return TICKER_MODEL_OVERRIDE[upper]
    sector = TICKER_SECTORS.get(upper)
    if sector is None:
        return "EGARCH"
    return SECTOR_BEST_MODEL.get(sector, "EGARCH")


# Default pipeline settings used by notebooks (fixed for reproducibility).
# main.py CLI uses dynamic defaults: start=5 years ago, end=today.
DEFAULT_START      = "2019-01-01"
DEFAULT_END        = "2024-12-31"   # notebooks use this fixed window
DEFAULT_HORIZON    = 5
DEFAULT_TRAIN_SIZE = 0.8
DEFAULT_GARCH_TYPE = "EGARCH"
