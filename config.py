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

SECTOR_COLORS = {
    "Semiconductors": "#e74c3c",
    "Financials":     "#3498db",
    "Energy":         "#2ecc71",
    "Tech":           "#9b59b6",
    "Consumer/Tech":  "#f39c12",
}

# Default pipeline settings (used by main.py and notebooks)
DEFAULT_START      = "2019-01-01"
DEFAULT_END        = "2024-12-31"
DEFAULT_HORIZON    = 5
DEFAULT_TRAIN_SIZE = 0.8
DEFAULT_GARCH_TYPE = "EGARCH"
