"""Run sector spillover analysis using cached price data."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding="utf-8")
import warnings
warnings.filterwarnings("ignore")
from datetime import date, timedelta
from src.data_loader import load_stock_data
from src.spillover import run_spillover_analysis

today = date.today().isoformat()
start = (date.today() - timedelta(days=5 * 365)).isoformat()

sector_groups = {
    "Semiconductors": ["MU", "NVDA", "AMD"],
    "Financials":     ["JPM", "BAC"],
    "Energy":         ["XOM", "CVX"],
}

print("Loading price data for spillover analysis...")
df_dict = {}
for tickers in sector_groups.values():
    for t in tickers:
        if t not in df_dict:
            df_dict[t] = load_stock_data(t, start, today)

run_spillover_analysis(df_dict, sector_groups, plot_dir=".")
