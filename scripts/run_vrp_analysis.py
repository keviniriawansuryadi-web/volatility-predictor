"""
Section 10 — Variance Risk Premium (VRP) analysis.

VRP = VIX - realized_vol_21d.
  Positive VRP (normal state): market pays a premium for vol insurance.
  Negative VRP (unusual): realized vol has exceeded implied — often a
  leading indicator of continued stress or mean reversion.

H9: Negative VRP (RV > VIX) predicts above-average vol in the next 10 days.
Test: Mann-Whitney U on next-10d vol split by VRP sign.
Run on SPY only (VIX is SPY-specific).

Usage:
    python scripts/run_vrp_analysis.py
"""

import sys
import json
import warnings
from pathlib import Path
from datetime import date, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.data_loader import load_stock_data, load_vix_data
from src.hypothesis_tests import analyze_variance_risk_premium

TODAY    = date.today().isoformat()
START    = (date.today() - timedelta(days=5 * 365)).isoformat()
PLOT_DIR = str(Path(__file__).parent.parent)

print(f"\n{'='*65}")
print(f"  SECTION 10 — VARIANCE RISK PREMIUM ANALYSIS (SPY)")
print(f"{'='*65}")

df = load_stock_data("SPY", START, TODAY, cache=True)
vix_df = load_vix_data(START, TODAY)
if not vix_df.empty:
    df = df.join(vix_df, how="left")
    df[["vix_level","vix_change"]] = df[["vix_level","vix_change"]].ffill()

result = analyze_variance_risk_premium(df, ticker="SPY", forward_days=10, plot_dir=PLOT_DIR)

if result.get("available"):
    print(f"\n{'='*65}")
    print(f"  VRP SUMMARY")
    print(f"{'='*65}")
    print(f"  Current state  : {result['current_state']}")
    print(f"  Current VRP    : {result['current_vrp']:+.1%}")
    print(f"  Current VIX    : {result['current_vix']:.1%}")
    print(f"  Current RV 21d : {result['current_rv']:.1%}")
    print(f"")
    print(f"  H9 result      : {'SUPPORTED' if result['significant'] else 'NOT SUPPORTED'}")
    print(f"  p-value        : {result['p_value']:.4f}")
    print(f"  After neg VRP  : {result['neg_mean_fwd_vol']:.1%} mean next-10d vol")
    print(f"  After pos VRP  : {result['pos_mean_fwd_vol']:.1%} mean next-10d vol")
    print(f"  Neg VRP episodes: {result['n_negative_vrp']} days")

    # Add current VRP to live signal for SPY
    live_json_path = Path("outputs") / f"live_signal_SPY_2026-05-15.json"
    if live_json_path.exists():
        live_data = json.loads(live_json_path.read_text(encoding="utf-8"))
        live_data["variance_risk_premium"] = {
            "current_vrp": float(result["current_vrp"]),
            "current_vix": float(result["current_vix"]),
            "current_rv":  float(result["current_rv"]),
            "state":       result["current_state"],
            "h9_supported": bool(result["significant"]),
            "h9_p_value":   round(float(result["p_value"]), 4),
        }
        live_json_path.write_text(json.dumps(live_data, indent=2), encoding="utf-8")
        print(f"\n  VRP appended to live signal: {live_json_path}")

print(f"\n  Section 10 COMPLETE.")
