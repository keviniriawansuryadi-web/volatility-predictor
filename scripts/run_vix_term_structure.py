"""
Section 8 — VIX term structure features.

Fetches VIX9D and VIX3M alongside the standard VIX30, computes:
  vix_term_slope    = VIX3M - VIX  (contango/backwardation)
  vix_short_squeeze = VIX9D - VIX  (near-term vs 30d fear)
  vix_rv_premium    = VIX - realized_vol_21d  (variance risk premium)

Adds all three as features and reruns XGBoost on SPY.
Reports whether vix_level importance drops when term structure is available
— if so, the ensemble was previously using vix_level as a proxy for
information that is now directly available via the term structure.

Usage:
    python scripts/run_vix_term_structure.py
"""

import sys
import warnings
from pathlib import Path
from datetime import date, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.data_loader import load_stock_data, load_vix_term_structure
from src.features import build_features, FEATURE_COLS
from src.garch_model import garch_in_sample_vol
from src.sentiment import fetch_sentiment, fetch_wsb_sentiment
from src.ml_model import train_and_predict, feature_importance
from src.evaluate import _metrics as eval_metrics
from config import DEFAULT_TRAIN_SIZE

TODAY = date.today().isoformat()
START = (date.today() - timedelta(days=5 * 365)).isoformat()

print(f"\n{'='*65}")
print(f"  VIX TERM STRUCTURE FEATURES — SPY")
print(f"{'='*65}")

# ── Load data ─────────────────────────────────────────────────────────────────
print(f"\nLoading SPY price data...")
df = load_stock_data("SPY", START, TODAY, cache=True)

print(f"\nLoading VIX term structure (VIX, VIX9D, VIX3M)...")
vix_ts = load_vix_term_structure(START, TODAY)

if not vix_ts.empty:
    df = df.join(vix_ts, how="left")
    ts_cols = [c for c in ["vix_level","vix_change","vix9d_level","vix3m_level",
                            "vix_term_slope","vix_short_squeeze"] if c in df.columns]
    df[ts_cols] = df[ts_cols].ffill()
    print(f"  VIX term structure loaded: {ts_cols}")
else:
    print("  VIX term structure unavailable — using vix_level only.")
    from src.data_loader import load_vix_data
    vix_base = load_vix_data(START, TODAY)
    if not vix_base.empty:
        df = df.join(vix_base, how="left")
        df[["vix_level", "vix_change"]] = df[["vix_level", "vix_change"]].ffill()

df["sentiment"]     = fetch_sentiment("SPY", df.index)
df["wsb_sentiment"] = fetch_wsb_sentiment("SPY", df.index)
df["garch_vol"]     = garch_in_sample_vol(df["log_return"])

feat_df = build_features(df, forecast_horizon=5)
active = [c for c in FEATURE_COLS if c in feat_df.columns]
print(f"\nFeature matrix: {feat_df.shape}  ({len(active)} features active)")
ts_feats = [c for c in ["vix_term_slope","vix_short_squeeze","vix_rv_premium"] if c in feat_df.columns]
print(f"VIX term structure features active: {ts_feats}")

# ── Baseline XGBoost (without term structure) ─────────────────────────────────
feat_base = feat_df.drop(columns=[c for c in ts_feats if c in feat_df.columns])
print(f"\nTraining XGBoost WITHOUT term structure ({feat_base.shape[1]-1} features)...")
xgb_base, model_base, feats_base = train_and_predict(feat_base, model_type="xgboost", train_size=DEFAULT_TRAIN_SIZE)
fi_base = feature_importance(model_base, feats_base)

split = int(len(feat_df) * DEFAULT_TRAIN_SIZE)
y_test = feat_df["target"].values[split:]
spike_thresh = float(np.percentile(y_test, 90))

base_m = eval_metrics(y_test, xgb_base.values, "XGBoost (no term struct)", spike_thresh)

# ── XGBoost with term structure ───────────────────────────────────────────────
print(f"\nTraining XGBoost WITH term structure ({len(active)} features)...")
xgb_ts, model_ts, feats_ts = train_and_predict(feat_df, model_type="xgboost", train_size=DEFAULT_TRAIN_SIZE)
fi_ts = feature_importance(model_ts, feats_ts)

ts_m = eval_metrics(y_test, xgb_ts.values, "XGBoost (+ term struct)", spike_thresh)

# ── Results ───────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  FEATURE IMPORTANCE — vix_level changes when term structure added")
print(f"{'='*65}")
vix_level_base = float(fi_base.loc[fi_base["feature"] == "vix_level", "importance"].values[0]) if "vix_level" in fi_base["feature"].values else float("nan")
vix_level_ts   = float(fi_ts.loc[fi_ts["feature"] == "vix_level", "importance"].values[0]) if "vix_level" in fi_ts["feature"].values else float("nan")

print(f"\n  vix_level importance (no term struct) : {vix_level_base:.4f}")
print(f"  vix_level importance (with term struct): {vix_level_ts:.4f}")
if not np.isnan(vix_level_base) and not np.isnan(vix_level_ts):
    drop = (vix_level_base - vix_level_ts) / (vix_level_base + 1e-10)
    print(f"  Importance drop: {drop:+.1%}")

print(f"\n  Term structure feature importances (with term struct):")
for feat in ts_feats:
    row = fi_ts.loc[fi_ts["feature"] == feat]
    if len(row):
        print(f"    {feat:25}: {row['importance'].values[0]:.4f}")

print(f"\n{'='*55}")
print(f"  MODEL COMPARISON — SPY")
print(f"{'='*55}")
print(f"  {'Model':>30}  {'QLIKE':>7}  {'Corr':>7}")
print(f"  {'-'*50}")
for m in [base_m, ts_m]:
    print(f"  {m['model']:>30}  {m['QLIKE']:>7.4f}  {m['Corr']:>7.4f}")

qlike_imp = (base_m["QLIKE"] - ts_m["QLIKE"]) / (base_m["QLIKE"] + 1e-10)
print(f"\n  QLIKE improvement from term structure: {qlike_imp:+.1%}")

print(f"\n  Top 10 features (WITH term structure):")
print(fi_ts.head(10).to_string(index=False))

print(f"\n  Section 8 COMPLETE.")
