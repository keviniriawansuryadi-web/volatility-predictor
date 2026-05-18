"""
Build outputs/plots/portfolio_summary.png — 2×2 summary figure.

Panel 1 (top-left):  Model QLIKE heatmap — all tickers × 6 models,
                      winner cells circled in gold.
Panel 2 (top-right): EGARCH QLIKE vs best-ML QLIKE scatter by sector.
Panel 3 (bot-left):  Current vol regime bar chart (realized_vol_21d per ticker).
Panel 4 (bot-right): MU EGARCH-ML disagreement history with vol overlay.

Data sources:
  - Panels 1-2: outputs/results/{ticker}/{ticker}_model_comparison.csv
  - Panel 3:    cached price data (load_stock_data cache=True)
  - Panel 4:    cached MU data + rolling GARCH/XGBoost forecasts
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding="utf-8")
import warnings
warnings.filterwarnings("ignore")

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

from src.data_loader import load_stock_data, load_vix_data
from src.sentiment import fetch_sentiment, fetch_wsb_sentiment
from src.features import build_features
from src.garch_model import rolling_garch_forecast, garch_in_sample_vol
from src.ml_model import train_and_predict
from config import TICKERS, DEFAULT_TRAIN_SIZE, DEFAULT_GARCH_TYPE

TODAY  = date.today().isoformat()
START  = (date.today() - timedelta(days=5 * 365)).isoformat()
OUT    = Path("outputs/plots/portfolio_summary.png")
OUT.parent.mkdir(parents=True, exist_ok=True)

MODELS  = ["EGARCH", "HAR-RV", "XGBoost", "XGB-Asymmetric", "RandomForest", "StackingEnsemble"]
SECTORS = {
    "MU":   "Semiconductor", "NVDA": "Semiconductor", "AMD":  "Semiconductor",
    "JPM":  "Financial",     "BAC":  "Financial",
    "XOM":  "Energy",        "CVX":  "Energy",
    "AAPL": "Tech",          "MSFT": "Tech",          "AMZN": "Tech",
}
SECTOR_COLORS = {
    "Semiconductor": "#4e79a7",
    "Financial":     "#f28e2b",
    "Energy":        "#e15759",
    "Tech":          "#59a14f",
}
REGIME_COLORS = {
    "Low":      "#2ca02c",
    "Elevated": "#ffbf00",
    "High":     "#ff7f0e",
    "Extreme":  "#d62728",
}

# ── Load all model comparison CSVs ───────────────────────────────────────────
print("Loading model comparison CSVs...")
frames = []
for t in TICKERS:
    p = Path(f"outputs/results/{t}/{t}_model_comparison.csv")
    if p.exists():
        frames.append(pd.read_csv(p))
all_df = pd.concat(frames, ignore_index=True)
print(f"  Loaded {len(all_df)} rows across {all_df['ticker'].nunique()} tickers.")

# Pivot QLIKE to tickers × models matrix
qlike_pivot = all_df.pivot(index="ticker", columns="model", values="QLIKE").reindex(
    index=TICKERS, columns=MODELS
)

# Winner mask
winner_pivot = all_df.pivot(index="ticker", columns="model", values="winner").reindex(
    index=TICKERS, columns=MODELS
).fillna(False)

# Best non-EGARCH ML QLIKE per ticker
ml_models = ["XGBoost", "XGB-Asymmetric", "RandomForest", "StackingEnsemble", "HAR-RV"]
best_ml_qlike = all_df[all_df["model"].isin(ml_models)].groupby("ticker")["QLIKE"].min()
egarch_qlike  = all_df[all_df["model"] == "EGARCH"].set_index("ticker")["QLIKE"]

# ── Load current realized vol (Panel 3) ──────────────────────────────────────
print("Loading cached price data for regime chart...")
current_rv = {}
for t in TICKERS:
    try:
        df = load_stock_data(t, START, TODAY, cache=True)
        rv = df["realized_vol_21d"].dropna()
        current_rv[t] = float(rv.iloc[-1]) if len(rv) > 0 else np.nan
    except Exception as e:
        print(f"  {t}: FAILED — {e}")
        current_rv[t] = np.nan

def _regime_label(v: float) -> str:
    if np.isnan(v):  return "Low"
    if v >= 0.35:    return "Extreme"
    if v >= 0.25:    return "High"
    if v >= 0.15:    return "Elevated"
    return "Low"

# ── Load MU disagreement history (Panel 4) ───────────────────────────────────
print("Computing MU disagreement history...")
mu_df = load_stock_data("MU", START, TODAY, cache=True)
vix_df = load_vix_data(START, TODAY)
if not vix_df.empty:
    mu_df = mu_df.join(vix_df, how="left")
    mu_df[["vix_level", "vix_change"]] = mu_df[["vix_level", "vix_change"]].ffill()
mu_df["sentiment"]     = fetch_sentiment("MU", mu_df.index)
mu_df["wsb_sentiment"] = fetch_wsb_sentiment("MU", mu_df.index)
mu_df["garch_vol"]     = garch_in_sample_vol(mu_df["log_return"], model_type=DEFAULT_GARCH_TYPE)

mu_feat   = build_features(mu_df, forecast_horizon=21)
garch_mu  = rolling_garch_forecast(mu_df["log_return"], DEFAULT_TRAIN_SIZE, 21, DEFAULT_GARCH_TYPE)
xgb_mu, _, _ = train_and_predict(mu_feat, model_type="xgboost", train_size=DEFAULT_TRAIN_SIZE)

split    = int(len(mu_feat) * DEFAULT_TRAIN_SIZE)
rv_mu    = mu_feat["target"].iloc[split:]
eg_common = garch_mu.index.intersection(xgb_mu.index).intersection(rv_mu.index)
eg = garch_mu.reindex(eg_common)
ml = xgb_mu.reindex(eg_common)
denom = (eg.abs() + ml.abs()) / 2 + 1e-8
disagree_series = (eg - ml).abs() / denom
rv_series       = rv_mu.reindex(eg_common)
thresh80        = float(disagree_series.quantile(0.80))

print("Rendering 2×2 figure...")

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 14))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.30)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

# ── Panel 1: QLIKE heatmap ────────────────────────────────────────────────────
qdata = qlike_pivot.values.astype(float)
vmax  = float(np.nanpercentile(qdata, 95))
im1   = ax1.imshow(qdata, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=vmax)
ax1.set_xticks(range(len(MODELS)))
ax1.set_yticks(range(len(TICKERS)))
ax1.set_xticklabels([m.replace("-", "-\n") for m in MODELS], fontsize=8, rotation=0)
ax1.set_yticklabels(TICKERS, fontsize=9)
ax1.set_title("QLIKE Loss — All Tickers × Models\n(lower = better; gold = best per ticker)", fontsize=11)

for i, t in enumerate(TICKERS):
    for j, m in enumerate(MODELS):
        v = qlike_pivot.loc[t, m]
        is_win = bool(winner_pivot.loc[t, m])
        txt_color = "white" if v > vmax * 0.6 else "black"
        txt = f"{v:.3f}" if not np.isnan(v) else "—"
        ax1.text(j, i, txt, ha="center", va="center",
                 fontsize=7.5, color=txt_color,
                 fontweight="bold" if is_win else "normal")
        if is_win:
            rect = mpatches.FancyBboxPatch(
                (j - 0.48, i - 0.48), 0.96, 0.96,
                boxstyle="round,pad=0.05", linewidth=2,
                edgecolor="gold", facecolor="none",
            )
            ax1.add_patch(rect)

plt.colorbar(im1, ax=ax1, label="QLIKE", fraction=0.035, pad=0.02)

# ── Panel 2: EGARCH vs best-ML QLIKE scatter ─────────────────────────────────
for t in TICKERS:
    eg_q  = float(egarch_qlike.get(t, np.nan))
    ml_q  = float(best_ml_qlike.get(t, np.nan))
    sect  = SECTORS.get(t, "Other")
    col   = SECTOR_COLORS.get(sect, "#999")
    ax2.scatter(eg_q, ml_q, color=col, s=120, zorder=3)
    ax2.annotate(t, (eg_q, ml_q), textcoords="offset points",
                 xytext=(6, 4), fontsize=8)

# Diagonal (breakeven line)
lo = min(egarch_qlike.min(), best_ml_qlike.min()) * 0.9
hi = max(egarch_qlike.max(), best_ml_qlike.max()) * 1.1
ax2.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.5, label="Breakeven")
ax2.fill_between([lo, hi], [lo, lo], [lo, hi], alpha=0.04, color="blue",
                 label="EGARCH wins region")
ax2.fill_between([lo, hi], [lo, hi], [hi, hi], alpha=0.04, color="red",
                 label="ML wins region")
ax2.set_xlabel("EGARCH QLIKE", fontsize=10)
ax2.set_ylabel("Best ML QLIKE", fontsize=10)
ax2.set_title("EGARCH vs Best-ML QLIKE\n(below diagonal = EGARCH wins)", fontsize=11)

legend_patches = [
    mpatches.Patch(color=c, label=s) for s, c in SECTOR_COLORS.items()
]
ax2.legend(handles=legend_patches, fontsize=8, loc="upper left")

# ── Panel 3: Regime bar chart ─────────────────────────────────────────────────
rv_vals   = [current_rv.get(t, np.nan) for t in TICKERS]
bar_cols  = [REGIME_COLORS[_regime_label(v)] for v in rv_vals]
bars = ax3.bar(TICKERS, rv_vals, color=bar_cols, edgecolor="white", linewidth=0.5)
ax3.set_ylabel("Realized Vol (21d, annualised)", fontsize=10)
ax3.set_title("Current Vol Regime by Ticker\n(as of latest trading day)", fontsize=11)
ax3.set_ylim(0, max([v for v in rv_vals if not np.isnan(v)] + [0.6]) * 1.15)

# Threshold lines
for thresh, label in [(0.15, "Low|Elev"), (0.25, "Elev|High"), (0.35, "High|Extreme")]:
    ax3.axhline(thresh, color="grey", linewidth=0.8, linestyle="--", alpha=0.7)
    ax3.text(len(TICKERS) - 0.3, thresh + 0.005, label, fontsize=7, color="grey", ha="right")

for bar, v in zip(bars, rv_vals):
    if not np.isnan(v):
        ax3.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                 f"{v:.0%}", ha="center", va="bottom", fontsize=8)

legend_regime = [
    mpatches.Patch(color=c, label=r) for r, c in REGIME_COLORS.items()
]
ax3.legend(handles=legend_regime, fontsize=8, loc="upper right")
ax3.tick_params(axis="x", rotation=30)

# ── Panel 4: MU disagreement history + realized vol overlay ──────────────────
ax4b = ax4.twinx()

idx = eg_common
ax4.fill_between(idx, 0, disagree_series.values, alpha=0.35, color="#4e79a7",
                 label="EGARCH-ML Disagreement")
ax4.axhline(thresh80, color="#4e79a7", linewidth=1.2, linestyle="--", alpha=0.8,
            label=f"80th-pct threshold ({thresh80:.3f})")
ax4b.plot(idx, rv_series.values, color="#d62728", linewidth=1.5, alpha=0.8,
          label="Realized Vol 21d")

ax4.set_xlabel("Date", fontsize=10)
ax4.set_ylabel("Normalised |EGARCH − XGB| / mean", fontsize=9, color="#4e79a7")
ax4b.set_ylabel("Realized Vol 21d (annualised)", fontsize=9, color="#d62728")
ax4.set_title("MU — EGARCH-ML Disagreement vs Realized Vol\n(test window)", fontsize=11)
ax4.tick_params(axis="y", labelcolor="#4e79a7")
ax4b.tick_params(axis="y", labelcolor="#d62728")

lines1, labels1 = ax4.get_legend_handles_labels()
lines2, labels2 = ax4b.get_legend_handles_labels()
ax4.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")
plt.setp(ax4.get_xticklabels(), rotation=25, ha="right", fontsize=8)

fig.suptitle(
    "Volatility Predictor — Portfolio Summary\n"
    "EGARCH / HAR-RV / XGBoost / RandomForest / StackingEnsemble   |   10-ticker multi-sector analysis",
    fontsize=13, y=0.995,
)

plt.savefig(OUT, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {OUT}")
