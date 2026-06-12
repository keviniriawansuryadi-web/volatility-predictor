"""
Run the selected L2 hypotheses across companies from every major sector.

Two kinds of test:
  * Per-sector  : H14 directional spillover hub (who is the vol *source* in each
                  sector's panel?).
  * Per-company : H16 leverage amplification (REAL data),
                  H9  negative-sentiment spike risk (SIMULATED sentiment),
                  H20 10-K language change -> regime (PROXY 10-K scores),
                  H22 triple-signal interaction (SIMULATED sentiment + EGARCH-ML
                      disagreement — the only test needing model fitting).

H9/H14/H16/H20 require no model fitting and run in seconds per name. H22 needs a
rolling EGARCH + XGBoost fit per ticker, so it is gated behind --with-h22 and is
meant to be run in the background.

Outputs:
  outputs/results/hyp_sector_spillover.csv   (H14, one row per sector)
  outputs/results/hyp_sector_perticker.csv   (H9/H16/H20[/H22], one row per name)
  outputs/hyp_sector_status.log              (progress)

Usage:
  python scripts/run_hypotheses_by_sector.py              # fast: H9,H14,H16,H20
  python scripts/run_hypotheses_by_sector.py --with-h22   # adds slow H22 (bg)
"""
import argparse
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.data_loader import load_stock_data
from src.data_helpers import (add_forward_vol, simulate_sentiment,
                              simulate_10k_risk_scores)
import modules.hypothesis_tests_L2 as L2

# 3 large-caps per GICS sector (all have cached price data from prior runs).
SECTORS = {
    "Technology":         ["AAPL", "MSFT", "NVDA"],
    "Financials":         ["JPM", "BAC", "GS"],
    "Energy":             ["XOM", "CVX", "COP"],
    "Health Care":        ["JNJ", "UNH", "LLY"],
    "Industrials":        ["CAT", "BA", "HON"],
    "Consumer Disc.":     ["AMZN", "HD", "TSLA"],
    "Consumer Staples":   ["PG", "KO", "COST"],
    "Utilities":          ["NEE", "DUK", "SO"],
    "Materials":          ["LIN", "FCX", "SHW"],
    "Real Estate":        ["AMT", "PLD", "SPG"],
    "Communication Svcs": ["GOOGL", "META", "NFLX"],
}

START, END = "2018-01-01", "2024-12-31"
LOG = ROOT / "outputs" / "hyp_sector_status.log"
SPILL_CSV = ROOT / "outputs" / "results" / "hyp_sector_spillover.csv"
PERTICK_CSV = ROOT / "outputs" / "results" / "hyp_sector_perticker.csv"


def logline(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def _sig(r: dict) -> bool:
    p = r.get("p_value")
    return bool(r.get("available") and isinstance(p, (int, float)) and p == p and p < 0.05)


def _disagreement(ticker: str, df: pd.DataFrame) -> pd.Series:
    """EGARCH-ML disagreement for H22 (the one slow input)."""
    from src.features import build_features
    from src.ml_model import train_and_predict
    from src.garch_model import rolling_garch_forecast
    feat_df = build_features(df, forecast_horizon=5)
    ml_preds, _, _ = train_and_predict(feat_df, model_type="xgboost", train_size=0.8)
    garch_preds = rolling_garch_forecast(df["log_return"], train_size=0.8,
                                         forecast_horizon=5, model_type="EGARCH")
    return L2.compute_disagreement(garch_preds, ml_preds)["norm_abs"]


def run(with_h22: bool) -> None:
    LOG.write_text("", encoding="utf-8")
    SPILL_CSV.parent.mkdir(parents=True, exist_ok=True)
    logline(f"START sectors={len(SECTORS)} with_h22={with_h22}")

    spill_rows, perticker_rows = [], []

    for sector, names in SECTORS.items():
        # --- load this sector's panel once (cached) ---
        df_dict, dfs = {}, {}
        for t in names:
            try:
                raw = load_stock_data(t, START, END, cache=True)
                df_dict[t] = raw
                dfs[t] = add_forward_vol(raw, horizon=5)
            except Exception as exc:
                logline(f"  [{sector}] load {t} FAILED: {exc}")

        # --- H14: per-sector spillover hub (REAL) ---
        # The test defaults to the semiconductor panel + NVDA as source; pass this
        # sector's own members. Two-pass: first find the hub, then re-test with the
        # hub as source so the bootstrap p-value is the hub's own.
        try:
            members = tuple(t for t in names if t in df_dict)
            h14 = L2.test_directional_spillover_hub(
                df_dict, source=members[0], members=members)
            hub = h14.get("hub")
            if hub and hub != members[0] and hub in members:
                h14 = L2.test_directional_spillover_hub(
                    df_dict, source=hub, members=members)
            spill_rows.append({
                "Sector": sector, "Hub": h14.get("hub"),
                "Hub_net_pts": round(float(h14.get("effect", np.nan)), 1),
                "Connectedness_%": round(float(h14.get("statistic", np.nan)), 0),
                "p_value": round(float(h14.get("p_value", np.nan)), 4),
                "Significant": "yes" if _sig(h14) else "no",
            })
            logline(f"  [{sector}] H14 hub={h14.get('hub')} p={h14.get('p_value')}")
        except Exception as exc:
            logline(f"  [{sector}] H14 FAILED: {exc}")

        # --- per-company tests ---
        for t in names:
            if t not in dfs:
                continue
            df = dfs[t]
            row = {"Sector": sector, "Ticker": t}
            # H16 (REAL)
            try:
                r = L2.test_leverage_amplification(df, t)
                row["H16_b"] = round(float(r.get("effect", np.nan)), 3)
                row["H16_p"] = round(float(r.get("p_value", np.nan)), 4)
                row["H16_sig"] = "yes" if _sig(r) else "no"
            except Exception as exc:
                logline(f"  [{sector}] {t} H16 FAILED: {exc}")
            # H9 (SIMULATED sentiment) — long history for power
            try:
                df_h9 = add_forward_vol(load_stock_data(t, "2014-01-01", END, cache=True), horizon=5)
                sent_h9 = simulate_sentiment(df_h9, t)
                r = L2.test_negative_sentiment_spike_risk(df_h9, sent_h9, t)
                row["H9_OR"] = round(float(r.get("odds_ratio", np.nan)), 3)
                row["H9_p"] = round(float(r.get("p_value", np.nan)), 4)
                row["H9_sig"] = "yes" if _sig(r) else "no"
            except Exception as exc:
                logline(f"  [{sector}] {t} H9 FAILED: {exc}")
            # H20 (PROXY 10-K)
            try:
                filings = pd.DatetimeIndex(sorted(set(
                    df.index[np.linspace(60, len(df) - 80, 5).astype(int)])))
                lm = simulate_10k_risk_scores(t, filings)
                r = L2.test_10k_language_change_predicts_regime(df, lm, t)
                row["H20_p"] = round(float(r.get("p_value", np.nan)), 4)
                row["H20_sig"] = "yes" if _sig(r) else "no"
            except Exception as exc:
                logline(f"  [{sector}] {t} H20 FAILED: {exc}")
            # H22 (SIMULATED sentiment + EGARCH-ML disagreement) — slow
            if with_h22:
                try:
                    sent = simulate_sentiment(df, t)
                    disagree = _disagreement(t, df)
                    r = L2.test_triple_signal_interaction(df, sent, disagree, t)
                    row["H22_ratio"] = round(float(r.get("effect", np.nan)), 2)
                    row["H22_p"] = round(float(r.get("p_value", np.nan)), 4)
                    row["H22_sig"] = "yes" if _sig(r) else "no"
                except Exception as exc:
                    logline(f"  [{sector}] {t} H22 FAILED: {exc}")
            perticker_rows.append(row)
            logline(f"  [{sector}] {t} done")

    pd.DataFrame(spill_rows).to_csv(SPILL_CSV, index=False)
    pd.DataFrame(perticker_rows).to_csv(PERTICK_CSV, index=False)
    logline(f"DONE spillover={len(spill_rows)} perticker={len(perticker_rows)}")

    print("\n=== H14 — VOLATILITY SOURCE HUB PER SECTOR (real data) ===")
    print(pd.DataFrame(spill_rows).to_string(index=False))
    print("\n=== PER-COMPANY HYPOTHESES BY SECTOR ===")
    print(pd.DataFrame(perticker_rows).to_string(index=False))
    print(f"\nSaved: {SPILL_CSV}\n       {PERTICK_CSV}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-h22", action="store_true", help="also run slow H22 (model fitting)")
    args = ap.parse_args()
    run(with_h22=args.with_h22)
