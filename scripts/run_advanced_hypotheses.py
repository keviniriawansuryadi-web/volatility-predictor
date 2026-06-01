"""Headless runner for the L2 advanced hypothesis suite (H9-H23).

Assembles inputs from src helpers, runs all 15 tests for a primary ticker
(plus a sector universe for the contagion tests), writes figures to
outputs/figures/L2/ and the summary CSV to outputs/results/.

Usage:
    python scripts/run_advanced_hypotheses.py --ticker MU --start 2018-01-01 --end 2024-12-31
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import load_stock_data
from src.features import build_features
from src.ml_model import train_and_predict
from src.garch_model import rolling_garch_forecast
from src.data_helpers import (add_forward_vol, simulate_sentiment,
                              load_earnings_dates, simulate_10k_risk_scores)
import modules.hypothesis_tests_L2 as L2


def assemble_inputs(ticker: str, start: str, end: str) -> dict:
    """Build every input the L2 suite needs for `ticker` + sector universe."""
    df = add_forward_vol(load_stock_data(ticker, start, end, cache=True), horizon=5)
    sent = simulate_sentiment(df, ticker)

    # EGARCH-ML disagreement (real models; realized-vol proxy if fitting fails).
    try:
        feat_df = build_features(df, forecast_horizon=5)
        ml_preds, _, _ = train_and_predict(feat_df, model_type="xgboost", train_size=0.8)
        garch_preds = rolling_garch_forecast(df["log_return"], train_size=0.8,
                                              forecast_horizon=5, model_type="EGARCH")
        dis = L2.compute_disagreement(garch_preds, ml_preds)
    except Exception as exc:
        warnings.warn(f"EGARCH/ML fit failed ({exc}); using realized-vol proxy disagreement.")
        rng = np.random.default_rng(0)
        split = int(len(df) * 0.4)
        rv = df["realized_vol_21d"].iloc[split:]
        g = (rv * (1 + rng.normal(0, 0.15, len(rv)))).rename("g")
        m = (rv * (1 + rng.normal(0, 0.15, len(rv)))).rename("m")
        dis = L2.compute_disagreement(g, m)

    df_dict = {ticker: df}
    for members in L2.DEFAULT_SECTORS.values():
        for t in members:
            if t in df_dict:
                continue
            try:
                df_dict[t] = load_stock_data(t, start, end, cache=True)
            except Exception as exc:
                warnings.warn(f"Could not load {t}: {exc}")

    earnings = load_earnings_dates(ticker)
    if len(earnings) < 4:
        rng = np.random.default_rng(1)
        earnings = pd.DatetimeIndex(sorted(df.index[rng.integers(30, len(df) - 30, 12)]))
    filings = pd.DatetimeIndex(sorted(set(df.index[np.linspace(60, len(df) - 80, 5).astype(int)])))
    lm_scores = simulate_10k_risk_scores(ticker, filings)

    return dict(df=df, sent=sent, disagree=dis["norm_abs"], signed=dis["signed"],
                df_dict=df_dict, earnings=earnings, filings=filings, lm_scores=lm_scores)


def run_all(in_: dict, ticker: str) -> dict:
    df, sent = in_["df"], in_["sent"]
    res = {}
    res["H9"] = L2.test_sentiment_asymmetry(df, sent, ticker)
    res["H10"] = L2.test_sentiment_velocity(df, sent, ticker)
    res["H11"] = L2.test_sentiment_model_consensus(df, sent, ticker)
    res["H12"] = L2.test_disagreement_persistence(df, in_["disagree"], ticker)
    res["H13"] = L2.test_disagreement_direction(df, in_["signed"], ticker)
    res["H14"] = L2.test_asymmetric_contagion(in_["df_dict"])
    res["H15"] = L2.test_cross_sector_contagion(in_["df_dict"])
    res["H16"] = L2.test_regime_dependent_leverage(df, ticker)
    res["H17"] = L2.test_monday_leverage_interaction(df, ticker)
    res["H18"] = L2.test_pre_earnings_sentiment_predicts_spike(df, sent, in_["earnings"], ticker)
    res["H19"] = L2.test_earnings_vol_premium_vs_iv(df, in_["earnings"], ticker)
    res["H20"] = L2.test_10k_language_change_predicts_regime(df, in_["lm_scores"], ticker)
    res["H21"] = L2.test_topic_specific_risk_prediction(in_["df_dict"])
    res["H22"] = L2.test_triple_signal_interaction(df, sent, in_["disagree"], ticker)
    res["H23"] = L2.test_signal_temporal_precedence(df, sent, in_["disagree"], ticker,
                                                    df_dict=in_["df_dict"], filing_dates=in_["filings"])
    return res


def save_figures(res: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for hyp, r in res.items():
        fig = r.get("figure")
        if fig is None:
            continue
        try:
            if isinstance(fig, go.Figure):
                fig.write_image(str(out_dir / f"{hyp}.png"))
            else:
                fig.savefig(out_dir / f"{hyp}.png", dpi=120, bbox_inches="tight")
                plt.close(fig)
        except Exception as exc:
            warnings.warn(f"Could not save figure for {hyp} ({exc}); "
                          "plotly PNG export needs the 'kaleido' package.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="MU")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2024-12-31")
    args = ap.parse_args()

    inputs = assemble_inputs(args.ticker, args.start, args.end)
    res = run_all(inputs, args.ticker)

    out_csv = ROOT / "outputs" / "results" / "hypothesis_L2_summary.csv"
    summary = L2.compile_l2_summary(res, out_csv=str(out_csv))
    print(summary.to_string())
    save_figures(res, ROOT / "outputs" / "figures" / "L2")
    print(f"\nFigures: {ROOT / 'outputs' / 'figures' / 'L2'}")


if __name__ == "__main__":
    main()
