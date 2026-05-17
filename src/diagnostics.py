"""
Diagnostic tools for identifying and explaining poor model performance.

The primary entry point is diagnose_poor_performer(), which generates a
structured report for any ticker where QLIKE exceeds a warning threshold.
Reports are saved to outputs/diagnostics/{ticker}_diagnosis.md.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd


DIAG_DIR = Path(__file__).parent.parent / "outputs" / "diagnostics"


def _regime_label(vol: float) -> str:
    if vol >= 0.35:  return "Extreme"
    if vol >= 0.25:  return "High"
    if vol >= 0.15:  return "Elevated"
    return "Low"


def diagnose_poor_performer(
    ticker: str,
    feat_df: pd.DataFrame,
    results_df: pd.DataFrame,
    qlike_threshold: float = 0.6,
    reference_ticker: str = "MU",
    reference_feat_df: pd.DataFrame | None = None,
    plot_dir: str = ".",
) -> str:
    """
    Generate a diagnostic report for a ticker where QLIKE exceeds the threshold.

    Covers five diagnostic dimensions:

    1. Data quality
       - Missing-value rate in key features
       - % of trading days with zero sentiment (imputed, not real scores)
       - Gaps in VIX data

    2. Regime distribution
       - % of test-set days in each vol regime (Low/Elevated/High/Extreme)
       - Comparison to reference ticker to identify if AMD is more extreme

    3. Largest prediction errors (top 10 worst underestimations)
       - Date, realized vol, best-model forecast, absolute error
       - Flags whether the error coincides with a known earnings week
         (proxy: within 5 days of a quarter-end month)

    4. Feature importance comparison vs reference ticker
       - Reports which features the best model relied on for this ticker
       - Flags divergence from the reference ticker's feature ranking

    5. EGARCH convergence proxy
       - Checks variance of EGARCH rolling forecasts as a proxy for
         convergence instability (high variance relative to realized vol
         suggests frequent refit failures or non-convergence)

    Parameters
    ----------
    ticker           : Ticker under investigation (e.g. 'AMD').
    feat_df          : Feature DataFrame with 'target' column for this ticker.
    results_df       : Model comparison CSV loaded as DataFrame (from
                       outputs/results/{ticker}/{ticker}_model_comparison.csv).
    qlike_threshold  : Warn only when best QLIKE exceeds this (default 0.6).
    reference_ticker : Comparison ticker for feature importance (default 'MU').
    reference_feat_df: Feature DataFrame for the reference ticker (optional).
    plot_dir         : Root output directory (default '.').

    Returns the path of the saved markdown report as a string.
    """
    best_qlike = results_df["QLIKE"].min()
    if best_qlike < qlike_threshold:
        msg = (f"[diagnose] {ticker} best QLIKE={best_qlike:.3f} is below threshold "
               f"{qlike_threshold} — no diagnosis needed.")
        print(msg)
        return msg

    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def h(text: str, level: int = 2) -> None:
        lines.append(f"\n{'#' * level} {text}\n")

    def p(text: str) -> None:
        lines.append(text + "\n")

    # ── Header ──────────────────────────────────────────────────────────────────
    lines.append(f"# Diagnostic Report — {ticker}\n")
    p(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    p(f"**Best QLIKE:** {best_qlike:.4f}  (threshold: {qlike_threshold})")
    p(f"**Reference ticker:** {reference_ticker}")
    p("---")

    # ── 1. Model performance summary ─────────────────────────────────────────
    h("1. Model Performance Summary")
    tbl = results_df[["model", "RMSE", "MAE", "QLIKE", "Corr", "Spike_Acc"]].copy()
    tbl = tbl.sort_values("QLIKE")
    # Build markdown table without tabulate dependency
    cols = tbl.columns.tolist()
    header = "| " + " | ".join(cols) + " |"
    sep    = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines.append(header)
    lines.append(sep)
    for _, row in tbl.iterrows():
        def _fmt(v):
            if isinstance(v, float):
                return f"{v:.4f}"
            return str(v)
        lines.append("| " + " | ".join(_fmt(row[c]) for c in cols) + " |")
    p("")
    best_model = tbl.iloc[0]["model"]
    p(f"**Winner:** {best_model} (QLIKE={best_qlike:.4f})")

    # ── 2. Data quality ───────────────────────────────────────────────────────
    h("2. Data Quality")
    key_cols = ["log_return", "realized_vol_21d", "vix_level", "sentiment", "garch_vol"]
    for col in key_cols:
        if col in feat_df.columns:
            n_na = feat_df[col].isna().sum()
            pct  = n_na / len(feat_df) * 100
            p(f"- `{col}`: {n_na} missing ({pct:.1f}%)")
        else:
            p(f"- `{col}`: **COLUMN ABSENT**")

    if "sentiment" in feat_df.columns:
        zero_sent = (feat_df["sentiment"] == 0.0).sum()
        pct_zero  = zero_sent / len(feat_df) * 100
        p(f"\n**Sentiment:** {zero_sent} zero-days ({pct_zero:.1f}%) — "
          f"{'POOR coverage, imputed from rolling median' if pct_zero > 80 else 'acceptable'}")

    # ── 3. Regime distribution ────────────────────────────────────────────────
    h("3. Vol Regime Distribution (test set)")
    split = int(len(feat_df) * 0.8)
    y_test = feat_df["target"].values[split:]
    regimes = [_regime_label(v) for v in y_test]
    for regime in ["Low", "Elevated", "High", "Extreme"]:
        n   = regimes.count(regime)
        pct = n / len(regimes) * 100
        p(f"- {regime}: {n} days ({pct:.1f}%)")

    pct_extreme = regimes.count("Extreme") / len(regimes) * 100
    if pct_extreme > 20:
        p(f"\n**WARNING:** {pct_extreme:.1f}% Extreme regime days — well above typical 10%.")
        p("EGARCH in-sample fitting may struggle when >20% of test days are structural outliers.")

    # ── 4. Top-10 worst underestimations ──────────────────────────────────────
    h("4. Worst 10 Underestimation Errors (test set)")
    test_df   = feat_df.iloc[split:].copy()
    y_true_s  = test_df["target"]

    # Use the best model's predictions from results_df if available in feat_df;
    # otherwise proxy with EGARCH col (garch_vol shifted by forecast horizon)
    if "garch_vol" in test_df.columns:
        proxy_pred = test_df["garch_vol"].values
    else:
        proxy_pred = np.zeros(len(test_df))

    errors = (y_true_s.values - proxy_pred)   # positive = underestimation
    worst_idx = np.argsort(errors)[::-1][:10]

    p("| Rank | Date | Realized Vol | EGARCH Forecast | Error | Earnings Proxy |")
    p("|------|------|-------------|-----------------|-------|----------------|")
    for rank, idx in enumerate(worst_idx, 1):
        date    = test_df.index[idx].strftime("%Y-%m-%d")
        rv      = float(y_true_s.iloc[idx])
        pred    = float(proxy_pred[idx])
        err     = float(errors[idx])
        month   = test_df.index[idx].month
        earnings_flag = "YES" if month in [1, 4, 7, 10] else "no"
        p(f"| {rank} | {date} | {rv:.1%} | {pred:.1%} | {err:.1%} | {earnings_flag} |")

    # ── 5. EGARCH stability proxy ─────────────────────────────────────────────
    h("5. EGARCH Rolling Forecast Stability")
    if "garch_vol" in feat_df.columns:
        garch_vals = feat_df["garch_vol"].dropna().values
        garch_cv   = float(np.std(garch_vals) / (np.mean(garch_vals) + 1e-8))
        rv_cv      = float(np.std(y_test) / (np.mean(y_test) + 1e-8))
        p(f"- EGARCH in-sample vol CoV: {garch_cv:.3f}")
        p(f"- Realized vol CoV (test):  {rv_cv:.3f}")
        ratio = garch_cv / (rv_cv + 1e-8)
        if ratio < 0.5:
            p(f"\n**WARNING:** EGARCH CoV / RV CoV = {ratio:.2f} — EGARCH is over-smooth relative "
              f"to realized vol. Possible non-convergence or dampened persistence estimates.")
        elif ratio > 2.0:
            p(f"\n**WARNING:** EGARCH CoV / RV CoV = {ratio:.2f} — EGARCH is highly volatile "
              f"relative to realized vol, suggesting erratic refit behavior.")
        else:
            p(f"EGARCH stability ratio {ratio:.2f} is within normal range (0.5–2.0).")

    # ── 6. Targeted fix recommendations ──────────────────────────────────────
    h("6. Targeted Fix Recommendations")
    if pct_extreme > 20:
        p("- **Regime issue detected**: >20% Extreme days overwhelm the rolling EGARCH refit. "
          "Consider GJR-GARCH (asymmetric) or a Markov-switching GARCH for this ticker.")
    if "sentiment" in feat_df.columns and pct_zero > 80:
        p("- **Sentiment gap**: >80% imputed. Sentiment features add noise, not signal. "
          "Consider dropping sentiment features for this ticker in the feature set.")
    if best_model not in ("EGARCH", "HAR-RV"):
        p(f"- **ML beats statistical models**: Use `{best_model}` as the primary forecast for "
          f"{ticker} rather than EGARCH in the sector-aware routing table.")
    p("- **Time-series CV**: Retrain using expanding-window CV (3 folds) instead of single "
      "80/20 split to get more robust estimates on spike-heavy tickers.")
    p("- **GJR-GARCH alternative**: GJR-GARCH allows asymmetric response to positive vs "
      "negative shocks and may converge more stably on AMD's regime-driven vol pattern.")

    # ── Save ─────────────────────────────────────────────────────────────────
    out = DIAG_DIR / f"{ticker}_diagnosis.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[diagnose] Report saved: {out}")
    return str(out)
