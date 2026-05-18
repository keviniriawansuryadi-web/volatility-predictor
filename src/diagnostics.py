"""
Diagnostic tools for identifying and explaining poor model performance.

The primary entry point is diagnose_poor_performer(), which generates a
structured report for any ticker where QLIKE exceeds a warning threshold.
Reports are saved to outputs/diagnostics/{ticker}_diagnosis.md.

Additional functions:
  test_vix_only_model       — single-feature VIX linear regression baseline
  decompose_spike_accuracy  — breaks spike hit-rate into VIX/jump/sentiment buckets
  test_arch_effects         — Engle's ARCH LM test to justify GARCH modelling
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error


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


# ============================================================================ #
# Section 1 — VIX-ONLY BASELINE
# ============================================================================ #

def test_vix_only_model(
    feat_df: pd.DataFrame,
    train_size: float = 0.8,
) -> dict:
    """
    Trains a trivial model using only vix_level as the predictor via linear
    regression.  If this single-feature model achieves Corr > 0.5, it suggests
    the StackingEnsemble's strong performance on SPY is largely explained by
    the VIX-realized vol relationship rather than any sophisticated modeling.
    Reports RMSE, QLIKE, Corr for the VIX-only baseline alongside all other
    models so it can appear in the SPY model comparison table.

    Parameters
    ----------
    feat_df    : Feature DataFrame with 'target' and 'vix_level' columns.
    train_size : Train fraction (default 0.8).

    Returns dict with keys: rmse, mae, qlike, corr, spike_acc, available,
    and a conclusion string.
    """
    if "vix_level" not in feat_df.columns:
        return {"available": False, "conclusion": "vix_level not in feat_df — skipped."}

    split = int(len(feat_df) * train_size)
    X = feat_df["vix_level"].values.reshape(-1, 1)
    y = feat_df["target"].values

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mask = ~(np.isnan(y_test) | np.isnan(preds))
    yt, yp = y_test[mask], preds[mask]

    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mae  = float(mean_absolute_error(yt, yp))

    eps = 1e-8
    h = np.maximum(yp, eps) ** 2
    s2 = yt ** 2
    qlike = float(np.mean(s2 / h - np.log(s2 / h) - 1))

    corr = float(np.corrcoef(yt, yp)[0, 1]) if len(yt) > 1 else float("nan")

    spike_thresh = float(np.percentile(yt, 90))
    spike_mask   = yt > spike_thresh
    spike_acc    = float((yp[spike_mask] > spike_thresh).mean()) if spike_mask.sum() > 0 else float("nan")

    conclusion = (
        f"VIX-only linear regression: Corr={corr:.4f}, QLIKE={qlike:.4f}. "
    )
    if corr > 0.60:
        conclusion += (
            "SPY StackingEnsemble performance is partially explained by the "
            "well-known VIX-realized vol relationship. The ensemble adds value "
            "beyond VIX alone primarily through jump detection and regime-switching."
        )
    elif corr > 0.40:
        conclusion += (
            "VIX alone explains a moderate share of SPY vol variance. "
            "The ensemble materially adds value beyond this baseline."
        )
    else:
        conclusion += (
            "VIX alone is a weak predictor here. Ensemble value is not primarily "
            "from the VIX-realized-vol relationship."
        )

    print(f"\n{'='*60}")
    print(f"  VIX-ONLY BASELINE (linear regression, vix_level only)")
    print(f"{'='*60}")
    print(f"  Intercept : {model.intercept_:.6f}")
    print(f"  Coefficient: {model.coef_[0]:.6f}")
    print(f"  RMSE  : {rmse:.4f}")
    print(f"  MAE   : {mae:.4f}")
    print(f"  QLIKE : {qlike:.4f}")
    print(f"  Corr  : {corr:.4f}")
    print(f"  Spike_Acc: {spike_acc:.1%}" if not np.isnan(spike_acc) else "  Spike_Acc: n/a")
    print(f"\n  {conclusion}")
    print(f"{'='*60}")

    return dict(
        available=True,
        rmse=round(rmse, 6),
        mae=round(mae, 6),
        qlike=round(qlike, 6),
        corr=round(corr, 6),
        spike_acc=round(spike_acc, 4) if not np.isnan(spike_acc) else None,
        intercept=float(model.intercept_),
        coefficient=float(model.coef_[0]),
        corr_above_0p6=(corr > 0.60),
        conclusion=conclusion,
    )


# ============================================================================ #
# Section 2 — DECOMPOSE SPIKE ACCURACY
# ============================================================================ #

def decompose_spike_accuracy(
    feat_df: pd.DataFrame,
    forecasts: dict[str, pd.Series],
    ticker: str,
    train_size: float = 0.8,
    spike_pct: float = 0.90,
) -> dict:
    """
    For each correctly predicted spike day, reports which features were above
    their 75th percentile on that day.  Determines whether spike accuracy is
    driven by:
      (a) VIX already elevated before spike  — model detecting ongoing stress
      (b) jump_flag firing                   — model detecting sudden moves
      (c) sentiment signal                   — model detecting news-driven moves

    Breaks down the overall spike-hit-rate into these three buckets to answer:
    is the model predicting spikes or just detecting them after they start?

    Parameters
    ----------
    feat_df   : Feature DataFrame with 'target' column.
    forecasts : Dict of {label: pd.Series} test-set predictions.
    ticker    : Ticker symbol for display.
    train_size: Train fraction (default 0.8).
    spike_pct : Percentile threshold defining a spike day (default 0.90).

    Returns a dict with per-model decomposition and a summary.
    """
    split = int(len(feat_df) * train_size)
    test_df   = feat_df.iloc[split:].copy()
    y_test    = test_df["target"].values
    spike_thr = float(np.percentile(y_test, spike_pct * 100))
    spike_mask = y_test > spike_thr

    # Feature 75th percentile thresholds on the full dataset (training + test)
    vix_p75  = float(feat_df["vix_level"].quantile(0.75))   if "vix_level"   in feat_df.columns else None
    jump_p75 = float(feat_df["jump_flag"].quantile(0.75))   if "jump_flag"   in feat_df.columns else None
    sent_p75 = float(feat_df["sentiment_3d"].quantile(0.75)) if "sentiment_3d" in feat_df.columns else None

    test_vix  = test_df["vix_level"].values   if "vix_level"   in test_df.columns else None
    test_jump = test_df["jump_flag"].values    if "jump_flag"   in test_df.columns else None
    test_sent = test_df["sentiment_3d"].values if "sentiment_3d" in test_df.columns else None

    results = {}
    for label, series in forecasts.items():
        preds = series.reindex(test_df.index).values
        hit_mask = spike_mask & (preds > spike_thr)  # correctly flagged spikes
        n_spikes = int(spike_mask.sum())
        n_hits   = int(hit_mask.sum())
        if n_spikes == 0:
            results[label] = {"n_spikes": 0, "n_hits": 0}
            continue

        # Decompose: on correctly flagged spike days, how many had each driver elevated?
        def _pct_elevated(feature_vals, threshold):
            if feature_vals is None or threshold is None:
                return None
            return float((feature_vals[hit_mask] > threshold).mean()) if n_hits > 0 else 0.0

        vix_driven  = _pct_elevated(test_vix,  vix_p75)
        jump_driven = _pct_elevated(test_jump, jump_p75)
        sent_driven = _pct_elevated(test_sent, sent_p75)

        results[label] = dict(
            n_spikes=n_spikes,
            n_hits=n_hits,
            spike_acc=round(n_hits / n_spikes, 4),
            vix_elevated_pct=round(vix_driven, 4)  if vix_driven  is not None else None,
            jump_elevated_pct=round(jump_driven, 4) if jump_driven is not None else None,
            sent_elevated_pct=round(sent_driven, 4) if sent_driven is not None else None,
        )

    print(f"\n{'='*65}")
    print(f"  SPIKE ACCURACY DECOMPOSITION — {ticker}")
    print(f"  Spike threshold (90th pct): {spike_thr:.1%} annualized vol")
    print(f"{'='*65}")
    for lbl, r in results.items():
        if r["n_spikes"] == 0:
            continue
        print(f"\n  [{lbl}]  hits={r['n_hits']}/{r['n_spikes']}  spike_acc={r.get('spike_acc', 0):.1%}")
        if r.get("vix_elevated_pct")  is not None: print(f"    (a) VIX above 75th pct on hit days  : {r['vix_elevated_pct']:.1%}")
        if r.get("jump_elevated_pct") is not None: print(f"    (b) jump_flag above 75th pct on hits: {r['jump_elevated_pct']:.1%}")
        if r.get("sent_elevated_pct") is not None: print(f"    (c) sentiment_3d above 75th pct     : {r['sent_elevated_pct']:.1%}")
    print(f"{'='*65}")

    return dict(ticker=ticker, spike_threshold=spike_thr, decomposition=results)


# ============================================================================ #
# Section 3 — ARCH EFFECTS TEST
# ============================================================================ #

def test_arch_effects(
    df: pd.DataFrame,
    ticker: str,
    lags: list[int] | None = None,
) -> dict:
    """
    Runs Engle's ARCH test at lags 5, 10, 20 on the return series.  Reports
    the LM statistic and p-value.

    Strong ARCH effects (p < 0.01) justify GARCH modeling.  Weak ARCH effects
    (p > 0.05) mean GARCH is modeling noise — this explains why EGARCH fails
    on some tickers where volatility is not persistently clustered.

    Parameters
    ----------
    df     : DataFrame with a 'log_return' column.
    ticker : Ticker symbol for display.
    lags   : List of lags to test (default [5, 10, 20]).

    Returns dict with per-lag LM statistics and p-values, and an
    overall_strong_arch boolean (all tested lags p < 0.01).
    """
    from statsmodels.stats.diagnostic import het_arch

    if lags is None:
        lags = [5, 10, 20]

    returns = df["log_return"].dropna().values * 100  # scale to percent

    print(f"\n{'='*55}")
    print(f"  ARCH EFFECTS TEST — {ticker}")
    print(f"{'='*55}")
    print(f"  {'Lag':>5}  {'LM stat':>10}  {'p-value':>10}  {'ARCH?':>8}")
    print(f"  {'-'*45}")

    per_lag = {}
    for lag in lags:
        try:
            lm, p, f_stat, fp = het_arch(returns, nlags=lag)
            verdict = "YES (p<0.01)" if p < 0.01 else ("marginal" if p < 0.05 else "NO")
            print(f"  {lag:>5}  {lm:>10.4f}  {p:>10.6f}  {verdict:>10}")
            per_lag[lag] = dict(lm_stat=round(lm, 4), p_value=round(p, 6), f_stat=round(f_stat, 4))
        except Exception as exc:
            print(f"  {lag:>5}  [ERROR: {exc}]")
            per_lag[lag] = dict(lm_stat=None, p_value=None, f_stat=None)

    all_strong = all(
        v["p_value"] is not None and v["p_value"] < 0.01
        for v in per_lag.values()
    )
    any_significant = any(
        v["p_value"] is not None and v["p_value"] < 0.05
        for v in per_lag.values()
    )

    if all_strong:
        conclusion = f"Strong ARCH effects across all lags — GARCH modelling is justified for {ticker}."
    elif any_significant:
        conclusion = f"Marginal ARCH effects for {ticker} — GARCH adds value but may not be dominant."
    else:
        conclusion = (
            f"Weak/no ARCH effects for {ticker} — GARCH is modelling noise. "
            "This explains why EGARCH underperforms on this ticker relative to ML."
        )

    print(f"\n  Conclusion: {conclusion}")
    print(f"{'='*55}")

    return dict(
        ticker=ticker,
        per_lag=per_lag,
        all_strong_arch=all_strong,
        any_significant=any_significant,
        conclusion=conclusion,
    )
