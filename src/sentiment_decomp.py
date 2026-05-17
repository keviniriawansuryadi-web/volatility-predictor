"""
Cross-ticker sentiment decomposition: systematic vs idiosyncratic components.

Decomposes daily sentiment into:
  1. Market-wide (systematic) component: equal-weighted average across all tickers
  2. Idiosyncratic component: ticker sentiment minus the market average

Tests which component better predicts vol spikes.  If >60% of sentiment
variance is systematic, the scrapers are pulling general market news rather
than ticker-specific news — a signal quality warning.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats


def decompose_sentiment(
    df_dict: dict[str, pd.DataFrame],
    tickers: list[str],
    sentiment_col: str = "sentiment",
    vol_col: str = "realized_vol_21d",
    spike_pct: float = 0.90,
) -> dict:
    """
    Decompose daily sentiment into systematic and idiosyncratic components.

    The decomposition follows the standard factor-model approach:
      sentiment_ticker(t) = market_avg(t) + idiosyncratic_ticker(t)

    where market_avg(t) is the equal-weighted cross-sectional mean across
    all tickers on day t, and idiosyncratic is the residual.

    Three tests are run:
      1. Variance decomposition: % of each ticker's sentiment variance that
         is explained by the market-wide component (R² of regression on market avg).
      2. Granger causality: does market-wide sentiment Granger-cause realized vol
         for each ticker?  Does idiosyncratic sentiment?
      3. Spike-day correlation: on 90th-pct vol spike days, is the correlation
         with market-wide or idiosyncratic sentiment higher?

    Parameters
    ----------
    df_dict       : Dict of {ticker: DataFrame} where each DataFrame has both
                    `sentiment_col` and `vol_col` columns indexed to trading days.
    tickers       : Subset of tickers to include.
    sentiment_col : Name of the sentiment column (default 'sentiment').
    vol_col       : Name of the realized vol column (default 'realized_vol_21d').
    spike_pct     : Percentile threshold for spike-day detection (default 90th).

    Returns a dict with keys:
      'systematic_var_pct'  : pd.Series — % of sentiment variance that is systematic
      'idio_var_pct'        : pd.Series — % of sentiment variance that is idiosyncratic
      'granger_systematic'  : pd.Series — min p-value: market sentiment → ticker vol
      'granger_idio'        : pd.Series — min p-value: idio sentiment → ticker vol
      'spike_corr_systematic': pd.Series — corr of market sentiment with spike-day vol
      'spike_corr_idio'     : pd.Series — corr of idio sentiment with spike-day vol
      'flag_systematic'     : list of tickers where systematic pct > 60%
      'market_sentiment'    : pd.Series — the market-wide daily sentiment series
    """
    from statsmodels.tsa.stattools import grangercausalitytests

    # ── Build aligned sentiment panel ─────────────────────────────────────────
    sent_frames = {}
    for t in tickers:
        if t not in df_dict or sentiment_col not in df_dict[t].columns:
            continue
        sent_frames[t] = df_dict[t][sentiment_col].rename(t)

    if len(sent_frames) < 2:
        warnings.warn("[sentiment_decomp] Need >= 2 tickers with sentiment data.")
        return {}

    panel = pd.concat(sent_frames.values(), axis=1, join="inner")
    panel = panel.dropna(how="all")

    # Market-wide component: cross-sectional mean per day
    market_sent = panel.mean(axis=1)

    # ── Variance decomposition ────────────────────────────────────────────────
    sys_var_pct  = {}
    idio_var_pct = {}
    idio_panels  = {}

    for t in panel.columns:
        ticker_sent = panel[t]
        total_var   = float(ticker_sent.var())
        sys_var     = float(ticker_sent.cov(market_sent) ** 2 / (market_sent.var() + 1e-12)
                            if market_sent.var() > 0 else 0.0)
        # R² = corr² between ticker sentiment and market sentiment
        corr = float(ticker_sent.corr(market_sent))
        r2   = corr ** 2
        sys_var_pct[t]  = r2 * 100
        idio_var_pct[t] = (1 - r2) * 100
        idio_panels[t]  = ticker_sent - market_sent

    # ── Granger causality: systematic and idiosyncratic → ticker vol ──────────
    granger_sys  = {}
    granger_idio = {}

    for t in panel.columns:
        if t not in df_dict or vol_col not in df_dict[t].columns:
            continue
        vol = df_dict[t][vol_col]
        common = vol.index.intersection(panel.index)
        if len(common) < 30:
            continue

        for label, sent_series, result_dict in [
            ("systematic", market_sent,  granger_sys),
            ("idio",       idio_panels[t], granger_idio),
        ]:
            data = pd.DataFrame({
                "vol":  vol.loc[common],
                "sent": sent_series.loc[common],
            }).dropna()
            if len(data) < 20:
                result_dict[t] = np.nan
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = grangercausalitytests(data[["vol", "sent"]], maxlag=5, verbose=False)
                min_p = min(res[lag][0]["ssr_ftest"][1] for lag in range(1, 6))
                result_dict[t] = min_p
            except Exception:
                result_dict[t] = np.nan

    # ── Spike-day correlation ─────────────────────────────────────────────────
    spike_corr_sys  = {}
    spike_corr_idio = {}

    for t in panel.columns:
        if t not in df_dict or vol_col not in df_dict[t].columns:
            continue
        vol = df_dict[t][vol_col]
        thresh = vol.quantile(spike_pct)
        spike_mask = vol > thresh
        common = vol.index.intersection(panel.index)

        spike_idx = common[spike_mask.reindex(common).fillna(False)]
        if len(spike_idx) < 5:
            spike_corr_sys[t] = spike_corr_idio[t] = np.nan
            continue

        vol_spike = vol.loc[spike_idx]
        spike_corr_sys[t]  = float(vol_spike.corr(market_sent.reindex(spike_idx)))
        spike_corr_idio[t] = float(vol_spike.corr(idio_panels[t].reindex(spike_idx)))

    # ── Flag tickers where >60% of sentiment variance is systematic ───────────
    flag_systematic = [t for t, v in sys_var_pct.items() if v > 60]

    if flag_systematic:
        warnings.warn(
            f"[sentiment_decomp] Tickers with >60% systematic sentiment: {flag_systematic}. "
            "Scrapers may be pulling general market news rather than ticker-specific content."
        )

    # ── Summary print ─────────────────────────────────────────────────────────
    print(f"\n  [sentiment_decomp] Variance decomposition:")
    print(f"  {'Ticker':<8} {'Systematic%':>12} {'Idiosyncratic%':>15} "
          f"{'Granger Sys p':>14} {'Granger Idio p':>15}")
    for t in panel.columns:
        sp  = sys_var_pct.get(t, np.nan)
        ip  = idio_var_pct.get(t, np.nan)
        gsp = granger_sys.get(t, np.nan)
        gip = granger_idio.get(t, np.nan)
        print(f"  {t:<8} {sp:>12.1f} {ip:>15.1f} "
              f"{(f'{gsp:.4f}' if not np.isnan(gsp) else 'n/a'):>14} "
              f"{(f'{gip:.4f}' if not np.isnan(gip) else 'n/a'):>15}")

    return {
        "systematic_var_pct":    pd.Series(sys_var_pct),
        "idio_var_pct":          pd.Series(idio_var_pct),
        "granger_systematic":    pd.Series(granger_sys),
        "granger_idio":          pd.Series(granger_idio),
        "spike_corr_systematic": pd.Series(spike_corr_sys),
        "spike_corr_idio":       pd.Series(spike_corr_idio),
        "flag_systematic":       flag_systematic,
        "market_sentiment":      market_sent,
    }
