"""
Portfolio-level volatility forecast aggregation.

Combines per-ticker vol forecasts into a portfolio volatility estimate
using the standard covariance matrix formula: port_vol = sqrt(w' * Σ * w)
where Σ is built from individual vol forecasts on the diagonal and
pairwise correlations off-diagonal.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd


def compute_portfolio_vol_forecast(
    tickers: list[str],
    weights: dict[str, float],
    forecasts: dict[str, float],
    corr_matrix: pd.DataFrame,
    spy_forecast: float | None = None,
) -> dict:
    """
    Given per-ticker vol forecasts and a correlation matrix estimated from
    the last 60 days of returns, computes the implied portfolio volatility
    using the standard matrix formula: port_vol = sqrt(w' * Sigma * w)
    where Sigma is built from individual vol forecasts on the diagonal and
    pairwise correlations off-diagonal.

    Compares this to SPY ensemble forecast as a sanity check — portfolio vol
    should be lower than average individual vol due to diversification.
    If portfolio vol > SPY vol something is wrong (logs as a data quality warning).

    Parameters
    ----------
    tickers     : Ordered list of ticker symbols.
    weights     : Dict of {ticker: weight}. Weights are normalised to sum to 1.
    forecasts   : Dict of {ticker: annualized_vol_forecast}. Missing tickers
                  are skipped with a warning.
    corr_matrix : Square DataFrame of pairwise correlations indexed/columned
                  by ticker symbols. Built from the last 60 days of returns.
    spy_forecast: SPY ensemble vol forecast for the sanity check (optional).

    Returns dict with:
      port_vol          — annualized portfolio vol (scalar)
      avg_individual_vol — weighted average of individual vol forecasts
      diversification_ratio — avg_individual_vol / port_vol (> 1 = diversification)
      tickers_used      — list of tickers included in the computation
      tickers_missing   — list of tickers with missing forecasts (skipped)
      sanity_ok         — bool (True if port_vol < SPY forecast, or SPY unavailable)
      weights_used      — normalised weights dict
      covariance_matrix — the Sigma matrix used (as DataFrame)
    """
    # ── 1. Filter to tickers with both weight and forecast ───────────────────
    available = [t for t in tickers if t in forecasts and not np.isnan(forecasts.get(t, np.nan))]
    missing   = [t for t in tickers if t not in available]
    if missing:
        warnings.warn(f"[portfolio] Skipping tickers with missing forecasts: {missing}")
    if len(available) < 2:
        return {
            "port_vol": float("nan"),
            "available": False,
            "error": f"Need at least 2 tickers with forecasts; got {len(available)}.",
        }

    # ── 2. Normalise weights ──────────────────────────────────────────────────
    raw_w   = np.array([weights.get(t, 1.0 / len(available)) for t in available])
    raw_w  /= raw_w.sum()
    w_dict  = {t: float(raw_w[i]) for i, t in enumerate(available)}

    # ── 3. Build covariance matrix Σ from vol forecasts + correlations ───────
    n = len(available)
    sigma = np.zeros((n, n))
    for i, ti in enumerate(available):
        for j, tj in enumerate(available):
            rho = 1.0 if ti == tj else float(
                corr_matrix.at[ti, tj]
                if (ti in corr_matrix.index and tj in corr_matrix.columns)
                else 0.0
            )
            rho = np.clip(rho, -1.0, 1.0)
            sigma[i, j] = forecasts[ti] * forecasts[tj] * rho

    sigma_df = pd.DataFrame(sigma, index=available, columns=available)

    # ── 4. Portfolio variance and vol ────────────────────────────────────────
    port_var = float(raw_w @ sigma @ raw_w)
    port_var = max(port_var, 1e-12)  # numerical floor
    port_vol = float(np.sqrt(port_var))

    avg_indiv = float(np.dot(raw_w, [forecasts[t] for t in available]))
    div_ratio = avg_indiv / port_vol if port_vol > 1e-8 else float("nan")

    # ── 5. Sanity check ──────────────────────────────────────────────────────
    sanity_ok = True
    sanity_note = ""
    if spy_forecast is not None and not np.isnan(spy_forecast):
        if port_vol > spy_forecast * 1.05:  # 5% tolerance
            sanity_ok = False
            sanity_note = (
                f"DATA QUALITY WARNING: portfolio vol ({port_vol:.1%}) > SPY forecast "
                f"({spy_forecast:.1%}). Check correlation inputs or weights."
            )
            warnings.warn(f"[portfolio] {sanity_note}")
        else:
            sanity_note = (
                f"Sanity OK: portfolio vol ({port_vol:.1%}) < SPY forecast "
                f"({spy_forecast:.1%}) — diversification is working."
            )

    result = dict(
        available=True,
        port_vol=round(port_vol, 6),
        avg_individual_vol=round(avg_indiv, 6),
        diversification_ratio=round(div_ratio, 4),
        tickers_used=available,
        tickers_missing=missing,
        weights_used=w_dict,
        covariance_matrix=sigma_df,
        sanity_ok=sanity_ok,
        sanity_note=sanity_note,
    )

    _print_portfolio_summary(result, spy_forecast)
    return result


def _print_portfolio_summary(result: dict, spy_forecast: float | None) -> None:
    print(f"\n{'='*65}")
    print(f"  PORTFOLIO VOL FORECAST")
    print(f"{'='*65}")
    print(f"  Tickers: {', '.join(result['tickers_used'])}")
    if result["tickers_missing"]:
        print(f"  Skipped: {', '.join(result['tickers_missing'])} (no forecast)")
    print(f"")
    print(f"  Portfolio vol        : {result['port_vol']:.2%}")
    print(f"  Avg individual vol   : {result['avg_individual_vol']:.2%}")
    print(f"  Diversification ratio: {result['diversification_ratio']:.3f}x")
    if spy_forecast:
        print(f"  SPY ensemble forecast: {spy_forecast:.2%}")
    print(f"")
    if result.get("sanity_note"):
        prefix = "  ⚠" if not result["sanity_ok"] else "  ✓"
        print(f"{prefix} {result['sanity_note']}")
    print(f"{'='*65}")


def build_correlation_matrix(
    returns_dict: dict[str, pd.Series],
    window: int = 60,
) -> pd.DataFrame:
    """
    Estimate a pairwise correlation matrix from the last `window` days of returns.

    Parameters
    ----------
    returns_dict : Dict of {ticker: pd.Series of log returns}.
    window       : Rolling window in trading days (default 60).

    Returns a square DataFrame of pairwise Pearson correlations.
    """
    tickers = list(returns_dict.keys())
    n = len(tickers)
    corr = pd.DataFrame(np.eye(n), index=tickers, columns=tickers)

    for i, ti in enumerate(tickers):
        ri = returns_dict[ti].dropna().tail(window)
        for j, tj in enumerate(tickers):
            if i >= j:
                continue
            rj = returns_dict[tj].dropna().tail(window)
            common_idx = ri.index.intersection(rj.index)
            if len(common_idx) < 10:
                c = 0.0
            else:
                c = float(ri.reindex(common_idx).corr(rj.reindex(common_idx)))
            if np.isnan(c):
                c = 0.0
            corr.at[ti, tj] = c
            corr.at[tj, ti] = c

    return corr
