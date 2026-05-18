"""
Section 12 — Unified market dashboard.

Loads all live signal JSONs from outputs/ and assembles a single
market_dashboard JSON with:
  - market_stress_score  : 0-100 index (cap-weighted ensemble vol, normalised
                           so 100 = every ticker in Extreme regime ≥35%)
  - regime_breakdown     : tickers bucketed into Extreme / High / Elevated / Low
  - top_disagreement_signals : top-3 by disagreement value with interpretation
  - tickers              : per-ticker summary (regime, forecasts, flags)

Usage:
    python scripts/daily_run.py
    python scripts/daily_run.py --date 2026-05-15
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import date
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

# ── Market-cap weights (10 core tickers; ~S&P 500 weight proxies) ────────────
MARKET_CAP_WEIGHTS: dict[str, float] = {
    "AAPL": 0.22,
    "MSFT": 0.20,
    "AMZN": 0.12,
    "NVDA": 0.11,
    "JPM":  0.08,
    "XOM":  0.06,
    "BAC":  0.05,
    "AMD":  0.04,
    "CVX":  0.04,
    "MU":   0.03,
}

# Regime thresholds (annualised daily RV)
REGIME_EXTREME  = 0.35
REGIME_HIGH     = 0.25
REGIME_ELEVATED = 0.15


def _regime_label(vol: float) -> str:
    if vol >= REGIME_EXTREME:
        return "Extreme"
    if vol >= REGIME_HIGH:
        return "High"
    if vol >= REGIME_ELEVATED:
        return "Elevated"
    return "Low"


def generate_market_dashboard(all_results: dict[str, dict]) -> dict[str, Any]:
    """
    Build a market dashboard dict from per-ticker live signal records.

    Parameters
    ----------
    all_results : Dict keyed by ticker, each value is a parsed live-signal JSON
                  (keys: forecasts, regime, disagreement, regime_persistence, …).

    Returns a dict with:
      as_of                    : signal date (str)
      market_stress_score      : int 0-100
      regime_breakdown         : dict {Extreme/High/Elevated/Low: [tickers]}
      top_disagreement_signals : list of top-3 disagreement records
      tickers                  : dict of per-ticker summaries

    Market stress score methodology
    --------------------------------
    Weighted average ensemble forecast across all tickers, normalised by the
    Extreme regime threshold (0.35).  Score of 100 means every ticker sits at
    or above the Extreme boundary after cap-weighting.

        stress = min(100, round(weighted_avg(ensemble_vol) / REGIME_EXTREME * 100))

    Tickers without an explicit cap weight receive an equal share of the
    residual weight after the 10-ticker core is accounted for.
    """
    if not all_results:
        return {}

    # ── Compute weights ───────────────────────────────────────────────────────
    core_weight_sum  = sum(MARKET_CAP_WEIGHTS.get(t.upper(), 0) for t in all_results)
    non_core_tickers = [t for t in all_results if t.upper() not in MARKET_CAP_WEIGHTS]
    residual_weight  = max(0.0, 1.0 - core_weight_sum)
    per_extra_weight = (residual_weight / len(non_core_tickers)) if non_core_tickers else 0.0

    weights: dict[str, float] = {}
    for ticker in all_results:
        w = MARKET_CAP_WEIGHTS.get(ticker.upper(), per_extra_weight)
        weights[ticker] = w

    total_w = sum(weights.values()) or 1.0
    weights = {t: w / total_w for t, w in weights.items()}  # normalise to sum=1

    # ── Stress score ──────────────────────────────────────────────────────────
    weighted_vol = 0.0
    for ticker, record in all_results.items():
        ensemble_vol = record.get("forecasts", {}).get("ensemble", 0.0)
        weighted_vol += weights[ticker] * ensemble_vol

    market_stress_score = min(100, round(weighted_vol / REGIME_EXTREME * 100))

    # ── Regime breakdown ──────────────────────────────────────────────────────
    regime_breakdown: dict[str, list[str]] = {
        "Extreme": [], "High": [], "Elevated": [], "Low": []
    }
    for ticker, record in all_results.items():
        ensemble_vol = record.get("forecasts", {}).get("ensemble", 0.0)
        label = _regime_label(ensemble_vol)
        regime_breakdown[label].append(ticker)

    # ── Disagreement signals (top-3 by value) ────────────────────────────────
    disagreement_records = []
    for ticker, record in all_results.items():
        dis = record.get("disagreement", {})
        dis_val = dis.get("value", 0.0)
        interpretation = (
            dis.get("live_pct", {}).get("interpretation") or
            ("HIGH disagreement" if dis.get("flag") else "LOW disagreement")
        )
        disagreement_records.append({
            "ticker":         ticker,
            "disagreement":   round(float(dis_val), 4),
            "flag":           bool(dis.get("flag", False)),
            "interpretation": interpretation,
            "backtest_signal": dis.get("backtest", {}).get("interpretation", ""),
        })
    disagreement_records.sort(key=lambda x: x["disagreement"], reverse=True)
    top_disagreement_signals = disagreement_records[:3]

    # ── Per-ticker summaries ──────────────────────────────────────────────────
    tickers_detail: dict[str, dict] = {}
    for ticker, record in all_results.items():
        forecasts = record.get("forecasts", {})
        ensemble  = forecasts.get("ensemble", 0.0)
        rv21d     = record.get("realized_vol_21d", 0.0)
        dis       = record.get("disagreement", {})
        pers      = record.get("regime_persistence", {})
        vrp       = record.get("variance_risk_premium", {})

        tickers_detail[ticker] = {
            "regime":           _regime_label(ensemble),
            "regime_label_raw": record.get("regime", ""),
            "ensemble_forecast": round(float(ensemble), 4),
            "realized_vol_21d": round(float(rv21d), 4),
            "forecasts": {
                k: round(float(v), 4) for k, v in forecasts.items()
            },
            "disagreement_flag":  bool(dis.get("flag", False)),
            "disagreement_value": round(float(dis.get("value", 0.0)), 4),
            "current_regime_run_days": pers.get("current_run_days", None),
            "expected_reversion_days": pers.get("expected_reversion_days", None),
            "vrp_state": vrp.get("state", None) if vrp else None,
            "sentiment_h1_significant": record.get("sentiment", {}).get("h1_significant", False),
            "cap_weight": round(weights[ticker], 4),
        }

    # ── Signal date from first record ─────────────────────────────────────────
    first_record = next(iter(all_results.values()))
    as_of = first_record.get("signal_date", date.today().isoformat())

    return {
        "as_of":                   as_of,
        "market_stress_score":     market_stress_score,
        "regime_breakdown":        regime_breakdown,
        "top_disagreement_signals": top_disagreement_signals,
        "tickers":                 tickers_detail,
    }


def print_dashboard(dashboard_data: dict[str, Any]) -> None:
    """
    Render the market dashboard as a formatted terminal table.

    Prints:
      - Market stress score and regime breakdown counts
      - Per-ticker table: regime, ensemble forecast, disagreement flag, VRP state
      - Top-3 disagreement signals
    """
    if not dashboard_data:
        print("  [dashboard] No data to display.")
        return

    as_of  = dashboard_data.get("as_of", "?")
    score  = dashboard_data.get("market_stress_score", 0)
    rb     = dashboard_data.get("regime_breakdown", {})
    top_d  = dashboard_data.get("top_disagreement_signals", [])
    tickers = dashboard_data.get("tickers", {})

    # Stress score bar
    bar_len = 40
    filled  = round(score / 100 * bar_len)
    bar     = "█" * filled + "░" * (bar_len - filled)

    stress_label = (
        "EXTREME STRESS"  if score >= 75 else
        "HIGH STRESS"     if score >= 50 else
        "ELEVATED STRESS" if score >= 25 else
        "CALM"
    )

    print(f"\n{'═'*65}")
    print(f"  MARKET DASHBOARD  ·  as of {as_of}")
    print(f"{'═'*65}")
    print(f"\n  Market Stress Score: {score:3d}/100  [{stress_label}]")
    print(f"  [{bar}]")
    print(f"\n  Regime Breakdown:")
    for label in ("Extreme", "High", "Elevated", "Low"):
        tlist = rb.get(label, [])
        bar_r = "■" * len(tlist)
        print(f"    {label:<10} {len(tlist):2d}  {bar_r}  {', '.join(tlist) if tlist else '—'}")

    # Per-ticker table
    print(f"\n  {'Ticker':<8} {'Regime':<10} {'Ensbl':>6} {'RV21d':>6} {'Dis':>5} {'VRP':>22}  Regime Run")
    print(f"  {'-'*8} {'-'*10} {'-'*6} {'-'*6} {'-'*5} {'-'*22}  {'-'*10}")
    for ticker, det in sorted(tickers.items(), key=lambda x: -x[1]["ensemble_forecast"]):
        regime    = det["regime"]
        ensemble  = det["ensemble_forecast"]
        rv21d     = det["realized_vol_21d"]
        dis_flag  = "⚠" if det["disagreement_flag"] else " "
        dis_val   = det["disagreement_value"]
        vrp_state = (det["vrp_state"] or "")[:22]
        run_days  = det.get("current_regime_run_days")
        run_str   = f"{run_days}d" if run_days is not None else "?"

        regime_color = ""
        print(f"  {ticker:<8} {regime:<10} {ensemble:>6.1%} {rv21d:>6.1%} {dis_flag}{dis_val:>4.2f} {vrp_state:<22}  {run_str}")

    # Top disagreement
    print(f"\n  Top-3 Disagreement Signals:")
    for i, sig in enumerate(top_d, 1):
        flag_str = "⚠ FLAG" if sig["flag"] else "      "
        print(f"    {i}. {sig['ticker']:<6} dis={sig['disagreement']:.3f} {flag_str}  {sig['interpretation']}")
        if sig["backtest_signal"]:
            print(f"           backtest: {sig['backtest_signal']}")

    print(f"\n{'═'*65}\n")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate market dashboard from live signals.")
    parser.add_argument("--date", default="2026-05-15",
                        help="Signal date suffix for live_signal_{TICKER}_{DATE}.json files")
    args = parser.parse_args()

    signal_date = args.date
    out_dir     = Path(__file__).parent.parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  SECTION 12 — MARKET DASHBOARD")
    print(f"{'='*65}")

    # ── Load all live signals for the given date ──────────────────────────────
    all_results: dict[str, dict] = {}
    pattern = f"live_signal_*_{signal_date}.json"
    for path in sorted(out_dir.glob(pattern)):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            ticker = record.get("ticker") or path.stem.split("_")[2]
            all_results[ticker] = record
        except Exception as exc:
            print(f"  [WARNING] Could not load {path.name}: {exc}")

    if not all_results:
        print(f"  No live signal files found matching: outputs/{pattern}")
        print(f"  Run the main pipeline first (e.g. python main.py --ticker SPY).")
        sys.exit(1)

    print(f"\n  Loaded {len(all_results)} live signals for {signal_date}:")
    print(f"  {sorted(all_results.keys())}")

    # ── Generate dashboard ────────────────────────────────────────────────────
    dashboard = generate_market_dashboard(all_results)

    # ── Print to terminal ──────────────────────────────────────────────────────
    print_dashboard(dashboard)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_path = out_dir / f"market_dashboard_{signal_date}.json"
    out_path.write_text(json.dumps(dashboard, indent=2), encoding="utf-8")
    print(f"  Saved: {out_path}")
    print(f"\n  Section 12 COMPLETE.")
