"""
Run a focused subset of the L2 advanced hypothesis suite: H9, H14, H16, H20, H22.

Reuses assemble_inputs() from run_advanced_hypotheses so the inputs (real price/
vol panel, EGARCH-ML disagreement, simulated sentiment / proxy 10-K scores) are
built exactly as the full suite builds them, then runs only the five requested
tests and prints each one's verdict.

Input provenance (flagged in the output so results aren't over-read):
  H14, H16  -> REAL price/vol data only           (stable run-to-run)
  H9, H22   -> depend on SIMULATED sentiment        (hypothesis-generating)
  H20       -> depends on PROXY 10-K risk scores    (hypothesis-generating)

Usage:
    python scripts/run_selected_hypotheses.py --ticker MU --start 2018-01-01 --end 2024-12-31
"""
import argparse
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

warnings.filterwarnings("ignore")

import modules.hypothesis_tests_L2 as L2
from run_advanced_hypotheses import assemble_inputs

SELECTED = ["H9", "H14", "H16", "H20", "H22"]
PROVENANCE = {
    "H9": "SIMULATED sentiment",
    "H14": "REAL price/vol",
    "H16": "REAL price/vol",
    "H20": "PROXY 10-K scores",
    "H22": "SIMULATED sentiment",
}


def run_selected(in_: dict, ticker: str) -> dict:
    """Run only H9/H14/H16/H20/H22 against the assembled inputs."""
    df, sent = in_["df"], in_["sent"]
    return {
        "H9": L2.test_negative_sentiment_spike_risk(in_["df_h9"], in_["sent_h9"], ticker),
        "H14": L2.test_directional_spillover_hub(in_["df_dict"]),
        "H16": L2.test_leverage_amplification(df, ticker),
        "H20": L2.test_10k_language_change_predicts_regime(df, in_["lm_scores"], ticker),
        "H22": L2.test_triple_signal_interaction(df, sent, in_["disagree"], ticker),
    }


def _fmt(res: dict, hyp: str) -> str:
    """One-line summary for a single hypothesis result dict."""
    r = res[hyp]
    p = r.get("p_value")
    p_str = f"p={p:.4f}" if isinstance(p, (int, float)) and p == p else "p=n/a"
    sig = r.get("significant")
    if sig is None and isinstance(p, (int, float)) and p == p:
        sig = p < 0.05
    verdict = "SIGNIFICANT" if sig else ("n/a" if sig is None else "not significant")
    return (f"  [{hyp}] {PROVENANCE[hyp]:<18} {p_str:<12} {verdict}\n"
            f"        {r.get('conclusion', '(no conclusion field)')}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="MU")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2024-12-31")
    args = ap.parse_args()

    print(f"\nAssembling inputs for {args.ticker} ({args.start} -> {args.end})...")
    inputs = assemble_inputs(args.ticker, args.start, args.end)
    res = run_selected(inputs, args.ticker)

    print("\n" + "=" * 78)
    print(f"  SELECTED L2 HYPOTHESES — {args.ticker}  (H9, H14, H16, H20, H22)")
    print("=" * 78)
    for hyp in SELECTED:
        print(_fmt(res, hyp))
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
