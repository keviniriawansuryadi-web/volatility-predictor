import pandas as pd
import numpy as np
from scipy import stats


def spike_sentiment_test(feat_df: pd.DataFrame, lookback: int = 3) -> dict:
    """
    H1: spike days (realized_vol > 90th pct) are preceded by significantly
    more negative VADER sentiment than non-spike days.

    Test: one-sided Mann-Whitney U (spike sentiment < non-spike sentiment).
    Also reports Cohen's d effect size.
    """
    if "sentiment" not in feat_df.columns or feat_df["sentiment"].abs().sum() == 0:
        return {
            "available": False,
            "reason": "No real sentiment data (all zero). Run with a ticker that has recent news.",
        }

    rv = feat_df["target"]
    spike_thresh = rv.quantile(0.90)
    spike_mask = rv > spike_thresh

    # Rolling mean of sentiment in the `lookback` days preceding each row
    sentiment_pre = feat_df["sentiment"].rolling(lookback).mean().shift(1)

    spike_scores = sentiment_pre[spike_mask].dropna()
    non_spike_scores = sentiment_pre[~spike_mask].dropna()

    if len(spike_scores) < 5 or len(non_spike_scores) < 5:
        return {
            "available": False,
            "reason": (
                f"Insufficient sentiment data: only {len(spike_scores)} spike days "
                f"with non-zero sentiment. Need >= 5."
            ),
        }

    # One-sided: H1 = spike sentiment < non-spike sentiment (more negative before spikes)
    u_stat, p_value = stats.mannwhitneyu(spike_scores, non_spike_scores, alternative="less")

    pooled_std = np.sqrt((spike_scores.std() ** 2 + non_spike_scores.std() ** 2) / 2)
    cohens_d = (spike_scores.mean() - non_spike_scores.mean()) / (pooled_std + 1e-10)

    return {
        "available": True,
        "spike_threshold": spike_thresh,
        "n_spike": len(spike_scores),
        "n_non_spike": len(non_spike_scores),
        "spike_mean_sentiment": spike_scores.mean(),
        "non_spike_mean_sentiment": non_spike_scores.mean(),
        "mann_whitney_u": u_stat,
        "p_value": p_value,
        "cohens_d": cohens_d,
        "significant": p_value < 0.05,
    }


def print_hypothesis_results(result: dict) -> None:
    print("\n" + "=" * 60)
    print("  HYPOTHESIS TEST: Sentiment Before Volatility Spikes")
    print("=" * 60)

    if not result.get("available"):
        print(f"  Result: INCONCLUSIVE\n  Reason: {result.get('reason')}")
        print("=" * 60)
        return

    print(f"  H1: Spike days preceded by more negative sentiment")
    print(f"  Spike threshold (90th pct): {result['spike_threshold']:.1%} annualized vol")
    print(f"  Spike days  (n={result['n_spike']}): "
          f"mean sentiment = {result['spike_mean_sentiment']:.4f}")
    print(f"  Non-spike   (n={result['n_non_spike']}): "
          f"mean sentiment = {result['non_spike_mean_sentiment']:.4f}")
    print(f"  Mann-Whitney U = {result['mann_whitney_u']:.1f}  |  "
          f"p-value = {result['p_value']:.4f}  |  "
          f"Cohen's d = {result['cohens_d']:.3f}")

    if result["significant"]:
        magnitude = (
            "large" if abs(result["cohens_d"]) > 0.8
            else "medium" if abs(result["cohens_d"]) > 0.5
            else "small"
        )
        print(f"\n  REJECT H0 (p < 0.05). Spike days ARE preceded by significantly")
        print(f"  more negative sentiment (effect size: {magnitude}).")
    else:
        print(f"\n  FAIL TO REJECT H0 (p >= 0.05). No statistically significant")
        print(f"  difference in pre-spike sentiment detected.")

    print("=" * 60)
