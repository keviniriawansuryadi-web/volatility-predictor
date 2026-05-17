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


def disagreement_vol_test(
    feat_df: pd.DataFrame,
    garch_preds: pd.Series,
    ml_preds: pd.Series,
    threshold_pct: float = 0.20,
) -> dict:
    """
    H2: High EGARCH-ML disagreement days predict above-median realized vol.

    Disagreement is defined as the absolute relative difference between the
    EGARCH and XGBoost forecasts, normalised by their mean:

        disagreement = |egarch - ml| / ((|egarch| + |ml|) / 2 + epsilon)

    Days where disagreement exceeds the `threshold_pct` percentile of all
    disagreement scores are classed as 'high-disagreement' days.

    Test: one-sided Mann-Whitney U (high-disagreement vol > low-disagreement vol).

    When the two models disagree strongly, at least one is surprised by the
    regime — this uncertainty itself is a signal that realized vol will be
    elevated relative to the calm baseline.

    Parameters
    ----------
    feat_df         : Feature DataFrame with a 'target' column.
    garch_preds     : pd.Series of EGARCH test-set forecasts.
    ml_preds        : pd.Series of XGBoost test-set forecasts.
    threshold_pct   : Percentile cutoff defining 'high disagreement' (default 0.20
                      = top 80th percentile of disagreement scores).

    Returns a result dict suitable for printing via print_disagreement_results().
    """
    common_index = feat_df.index.intersection(garch_preds.index).intersection(ml_preds.index)
    if len(common_index) < 20:
        return {
            "available": False,
            "reason": f"Only {len(common_index)} overlapping rows — need >= 20.",
        }

    y_true = feat_df.loc[common_index, "target"]
    eg = garch_preds.reindex(common_index)
    ml = ml_preds.reindex(common_index)

    denom = (eg.abs() + ml.abs()) / 2 + 1e-8
    disagreement = (eg - ml).abs() / denom

    cutoff = disagreement.quantile(1.0 - threshold_pct)
    high_mask = disagreement >= cutoff

    high_vol = y_true[high_mask].dropna()
    low_vol  = y_true[~high_mask].dropna()

    if len(high_vol) < 5 or len(low_vol) < 5:
        return {
            "available": False,
            "reason": (
                f"Insufficient samples: high_disagreement n={len(high_vol)}, "
                f"low_disagreement n={len(low_vol)}. Need >= 5 each."
            ),
        }

    u_stat, p_value = stats.mannwhitneyu(high_vol, low_vol, alternative="greater")

    pooled_std = np.sqrt((high_vol.std() ** 2 + low_vol.std() ** 2) / 2)
    cohens_d = (high_vol.mean() - low_vol.mean()) / (pooled_std + 1e-10)

    return {
        "available":            True,
        "threshold_percentile": threshold_pct,
        "disagreement_cutoff":  float(cutoff),
        "n_high":               int(high_mask.sum()),
        "n_low":                int((~high_mask).sum()),
        "high_mean_vol":        float(high_vol.mean()),
        "low_mean_vol":         float(low_vol.mean()),
        "mann_whitney_u":       float(u_stat),
        "p_value":              float(p_value),
        "cohens_d":             float(cohens_d),
        "significant":          p_value < 0.05,
        "disagreement_series":  disagreement,  # for adding as feature / signal
    }


def print_disagreement_results(result: dict) -> None:
    """Pretty-print the output of disagreement_vol_test()."""
    print("\n" + "=" * 60)
    print("  HYPOTHESIS TEST: EGARCH-ML Disagreement vs Realized Vol")
    print("=" * 60)

    if not result.get("available"):
        print(f"  Result: INCONCLUSIVE\n  Reason: {result.get('reason')}")
        print("=" * 60)
        return

    print(f"  H2: High EGARCH-ML disagreement days have elevated realized vol")
    print(f"  Disagreement cutoff (top {result['threshold_percentile']:.0%}): "
          f"{result['disagreement_cutoff']:.4f}")
    print(f"  High-disagreement (n={result['n_high']}): "
          f"mean vol = {result['high_mean_vol']:.1%}")
    print(f"  Low-disagreement  (n={result['n_low']}): "
          f"mean vol = {result['low_mean_vol']:.1%}")
    print(f"  Mann-Whitney U = {result['mann_whitney_u']:.1f}  |  "
          f"p-value = {result['p_value']:.4f}  |  "
          f"Cohen's d = {result['cohens_d']:.3f}")

    if result["significant"]:
        magnitude = (
            "large" if abs(result["cohens_d"]) > 0.8
            else "medium" if abs(result["cohens_d"]) > 0.5
            else "small"
        )
        print(f"\n  REJECT H0 (p < 0.05). High EGARCH-ML disagreement days have")
        print(f"  significantly higher realized vol (effect: {magnitude}).")
    else:
        print(f"\n  FAIL TO REJECT H0 (p >= 0.05). No significant vol difference")
        print(f"  between high- and low-disagreement days.")

    print("=" * 60)


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
