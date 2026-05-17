# Hypothesis Test Results — MU Volatility Research

Ticker: MU | Period: 2015-01-01 to 2024-12-31 | Horizon: 5 trading days

| Hypothesis | Test Used | p-value | Effect Size | Significant? | Conclusion |
|------------|-----------|---------|-------------|--------------|------------|
| H1 — Spike days preceded by negative sentiment | Two-sample t-test + Bootstrap CI (Cohen's d) | 0.3266 | d = −0.061 | No | No detectable pre-spike sentiment difference on MU |
| H2 — Leverage effect | Mann-Whitney U (rank-biserial r) | 0.0651 | r = −0.035 | No (marginal) | Weak leverage signal, not significant at α=0.05 |
| H3 — Sentiment Granger-causes vol | Granger F-test (lags 1–5) | min 0.1465 | — | No | Dominant direction: Sentiment→Vol, but not significant |
| H4 — Monday Effect | Kruskal-Wallis + Dunn post-hoc (η²) | 0.9863 | η² = 0.000 | No | Zero weekday vol pattern on MU |
| H5 — Earnings week vol regime | Permutation test (n=5000) | — | — | Inconclusive | No earnings date data from yfinance |
| H6 — VIX regime and ML accuracy | Levene test | 0.3719 | — | No | RMSE not significantly different across VIX regimes |
| H7 — 10-K risk language | Mann-Whitney U + Bootstrap CI | — | — | Inconclusive | Insufficient 10-K filing data |
| **H8 — Sentiment mean reversion** | **Wilcoxon signed-rank (paired)** | **< 0.0001** | **W = 692.5** | **Yes ✓** | **Sentiment recovers −0.723 → −0.061 by t+3** |

**Only H8 is statistically significant.** After extreme negative sentiment days (VADER < -0.5),
sentiment recovers significantly by t+3, suggesting textual overreaction before price recovery.
