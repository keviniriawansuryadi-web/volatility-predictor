# Hypothesis Test Results -- MU Volatility Research
Generated from hypotheses.ipynb
Ticker: MU | Period: 2015-01-01 to 2024-12-31

| Hypothesis | Test Used | p-value | Effect Size | Significant? | Conclusion |
|------------|-----------|---------|-------------|--------------|------------|
| H1 | Two-sample t-test + Bootstrap CI (Cohen's d) | 0.3266 | -0.061 | No | Spike days show more negative prior sentiment (d=-0.061, p=0.3266). |
| H2 | Mann-Whitney U (rank-biserial r) | 0.0651 | -0.035 | No | Negative return days show not significantly higher future vol (U=789490, p=0.0651, r=-0.035). |
| H3 | Granger causality F-test (lags 1,2,3,5) | -- | -- | -- | Minimum p: Sentiment->Vol=0.1465, Vol->Sentiment=0.8517. Dominant Granger direction: Sentiment -> Vo |
| H4 | Kruskal-Wallis + Dunn post-hoc (eta2) | 0.9863 | 0.000 | No | Kruskal-Wallis H=0.351, p=0.9863 (not significant at a=0.05). eta2=0.000. |
| H5 | Permutation test n=5000 | -- | -- | -- | No earnings data. |
| H6 | Levene test for equal variances | 0.3719 | -- | No | RMSE: Low VIX=0.2093, High VIX=0.2259. Levene W=0.799, p=0.3719 (no significant difference). |
| H7 | Mann-Whitney U + Bootstrap CI | -- | -- | -- | Insufficient filing data for H7. |
| H8 | Wilcoxon signed-rank (paired) | 0.0000 | -- | Yes | Extreme negative days: mean sentiment = -0.723 -> -0.061 at t+3. Wilcoxon W=692.5, p=0.0000 (signifi |
