# Diagnostic Report — MSFT

**Generated:** 2026-05-18 03:01

**Best QLIKE:** 0.3457  (threshold: 0.3)

**Reference ticker:** MU

---


## 1. Model Performance Summary

| model | RMSE | MAE | QLIKE | Corr | Spike_Acc |
| --- | --- | --- | --- | --- | --- |
| EGARCH | 0.1027 | 0.0789 | 0.3457 | 0.1394 | 0.0000 |
| XGB-Asymmetric | 0.1156 | 0.0871 | 0.4282 | 0.0887 | 0.0435 |
| HAR-RV | 0.1135 | 0.0833 | 0.4517 | 0.1163 | 0.0000 |
| RandomForest | 0.1191 | 0.0898 | 0.4562 | 0.0721 | 0.0000 |
| XGBoost | 0.1196 | 0.0898 | 0.4631 | 0.0288 | 0.0870 |


**Winner:** EGARCH (QLIKE=0.3457)


## 2. Data Quality

- `log_return`: 0 missing (0.0%)

- `realized_vol_21d`: 0 missing (0.0%)

- `vix_level`: 0 missing (0.0%)

- `sentiment`: 0 missing (0.0%)

- `garch_vol`: 0 missing (0.0%)


**Sentiment:** 1150 zero-days (100.0%) — POOR coverage, imputed from rolling median


## 3. Vol Regime Distribution (test set)

- Low: 50 days (21.7%)

- Elevated: 128 days (55.7%)

- High: 31 days (13.5%)

- Extreme: 21 days (9.1%)


## 4. Worst 10 Underestimation Errors (test set)

| Rank | Date | Realized Vol | EGARCH Forecast | Error | Earnings Proxy |

|------|------|-------------|-----------------|-------|----------------|

| 1 | 2026-01-13 | 48.5% | 20.4% | 28.2% | YES |

| 2 | 2026-01-12 | 48.6% | 20.6% | 28.0% | YES |

| 3 | 2026-01-08 | 48.4% | 20.9% | 27.5% | YES |

| 4 | 2026-01-14 | 48.2% | 20.9% | 27.3% | YES |

| 5 | 2026-01-09 | 48.3% | 21.2% | 27.2% | YES |

| 6 | 2026-01-20 | 48.2% | 21.8% | 26.4% | YES |

| 7 | 2026-01-16 | 48.2% | 22.0% | 26.2% | YES |

| 8 | 2026-01-15 | 48.2% | 22.2% | 26.0% | YES |

| 9 | 2026-01-21 | 47.8% | 22.0% | 25.8% | YES |

| 10 | 2026-01-07 | 46.4% | 20.8% | 25.6% | YES |


## 5. EGARCH Rolling Forecast Stability

- EGARCH in-sample vol CoV: 0.206

- Realized vol CoV (test):  0.438


**WARNING:** EGARCH CoV / RV CoV = 0.47 — EGARCH is over-smooth relative to realized vol. Possible non-convergence or dampened persistence estimates.


## 6. Targeted Fix Recommendations

- **Sentiment gap**: >80% imputed. Sentiment features add noise, not signal. Consider dropping sentiment features for this ticker in the feature set.

- **Time-series CV**: Retrain using expanding-window CV (3 folds) instead of single 80/20 split to get more robust estimates on spike-heavy tickers.

- **GJR-GARCH alternative**: GJR-GARCH allows asymmetric response to positive vs negative shocks and may converge more stably on AMD's regime-driven vol pattern.
