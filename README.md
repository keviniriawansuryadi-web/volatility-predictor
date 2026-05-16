# Volatility Predictor — MU / Equity Vol Forecasting Pipeline

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Models](https://img.shields.io/badge/models-EGARCH%20%7C%20HAR--RV%20%7C%20XGBoost%20%7C%20RF-green)

> **Forecast realized volatility, detect spike regimes, and generate live scalping signals** using EGARCH, HAR-RV, XGBoost, and Random Forest — with VIX integration, VADER sentiment, SHAP explainability, and eight formal statistical hypothesis tests.

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Quickstart](#quickstart)
4. [Models](#models)
5. [Features](#features)
6. [Exploratory Data Analysis](#exploratory-data-analysis)
7. [Hypothesis Tests](#hypothesis-tests)
8. [Evaluation Metrics](#evaluation-metrics)
9. [Results — MU (Micron Technology)](#results--mu-micron-technology)
10. [Academic References](#academic-references)

---

## Overview

This pipeline downloads daily OHLCV data from Yahoo Finance, engineers a rich feature set (realized vol at multiple lookbacks, VIX, VADER sentiment, jump flags, GARCH hybrid features), trains five volatility models, and outputs:

- A **model comparison table** (RMSE, MAE, QLIKE, Pearson correlation, spike detection accuracy)
- **Three diagnostic plots** per ticker — time series, error over time, predicted-vs-realized scatter
- A **SHAP feature importance chart** for the XGBoost model
- A **live forward signal** with vol regime classification for scalping decisions
- A **hypothesis test** on sentiment before vol spikes (Mann-Whitney U + Cohen's d)

---

## Project Structure

```
volatility-predictor/
│
├── main.py                      # CLI entry point
├── requirements.txt
│
├── src/
│   ├── data_loader.py           # yfinance download + CSV cache + VIX loader
│   ├── features.py              # Feature engineering (28 features)
│   ├── garch_model.py           # EGARCH / GARCH rolling + in-sample + latest forecast
│   ├── har_model.py             # HAR-RV linear model
│   ├── ml_model.py              # XGBoost (standard + asymmetric), Random Forest
│   ├── sentiment.py             # VADER sentiment from yfinance news
│   ├── evaluate.py              # RMSE, QLIKE, spike accuracy, 3-panel plot, SHAP
│   └── hypothesis.py            # Mann-Whitney U + Cohen's d spike-sentiment test
│
├── modules/
│   ├── data_helpers.py          # VIX loader, earnings dates, sentiment simulation
│   ├── eda_plots.py             # 10 interactive Plotly EDA functions
│   └── hypothesis_tests.py      # 8 formal statistical tests (H1–H8)
│
├── eda.ipynb                    # Exploratory Data Analysis notebook (10 analyses)
├── hypotheses.ipynb             # Hypothesis testing notebook (8 tests)
└── results_summary.md           # Auto-generated results table
```

---

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Run on any ticker — defaults to today's data (5-year rolling window)
python main.py --ticker MU

# Custom date range and forecast horizon
python main.py --ticker NVDA --start 2020-01-01 --end 2025-05-01 --horizon 3

# Force fresh download (skip CSV cache)
python main.py --ticker SPY --no-cache
```

**Output files** are saved to `outputs/plots/`:
- `{TICKER}_volatility_comparison.png` — 3-panel forecast chart
- `{TICKER}_shap_importance.png` — SHAP feature importance bar chart

---

## Models

| Model | Type | Description |
|-------|------|-------------|
| **EGARCH** | Statistical | Rolling EGARCH(1,1) via the `arch` library. Captures leverage effect and volatility clustering. |
| **HAR-RV** | Linear | Heterogeneous Autoregressive Realized Volatility (Corsi 2009). Regresses forward vol on daily, weekly (5d), and monthly (22d) averages. |
| **XGBoost** | ML | Gradient boosting on 28 engineered features including the GARCH fitted vol as a hybrid feature. |
| **XGBoost Asymmetric** | ML | Same architecture with a custom spike-weighted loss: 3× gradient penalty when the model underestimates days where realized vol > 50%. |
| **Random Forest** | ML | Ensemble of 200 trees; provides a robust nonlinear baseline. |

### GARCH + XGBoost Hybrid

The EGARCH in-sample conditional volatility is included as the feature `garch_vol` in the XGBoost feature matrix. This lets XGBoost learn *what GARCH misses* — regime shifts, jumps, and sentiment signals — rather than re-learning what GARCH already captures well.

### Asymmetric Spike Loss

The custom XGBoost objective applies a **3× gradient weight** whenever:
- The model **underestimates** (prediction < actual), AND
- Realized vol **exceeds 50% annualized** (a spike day)

This pushes the model to err on the side of over-forecasting during high-volatility environments — critical for scalping risk management.

---

## Features

28 features are active when VIX and sentiment data are available:

| Category | Features |
|----------|----------|
| **Lagged returns** | `ret_lag1`, `ret_lag2`, `ret_lag3`, `ret_lag5`, `ret_lag10` |
| **Realized vol (multi-lookback)** | `vol_5d`, `vol_10d`, `vol_20d`, `vol_21d`, `vol_60d`, `vol_63d` |
| **Vol-of-vol** | `vol_of_vol` (rolling std of 21d vol) |
| **Jump indicator** | `jump_flag` — binary: \|return\| > 2.5σ |
| **Volume** | `volume_ratio` (vs 10d MA) |
| **Statistical** | `ret_skew_21d`, `ret_kurt_21d` |
| **Technical** | `rsi_14`, `bb_width` |
| **VIX** | `vix_level`, `vix_change` |
| **Sentiment** | `sentiment_3d`, `sentiment_lag1/2/3` |
| **Vol lags** | `vol_lag1`, `vol_lag2`, `vol_lag3` |
| **GARCH hybrid** | `garch_vol` (EGARCH in-sample conditional vol) |

---

## Exploratory Data Analysis

The `eda.ipynb` notebook runs **10 interactive analyses** on MU (2015–2024):

| EDA | Analysis | Key Finding |
|-----|----------|-------------|
| **EDA 1** | Vol distribution + Q-Q plot | Fat-tailed, right-skewed → non-parametric tests required |
| **EDA 2** | Rolling correlation: sentiment vs vol | Spikes to strongly negative in High VIX periods → regime-aware features needed |
| **EDA 3** | Volatility ACF / PACF | Slow ACF decay + ARCH effects → GARCH well-motivated; long-memory dynamics |
| **EDA 4** | Sentiment model agreement (VADER / LM / FinBERT / TextBlob) | VADER–LM agree (r≈0.35); TextBlob diverges on financial language |
| **EDA 5** | 10-K filing event study | High-LM-Risk filings → slower post-filing vol decay |
| **EDA 6** | Weekday volatility pattern | Monday and Friday elevated → weekend news accumulation effect |
| **EDA 7** | Feature–target Spearman correlation ranking | Short-window vol lags dominate; sentiment meaningful but unstable |
| **EDA 8** | Joint sentiment–vol density | Mild U-shape → both raw VADER and \|VADER\| useful as features |
| **EDA 9** | Vol regime transition matrix | High-vol regime very persistent (diagonal ≥ 0.7); rare Low→High jumps |
| **EDA 10** | News volume vs realized vol | News volume leads vol by 1–2 days around earnings |

---

## Hypothesis Tests

The `hypotheses.ipynb` notebook runs **8 formal statistical tests** on MU (2015–2024):

| # | Hypothesis | Test | p-value | Effect Size | Significant? |
|---|-----------|------|---------|-------------|--------------|
| **H1** | Spike days preceded by negative sentiment | Two-sample t-test + Bootstrap CI (Cohen's d) | 0.3266 | d = −0.061 | No |
| **H2** | Leverage effect — negative days → higher future vol | Mann-Whitney U (rank-biserial r) | 0.0651 | r = −0.035 | No (marginal) |
| **H3** | Sentiment Granger-causes volatility | Granger F-test, lags 1–5 | min p = 0.1465 | — | No |
| **H4** | Monday Effect — weekday vol differences | Kruskal-Wallis + Dunn post-hoc (η²) | 0.9863 | η² = 0.000 | No |
| **H5** | Earnings week vol regime | Permutation test (n=5000) | — | — | Inconclusive (no data) |
| **H6** | VIX regime breaks ML model accuracy | Levene test for equal variances | 0.3719 | — | No |
| **H7** | 10-K risk language predicts future vol | Mann-Whitney U + Bootstrap CI | — | — | Inconclusive (no data) |
| **H8** | Sentiment mean reversion after extreme negative days | Wilcoxon signed-rank (paired) | **0.0000** | — | **Yes ✓** |

**H8** is the one statistically significant result: after extreme negative sentiment days (VADER < −0.5), sentiment recovers significantly by t+3 (mean: −0.723 → −0.061, Wilcoxon W=692.5, p≈0). This suggests market overreaction at the textual level before price recovery — a potential contrarian signal.

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **RMSE** | Root Mean Squared Error — standard accuracy measure |
| **MAE** | Mean Absolute Error — less sensitive to outliers |
| **QLIKE** | Quasi-Likelihood loss: mean(σ²/h − ln(σ²/h) − 1). Standard in vol forecasting; heavily penalises underestimating spikes (Patton 2011) |
| **Corr** | Pearson correlation between predicted and realized vol |
| **Spike Accuracy** | % of 90th-percentile spike days where the model also predicted above the threshold |

---

## Results — MU (Micron Technology)

**Period:** 2021-05-18 → 2026-05-17 | **Horizon:** 5 trading days | **Train/Test:** 80/20

| Model | RMSE | MAE | QLIKE | Corr | Spike Acc |
|-------|------|-----|-------|------|-----------|
| **EGARCH** | **0.2200** | 0.1755 | **0.2913** | **0.4364** | 0% |
| **HAR-RV** | 0.2246 | 0.1756 | 0.3073 | 0.4186 | 0% |
| XGBoost | 0.2574 | 0.1970 | 0.5067 | 0.1573 | 0% |
| XGB-Asymmetric | 0.2577 | 0.1988 | 0.5018 | 0.1009 | 0% |
| RandomForest | 0.2607 | **0.1867** | 0.5393 | 0.0864 | 0% |

**Key findings:**
- EGARCH leads on RMSE, QLIKE, and correlation for MU — statistical models dominate on a high-volatility cyclical stock
- HAR-RV nearly matches EGARCH despite being a simple linear model, confirming Corsi (2009)
- ML models underperform: MU's vol spikes are driven by DRAM supply shocks and macro events poorly captured in technical features alone
- **Spike Accuracy = 0%** across all models: MU's 90th-percentile threshold is **92.7% annualized vol** — structurally too extreme for any model to reliably flag
- **SHAP analysis** shows `vol_60d` dominates XGBoost predictions by 2×, confirming long-memory vol is the key signal; `vix_level` and `ret_skew_21d` also rank highly
- **Live signal (May 2026):** Ensemble forward vol = 55.7% → **EXTREME regime**, tight stops essential

---

## Academic References

The models and methods in this project are grounded in the following papers:

**Volatility Modelling**
- Bollerslev, T. (1986). *Generalized autoregressive conditional heteroskedasticity.* Journal of Econometrics, 31(3), 307–327.
- Nelson, D. B. (1991). *Conditional heteroskedasticity in asset returns: A new approach.* Econometrica, 59(2), 347–370. *(EGARCH)*
- Corsi, F. (2009). *A simple approximate long-memory model of realized volatility.* Journal of Financial Econometrics, 7(2), 174–196. *(HAR-RV)*
- Andersen, T. G., & Bollerslev, T. (1998). *Answering the skeptics: Yes, standard volatility models do provide accurate forecasts.* International Economic Review, 39(4), 885–905.

**Loss Functions & Evaluation**
- Patton, A. J. (2011). *Volatility forecast comparison using imperfect volatility proxies.* Journal of Econometrics, 160(1), 246–256. *(QLIKE loss)*
- Hansen, P. R., & Lunde, A. (2005). *A forecast comparison of volatility models: Does anything beat a GARCH(1,1)?* Journal of Applied Econometrics, 20(7), 873–889.

**Sentiment & Text Analysis**
- Tetlock, P. C. (2007). *Giving content to investor sentiment: The role of media in the stock market.* Journal of Finance, 62(3), 1139–1168.
- Loughran, T., & McDonald, B. (2011). *When is a liability not a liability? Textual analysis, dictionaries, and 10-Ks.* Journal of Finance, 66(1), 35–65. *(LM sentiment)*
- Hutto, C. J., & Gilbert, E. (2014). *VADER: A parsimonious rule-based model for sentiment analysis of social media text.* ICWSM. *(VADER)*

**Leverage Effect & Stylized Facts**
- Black, F. (1976). *Studies of stock price volatility changes.* Proceedings of the 1976 Meetings of the American Statistical Association.
- Cont, R. (2001). *Empirical properties of asset returns: Stylized facts and statistical issues.* Quantitative Finance, 1(2), 223–236.

**Machine Learning for Finance**
- Chen, T., & Guestrin, C. (2016). *XGBoost: A scalable tree boosting system.* KDD 2016.
- Lundberg, S. M., & Lee, S. I. (2017). *A unified approach to interpreting model predictions (SHAP).* NeurIPS 2017.

---

*Built with Python 3.14 · yfinance · arch · XGBoost · scikit-learn · SHAP · VADER · SciPy*
