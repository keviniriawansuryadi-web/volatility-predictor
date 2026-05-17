# Volatility Predictor

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Models](https://img.shields.io/badge/models-EGARCH%20%7C%20HAR--RV%20%7C%20XGBoost%20%7C%20RF-green)

**Forecast stock volatility and identify high-risk trading regimes using a five-model ensemble.**

Given any ticker and date range, this tool:
- Downloads price data from Yahoo Finance
- Trains EGARCH, HAR-RV, XGBoost (×2), and Random Forest models
- Compares their accuracy on a held-out test set
- Outputs a live forward signal classifying the current volatility regime (Low / Moderate / Elevated / High / Extreme)

Built for equity traders who need a quantitative read on upcoming volatility before placing scalp or swing trades.

---

## Quickstart

```bash
pip install -r requirements.txt
```

**Option A — Interactive web app (recommended)**
```bash
streamlit run app.py
```
Opens a browser UI where you pick the ticker, date range, and model settings, then click Run.

**Option B — Command line**
```bash
# Analyze MU with default settings (5-year window, 5-day forecast horizon)
python main.py --ticker MU

# Custom range and horizon
python main.py --ticker NVDA --start 2020-01-01 --end 2025-05-01 --horizon 3

# Skip the local CSV cache and re-download
python main.py --ticker SPY --no-cache
```

---

## How It Works

```
Yahoo Finance ──► Feature Engineering ──► Model Training ──► Evaluation & Signal
     │                   │                      │
  OHLCV + VIX      28 features:           EGARCH (rolling)
  News headlines   - Realized vol         HAR-RV
                   - VIX level/change     XGBoost (standard)
                   - VADER sentiment      XGBoost (spike-weighted loss)
                   - RSI, Bollinger       Random Forest
                   - GARCH hybrid vol
```

The **GARCH hybrid feature** is the key design choice: instead of having XGBoost re-learn what GARCH already captures (volatility clustering, leverage effect), it's given the EGARCH fitted vol as an input feature. This lets ML models focus on residual signals — regime shifts, sentiment, macro events — that GARCH misses.

---

## Project Structure

```
volatility-predictor/
│
├── app.py                       # Streamlit web app
├── main.py                      # CLI entry point
├── requirements.txt
│
├── src/                         # Core pipeline (used by both app.py and main.py)
│   ├── data_loader.py           # Yahoo Finance download + CSV cache + VIX
│   ├── features.py              # Feature engineering (28 features)
│   ├── garch_model.py           # EGARCH/GARCH rolling + in-sample + forecast
│   ├── har_model.py             # HAR-RV linear model
│   ├── ml_model.py              # XGBoost (standard + asymmetric), Random Forest
│   ├── sentiment.py             # VADER sentiment from Yahoo Finance news
│   ├── evaluate.py              # Metrics, 3-panel forecast plot, SHAP chart
│   └── hypothesis.py            # Mann-Whitney U spike-sentiment test
│
├── modules/                     # Notebook helpers (used by the .ipynb files)
│   ├── data_helpers.py          # VIX loader, earnings dates, sentiment simulation
│   ├── eda_plots.py             # 10 interactive Plotly EDA functions
│   └── hypothesis_tests.py      # 8 formal statistical tests (H1–H8)
│
├── eda.ipynb                    # Exploratory Data Analysis (10 analyses)
├── hypotheses.ipynb             # Hypothesis testing (8 formal tests)
├── docs/images/                 # Charts for this README
└── papers/references.bib        # Annotated BibTeX references
```

---

## Models

| Model | Type | Notes |
|-------|------|-------|
| **EGARCH** | Statistical | Rolling EGARCH(1,1). Captures the leverage effect (negative returns → more vol than positive) and volatility clustering. |
| **HAR-RV** | Linear | Corsi (2009). Regresses forward vol on its 1-day, 5-day, and 22-day averages. Surprisingly competitive despite simplicity. |
| **XGBoost** | ML | Gradient boosting on 28 features including the EGARCH in-sample vol as a hybrid input. |
| **XGBoost Asymmetric** | ML | Same as above, but with a custom loss that applies a **3× gradient penalty** when the model underestimates on spike days (realized vol > 50% annualized). Designed to make the model conservative during high-risk periods. |
| **Random Forest** | ML | 200-tree ensemble; nonlinear baseline. |

---

## Features (28 total)

| Category | Features |
|----------|----------|
| Lagged returns | `ret_lag1/2/3/5/10` |
| Realized vol | `vol_5d`, `vol_10d`, `vol_20d`, `vol_21d`, `vol_60d`, `vol_63d` |
| Vol-of-vol | `vol_of_vol` (rolling std of 21d vol) |
| Jump indicator | `jump_flag` — 1 if \|return\| > 2.5σ |
| Volume | `volume_ratio` vs 10-day moving average |
| Statistical moments | `ret_skew_21d`, `ret_kurt_21d` |
| Technical | `rsi_14`, `bb_width` (Bollinger band width) |
| VIX | `vix_level`, `vix_change` |
| Sentiment (VADER) | `sentiment_3d`, `sentiment_lag1/2/3` |
| Vol lags | `vol_lag1/2/3` |
| GARCH hybrid | `garch_vol` — EGARCH in-sample conditional volatility |

---

## Evaluation Metrics

| Metric | What it measures |
|--------|-----------------|
| **RMSE** | Average forecast error (penalizes large misses more) |
| **MAE** | Average absolute error (treats all errors equally) |
| **QLIKE** | Quasi-likelihood loss — the standard for vol forecasting. Heavily penalizes underestimating spikes. (Patton 2011) |
| **Corr** | Pearson correlation between predicted and realized vol |
| **Spike Acc** | % of 90th-percentile vol spike days correctly flagged by the model |

---

## Results — MU (Micron Technology, 2021–2026)

**Settings:** 5-year window · 5-day horizon · 80/20 train-test split

| Model | RMSE | MAE | QLIKE | Corr | Spike Acc |
|-------|------|-----|-------|------|-----------|
| **EGARCH** | **0.2200** | 0.1755 | **0.2913** | **0.4364** | 0% |
| **HAR-RV** | 0.2246 | 0.1756 | 0.3073 | 0.4186 | 0% |
| XGBoost | 0.2574 | 0.1970 | 0.5067 | 0.1573 | 0% |
| XGB-Asymmetric | 0.2577 | 0.1988 | 0.5018 | 0.1009 | 0% |
| RandomForest | 0.2607 | **0.1867** | 0.5393 | 0.0864 | 0% |

**Key takeaways:**
- EGARCH wins on RMSE, QLIKE, and correlation — statistical models beat ML on a high-volatility cyclical stock where price action is driven by DRAM supply shocks and macro events, not technical patterns.
- HAR-RV almost matches EGARCH despite being a simple linear regression, confirming Corsi (2009).
- Spike Accuracy is 0% across all models because MU's 90th-percentile vol threshold is ~93% annualized — structurally too extreme for any model to reliably pre-flag.
- SHAP analysis shows `vol_60d` dominates XGBoost predictions by 2×; long-memory vol is the key signal.

### Forecast Chart

![MU Volatility Comparison](docs/images/MU_volatility_comparison.png)

*Top: all five forecasts vs realized vol. Red dashed line = 90th-percentile spike threshold.
Middle: absolute error over time. Bottom: predicted vs realized scatter (★ = spike days).*

### SHAP Feature Importance

![MU SHAP Feature Importance](docs/images/MU_shap_importance.png)

*`vol_60d` contributes more than twice the next feature. VIX level and 21-day return skewness round out the top 3.*

---

## Research Notebooks

### EDA (`eda.ipynb`) — 10 analyses on MU (2015–2024)

| # | Analysis | Key Finding |
|---|----------|-------------|
| 1 | Vol distribution + Q-Q plot | Fat-tailed, right-skewed → non-parametric tests required |
| 2 | Rolling sentiment–vol correlation | Strongly negative during high-VIX periods |
| 3 | ACF / PACF | Slow decay + ARCH effects → GARCH well-motivated |
| 4 | Sentiment model agreement | VADER and LM agree (r≈0.35); TextBlob diverges on financial language |
| 5 | 10-K filing event study | High-risk filings → slower post-filing vol decay |
| 6 | Weekday vol pattern | Monday and Friday elevated — weekend news accumulation |
| 7 | Feature–target Spearman ranking | Short-window vol lags dominate; sentiment meaningful but noisy |
| 8 | Sentiment–vol joint density | Mild U-shape → include both raw VADER and \|VADER\| |
| 9 | Vol regime transition matrix | High-vol regime very persistent (diagonal ≥ 0.7) |
| 10 | News volume vs realized vol | News volume leads vol by 1–2 days around earnings |

### Hypothesis Tests (`hypotheses.ipynb`) — 8 formal tests on MU (2015–2024)

| # | Hypothesis | Test | p-value | Significant? |
|---|-----------|------|---------|--------------|
| H1 | Spike days preceded by negative sentiment | t-test + Bootstrap (Cohen's d) | 0.3266 | No |
| H2 | Leverage effect — negative days → higher future vol | Mann-Whitney U | 0.0651 | No (marginal) |
| H3 | Sentiment Granger-causes volatility | Granger F-test (lags 1–5) | 0.1465 | No |
| H4 | Monday Effect — weekday vol differences | Kruskal-Wallis + Dunn (η²) | 0.9863 | No |
| H5 | Earnings week volatility | Permutation test (n=5000) | — | Inconclusive |
| H6 | VIX regime breaks ML accuracy | Levene test | 0.3719 | No |
| H7 | 10-K risk language predicts future vol | Mann-Whitney U | — | Inconclusive |
| **H8** | **Sentiment mean reversion after extreme negative days** | **Wilcoxon signed-rank** | **< 0.0001** | **Yes ✓** |

**H8** is the only statistically significant result: after extreme negative sentiment days (VADER < −0.5), sentiment recovers significantly by t+3 (mean: −0.723 → −0.061, W=692.5, p≈0). This suggests textual overreaction before price recovery — a potential contrarian signal.

---

## Academic References

**Volatility modelling**
- Bollerslev (1986) — GARCH. *Journal of Econometrics.*
- Nelson (1991) — EGARCH. *Econometrica.*
- Corsi (2009) — HAR-RV. *Journal of Financial Econometrics.*
- Andersen & Bollerslev (1998) — Realized vol forecasting. *International Economic Review.*

**Loss functions & evaluation**
- Patton (2011) — QLIKE loss. *Journal of Econometrics.*
- Hansen & Lunde (2005) — GARCH(1,1) benchmark study. *Journal of Applied Econometrics.*

**Sentiment & text analysis**
- Tetlock (2007) — Media sentiment & stock returns. *Journal of Finance.*
- Loughran & McDonald (2011) — LM financial sentiment dictionary. *Journal of Finance.*
- Hutto & Gilbert (2014) — VADER. *ICWSM.*

**Stylized facts**
- Black (1976) — Leverage effect. *ASA Proceedings.*
- Cont (2001) — Empirical properties of asset returns. *Quantitative Finance.*

**Machine learning**
- Chen & Guestrin (2016) — XGBoost. *KDD.*
- Lundberg & Lee (2017) — SHAP. *NeurIPS.*

Full annotated BibTeX: [`papers/references.bib`](papers/references.bib)

---

*Python 3.10+ · yfinance · arch · XGBoost · scikit-learn · SHAP · VADER · Streamlit · SciPy*
