# src/

Core library modules for the volatility predictor.

| Module | Purpose |
|--------|---------|
| `data_loader.py` | Download/cache price and VIX data (yfinance + local CSV cache) |
| `features.py` | Feature engineering (returns, realized vols, jumps, RSI, sentiment, VIX) |
| `garch_model.py` / `har_model.py` | GARCH/EGARCH and HAR-RV volatility forecasts |
| `ml_model.py` | XGBoost (standard + asymmetric), Random Forest, quantile & stacking models |
| `evaluate.py` | RMSE/MAE/QLIKE metrics, comparison plots, SHAP |
| `sentiment.py` / `scraper_news.py` | News scraping (8 free sources) and VADER sentiment |
| `regime.py` / `regime_perf.py` | Volatility regime labelling and persistence analysis |
| `hypothesis.py` / `hypothesis_tests.py` | Statistical hypothesis tests (H1–H9) |
| `walk_forward.py` / `validation.py` | Walk-forward CV and model validation |
| `diagnostics.py` / `eda_plots.py` | Diagnostic reports and exploratory plots |
| `portfolio.py` / `spillover.py` / `sentiment_decomp.py` / `disagree_backtest.py` | Specialised analyses |
