"""
Volatility Predictor — Streamlit App
Live equity volatility forecasting with EGARCH, HAR-RV, XGBoost, and Random Forest.
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta

from src.data_loader import load_stock_data, load_vix_data
from src.sentiment import fetch_sentiment
from src.features import build_features, latest_feature_row, FEATURE_COLS
from src.garch_model import rolling_garch_forecast, garch_latest_forecast, garch_in_sample_vol
from src.har_model import har_rv_forecast
from src.ml_model import train_and_predict, predict_latest, feature_importance
from src.hypothesis import spike_sentiment_test


# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Volatility Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── helpers ───────────────────────────────────────────────────────────────────

def _qlike(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    h = np.maximum(y_pred, 1e-8) ** 2
    s2 = y_true ** 2
    return float(np.nanmean(s2 / h - np.log(s2 / h) - 1))


def _regime(v: float):
    if v > 0.35: return "EXTREME",  "#c0392b"
    if v > 0.25: return "HIGH",     "#e67e22"
    if v > 0.15: return "ELEVATED", "#f1c40f"
    if v > 0.10: return "MODERATE", "#2980b9"
    return              "LOW",      "#7f8c8d"


def build_metrics(y_true, aligned_preds, spike_thresh):
    rows = []
    for name, yp in aligned_preds.items():
        mask = ~(np.isnan(y_true) | np.isnan(yp))
        yt, ypm = y_true[mask], yp[mask]
        if len(yt) == 0:
            continue
        sm = yt > spike_thresh
        sa = float((ypm[sm] > spike_thresh).mean()) if sm.sum() > 0 else float("nan")
        rows.append({
            "Model": name,
            "RMSE":  round(float(np.sqrt(np.mean((yt - ypm) ** 2))), 4),
            "MAE":   round(float(np.mean(np.abs(yt - ypm))), 4),
            "QLIKE": round(_qlike(yt, ypm), 4),
            "Corr":  round(float(np.corrcoef(yt, ypm)[0, 1]), 4),
            "Spike Acc": f"{sa:.1%}" if not np.isnan(sa) else "n/a",
        })
    return pd.DataFrame(rows).set_index("Model")


def make_forecast_fig(index, y_true, aligned_preds, spike_thresh, ticker):
    colors = ["steelblue", "darkorange", "forestgreen", "crimson", "mediumpurple"]
    spike = y_true > spike_thresh
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), facecolor="#0e1117")
    for ax in axes:
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    # 1 — time series
    ax = axes[0]
    ax.plot(index, y_true, color="white", lw=1.5, label="Realized Vol")
    ax.axhline(spike_thresh, color="red", ls="--", lw=0.9, alpha=0.7, label="90th pct")
    for (name, yp), c in zip(aligned_preds.items(), colors):
        ax.plot(index, yp, label=name, color=c, alpha=0.8, lw=1.2)
    ax.set_title(f"{ticker}  —  Forecast vs Realized (Test Set)", fontsize=11)
    ax.set_ylabel("Annualized Vol", fontsize=9)
    ax.legend(fontsize=8, facecolor="#0e1117", labelcolor="white")
    ax.grid(alpha=0.15)

    # 2 — absolute error
    ax2 = axes[1]
    for (name, yp), c in zip(aligned_preds.items(), colors):
        ax2.plot(index, np.abs(yp - y_true), label=name, color=c, alpha=0.7, lw=1)
    ax2.set_title("Absolute Error Over Time", fontsize=11)
    ax2.set_ylabel("|Forecast − Realized|", fontsize=9)
    ax2.legend(fontsize=8, facecolor="#0e1117", labelcolor="white")
    ax2.grid(alpha=0.15)

    # 3 — scatter
    ax3 = axes[2]
    vmax = max(float(y_true.max()), max(float(np.nanmax(yp)) for yp in aligned_preds.values()))
    ax3.plot([0, vmax], [0, vmax], color="white", ls="--", lw=1, alpha=0.6, label="45° line")
    for (name, yp), c in zip(aligned_preds.items(), colors):
        mv = ~np.isnan(yp)
        ax3.scatter(y_true[~spike & mv], yp[~spike & mv], color=c, alpha=0.2, s=10)
        ax3.scatter(y_true[spike & mv],  yp[spike & mv],  color=c, alpha=0.9, s=40, marker="*")
    ax3.set_xlabel("Realized Vol", fontsize=9)
    ax3.set_ylabel("Predicted Vol", fontsize=9)
    ax3.set_title("Predicted vs Realized  (★ = spike days > 90th pct)", fontsize=11)
    ax3.grid(alpha=0.15)

    plt.tight_layout(pad=2)
    return fig


def make_shap_fig(model, X_test, feature_names, ticker):
    try:
        import shap
        underlying = model.get_booster() if hasattr(model, "get_booster") else \
                     (model.booster if hasattr(model, "booster") else model)
        explainer = shap.TreeExplainer(underlying)
        sv = explainer.shap_values(X_test)
        n = min(len(sv[0]), len(feature_names))
        mean_abs = np.abs(sv).mean(axis=0)[:n]
        df_s = (
            pd.DataFrame({"feature": feature_names[:n], "shap": mean_abs})
            .sort_values("shap", ascending=True).tail(15)
        )
        fig, ax = plt.subplots(figsize=(8, 5), facecolor="#0e1117")
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        ax.barh(df_s["feature"], df_s["shap"], color="steelblue")
        ax.set_title(f"{ticker}  —  XGBoost SHAP Feature Importance", fontsize=11, color="white")
        ax.set_xlabel("Mean |SHAP value|", fontsize=9)
        plt.tight_layout()
        return fig
    except Exception as e:
        return None


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 Volatility Predictor")
    st.caption("EGARCH · HAR-RV · XGBoost · RF")
    st.divider()

    ticker = st.text_input("Ticker Symbol", value="MU", max_chars=10).upper().strip()

    c1, c2 = st.columns(2)
    start_d = c1.date_input("Start", value=date.today() - timedelta(days=5 * 365))
    end_d   = c2.date_input("End",   value=date.today())

    horizon    = st.slider("Forecast Horizon (days)", 1, 21, 5)
    train_pct  = st.slider("Training Data (%)", 50, 95, 80, step=5)
    train_size = train_pct / 100
    garch_type = st.selectbox("GARCH Variant", ["EGARCH", "GARCH"])

    st.divider()
    run_btn = st.button("▶  Run Analysis", type="primary", use_container_width=True)
    st.caption("First run ~60s (GARCH). Cached instantly on re-run.")

    st.divider()
    st.markdown("""
**Regimes**
🔴 EXTREME > 35%
🟠 HIGH > 25%
🟡 ELEVATED > 15%
🔵 MODERATE > 10%
⚫ LOW ≤ 10%
    """)


# ── header ────────────────────────────────────────────────────────────────────
st.markdown("# 📈 Volatility Predictor")
st.markdown("Live equity vol forecasting — EGARCH · HAR-RV · XGBoost · Random Forest · SHAP")

if not run_btn and "results" not in st.session_state:
    st.info("👈  Configure parameters in the sidebar and click **Run Analysis**.")
    st.markdown("""
| Feature | Description |
|---------|-------------|
| **5 models** | EGARCH, HAR-RV, XGBoost (standard + asymmetric spike loss), Random Forest |
| **28 features** | Realized vol at 6 lookbacks, VIX, VADER sentiment, jump flags, GARCH hybrid |
| **QLIKE loss** | Standard vol-forecasting metric that penalises spike underestimation |
| **SHAP explainability** | Feature attribution for the XGBoost model |
| **Live signal** | Ensemble forward vol + regime classification for scalping decisions |
| **Hypothesis test** | Mann-Whitney U on sentiment before vol spikes |
    """)
    st.stop()


# ── run pipeline ──────────────────────────────────────────────────────────────
if run_btn:
    start_str = start_d.isoformat()
    end_str   = end_d.isoformat()

    prog = st.progress(0)
    msg  = st.empty()

    try:
        msg.text("⬇  Downloading price data...")
        prog.progress(5)
        df = load_stock_data(ticker, start_str, end_str)

        msg.text("⬇  Loading VIX...")
        prog.progress(12)
        vix_df = load_vix_data(start_str, end_str)
        if not vix_df.empty:
            df = df.join(vix_df, how="left")
            df[["vix_level", "vix_change"]] = df[["vix_level", "vix_change"]].ffill()

        msg.text("📰  Fetching VADER sentiment from news...")
        prog.progress(17)
        df["sentiment"] = fetch_sentiment(ticker, df.index)

        msg.text(f"⚙  Fitting {garch_type} in-sample (hybrid feature)...")
        prog.progress(22)
        df["garch_vol"] = garch_in_sample_vol(df["log_return"], model_type=garch_type)

        msg.text("🔧  Building feature matrix...")
        prog.progress(26)
        feat_df = build_features(df, forecast_horizon=horizon)

        msg.text(f"🔄  Running {garch_type} rolling forecast — slowest step, ~60s...")
        prog.progress(30)
        garch_preds = rolling_garch_forecast(
            df["log_return"], train_size=train_size,
            forecast_horizon=horizon, model_type=garch_type,
        )

        msg.text("📐  Fitting HAR-RV...")
        prog.progress(60)
        har_preds = har_rv_forecast(
            df["realized_vol_21d"], train_size=train_size, forecast_horizon=horizon)

        msg.text("🤖  Training XGBoost...")
        prog.progress(68)
        xgb_preds, xgb_model, xgb_feats = train_and_predict(
            feat_df, model_type="xgboost", train_size=train_size)

        msg.text("🤖  Training XGBoost (asymmetric spike loss)...")
        prog.progress(76)
        xgb_asym_preds, xgb_asym_model, _ = train_and_predict(
            feat_df, model_type="xgboost_asymmetric", train_size=train_size)

        msg.text("🌲  Training Random Forest...")
        prog.progress(83)
        rf_preds, rf_model, _ = train_and_predict(
            feat_df, model_type="random_forest", train_size=train_size)

        msg.text("📊  Computing metrics & live signal...")
        prog.progress(90)
        split    = int(len(feat_df) * train_size)
        test_df  = feat_df.iloc[split:]
        y_true   = test_df["target"].values
        spike_th = float(np.nanpercentile(y_true, 90))

        forecasts = {
            garch_type:      garch_preds,
            "HAR-RV":        har_preds,
            "XGBoost":       xgb_preds,
            "XGB-Asymmetric": xgb_asym_preds,
            "RandomForest":  rf_preds,
        }
        aligned = {n: s.reindex(test_df.index).values for n, s in forecasts.items()}
        metrics_df = build_metrics(y_true, aligned, spike_th)

        # live signal
        latest_row  = latest_feature_row(df)
        xgb_now     = predict_latest(xgb_model, latest_row)
        xgb_asym_now = predict_latest(xgb_asym_model, latest_row)
        rf_now      = predict_latest(rf_model, latest_row)
        garch_now   = garch_latest_forecast(df["log_return"], horizon, garch_type)
        ensemble    = float(np.nanmean([xgb_now, xgb_asym_now, rf_now, garch_now]))

        hyp = spike_sentiment_test(feat_df)

        prog.progress(100)
        msg.empty()
        prog.empty()

        st.session_state["results"] = dict(
            ticker=ticker, df=df, feat_df=feat_df,
            test_index=test_df.index, y_true=y_true,
            aligned=aligned, spike_th=spike_th,
            metrics_df=metrics_df,
            xgb_model=xgb_model, xgb_feats=xgb_feats,
            X_test=feat_df[xgb_feats].values[split:],
            fi_df=feature_importance(xgb_model, xgb_feats),
            live=dict(
                date=latest_row.index[-1].strftime("%Y-%m-%d"),
                price=float(df["close"].iloc[-1]),
                rv=float(df["realized_vol_21d"].iloc[-1]),
                xgb=xgb_now, xgb_asym=xgb_asym_now,
                rf=rf_now, garch=garch_now, ensemble=ensemble,
            ),
            hyp=hyp, garch_type=garch_type, horizon=horizon,
        )
        st.success(f"Pipeline complete — {len(df)} trading days loaded.")

    except Exception as e:
        prog.empty(); msg.empty()
        st.error(f"Pipeline error: {e}")
        st.exception(e)
        st.stop()


# ── display ───────────────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.stop()

R   = st.session_state["results"]
L   = R["live"]
gt  = R["garch_type"]
tkr = R["ticker"]

regime_label, regime_color = _regime(L["ensemble"])

st.markdown(f"## {tkr}  ·  {L['date']}")

# ── live signal cards ─────────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Price",          f"${L['price']:.2f}")
c2.metric("21d Realized Vol", f"{L['rv']:.1%}")
c3.metric("XGBoost",        f"{L['xgb']:.1%}")
c4.metric("XGB-Asymmetric", f"{L['xgb_asym']:.1%}")
c5.metric(gt,               f"{L['garch']:.1%}")
c6.metric("Ensemble",       f"{L['ensemble']:.1%}")

st.markdown(
    f"<div style='background:{regime_color};color:white;padding:10px 22px;"
    f"border-radius:8px;font-size:1.15em;font-weight:700;display:inline-block;"
    f"margin:6px 0 14px 0;letter-spacing:1px;'>"
    f"REGIME: {regime_label}"
    f"</div>",
    unsafe_allow_html=True,
)
st.divider()

# ── tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Model Comparison",
    "🔍 Feature Analysis",
    "🧪 Hypothesis Tests",
    "📚 About",
])

# ── tab 1: model comparison ───────────────────────────────────────────────────
with tab1:
    st.subheader("Performance Metrics")
    st.caption(
        f"Test set: {len(R['y_true'])} trading days  |  "
        f"Spike threshold (90th pct): {R['spike_th']:.1%} annualized vol  |  "
        f"Horizon: {R['horizon']} days"
    )

    # Color best in each column green
    mdf = R["metrics_df"].copy()
    st.dataframe(
        mdf.style
           .highlight_min(subset=["RMSE", "MAE", "QLIKE"], color="#1a4a1a")
           .highlight_max(subset=["Corr"], color="#1a4a1a")
           .format({"RMSE": "{:.4f}", "MAE": "{:.4f}", "QLIKE": "{:.4f}", "Corr": "{:.4f}"}),
        use_container_width=True,
    )
    st.caption("Green = best model for that metric. QLIKE heavily penalises underestimating spikes.")

    st.subheader("Forecast Charts")
    with st.spinner("Rendering charts..."):
        fig = make_forecast_fig(
            R["test_index"], R["y_true"], R["aligned"], R["spike_th"], tkr)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

# ── tab 2: feature analysis ───────────────────────────────────────────────────
with tab2:
    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.subheader("SHAP Feature Importance")
        with st.spinner("Computing SHAP values..."):
            shap_fig = make_shap_fig(R["xgb_model"], R["X_test"], R["xgb_feats"], tkr)
        if shap_fig:
            st.pyplot(shap_fig, use_container_width=True)
            plt.close(shap_fig)
        else:
            st.warning("SHAP plot unavailable.")

    with col_b:
        st.subheader("Feature Importance Table")
        st.dataframe(
            R["fi_df"].head(20)
              .style.bar(subset=["importance"], color="#2980b9"),
            use_container_width=True,
        )
        st.caption("Gain-based importance from XGBoost. SHAP chart (left) uses model-level attribution.")

# ── tab 3: hypothesis tests ───────────────────────────────────────────────────
with tab3:
    st.subheader("H1 — Live Test: Spike Days & Negative Sentiment")
    st.markdown(
        "**Test:** Mann-Whitney U (one-sided) + Cohen's d  |  "
        "**H₁:** Spike days (vol > 90th pct) are preceded by more negative VADER sentiment"
    )

    hyp = R["hyp"]
    if not hyp.get("available"):
        st.warning(f"**Inconclusive:** {hyp.get('reason')}")
        st.info(
            "yfinance only returns ~30 days of recent news headlines. "
            "To run the full historical test, plug in a paid news API "
            "(e.g. Tiingo, RavenPack, or Bloomberg)."
        )
    else:
        hc1, hc2, hc3 = st.columns(3)
        sig = hyp["significant"]
        hc1.metric("p-value", f"{hyp['p_value']:.4f}",
                   delta="Significant ✓" if sig else "Not significant",
                   delta_color="normal" if sig else "off")
        hc2.metric("Cohen's d", f"{hyp['cohens_d']:.3f}")
        hc3.metric("Spike days (n)", hyp["n_spike"])

        if sig:
            st.success("**REJECT H₀** — Spike days are preceded by significantly more negative sentiment.")
        else:
            st.info("**FAIL TO REJECT H₀** — No statistically significant pre-spike sentiment difference.")

        st.dataframe(pd.DataFrame([
            {"Group": f"Spike days (vol > {hyp['spike_threshold']:.0%})",
             "N": hyp["n_spike"], "Mean VADER": f"{hyp['spike_mean_sentiment']:.4f}"},
            {"Group": "Non-spike days",
             "N": hyp["n_non_spike"], "Mean VADER": f"{hyp['non_spike_mean_sentiment']:.4f}"},
        ]), hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("Full Hypothesis Test Results (MU, 2015–2024)")
    st.caption("From `hypotheses.ipynb` — 8 tests across sentiment, leverage effect, calendar anomalies, and more.")

    st.markdown("""
| # | Hypothesis | Test | p-value | Effect | Sig? |
|---|-----------|------|---------|--------|------|
| H1 | Spike days preceded by negative sentiment | t-test + Bootstrap (Cohen's d) | 0.3266 | d = −0.061 | No |
| H2 | Leverage effect — negative days → higher future vol | Mann-Whitney U | 0.0651 | r = −0.035 | Marginal |
| H3 | Sentiment Granger-causes volatility | Granger F-test (lags 1–5) | min p = 0.147 | — | No |
| H4 | Monday Effect — weekday vol differences | Kruskal-Wallis + Dunn (η²) | 0.9863 | η² = 0.000 | No |
| H5 | Earnings week volatility regime | Permutation test (n=5000) | — | — | Inconclusive |
| H6 | VIX regime breaks ML model accuracy | Levene equal-variance test | 0.3719 | — | No |
| H7 | 10-K risk language predicts future vol | Mann-Whitney U + Bootstrap | — | — | Inconclusive |
| **H8** | **Sentiment mean reversion after extreme negative days** | **Wilcoxon signed-rank** | **< 0.0001** | — | **Yes ✓** |
    """)

    st.success(
        "**H8 is the only significant result.** After extreme negative sentiment days "
        "(VADER < −0.5), sentiment recovers significantly by t+3 "
        "(mean: −0.723 → −0.061, Wilcoxon W=692.5, p≈0). "
        "This suggests market overreaction at the textual level before price recovery."
    )

# ── tab 4: about ──────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Models")
    st.markdown("""
| Model | Type | Notes |
|-------|------|-------|
| **EGARCH** | Statistical | Captures leverage effect & vol clustering. Best on high-vol cyclical stocks (MU). |
| **HAR-RV** | Linear | Corsi (2009). Daily + weekly + monthly vol averages. Nearly matches EGARCH. |
| **XGBoost** | ML | Trained on 28 features including GARCH hybrid vol, VIX, VADER sentiment. |
| **XGB-Asymmetric** | ML | Custom loss: 3× gradient penalty when underestimating vol > 50% annualized. |
| **Random Forest** | ML | 200 trees, ensemble baseline. |
    """)

    st.subheader("Key Academic References")
    st.markdown("""
- **Bollerslev (1986)** — GARCH. *Journal of Econometrics.*
- **Nelson (1991)** — EGARCH. *Econometrica.*
- **Corsi (2009)** — HAR-RV. *Journal of Financial Econometrics.*
- **Patton (2011)** — QLIKE loss. *Journal of Econometrics.*
- **Tetlock (2007)** — Media sentiment & stock returns. *Journal of Finance.*
- **Hutto & Gilbert (2014)** — VADER sentiment. *ICWSM.*
- **Chen & Guestrin (2016)** — XGBoost. *KDD.*
- **Lundberg & Lee (2017)** — SHAP. *NeurIPS.*

Full annotated BibTeX: [`papers/references.bib`](https://github.com/keviniriawansuryadi-web/volatility-predictor/blob/master/papers/references.bib)
    """)

    st.subheader("EDA Summary (10 Analyses, MU 2015–2024)")
    st.markdown("""
| EDA | Key Finding |
|-----|-------------|
| **EDA 1** Volatility distribution | Fat-tailed, right-skewed — non-parametric tests required |
| **EDA 2** Rolling sentiment–vol correlation | Spikes strongly negative in High VIX periods |
| **EDA 3** ACF / PACF | Slow ACF decay + ARCH effects — GARCH well-motivated |
| **EDA 4** Sentiment model agreement | VADER–LM agree (r≈0.35); TextBlob diverges on financial language |
| **EDA 5** 10-K event study | High-LM-Risk filings → slower post-filing vol decay |
| **EDA 6** Weekday vol pattern | Monday & Friday elevated — weekend news accumulation |
| **EDA 7** Feature–target Spearman ranking | Short-window vol lags dominate; sentiment unstable |
| **EDA 8** Sentiment–vol joint density | Mild U-shape — include both raw VADER and abs(VADER) |
| **EDA 9** Regime transition matrix | High-vol regime persistent (diagonal ≥ 0.7) |
| **EDA 10** News volume vs realized vol | News volume leads vol by 1–2 days around earnings |
    """)
