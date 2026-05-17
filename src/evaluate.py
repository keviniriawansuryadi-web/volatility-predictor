import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _qlike(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """QLIKE loss: mean(sigma^2/h - ln(sigma^2/h) - 1). Penalises underestimation heavily."""
    h = np.maximum(y_pred, 1e-8) ** 2
    s2 = y_true ** 2
    return float(np.mean(s2 / h - np.log(s2 / h) - 1))


def _spike_accuracy(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> float:
    """% of spike days (y_true > threshold) where the model also predicted > threshold."""
    spike_mask = y_true > threshold
    if spike_mask.sum() == 0:
        return float("nan")
    return float((y_pred[spike_mask] > threshold).mean())


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, name: str, spike_thresh: float) -> dict:
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) == 0:
        return {"model": name, "RMSE": np.nan, "MAE": np.nan,
                "QLIKE": np.nan, "Corr": np.nan, "Spike_Acc": np.nan}
    return {
        "model": name,
        "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
        "MAE": float(mean_absolute_error(yt, yp)),
        "QLIKE": _qlike(yt, yp),
        "Corr": float(np.corrcoef(yt, yp)[0, 1]),
        "Spike_Acc": _spike_accuracy(yt, yp, spike_thresh),
    }


# ---------------------------------------------------------------------------
# Main comparison function
# ---------------------------------------------------------------------------

def compare_models(
    feat_df: pd.DataFrame,
    forecasts: dict,          # {label: pd.Series}
    train_size: float = 0.8,
    ticker: str = "TICKER",
    plot_dir: str = ".",
) -> pd.DataFrame:
    split = int(len(feat_df) * train_size)
    test_df = feat_df.iloc[split:]
    y_true = test_df["target"].values
    spike_thresh = np.nanpercentile(y_true, 90)

    def _align(series: pd.Series) -> np.ndarray:
        return series.reindex(test_df.index).values

    results = []
    aligned_preds = {}
    for name, series in forecasts.items():
        yp = _align(series)
        aligned_preds[name] = yp
        results.append(_metrics(y_true, yp, name, spike_thresh))

    metrics_df = pd.DataFrame(results).set_index("model")

    print("\n--- Model Comparison ---")
    fmt = metrics_df.copy()
    fmt["RMSE"] = fmt["RMSE"].map("{:.4f}".format)
    fmt["MAE"] = fmt["MAE"].map("{:.4f}".format)
    fmt["QLIKE"] = fmt["QLIKE"].map("{:.4f}".format)
    fmt["Corr"] = fmt["Corr"].map("{:.4f}".format)
    fmt["Spike_Acc"] = fmt["Spike_Acc"].map(lambda v: f"{v:.1%}" if not pd.isna(v) else "n/a")
    print(fmt.to_string())
    print(f"\n  [Spike threshold (90th pct): {spike_thresh:.1%} annualized vol]")

    _plot_forecasts(test_df.index, y_true, aligned_preds, spike_thresh, ticker, plot_dir)
    _save_metrics(metrics_df, ticker, plot_dir)

    return metrics_df


def _save_metrics(metrics_df: pd.DataFrame, ticker: str, plot_dir: str = ".") -> None:
    """Persist per-ticker metrics to outputs/results/{ticker}/metrics.csv."""
    out_dir = Path(plot_dir) / "outputs" / "results" / ticker
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "metrics.csv"
    metrics_df.to_csv(out)
    print(f"Metrics saved: {out}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_forecasts(index, y_true, aligned_preds, spike_thresh, ticker, plot_dir):
    out_dir = Path(plot_dir) / "outputs" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    colors = ["blue", "orange", "green", "red", "purple", "brown"]
    spike_mask = y_true > spike_thresh

    fig, axes = plt.subplots(3, 1, figsize=(14, 14))

    # --- Subplot 1: time series ---
    ax = axes[0]
    ax.plot(index, y_true, label="Realized Vol", color="black", linewidth=1.5)
    ax.axhline(spike_thresh, color="red", linestyle="--", linewidth=0.8, alpha=0.6, label="90th pct")
    for (name, yp), color in zip(aligned_preds.items(), colors):
        ax.plot(index, yp, label=name, color=color, alpha=0.7)
    ax.set_title(f"{ticker} — Volatility Forecast vs Realized (Test Set)")
    ax.set_ylabel("Annualized Volatility")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # --- Subplot 2: absolute error ---
    ax2 = axes[1]
    for (name, yp), color in zip(aligned_preds.items(), colors):
        err = np.abs(yp - y_true)
        ax2.plot(index, err, label=name, color=color, alpha=0.6)
    ax2.set_title("Absolute Error Over Time")
    ax2.set_ylabel("|Forecast - Realized|")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # --- Subplot 3: scatter predicted vs realized, colored by spike ---
    ax3 = axes[2]
    vmin = 0
    vmax = max(y_true.max(), max(yp.max() for yp in aligned_preds.values() if len(yp)))
    ax3.plot([vmin, vmax], [vmin, vmax], "k--", linewidth=1, label="45° line")

    for (name, yp), color in zip(aligned_preds.items(), colors):
        mask_valid = ~np.isnan(yp)
        # Non-spike
        ns = ~spike_mask & mask_valid
        ax3.scatter(y_true[ns], yp[ns], color=color, alpha=0.3, s=12, label=f"{name} (normal)")
        # Spike days
        sp = spike_mask & mask_valid
        ax3.scatter(y_true[sp], yp[sp], color=color, alpha=0.9, s=40, marker="*", label=f"{name} (spike)")

    ax3.set_xlabel("Realized Vol")
    ax3.set_ylabel("Predicted Vol")
    ax3.set_title("Predicted vs Realized (stars = spike days above 90th pct)")
    ax3.legend(fontsize=7, ncol=2)
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    out = out_dir / f"{ticker}_volatility_comparison.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Plot saved: {out}")


def plot_shap(model, X_test: np.ndarray, feature_names: list, ticker: str, plot_dir: str) -> None:
    try:
        import shap

        # Resolve to an xgb.Booster regardless of wrapper type
        if hasattr(model, "get_booster"):          # XGBRegressor
            underlying = model.get_booster()
        elif hasattr(model, "booster"):            # _BoosterWrapper
            underlying = model.booster
        else:
            underlying = model
        explainer = shap.TreeExplainer(underlying)
        shap_vals = explainer.shap_values(X_test)

        mean_abs = np.abs(shap_vals).mean(axis=0)
        n = min(len(mean_abs), len(feature_names))
        shap_df = (
            pd.DataFrame({"feature": feature_names[:n], "shap": mean_abs[:n]})
            .sort_values("shap", ascending=True)
            .tail(15)
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(shap_df["feature"], shap_df["shap"], color="steelblue")
        ax.set_title(f"{ticker} — XGBoost SHAP Feature Importance (mean |SHAP|)")
        ax.set_xlabel("Mean |SHAP value|")
        plt.tight_layout()

        out_dir = Path(plot_dir) / "outputs" / "plots"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{ticker}_shap_importance.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"SHAP plot saved: {out}")

    except ImportError:
        print("  [SHAP] shap not installed — skipping.")
    except Exception as e:
        print(f"  [SHAP] Error: {e}")
