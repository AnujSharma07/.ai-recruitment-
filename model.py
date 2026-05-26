"""
model.py
--------
Handles the complete modelling workflow:
  1. Strict chronological train/test split
  2. Training XGBoost and LightGBM regressors
  3. Evaluation (MAPE, MAE, RMSE)
  4. Visualisations:
       - Actual vs Predicted (2-week window)
       - Percentage Error Distribution + Scatter
       - Feature Importance (colour-coded by category)
       - MAPE by Hour of Day

No data normalisation is required for tree-based models — they are
invariant to monotonic feature scaling.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import lightgbm as lgb


# ── Constants ──────────────────────────────────────────────────────────────────
TEST_YEAR = 2023

XGBOOST_PARAMS = dict(
    n_estimators=800,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_lambda=1.0,
    n_jobs=-1,
    random_state=42,
    tree_method="hist",
)

LGBM_PARAMS = dict(
    n_estimators=1000,
    learning_rate=0.04,
    max_depth=8,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    reg_lambda=1.0,
    n_jobs=-1,
    random_state=42,
    verbose=-1,
)

# Colour scheme for feature-importance chart
_FEAT_COLOR = {
    "lag":      "#2196F3",   # blue   — lag / rolling
    "roll":     "#2196F3",
    "hour":     "#FF9800",   # orange — calendar
    "day":      "#FF9800",
    "month":    "#FF9800",
    "week":     "#FF9800",
    "sin":      "#FF9800",
    "cos":      "#FF9800",
    "weekend":  "#FF9800",
    "friday":   "#FF9800",
    "quarter":  "#FF9800",
    "temp":     "#4CAF50",   # green  — weather
    "humid":    "#4CAF50",
    "prec":     "#4CAF50",
    "cloud":    "#4CAF50",
    "sun":      "#4CAF50",
    "wind":     "#4CAF50",
    "dew":      "#4CAF50",
    "soil":     "#4CAF50",
    "apparent": "#4CAF50",
}


# ── Metric helpers ─────────────────────────────────────────────────────────────
def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error, ignoring zero-valued ground truth."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


# ── Split ──────────────────────────────────────────────────────────────────────
def chronological_split(
    df_feat: pd.DataFrame,
    features: list[str],
    test_year: int = TEST_YEAR,
) -> tuple:
    """
    Strict chronological split: train = [start, test_year), test = test_year.

    Returns
    -------
    X_train, y_train, X_test, y_test, train, test
    """
    train = df_feat[df_feat.index.year <  test_year].copy()
    test  = df_feat[df_feat.index.year == test_year].copy()

    X_train = train[features].ffill().fillna(0)
    y_train = train["target"]
    X_test  = test[features].ffill().fillna(0)
    y_test  = test["target"]

    print(f"[split]  Train: {len(X_train):,} rows  "
          f"({train.index.min().date()} → {train.index.max().date()})")
    print(f"         Test : {len(X_test):,}  rows  "
          f"({test.index.min().date()} → {test.index.max().date()})")
    return X_train, y_train, X_test, y_test, train, test


# ── Training & Evaluation ──────────────────────────────────────────────────────
def train_and_evaluate(
    X_train, y_train, X_test, y_test
) -> dict:
    """
    Train XGBoost and LightGBM, print metrics, return results dict.
    """
    print("\n── Model Training & Evaluation ─────────────────────────")
    results = {}

    for name, model in [
        ("XGBoost",          xgb.XGBRegressor(**XGBOOST_PARAMS)),
        ("LightGBM",         lgb.LGBMRegressor(**LGBM_PARAMS)),
    ]:
        model.fit(X_train, y_train)
        preds   = model.predict(X_test)
        m       = mape(y_test, preds)
        mae     = mean_absolute_error(y_test, preds)
        rmse    = mean_squared_error(y_test, preds) ** 0.5

        results[name] = {
            "model": model,
            "preds": preds,
            "mape":  m,
            "mae":   mae,
            "rmse":  rmse,
        }
        print(f"  {name:12s}  MAPE={m:.3f}%   MAE={mae:.1f} MW   RMSE={rmse:.1f} MW")

    print("── Done ────────────────────────────────────────────────\n")
    return results


def print_summary(results: dict) -> None:
    """Print a formatted results table and interpret the best MAPE."""
    rows = sorted(results.items(), key=lambda kv: kv[1]["mape"])
    print("=" * 55)
    print("  FINAL TEST RESULTS  (test year: 2023)")
    print("=" * 55)
    for name, r in rows:
        print(f"  {name:12s}  MAPE={r['mape']:.3f}%  "
              f"MAE={r['mae']:.1f} MW  RMSE={r['rmse']:.1f} MW")
    print("=" * 55)

    best_name, best = rows[0]
    print(f"\n★  Best model : {best_name}   MAPE = {best['mape']:.3f}%")
    if best["mape"] < 3:
        print("   Interpretation: Excellent — < 3% MAPE is publication-quality.")
    elif best["mape"] < 5:
        print("   Interpretation: Very good — operationally useful for dispatch.")
    elif best["mape"] < 10:
        print("   Interpretation: Good — acceptable for day-ahead scheduling.")
    else:
        print("   Interpretation: Consider more feature engineering.")


# ── Visualisations ─────────────────────────────────────────────────────────────
def _colour_feature(name: str) -> str:
    for key, col in _FEAT_COLOR.items():
        if key in name:
            return col
    return "#9C27B0"   # purple — economic / other


def plot_actual_vs_predicted(
    test: pd.DataFrame,
    best_preds: np.ndarray,
    n_days: int = 14,
    save_path: Path | None = None,
) -> None:
    """Line chart: actual vs predicted for the first n_days of the test set."""
    plot_df            = test.copy()
    plot_df["predicted"] = best_preds
    subset             = plot_df.iloc[: 24 * n_days]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(subset.index, subset["target"],    label="Actual",    color="steelblue",  lw=1.5)
    ax.plot(subset.index, subset["predicted"], label="Predicted", color="darkorange", lw=1.5, ls="--")
    ax.set_title(f"Actual vs Predicted — First {n_days} days of 2023 (LightGBM)",
                 fontweight="bold")
    ax.set_ylabel("Demand (MW)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_error_analysis(
    y_test: pd.Series,
    best_preds: np.ndarray,
    save_path: Path | None = None,
) -> None:
    """Percentage error histogram + actual-vs-predicted scatter."""
    errors  = y_test.values - best_preds
    pct_err = (errors / y_test.values) * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(pct_err, bins=60, color="steelblue", edgecolor="white")
    axes[0].axvline(0, color="red", lw=1.5, ls="--", label="Zero error")
    axes[0].set_xlabel("% Error")
    axes[0].set_title("Percentage Error Distribution", fontweight="bold")
    axes[0].legend()

    axes[1].scatter(y_test.values, best_preds, alpha=0.1, s=4, color="steelblue")
    lims = [y_test.min() * 0.9, y_test.max() * 1.05]
    axes[1].plot(lims, lims, "r--", lw=1.5, label="Perfect fit")
    axes[1].set_xlabel("Actual (MW)")
    axes[1].set_ylabel("Predicted (MW)")
    axes[1].set_title("Actual vs Predicted (scatter)", fontweight="bold")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_feature_importance(
    model: lgb.LGBMRegressor,
    features: list[str],
    top_n: int = 30,
    save_path: Path | None = None,
) -> None:
    """Horizontal bar chart of top_n feature importances, colour-coded by type."""
    fi = (
        pd.DataFrame({"feature": features, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )
    colours = [_colour_feature(f) for f in fi["feature"]]

    fig, ax = plt.subplots(figsize=(9, 10))
    ax.barh(fi["feature"], fi["importance"], color=colours)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance (Gain)", fontsize=12)
    ax.set_title(
        f"Top {top_n} Feature Importances — LightGBM\n"
        "(Blue=Lag/Rolling | Orange=Calendar | Green=Weather | Purple=Econ/Other)",
        fontweight="bold", fontsize=11,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

    print("\nTop 15 features:")
    print(fi.head(15).to_string(index=False))


def plot_mape_by_hour(
    test: pd.DataFrame,
    best_preds: np.ndarray,
    save_path: Path | None = None,
) -> None:
    """Bar chart: MAPE per hour of the day — highlights challenging peak hours."""
    plot_df              = test.copy()
    plot_df["predicted"] = best_preds
    plot_df["hour"]      = plot_df.index.hour

    hourly_mape = (
        plot_df.groupby("hour")
        .apply(lambda g: mape(g["target"], g["predicted"]))
        .reset_index()
    )
    hourly_mape.columns = ["hour", "mape"]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(hourly_mape["hour"], hourly_mape["mape"], color="steelblue", edgecolor="white")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("MAPE by Hour of Day  (LightGBM on 2023 test set)", fontweight="bold")
    ax.set_xticks(range(0, 24))
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


# ── Master run function ────────────────────────────────────────────────────────
def run_modelling(
    df_feat: pd.DataFrame,
    features: list[str],
    figures_dir: Path | None = None,
) -> dict:
    """
    End-to-end modelling: split → train → evaluate → visualise.

    Parameters
    ----------
    df_feat      : output of feature_engineering.build_features()
    features     : feature column names
    figures_dir  : if provided, all figures are saved here as PNG

    Returns
    -------
    results dict (same structure as train_and_evaluate output)
    """
    save = lambda name: (figures_dir / name) if figures_dir else None
    if figures_dir:
        figures_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_test, y_test, train, test = chronological_split(
        df_feat, features
    )
    results = train_and_evaluate(X_train, y_train, X_test, y_test)
    print_summary(results)

    best_model = results["LightGBM"]["model"]
    best_preds = results["LightGBM"]["preds"]

    plot_actual_vs_predicted(test, best_preds, save_path=save("fig_pred_vs_actual.png"))
    plot_error_analysis(y_test, best_preds,    save_path=save("fig_error_dist.png"))
    plot_feature_importance(best_model, features, save_path=save("fig_feature_importance.png"))
    plot_mape_by_hour(test, best_preds,        save_path=save("fig_mape_by_hour.png"))

    return results
