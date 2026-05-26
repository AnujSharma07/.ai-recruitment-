"""
feature_engineering.py
-----------------------
Transforms the cleaned master DataFrame into a supervised-learning
tabular structure by adding:

  A. Calendar features  — cyclical encoding of time components
  B. Lag features       — historical demand at specific past offsets
  C. Rolling statistics — short/medium/long-window aggregates
  D. Differential features — demand rate-of-change
  E. Target column      — demand_mw shifted by -1 (next hour)

Why these features?
~~~~~~~~~~~~~~~~~~~
Classical tree-based models have no built-in notion of time or
sequential order.  Every row is treated as an independent observation.
We therefore must *manually inject* temporal context so the model can
learn patterns such as:
  - "It's 6 PM on a weekday → peak demand"
  - "Demand was rising for the past 3 hours → likely still rising"
  - "Same time last week had demand X → expect roughly the same"
"""

import numpy as np
import pandas as pd


# ── Lag offsets (hours back from current time) ─────────────────────────────
LAGS = [
    1, 2, 3, 6, 12,     # within the same day
    24, 48,             # yesterday / two days ago
    24 * 7,             # same hour last week
    24 * 14,            # same hour two weeks ago
]

# ── Rolling windows (hours) ────────────────────────────────────────────────
WINDOWS = [3, 6, 12, 24, 48, 24 * 7]

# ── Cyclical encodings ─────────────────────────────────────────────────────
CYCLIC_PAIRS = [
    ("hour",        24),
    ("day_of_week",  7),
    ("month",       12),
    ("day_of_year", 365),
]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add raw and sine/cosine-encoded calendar columns.

    Cyclical encoding is important because simple integers like hour=0
    and hour=23 appear very far apart numerically but are only 1 hour
    apart in real time.  sin/cos projection onto the unit circle
    preserves that circularity.
    """
    df = df.copy()
    df["hour"]         = df.index.hour
    df["day_of_week"]  = df.index.dayofweek
    df["month"]        = df.index.month
    df["quarter"]      = df.index.quarter
    df["day_of_year"]  = df.index.dayofyear
    df["week_of_year"] = df.index.isocalendar().week.astype(int)
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
    df["is_friday"]    = (df["day_of_week"] == 4).astype(int)

    for col, period in CYCLIC_PAIRS:
        df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / period)
        df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / period)

    print(f"[calendar]  added {2 + len(CYCLIC_PAIRS) * 2 + 6} columns")
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add direct lag columns and demand-delta (rate-of-change) features.

    We shift demand_mw forward in time (positive shift = looking back),
    so each row knows what demand was 1h, 24h, 168h, etc. ago.
    This gives the model access to recent history without violating the
    supervised-learning constraint that no future values are used.
    """
    df = df.copy()
    for lag in LAGS:
        df[f"lag_{lag}h"] = df["demand_mw"].shift(lag)

    df["demand_diff_1h"]  = df["demand_mw"].diff(1)
    df["demand_diff_24h"] = df["demand_mw"].diff(24)

    print(f"[lags]      added {len(LAGS) + 2} lag/delta columns")
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling mean, std, max, and min over multiple windows.

    The `.shift(1)` before `.rolling(...)` ensures we never include the
    current hour in any window — only past values are used.  This is a
    critical anti-leakage step.
    """
    df = df.copy()
    n_added = 0
    for w in WINDOWS:
        roll = df["demand_mw"].shift(1).rolling(w, min_periods=w // 2)
        df[f"roll_mean_{w}h"] = roll.mean()
        df[f"roll_std_{w}h"]  = roll.std()
        df[f"roll_max_{w}h"]  = roll.max()
        df[f"roll_min_{w}h"]  = roll.min()
        n_added += 4

    # Demand ratio: current vs. 24-hour average (captures intra-day position)
    df["demand_ratio_24h"] = df["demand_mw"] / (df["roll_mean_24h"] + 1e-6)
    n_added += 1

    print(f"[rolling]   added {n_added} rolling-aggregate columns")
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Define the supervised learning target as next-hour demand.

    Shift demand_mw by -1 so each row's 'target' is the demand in the
    *following* hour.  This is the variable we want to predict.
    """
    df = df.copy()
    df["target"] = df["demand_mw"].shift(-1)
    return df


def build_features(df_master: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Full feature-engineering pipeline.

    Parameters
    ----------
    df_master : output of preprocessing.run_preprocessing()

    Returns
    -------
    df_feat : pd.DataFrame  complete feature matrix with target
    features : list[str]    ordered list of input feature names (X columns)
    """
    print("\n── Feature Engineering ─────────────────────────────────")
    df = add_calendar_features(df_master)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_target(df)

    # Drop rows where key lags or the target are NaN
    n_before = len(df)
    df = df.dropna(subset=["target", "lag_1h", "lag_24h"])
    print(f"[dropna]    removed {n_before - len(df):,} rows with key NaN  "
          f"→ {len(df):,} rows remaining")

    # Build feature list (exclude leakage columns and the target)
    drop_from_x = {"target", "demand_mw", "remarks"}
    features = [c for c in df.columns if c not in drop_from_x]
    print(f"[features]  total feature count = {len(features)}")
    print("── Done ────────────────────────────────────────────────\n")
    return df, features
