"""
preprocessing.py
----------------
Transforms the three raw DataFrames (PGCB, Weather, Economic) into a
single, clean, hourly-reindexed master DataFrame ready for feature
engineering.

Pipeline steps
~~~~~~~~~~~~~~
1. De-duplicate and reindex PGCB to a strict 1-hour frequency.
2. Forward-fill short gaps (≤ 2 h) in demand_mw.
3. Remove demand outliers via a rolling IQR filter (window=72 h, k=4.0).
4. Left-join hourly weather; interpolate gaps (≤ 6 h).
5. Merge annual economic data by calendar year.
"""

import numpy as np
import pandas as pd


# ── Constants ──────────────────────────────────────────────────────────────────
GRID_COLS = [
    "demand_mw", "generation_mw", "load_shedding",
    "gas", "liquid_fuel", "coal", "hydro", "solar", "wind",
]

WEATHER_COLS = [
    "temperature", "humidity", "apparent_temp",
    "precipitation", "dew_point", "soil_temp",
    "wind_dir", "cloud_cover", "sunshine_sec",
]

IQR_WINDOW  = 72   # hours for rolling IQR baseline
IQR_K       = 4.0  # multiplier — only very extreme spikes are flagged
INTERP_LIMIT = 6   # max hours to interpolate weather gaps


# ── Step 1 & 2: PGCB cleaning ─────────────────────────────────────────────────
def clean_pgcb(df_pgcb: pd.DataFrame) -> pd.DataFrame:
    """
    De-duplicate, reindex to hourly, and forward-fill small gaps.

    Parameters
    ----------
    df_pgcb : raw DataFrame from data_loader.load_pgcb()

    Returns
    -------
    pd.DataFrame  (datetime index, continuous 1-hour frequency)
    """
    df = df_pgcb.copy()
    df = df.drop_duplicates(subset="datetime", keep="first")
    df = df.set_index("datetime")

    full_range = pd.date_range(
        start=df.index.min(), end=df.index.max(), freq="1h"
    )
    df = df.reindex(full_range)
    df.index.name = "datetime"

    # Fill only very short gaps (telemetry glitches) — longer ones stay NaN
    df["demand_mw"] = df["demand_mw"].ffill(limit=2)

    n_missing = df["demand_mw"].isna().sum()
    print(f"[clean_pgcb]   rows={len(df):,}  "
          f"still-missing demand={n_missing} ({n_missing/len(df)*100:.2f}%)")
    return df


# ── Step 3: Outlier removal ───────────────────────────────────────────────────
def rolling_iqr_clean(
    series: pd.Series,
    window: int = IQR_WINDOW,
    k: float = IQR_K,
) -> pd.Series:
    """
    Replace extreme spikes with the rolling median.

    A value is flagged as anomalous when it falls outside:
        [Q1 - k·IQR ,  Q3 + k·IQR]
    where Q1, Q3, and IQR are computed over a rolling window centred on
    the current observation.  k=4.0 is intentionally conservative so we
    only suppress genuine telemetry errors, not real demand peaks.

    Parameters
    ----------
    series : pd.Series   (demand_mw column)
    window : int         rolling window width in hours (default 72)
    k      : float       IQR multiplier (default 4.0)

    Returns
    -------
    pd.Series  cleaned series
    """
    s    = series.copy()
    roll = s.rolling(window, center=True, min_periods=window // 2)
    q1   = roll.quantile(0.25)
    q3   = roll.quantile(0.75)
    med  = roll.median()
    iqr  = q3 - q1
    mask = (s < q1 - k * iqr) | (s > q3 + k * iqr)
    print(f"[rolling_iqr]  anomalies={mask.sum()} "
          f"({mask.mean()*100:.3f}% of series)  window={window}h  k={k}")
    s[mask] = med[mask]
    return s


# ── Step 4 & 5: Merge all sources ─────────────────────────────────────────────
def build_master(
    df_pgcb_clean: pd.DataFrame,
    df_weather: pd.DataFrame,
    df_econ: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join cleaned PGCB, weather, and economic data into one hourly DataFrame.

    Economic data is annual; we broadcast each year's values to every
    hour in that calendar year.  This is the correct approach because
    macroeconomic indicators change only on a yearly cadence and are
    published retrospectively — there is no intra-year temporal leakage.

    Parameters
    ----------
    df_pgcb_clean : output of clean_pgcb() (already outlier-removed)
    df_weather    : output of data_loader.load_weather()
    df_econ       : output of data_loader.load_economic()

    Returns
    -------
    pd.DataFrame  master merged frame
    """
    # Keep only the columns we need from the grid data
    available_grid = [c for c in GRID_COLS if c in df_pgcb_clean.columns]
    df = df_pgcb_clean[available_grid].copy()

    # Left-join weather (some hours may not overlap — that's OK)
    df = df.join(df_weather, how="left")

    # Fill short weather gaps via linear interpolation
    df[WEATHER_COLS] = (
        df[WEATHER_COLS]
        .interpolate("linear", limit=INTERP_LIMIT, limit_direction="forward")
        .ffill()
        .bfill()
    )

    # Merge annual economic data by year
    df["year"] = df.index.year
    df = df.merge(df_econ, on="year", how="left")
    df.index = df_pgcb_clean.index  # restore datetime index after merge
    df = df.drop(columns=["year"])

    n_missing = df.isna().sum()
    print(f"[build_master] shape={df.shape}")
    print(f"  Remaining NaN counts (top 5):")
    print(f"  {n_missing[n_missing > 0].sort_values(ascending=False).head(5).to_dict()}")
    return df


# ── High-level convenience function ───────────────────────────────────────────
def run_preprocessing(df_pgcb, df_weather, df_econ) -> pd.DataFrame:
    """
    Full preprocessing pipeline in one call.

    Returns the cleaned, merged master DataFrame.
    """
    print("\n── Preprocessing ──────────────────────────────────────")
    df_clean = clean_pgcb(df_pgcb)
    df_clean["demand_mw"] = rolling_iqr_clean(df_clean["demand_mw"])
    df_master = build_master(df_clean, df_weather, df_econ)
    print("── Done ────────────────────────────────────────────────\n")
    return df_master
