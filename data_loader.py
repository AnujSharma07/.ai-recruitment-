"""
data_loader.py
--------------
Handles loading and initial parsing of all three raw data sources:
  1. PGCB_date_power_demand.xlsx  — hourly grid telemetry
  2. weather_data.xlsx            — hourly environmental readings
  3. economic_full_1.csv          — annual World Bank macro-indicators

Returns clean DataFrames ready for preprocessing.
"""

import pandas as pd
from pathlib import Path


# ── Default paths (override via arguments) ────────────────────────────────────
DEFAULT_DATA_DIR = Path("./data")

PGCB_FILE     = "PGCB_date_power_demand.xlsx"
WEATHER_FILE  = "weather_data.xlsx"
ECONOMIC_FILE = "economic_full_1.csv"

# World Bank indicators we actually care about
ECON_INDICATORS = [
    "GDP growth (annual %)",
    "Population, total",
    "Urban population",
    "Access to electricity (% of population)",
    "Inflation, consumer prices (annual %)",
    "Energy use (kg of oil equivalent) per $1,000 GDP (constant 2021 PPP)",
]

# Rename map for economic columns (long names → short feature names)
ECON_RENAME = {
    "Access to electricity (% of population)":                        "econ_elec_access",
    "Energy use (kg of oil equivalent) per $1,000 GDP (constant 2021 PPP)": "econ_energy_intensity",
    "GDP growth (annual %)":                                           "econ_gdp_growth",
    "Inflation, consumer prices (annual %)":                          "econ_inflation_cpi",
    "Population, total":                                              "econ_population",
    "Urban population":                                               "econ_urban_pop",
}

# Weather column rename (positional — raw file has no header row)
WEATHER_COLUMNS = [
    "temperature", "humidity", "apparent_temp",
    "precipitation", "dew_point", "soil_temp",
    "wind_dir", "cloud_cover", "sunshine_sec",
]


def load_pgcb(data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """
    Load hourly grid demand/generation data from PGCB Excel file.

    Returns
    -------
    pd.DataFrame
        Sorted by datetime index, all original columns preserved.
    """
    path = data_dir / PGCB_FILE
    df = pd.read_excel(path, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    print(f"[PGCB]     shape={df.shape}  "
          f"range={df['datetime'].min().date()} → {df['datetime'].max().date()}  "
          f"duplicates={df['datetime'].duplicated().sum()}")
    return df


def load_weather(data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """
    Load hourly weather data.
    The raw file has a 4-row preamble; row index 3 is the header.

    Returns
    -------
    pd.DataFrame  (datetime index, numeric columns)
    """
    path = data_dir / WEATHER_FILE
    raw  = pd.read_excel(path, header=None)

    # Row 3 holds the column names; data starts at row 4
    cols  = raw.iloc[3].tolist()
    df    = raw.iloc[4:].copy()
    df.columns = cols
    df    = df.reset_index(drop=True)

    # Parse and set datetime index
    df["datetime"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["datetime"]).set_index("datetime")
    df = df.drop(columns=["time"], errors="ignore")

    # Assign readable column names
    df.columns = WEATHER_COLUMNS[: len(df.columns)]
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df[~df.index.duplicated(keep="first")]

    print(f"[Weather]  shape={df.shape}  "
          f"range={df.index.min().date()} → {df.index.max().date()}  "
          f"missing={df.isna().sum().sum()} cells")
    return df


def load_economic(data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """
    Load and pivot annual World Bank macro-economic indicators.
    Rows = indicator names, columns = years.  We transpose so rows = years.

    Returns
    -------
    pd.DataFrame  (year integer index, one column per indicator)
    """
    path = data_dir / ECONOMIC_FILE
    raw  = pd.read_csv(path)

    year_cols = [
        c for c in raw.columns
        if str(c).isdigit() and int(str(c)) >= 2014
    ]
    df = (
        raw[raw["Indicator Name"].isin(ECON_INDICATORS)][
            ["Indicator Name"] + year_cols
        ]
        .set_index("Indicator Name")[year_cols]
        .T
    )
    df.index = df.index.astype(int)
    df.index.name = "year"
    df = df.sort_index().ffill()
    df = df.rename(columns=ECON_RENAME)

    print(f"[Economic] shape={df.shape}  years={df.index.min()}–{df.index.max()}")
    return df


def load_all(data_dir: Path = DEFAULT_DATA_DIR):
    """
    Convenience wrapper — returns (df_pgcb, df_weather, df_econ) in one call.
    """
    print("=" * 55)
    print("  Loading raw data sources")
    print("=" * 55)
    df_pgcb    = load_pgcb(data_dir)
    df_weather = load_weather(data_dir)
    df_econ    = load_economic(data_dir)
    print("=" * 55)
    return df_pgcb, df_weather, df_econ
