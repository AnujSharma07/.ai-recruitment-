# Predictive Paradox — Power Grid Demand Forecasting

> **IITG.ai Recruitment Task** | Forecasting hourly electricity demand on the Bangladesh national grid using classical machine learning.

---

## Problem Statement

Accurate electricity demand forecasting is critical for grid stability.  
**Goal:** Predict `demand_mw` at time `t+1` using only information available at time `t`.

**Constraint:** Classical ML only — no LSTMs, Transformers, ARIMA, or Prophet.

---

## Project Structure

```
PredictiveParadox/
│
├── predictive_paradox.ipynb   # Main notebook — full end-to-end walkthrough
│
├── src/
│   ├── __init__.py            # Package exports
│   ├── data_loader.py         # Raw data ingestion (PGCB, Weather, Economic)
│   ├── preprocessing.py       # Cleaning, outlier removal, merging
│   ├── feature_engineering.py # Temporal features, lags, rolling stats
│   └── model.py               # Training, evaluation, visualisations
│
├── data/                      # (not committed) place datasets here
│   ├── PGCB_date_power_demand.xlsx
│   ├── weather_data.xlsx
│   └── economic_full_1.csv
│
├── figures/                   # Auto-generated plots
└── README.md
```

---

## Data Sources

| File | Granularity | Key Columns |
|------|-------------|-------------|
| `PGCB_date_power_demand.xlsx` | Hourly | `demand_mw`, `generation_mw`, fuel-type breakdown |
| `weather_data.xlsx` | Hourly | Temperature, humidity, precipitation, cloud cover |
| `economic_full_1.csv` | Annual | GDP growth, population, electricity access (World Bank) |

---

## Methodology

### 1. Exploratory Data Analysis
- Visualised raw demand time series (identified severe telemetry spikes)
- Plotted average demand by hour-of-day and month (confirmed daily and seasonal cycles)
- Computed correlation heatmap between demand and weather variables

### 2. Data Preprocessing

| Problem | Solution | Rationale |
|---------|----------|-----------|
| Duplicate timestamps | Keep first occurrence | Grid telemetry can double-log intervals |
| Missing hours | Hourly reindex + ffill (≤2 h) | Short telemetry dropout; longer gaps stay NaN |
| Extreme demand spikes | Rolling IQR filter (window=72 h, k=4.0) | Conservative — flags only genuine telemetry errors, not real peak-load events |
| Missing weather data | Linear interpolation (≤6 h) | Weather changes smoothly; longer gaps are filled with nearest valid reading |
| Annual economic data | Broadcast by calendar year | Macro-indicators change only yearly; no sub-annual information available |

**Why k=4.0?** A smaller multiplier (e.g. k=1.5) would incorrectly flag genuine extreme-heat peak days as anomalies, removing real signal the model needs to learn from.

### 3. Feature Engineering

Since tree-based models treat every row independently, temporal context must be injected explicitly.

**A. Calendar Features (cyclical encoding)**  
Raw integer hour/month encodes `23` and `0` as far apart; sin/cos projection fixes this:
```
hour_sin = sin(2π · hour / 24)
hour_cos = cos(2π · hour / 24)
```

**B. Lag Features**  
Historical demand at 1h, 2h, 3h, 6h, 12h, 24h, 48h, 168h (1 week), 336h (2 weeks) back.  
The weekly lag (`lag_168h`) is particularly powerful — demand at the same time last week is a strong predictor.

**C. Rolling Statistics**  
Mean, std, max, min over windows of 3h, 6h, 12h, 24h, 48h, and 168h.  
All rolling features use `.shift(1)` before `.rolling()` to prevent current-hour leakage.

**D. Differential Features**  
`demand_diff_1h` and `demand_diff_24h` — rate-of-change captures ramp-up/ramp-down patterns.

**Total features: ~80**

### 4. Train/Test Split (Strict Chronological)

- **Training:** All data before 2023
- **Test:** Full calendar year 2023 (hold-out, never touched during training)

A random split is **incorrect** here — lag/rolling features computed from future rows would leak test-period information into training.

### 5. Models

| Model | Config | Role |
|-------|--------|------|
| **LightGBM** | 1000 trees, lr=0.04, 63 leaves | Primary model |
| **XGBoost** | 800 trees, lr=0.05, depth=7 | Benchmark |

No feature scaling is applied — tree-based models are invariant to monotonic transformations.

---

## Results

| Model | MAPE (%) | MAE (MW) | RMSE (MW) |
|-------|----------|----------|-----------|
| LightGBM | 2.325% | 243.6 MW | 391.1 MW |
| XGBoost  | 2.560% | 274.2 MW | 426.4 MW |

**MAPE interpretation:**
- < 3%: Publication-quality for grid forecasting
- < 5%: Operationally useful for dispatch planning
- < 10%: Acceptable for day-ahead scheduling

---

## Key Findings from Feature Importance

1. **Lag features dominate** — `lag_1h`, `lag_2h`, and `lag_24h` are consistently the top three features. The most powerful predictor of next-hour demand is current-hour demand (strong short-range autocorrelation).

2. **Weekly lag matters** — `lag_168h` (same hour last week) ranks in the top 10, capturing the strong weekly demand cycle.

3. **Hour-of-day is the top calendar feature** — `hour_sin`/`hour_cos` reflect the clear daily peak-demand cycle seen in the EDA (~18:00–22:00).

4. **Temperature** is the most important weather variable, consistent with AC-driven cooling demand during summer months.

5. **Economic features** contribute modestly — they shift the long-run demand baseline but are constant within a year, so their per-hour signal is low.

---

## Running the Project

```bash
# 1. Install dependencies
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn openpyxl

# 2. Place the three data files in ./data/

# 3. Open the notebook
jupyter notebook predictive_paradox.ipynb
```

---

## Limitations

- The model does not have a holiday calendar (Eid, national holidays cause sharp demand dips not captured by day-of-week features alone)
- Load-shedding events in the training data distort the measured demand signal
- Economic features are only available up to 2022 — 2023 test-year values are forward-filled

---

*IITG.ai Recruitment — Predictive Paradox | Submission by Anuj*
