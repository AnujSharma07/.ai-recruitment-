# Predictive Paradox: Power Grid Demand Forecasting

This document provides a comprehensive technical breakdown of the `predictive_paradox.py` script, designed for high-precision hourly electricity demand forecasting using the PGCB dataset.

## 1. Project Overview
The objective of this project is to build a robust machine learning pipeline capable of predicting next-hour power demand. It integrates multi-domain data to capture the complex relationship between human activity (economics), environmental conditions (weather), and historical consumption patterns.

## 2. Integrated Data Architecture
The pipeline merges three distinct data sources into a unified hourly time-series:

| Source | Category | Key Variables |
| :--- | :--- | :--- |
| **Grid Data** | Target & Operations | `demand_mw`, `generation_mw`, `load_shedding`, fuel types (Gas, Coal, Solar, etc.) |
| **Weather Data** | Environmental | Temperature, Humidity, Apparent Temp, Precipitation, Cloud Cover, Wind Direction |
| **Economic Data** | Macro Trends | GDP Growth, Population, Urbanization, Electricity Access, Inflation (CPI) |

## 3. Data Cleaning & Preprocessing
To handle real-world data noise and telemetry errors, the script implements:
* **Hourly Reindexing:** Ensures a continuous timeline, filling small gaps (<2 hours) with forward-filling.
* **Weather Interpolation:** Uses linear interpolation (limit of 6 hours) to maintain environmental continuity.
* **Rolling IQR Outlier Filter:** * Calculates rolling statistics with a **72-hour window**.
    * Flags values outside $Q1 - 4k$ or $Q3 + 4k$ (where $k=4.0$).
    * Replaces anomalies with the rolling median to prevent model distortion from spikes.

## 4. Feature Engineering
The model's performance is driven by a high-dimensional feature set:

### A. Calendar & Temporal Features
* **Cyclical Encoding:** Hour, Day, Month, and Day of Year are transformed using Sine/Cosine functions to preserve circularity (e.g., 23:00 is close to 00:00).
* **Binary Indicators:** Identification of weekends and Fridays (specific to local grid significance).

### B. Lagged & Rolling Variables
* **Direct Lags:** Historical demand from 1, 2, 3, 6, 12, 24, 48, 168 (1 week), and 336 (2 weeks) hours prior.
* **Rolling Statistics:** Mean, Std, Max, and Min across windows of 3, 6, 12, 24, 48, and 168 hours.
* **Differential Features:** 1-hour and 24-hour demand changes (deltas).

## 5. Modeling Methodology
The script follows a strict **chronological split** to avoid data leakage:
* **Training Data:** All observations prior to 2023.
* **Testing Data:** The full calendar year of 2023.

### Algorithms
1.  **LightGBM (Primary):** Optimized for speed and large feature spaces. 
    * *Params:* 1000 estimators, 63 leaves, 0.04 learning rate.
2.  **XGBoost:** Used for performance benchmarking and validation.
    * *Params:* 800 estimators, depth 7, 0.05 learning rate.

## 6. Evaluation Metrics
The models are evaluated on the 2023 test set using:
* **MAPE (Mean Absolute Percentage Error):** The primary indicator of accuracy.
* **MAE (Mean Absolute Error):** Average error in Megawatts.
* **RMSE (Root Mean Squared Error):** Penalty for large prediction errors.

## 7. Results & Interpretation
The script generates several visualizations to interpret the findings:
* **Feature Importance:** Highlights which variables (usually Lags and Hour of Day) contribute most to the forecast.
* **Error Distribution:** Analyzes the percentage error frequency.
* **Hourly MAPE:** Identifies which hours of the day (e.g., peak evening hours) are the most difficult to predict.
