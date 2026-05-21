# ⚡ EnergyForecaster AI: Enterprise Time-Series energy Analytics System

EnergyForecaster AI is a production-grade, highly modular, time-series forecasting and anomaly detection system. It generates realistic utility energy loads across industrial, commercial, and residential sectors, engineers high-dimensional features, runs real-time statistical & ML-based anomaly detectors, validates predictive algorithms chronologically (without look-ahead bias), and projections future energy demands under stress-test scenarios (e.g. heatwaves, freeze events, outages).

The entire system is controlled via a state-of-the-art interactive **Streamlit Dashboard** featuring sleek glassmorphic aesthetics.

---

## 🚀 Key Features

1. **Realistic Synthetic Data Generator (`data/generate_data.py`)**:
   - Generates hourly load readings spanning multiple years.
   - Models realistic baseline trends, diurnal hourly patterns, weekday/weekend cycles, and major holiday drops.
   - Incorporates weather variables (Temperature, Humidity) with seasonal dependencies.
   - Injects realistic grid anomalies: extreme energy draws (spikes), grid failures (blackouts/drops), and progressive efficiency drift.

2. **Modular Data Pipeline (`src/preprocessing.py`)**:
   - Cleans raw indices, checks for duplicate rows, and handles seasonal-aware missing value imputations.
   - Implements strict **chronological train/test splitting** (no look-ahead/leakage).
   - Standardizes features using training set properties.

3. **Advanced Feature Engineering (`src/features.py`)**:
   - Extracted calendar features (`hour`, `day_of_week`, `month`, `is_weekend`, `is_holiday`).
   - Encodes cyclical features via trigonometric sine/cosine transformations to retain distance relations.
   - Multi-scale lags (`t-1`, `t-2`, `t-24`, `t-168`).
   - Rolling window statistical descriptors (means, deviations, mins, maxes over 6h, 24h, 168h).
   - Thermodynamic domain terms: **Heating Degree Days (HDD)** and **Cooling Degree Days (CDD)** with dynamic thermal inertia rolling averages.

4. **Multi-Algorithm Anomaly Diagnostic Engine (`src/anomaly.py`)**:
   - **Rolling Z-Score**: Evaluates deviations against localized rolling stats.
   - **Robust IQR**: Outlier detection using Interquartile Ranges.
   - **Isolation Forest**: Multi-dimensional ML detection to flag anomalies based on isolation paths.

5. **ML & Statistical Forecasting (`src/forecasting.py`)**:
   - Baselines: Seasonal Naive models.
   - Supervised ML: Linear Ridge Regression and Non-linear XGBoost Regressor.
   - Ensembles: Meta-averaging models to maximize generalization.
   - **Recursive Forecaster**: Dynamic rolling forecaster updating predictions and re-computing feature structures sequentially across future horizons.

6. **Scenario Stress-Testing Sandbox**:
   - Simulates hypothetical future weather scenarios (Heatwave peaks, polar freezes) and operational scenarios (factory shutdowns) to analyze grid capacity margins.

---

## 📂 Project Structure

```
energy_forecasting/
├── app.py                  # Main Streamlit dashboard UI and layout controller
├── requirements.txt        # Package dependencies
├── README.md               # User guide & system manual (This file)
├── data/
│   ├── generate_data.py    # Synthetic energy & weather generator
│   └── energy_data.csv     # Generated CSV database file
└── src/
    ├── preprocessing.py    # Cleaning, imputation, and scaling pipeline
    ├── features.py         # Advanced temporal, lag, and weather features
    ├── anomaly.py          # Z-score, IQR, and Isolation Forest detectors
    ├── forecasting.py      # Ridge, XGBoost, and recursive forecasting
    └── visualizations.py   # Reusable Plotly chart builders
```

---

## 🛠️ Installation & Setup

1. **Verify Python Environment**:
   Ensure you have Python 3.9 - 3.11 installed.

2. **Navigate to the Project Directory**:
   ```bash
   cd C:\Users\ASUS\.gemini\antigravity\scratch\energy_forecasting
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Generate Initial Dataset** *(Note: The Streamlit app will auto-generate this on startup if missing)*:
   ```bash
   python data/generate_data.py
   ```

5. **Launch the Dashboard**:
   ```bash
   streamlit run app.py
   ```

---

## 📊 Dashboard Workspaces

- **System Overview**: Interactive historical grids, KPI summary cards, and aggregate daily/monthly seasonal fingerprint shapes.
- **Anomaly Intelligence**: Real-time outlier tuning sliders and diagnostic log outputs with CSV export support.
- **Forecast Studio**: Model hyperparameter configuration, lag adjustments, progress-tracked training, R², MAE, RMSE metrics, and error residual diagnostics.
- **Future Scenario Planner**: Week-ahead recursive stress-tests against climate anomalies or load drops with comparative load delta calculations.
- **Data Simulator**: Full control over simulated database time ranges, noise scales, and contamination rates.
