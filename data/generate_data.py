import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_synthetic_data(start_date="2024-01-01", end_date="2025-12-31", seed=42):
    """
    Generates a highly realistic synthetic energy consumption dataset.
    
    Parameters:
    -----------
    start_date : str
        Start date of the dataset in YYYY-MM-DD format.
    end_date : str
        End date of the dataset in YYYY-MM-DD format.
    seed : int
        Random seed for reproducibility.
        
    Returns:
    --------
    pd.DataFrame: DataFrame containing energy consumption, weather, and calendar features.
    """
    np.random.seed(seed)
    
    # Generate hourly datetime range
    date_range = pd.date_range(start=start_date, end=end_date, freq='h')
    n_hours = len(date_range)
    
    # Create base dataframe
    df = pd.DataFrame(index=date_range)
    df.index.name = 'timestamp'
    
    # 1. Generate Weather Features (Temperature & Humidity)
    # Annual cycle: temperature peaks in late July (day of year ~205)
    day_of_year = df.index.dayofyear
    hour_of_day = df.index.hour
    
    # Annual temperature variation (Sine wave with peak in summer)
    annual_temp = 15 + 12 * np.sin(2 * np.pi * (day_of_year - 100) / 365)
    # Daily temperature variation (peaks at 15:00)
    daily_temp = 5 * np.sin(2 * np.pi * (hour_of_day - 9) / 24)
    # Weather noise (multi-day weather systems modeled via low-pass filter of normal noise)
    raw_noise = np.random.normal(0, 3, n_hours)
    weather_noise = pd.Series(raw_noise).ewm(span=48).mean().values * 2.5
    
    df['temperature'] = annual_temp + daily_temp + weather_noise
    
    # Humidity: Inversely correlated with temperature, with daily fluctuations
    base_humidity = 65 - 15 * np.sin(2 * np.pi * (hour_of_day - 9) / 24)
    humidity_noise = pd.Series(np.random.normal(0, 8, n_hours)).ewm(span=24).mean().values
    df['humidity'] = np.clip(base_humidity - 0.6 * (df['temperature'] - 15) + humidity_noise, 15, 100)
    
    # 2. Calendar Features (Holidays)
    # Simple US holiday logic (approximate)
    is_holiday = np.zeros(n_hours)
    # New Year (Jan 1)
    is_holiday[(df.index.month == 1) & (df.index.day == 1)] = 1
    # Independence Day (Jul 4)
    is_holiday[(df.index.month == 7) & (df.index.day == 4)] = 1
    # Thanksgiving (Fourth Thursday of November - approximate as late Nov)
    is_holiday[(df.index.month == 11) & (df.index.dayofweek == 3) & (df.index.day >= 22) & (df.index.day <= 28)] = 1
    # Christmas (Dec 25)
    is_holiday[(df.index.month == 12) & (df.index.day == 25)] = 1
    
    df['is_holiday'] = is_holiday.astype(int)
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    
    # Helper coefficients for sector consumption modeling
    # Temperature heating/cooling thresholds
    heating_threshold = 15.0  # Under 15°C, heating turns on
    cooling_threshold = 22.0  # Above 22°C, cooling turns on
    
    cdd = np.maximum(0, df['temperature'] - cooling_threshold)
    hdd = np.maximum(0, heating_threshold - df['temperature'])
    
    # 3. Sector 1: Industrial Consumption (High base load, weekend drop, low weather dependency)
    # Base load of 150 MWh
    ind_base = 150.0
    # Day/Night shift cycle: peaks during daytime (08:00 - 18:00)
    ind_daily = 15 * np.where((hour_of_day >= 8) & (hour_of_day <= 18), 1.0, 0.0)
    # Weekend drop: 35% lower load
    ind_weekend_mult = np.where(df['is_weekend'] == 1, 0.65, 1.0)
    # Holiday drop: 40% lower load
    ind_holiday_mult = np.where(df['is_holiday'] == 1, 0.60, 1.0)
    # Minor weather dependence (cooling large factories)
    ind_weather = 0.8 * cdd
    # Combine
    ind_noise = np.random.normal(0, 4, n_hours)
    df['industrial_consumption'] = (ind_base + ind_daily + ind_weather) * ind_weekend_mult * ind_holiday_mult + ind_noise
    
    # 4. Sector 2: Commercial Consumption (Moderate base load, strict business hours, high HVAC dependency)
    # Base load of 40 MWh
    com_base = 40.0
    # Strict business hours curve: peaks during 08:00 - 19:00
    com_daily = 35 * np.sin(np.pi * np.clip((hour_of_day - 7) / 12, 0, 1))
    # Weekend drop: 60% lower load
    com_weekend_mult = np.where(df['is_weekend'] == 1, 0.40, 1.0)
    # Holiday drop: 75% lower load
    com_holiday_mult = np.where(df['is_holiday'] == 1, 0.25, 1.0)
    # HVAC dependency (heating in winter, cooling in summer)
    com_weather = 3.5 * cdd + 2.0 * hdd
    # Combine
    com_noise = np.random.normal(0, 2.5, n_hours)
    df['commercial_consumption'] = (com_base + com_daily + com_weather) * com_weekend_mult * com_holiday_mult + com_noise
    
    # 5. Sector 3: Residential Consumption (Low base load, double peak, high temperature sensitivity)
    # Base load of 25 MWh
    res_base = 25.0
    # Double peak: morning peak (06:00-09:00) and evening peak (17:00-22:00)
    res_daily = np.zeros(n_hours)
    res_daily[(hour_of_day >= 6) & (hour_of_day <= 9)] = 12
    res_daily[(hour_of_day >= 17) & (hour_of_day <= 22)] = 22
    # Weekend increase (+15% higher load since people are at home)
    res_weekend_mult = np.where(df['is_weekend'] == 1, 1.15, 1.0)
    # Holiday increase (+10%)
    res_holiday_mult = np.where(df['is_holiday'] == 1, 1.10, 1.0)
    # Massive weather dependence (cooling and heating)
    res_weather = 5.0 * cdd + 4.5 * hdd
    # Combine
    res_noise = np.random.normal(0, 3, n_hours)
    df['residential_consumption'] = (res_base + res_daily + res_weather) * res_weekend_mult * res_holiday_mult + res_noise
    
    # Ensure no negative energy consumption values
    df['industrial_consumption'] = np.clip(df['industrial_consumption'], 5, None)
    df['commercial_consumption'] = np.clip(df['commercial_consumption'], 2, None)
    df['residential_consumption'] = np.clip(df['residential_consumption'], 5, None)
    
    # Total consumption
    df['total_consumption'] = df['industrial_consumption'] + df['commercial_consumption'] + df['residential_consumption']
    
    # 6. Inject Trends / Efficiency Drift (e.g. -1.5% consumption per year due to efficiency)
    years_elapsed = (df.index - df.index[0]).days / 365.25
    drift_factor = 1.0 - 0.015 * years_elapsed
    for col in ['industrial_consumption', 'commercial_consumption', 'residential_consumption', 'total_consumption']:
        df[col] = df[col] * drift_factor
        
    # 7. Inject Anomalies (Spikes, Drops)
    # We will randomly select timestamps to inject anomalies, and return which indices were anomalous
    df['anomaly_label'] = 0  # 0: normal, 1: spike, -1: drop
    
    # Inject Spikes (Sudden extremely high loads, ~0.5% rate)
    n_spikes = int(n_hours * 0.005)
    spike_indices = np.random.choice(n_hours, n_spikes, replace=False)
    for idx in spike_indices:
        # Avoid putting anomalies in first or last 48 hours to preserve model margins
        if idx < 48 or idx > n_hours - 48:
            continue
        # We increase total and sector consumptions by a factor of 1.6 to 2.2
        multiplier = np.random.uniform(1.6, 2.2)
        df.iloc[idx, df.columns.get_loc('total_consumption')] *= multiplier
        
        # Distribute the spike to a random sector as well
        sec_col = np.random.choice(['industrial_consumption', 'commercial_consumption', 'residential_consumption'])
        df.iloc[idx, df.columns.get_loc(sec_col)] *= multiplier
        df.iloc[idx, df.columns.get_loc('anomaly_label')] = 1
        
    # Inject Drops/Blackouts (Sudden extreme drops, ~0.4% rate, can last 2-6 hours)
    n_drops = int(n_hours * 0.0015)
    drop_indices = np.random.choice(n_hours, n_drops, replace=False)
    for idx in drop_indices:
        if idx < 48 or idx > n_hours - 48:
            continue
        duration = np.random.randint(2, 7)
        # Drop consumption by 70% to 90%
        multiplier = np.random.uniform(0.1, 0.3)
        for d in range(duration):
            curr_idx = idx + d
            if curr_idx >= n_hours:
                break
            df.iloc[curr_idx, df.columns.get_loc('total_consumption')] *= multiplier
            df.iloc[curr_idx, df.columns.get_loc('industrial_consumption')] *= multiplier
            df.iloc[curr_idx, df.columns.get_loc('commercial_consumption')] *= multiplier
            df.iloc[curr_idx, df.columns.get_loc('residential_consumption')] *= multiplier
            df.iloc[curr_idx, df.columns.get_loc('anomaly_label')] = -1
            
    # Clean up column data types
    df = df.round(2)
    return df

if __name__ == "__main__":
    # Script execution block to generate and save default dataset
    data_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(data_dir, "energy_data.csv")
    
    print(f"Generating synthetic energy dataset...")
    energy_df = generate_synthetic_data(start_date="2024-01-01", end_date="2025-12-31")
    energy_df.to_csv(output_path)
    print(f"Dataset successfully created and saved to: {output_path}")
    print(f"Shape: {energy_df.shape}")
    print(f"Detected anomalies injected: {len(energy_df[energy_df['anomaly_label'] != 0])} hours")
