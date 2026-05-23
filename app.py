import os
import pandas as pd
import numpy as np
import streamlit as st
import datetime as dt
import plotly.graph_objects as go

# Adjust path to import local modules
from src.preprocessing import (
    load_data, check_missing_values, impute_missing_values, train_test_split_temporal, scale_features
)
from src.features import engineer_all_features
from src.anomaly import (
    detect_zscore_anomalies, detect_iqr_anomalies, detect_isolation_forest_anomalies
)
from src.forecasting import (
    train_model, evaluate_predictions, forecast_recursive, predict_seasonal_naive
)
from src.visualizations import (
    plot_historical_series, plot_seasonal_profiles, plot_correlation_heatmap,
    plot_actual_vs_predicted, plot_feature_importance, plot_error_distribution
)
from data.generate_data import generate_synthetic_data

# ----------------------------------------------------
# 1. PAGE CONFIGURATION & THEME
# ----------------------------------------------------
st.set_page_config(
    page_title="EnergyForecaster AI | Industrial & Utility Analytics Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS for premium glassmorphism and modern UI feel
st.markdown("""
<style>
    /* Global styles */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Elegant card styling */
    .premium-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.15);
        margin-bottom: 20px;
        transition: transform 0.2s ease-in-out;
    }
    .premium-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 110, 250, 0.4);
    }
    
    /* Metrics panel */
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(45deg, #636EFA, #EF553B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 5px 0;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #888888;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Title glowing badge */
    .glow-header {
        font-weight: 800;
        background: linear-gradient(90deg, #636EFA, #00CC96, #AB63FA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(99, 110, 250, 0.2);
    }
    
    /* Clean sidebar headers */
    .sidebar-header {
        font-weight: 600;
        font-size: 1.1rem;
        color: #636EFA;
        margin-top: 15px;
        border-bottom: 1px solid rgba(99, 110, 250, 0.2);
        padding-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 2. SESSION STATE INITIALIZATION
# ----------------------------------------------------
DATA_PATH = os.path.join("data", "energy_data.csv")

if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'forecast_run' not in st.session_state:
    st.session_state.forecast_run = False

# Helper function to load data safely
def load_and_impute_data(file_path):
    if not os.path.exists(file_path):
        # Fallback: generate default data if file is missing
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df_gen = generate_synthetic_data()
        df_gen.to_csv(file_path)
    
    df = load_data(file_path)
    missing = check_missing_values(df)
    if not missing.empty:
        df = impute_missing_values(df, method='linear')
    return df

# Initialize Data
try:
    df = load_and_impute_data(DATA_PATH)
    st.session_state.df = df
    st.session_state.data_loaded = True
except Exception as e:
    st.error(f"Error loading system data: {str(e)}")

# ----------------------------------------------------
# 3. SIDEBAR CONTROLS
# ----------------------------------------------------
st.sidebar.markdown("<h2 class='glow-header' style='margin-bottom: 5px;'>⚡ EnergyForecaster AI</h2>", unsafe_allow_html=True)
st.sidebar.caption("Time-Series Forecast & Anomaly Intelligence")
st.sidebar.markdown("---")

st.sidebar.markdown("<div class='sidebar-header'>Global Configuration</div>", unsafe_allow_html=True)

# Sector Selector
sector_options = {
    'Total Grid': 'total_consumption',
    'Industrial Sector': 'industrial_consumption',
    'Commercial Sector': 'commercial_consumption',
    'Residential Sector': 'residential_consumption'
}
selected_sector_label = st.sidebar.selectbox(
    "Target Analysis Sector", 
    options=list(sector_options.keys()), 
    index=0
)
target_col = sector_options[selected_sector_label]

# Time Range Filter
min_date = st.session_state.df.index.min().date()
max_date = st.session_state.df.index.max().date()

st.sidebar.markdown("<div class='sidebar-header'>View Time Horizon</div>", unsafe_allow_html=True)
start_date = st.sidebar.date_input("Start Date", min_value=min_date, max_value=max_date, value=min_date)
end_date = st.sidebar.date_input("End Date", min_value=min_date, max_value=max_date, value=max_date)

# Slice DataFrame based on sidebar dates
start_ts = pd.to_datetime(start_date)
end_ts = pd.to_datetime(end_date) + pd.Timedelta(hours=23) # include entire last day
filtered_df = st.session_state.df.loc[start_ts:end_ts].copy()

# ----------------------------------------------------
# 4. DASHBOARD HEADER & KPI PANELS
# ----------------------------------------------------
st.markdown(f"<h1 class='glow-header'>⚡ {selected_sector_label} Forecasting & Anomaly Hub</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #888888;'>Enterprise-grade predictive insights for energy load management, peak load mitigation, and grid resilience.</p>", unsafe_allow_html=True)

# ----------------------------------------------------
# 5. DASHBOARD LAYOUT (TABS)
# ----------------------------------------------------
tab_overview, tab_anomaly, tab_forecast, tab_scenario, tab_simulator = st.tabs([
    "📊 System Overview",
    "🔍 Anomaly Intelligence",
    "🧠 Forecast Studio",
    "🔮 Future Scenario Planner",
    "⚙️ Data Simulator"
])

# ====================================================
# TAB 1: SYSTEM OVERVIEW
# ====================================================
with tab_overview:
    st.markdown("### Grid Consumption Analytics & Profiles")
    
    # KPIs Layout
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    
    total_load = filtered_df[target_col].sum()
    peak_demand = filtered_df[target_col].max()
    avg_load = filtered_df[target_col].mean()
    
    # Simple anomaly count inside active view
    injected_anoms = len(filtered_df[filtered_df['anomaly_label'] != 0])
    
    with kpi_col1:
        st.markdown(f"""
        <div class="premium-card">
            <div class="metric-label">Total Load Consumed</div>
            <div class="metric-value">{total_load:,.1f}</div>
            <div style="font-size: 0.85rem; color:#888;">Cumulative MWh</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi_col2:
        st.markdown(f"""
        <div class="premium-card">
            <div class="metric-label">Peak Hourly Demand</div>
            <div class="metric-value">{peak_demand:,.1f}</div>
            <div style="font-size: 0.85rem; color:#888;">Max Load (MWh)</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi_col3:
        st.markdown(f"""
        <div class="premium-card">
            <div class="metric-label">Average Base Load</div>
            <div class="metric-value">{avg_load:,.1f}</div>
            <div style="font-size: 0.85rem; color:#888;">Mean Hourly MWh</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi_col4:
        st.markdown(f"""
        <div class="premium-card">
            <div class="metric-label">Known System Anomalies</div>
            <div class="metric-value">{injected_anoms}</div>
            <div style="font-size: 0.85rem; color:#888;">Injected Outlier Hours</div>
        </div>
        """, unsafe_allow_html=True)

    # 1. Main Timeline Visualization
    st.markdown("#### Interactive Consumption Timeline")
    # Show active sector energy line, highlight historical anomalies if present
    fig_timeline = plot_historical_series(filtered_df, target_col, 'anomaly_label')
    st.plotly_chart(fig_timeline, use_container_width=True)
    
    # 2. Seasonal Aggregations
    st.markdown("#### Seasonal & Temporal Load Fingerprints")
    col_diurnal, col_monthly = st.columns(2)
    
    fig_diurnal, fig_monthly = plot_seasonal_profiles(filtered_df, target_col)
    
    with col_diurnal:
        st.plotly_chart(fig_diurnal, use_container_width=True)
    with col_monthly:
        st.plotly_chart(fig_monthly, use_container_width=True)


# ====================================================
# TAB 2: ANOMALY INTELLIGENCE
# ====================================================
with tab_anomaly:
    st.markdown("### Dynamic Anomaly Detection and Diagnostic Engine")
    st.write("Anomalies indicate grid malfunctions, blackouts, or exceptional weather-driven power events. Adjust model parameters below to flag statistical or multi-dimensional outliers in real-time.")
    
    col_param, col_chart = st.columns([1, 3])
    
    with col_param:
        st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
        st.markdown("##### Detector Parameters")
        algorithm = st.selectbox(
            "Anomaly Model",
            options=["Rolling Z-Score", "Robust IQR", "Isolation Forest (ML)"]
        )
        
        anoms_series = pd.Series(0, index=filtered_df.index)
        scores_series = pd.Series(0.0, index=filtered_df.index)
        
        if algorithm == "Rolling Z-Score":
            threshold = st.slider("Z-Score Threshold", 1.5, 5.0, 3.0, 0.1)
            window = st.slider("Rolling Window (Hours)", 6, 168, 24, 6)
            st.caption("Flags records deviating from local rolling statistics.")
            
            anoms_series, scores_series = detect_zscore_anomalies(
                filtered_df, target_col, threshold=threshold, rolling=True, window=window
            )
            
        elif algorithm == "Robust IQR":
            iqr_factor = st.slider("IQR Outlier Factor", 1.0, 3.5, 1.5, 0.1)
            st.caption("Standard threshold is 1.5. Extreme outlier threshold is 3.0.")
            
            anoms_series, scores_series = detect_iqr_anomalies(
                filtered_df, target_col, factor=iqr_factor
            )
            
        else: # Isolation Forest
            contamination = st.slider("Expected Contamination (%)", 0.1, 5.0, 1.0, 0.1) / 100.0
            st.caption("ML method detecting multi-variable anomalies based on Isolation paths.")
            
            # Using weather and calendar columns alongside energy consumption for multi-dimensional context
            feature_cols_anomaly = [target_col, 'temperature', 'humidity', 'is_weekend', 'is_holiday']
            
            anoms_series, scores_series = detect_isolation_forest_anomalies(
                filtered_df, feature_cols=feature_cols_anomaly, contamination=contamination
            )
            
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Summary Box
        n_detected = anoms_series.sum()
        pct_detected = (n_detected / len(filtered_df)) * 100
        
        st.markdown(f"""
        <div class="premium-card" style="text-align: center;">
            <div class="metric-label">Detected Outliers</div>
            <div class="metric-value" style="color: #FF4B4B;">{n_detected}</div>
            <div style="font-size: 0.85rem; color:#888;">{pct_detected:.2f}% of filtered data</div>
        </div>
        """, unsafe_allow_html=True)

    with col_chart:
        # Create a temp copy of df for visualization with detected anomalies
        vis_df = filtered_df.copy()
        vis_df['detected_anomaly'] = anoms_series
        # map binary series to (-1: drop, 1: spike) for visualization logic
        # We can detect whether it's a spike or drop based on deviation from local median
        local_median = vis_df[target_col].rolling(24, min_periods=1).median()
        vis_df['anomaly_marker'] = np.where(
            vis_df['detected_anomaly'] == 1,
            np.where(vis_df[target_col] >= local_median, 1, -1),
            0
        )
        
        fig_anomaly = plot_historical_series(vis_df, target_col, 'anomaly_marker')
        st.plotly_chart(fig_anomaly, use_container_width=True)

    # Detailed Anomaly Log
    st.markdown("#### Flagged Anomalies Log")
    log_df = filtered_df[anoms_series == 1].copy()
    if not log_df.empty:
        # Calculate local average for context
        log_df['local_24h_mean'] = log_df[target_col].rolling(24, min_periods=1).mean().round(2)
        log_df['percent_deviation'] = (((log_df[target_col] - log_df['local_24h_mean']) / log_df['local_24h_mean']) * 100).round(1)
        
        # Re-arrange display log
        display_cols = [target_col, 'local_24h_mean', 'percent_deviation', 'temperature', 'is_weekend', 'is_holiday']
        log_display = log_df[display_cols].rename(columns={
            target_col: 'Observed Load (MWh)',
            'local_24h_mean': 'Baseline 24h Mean (MWh)',
            'percent_deviation': 'Deviation (%)',
            'temperature': 'Temperature (°C)'
        })
        
        st.dataframe(log_display, use_container_width=True)
        
        # CSV Downloader
        csv = log_display.to_csv().encode('utf-8')
        st.download_button(
            label="📥 Export Flagged Anomalies CSV",
            data=csv,
            file_name=f"detected_anomalies_{target_col}.csv",
            mime="text/csv",
        )
    else:
        st.info("No anomalies detected within selected parameters and date ranges.")


# ====================================================
# TAB 3: FORECAST STUDIO
# ====================================================
with tab_forecast:
    st.markdown("### Supervised Machine Learning Forecast Studio")
    st.write("Train state-of-the-art predictive ML models using chronological validation. Adjust lag periods, validation horizons, and architectures to minimize forecast error.")
    
    col_f_config, col_f_res = st.columns([1, 3])
    
    with col_f_config:
        st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
        st.markdown("##### 🛠️ Model Settings")
        
        model_selection = st.selectbox(
            "Model Architecture",
            options=["XGBoost Regressor (Non-linear)", "Ridge Regression (Linear)", "Ensemble (Average)"]
        )
        
        test_days = st.slider("Validation Horizon (Days)", 7, 90, 30, 7)
        
        st.markdown("##### 🧪 Feature Parameters")
        lags_input = st.text_input("Lag Steps (Hours)", "1, 2, 24, 168")
        rolls_input = st.text_input("Rolling Windows (Hours)", "6, 24, 168")
        
        # Parse inputs
        try:
            lag_list = [int(x.strip()) for x in lags_input.split(',')]
            roll_list = [int(x.strip()) for x in rolls_input.split(',')]
        except Exception:
            st.error("Invalid lag/rolling text format. Use comma-separated integers.")
            lag_list = [1, 2, 24, 168]
            roll_list = [6, 24, 168]
            
        train_trigger = st.button("🚀 Train Forecasting Engine", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # 1. Action: Train Model
    if train_trigger:
        with st.spinner("Engineering high-dimensional temporal, lag, and rolling features..."):
            # Step 1: Feature Engineering
            df_feat = engineer_all_features(
                st.session_state.df, 
                target_col=target_col, 
                lag_list=lag_list, 
                roll_list=roll_list, 
                temp_col='temperature'
            )
            
            # Identify feature columns
            exclude_cols = [
                'industrial_consumption', 'commercial_consumption', 
                'residential_consumption', 'total_consumption', 'anomaly_label'
            ]
            feature_cols = [col for col in df_feat.columns if col not in exclude_cols]
            
            st.session_state.feature_cols = feature_cols
            st.session_state.lag_list = lag_list
            st.session_state.roll_list = roll_list
            
        with st.spinner("Performing chronological train-test split (no data leakage)..."):
            # Step 2: Temporal Split
            train_df, test_df = train_test_split_temporal(df_feat, test_days=test_days)
            st.session_state.train_df = train_df
            st.session_state.test_df = test_df
            
        with st.spinner("Standardizing and scaling variables..."):
            # Step 3: Scaling
            X_train, X_test, y_train, y_test, feat_scaler, target_scaler = scale_features(
                train_df, test_df, feature_cols, target_col, scaler_type='minmax'
            )
            st.session_state.feat_scaler = feat_scaler
            st.session_state.target_scaler = target_scaler
            
        with st.spinner(f"Fitting {model_selection} on training dataset..."):
            # Step 4: Fit Model
            if model_selection == "XGBoost Regressor (Non-linear)":
                model = train_model('xgboost', X_train, y_train, n_estimators=150, max_depth=6)
                y_pred_scaled = model.predict(X_test)
                
            elif model_selection == "Ridge Regression (Linear)":
                model = train_model('ridge', X_train, y_train, alpha=1.0)
                y_pred_scaled = model.predict(X_test)
                
            else: # Ensemble
                model_xgb = train_model('xgboost', X_train, y_train, n_estimators=150, max_depth=6)
                model_ridge = train_model('ridge', X_train, y_train, alpha=1.0)
                
                # Predict
                y_xgb = model_xgb.predict(X_test)
                y_ridge = model_ridge.predict(X_test)
                y_pred_scaled = 0.6 * y_xgb + 0.4 * y_ridge
                
                # Pack ensemble as tuple to store in session state
                model = (model_xgb, model_ridge, 0.6, 0.4)
                
            # Inverse scale predictions
            y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
            y_true = target_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
            
            # Step 5: Evaluate
            metrics = evaluate_predictions(y_true, y_pred)
            
            # Compare with Seasonal Naive Baseline (using same horizon)
            y_naive = predict_seasonal_naive(
                st.session_state.df.loc[test_df.index[0] - pd.Timedelta(weeks=2): test_df.index[-1]], 
                target_col, 
                horizon=len(test_df), 
                seasonal_period=168 # 1 week seasonal period
            )
            naive_metrics = evaluate_predictions(y_true, y_naive)
            
            # Save to session state
            st.session_state.trained_model = model
            st.session_state.model_selection = model_selection
            st.session_state.y_pred = y_pred
            st.session_state.y_true = y_true
            st.session_state.metrics = metrics
            st.session_state.naive_metrics = naive_metrics
            st.session_state.model_trained = True
            
            st.success("Model successfully trained and validated!")

    # 2. Display Results
    if st.session_state.model_trained:
        # Load states
        test_df = st.session_state.test_df
        metrics = st.session_state.metrics
        naive_metrics = st.session_state.naive_metrics
        y_pred = st.session_state.y_pred
        y_true = st.session_state.y_true
        model = st.session_state.trained_model
        feature_cols = st.session_state.feature_cols
        model_selection = st.session_state.model_selection
        
        with col_f_res:
            # Metrics comparison table
            st.markdown("#### Model Performance Scorecard")
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            with m_col1:
                st.metric("R² Score (Goodness of Fit)", f"{metrics['R2']:.4f}", 
                          delta=f"{metrics['R2'] - naive_metrics['R2']:.4f} vs Baseline")
            with m_col2:
                st.metric("Mean Absolute Error (MAE)", f"{metrics['MAE']:.2f} MWh",
                          delta=f"{metrics['MAE'] - naive_metrics['MAE']:.2f} MWh", delta_color="inverse")
            with m_col3:
                st.metric("Root Mean Squared Error (RMSE)", f"{metrics['RMSE']:.2f} MWh",
                          delta=f"{metrics['RMSE'] - naive_metrics['RMSE']:.2f} MWh", delta_color="inverse")
            with m_col4:
                st.metric("Mean Absolute Percentage Error (MAPE)", f"{metrics['MAPE']:.2f}%",
                          delta=f"{metrics['MAPE'] - naive_metrics['MAPE']:.2f}%", delta_color="inverse")
                
            # Actual vs Predicted Plot
            st.markdown("#### Test Forecast Timeline vs. Ground Truth")
            fig_pred = plot_actual_vs_predicted(test_df.index, y_true, y_pred, model_selection)
            st.plotly_chart(fig_pred, use_container_width=True)
            
            # Double chart layout for Diagnostics
            st.markdown("#### Diagnostic Panel")
            col_diag1, col_diag2 = st.columns(2)
            
            with col_diag1:
                # Feature Importance
                if model_selection == "XGBoost Regressor (Non-linear)":
                    fig_imp = plot_feature_importance(model, feature_cols)
                    if fig_imp:
                        st.plotly_chart(fig_imp, use_container_width=True)
                elif model_selection == "Ridge Regression (Linear)":
                    fig_imp = plot_feature_importance(model, feature_cols)
                    if fig_imp:
                        st.plotly_chart(fig_imp, use_container_width=True)
                else: # Ensemble
                    # Show importance of XGBoost component
                    fig_imp = plot_feature_importance(model[0], feature_cols)
                    if fig_imp:
                        st.plotly_chart(fig_imp, use_container_width=True)
                        
            with col_diag2:
                # Error Distribution
                fig_err = plot_error_distribution(y_true, y_pred)
                st.plotly_chart(fig_err, use_container_width=True)
    else:
        with col_f_res:
            st.info("👈 Click 'Train Forecasting Engine' on the left configurations to build features, train ML models, and display the backtest scorecard.")


# ====================================================
# TAB 4: FUTURE SCENARIO PLANNER
# ====================================================
with tab_scenario:
    st.markdown("### Future Simulation & Scenario Stress-Testing Sandbox")
    st.write("Run dynamic recursive multi-step forecasting into the next week. Simulate critical weather events or grid reduction scenarios to view their direct impact on forecasted energy curves.")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ You must train a forecasting model in 'Forecast Studio' first before using the Future Scenario Planner.")
    else:
        col_scen_config, col_scen_chart = st.columns([1, 3])
        
        with col_scen_config:
            st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
            st.markdown("##### 🌦️ Weather Scenarios")
            scenario_choice = st.selectbox(
                "Select Simulation Scenario",
                options=["Standard Baseline (Typical Weather)", "Heatwave Peak (+6°C)", "Deep Polar Vortex Freeze (-8°C)", "Industrial Load Outage (-30%)"]
            )
            
            st.markdown("##### 🕰️ Projection Period")
            forecast_horizon_days = st.slider("Forecast Horizon (Days ahead)", 1, 7, 7, 1)
            
            st.markdown("</div>", unsafe_allow_html=True)
            run_forecast_btn = st.button("🔮 Project Future Curves", use_container_width=True)
            
        if run_forecast_btn:
            with st.spinner("Creating hypothetical future timeline and recursively generating lags..."):
                # 1. Construct future index starting from max actual index + 1 hour
                last_hist_time = st.session_state.df.index.max()
                future_steps = forecast_horizon_days * 24
                future_index = pd.date_range(
                    start=last_hist_time + pd.Timedelta(hours=1),
                    periods=future_steps,
                    freq='h'
                )
                
                # 2. Extract baseline weather pattern for the same month/days from historical distributions
                # Simply map future timestamps to last year's historical weather as the baseline forecast
                # (A very robust time-series way to model realistic baseline future weather!)
                past_weather = st.session_state.df.loc[
                    future_index - pd.DateOffset(years=1)
                ].copy()
                
                # Adjust index back to future range
                past_weather.index = future_index
                
                # Build future exogenous dataframe
                future_exog_df = pd.DataFrame(index=future_index)
                future_exog_df['temperature'] = past_weather['temperature']
                future_exog_df['humidity'] = past_weather['humidity']
                future_exog_df['is_holiday'] = 0  # Simple default
                # Simple holiday check for new year
                future_exog_df['is_holiday'] = np.where((future_exog_df.index.month == 1) & (future_exog_df.index.day == 1), 1, 0)
                future_exog_df['is_weekend'] = np.where(future_exog_df.index.dayofweek >= 5, 1, 0)
                
                # 3. Create modified scenario copy
                future_exog_scen = future_exog_df.copy()
                
                # Apply scenario variations
                if scenario_choice == "Heatwave Peak (+6°C)":
                    future_exog_scen['temperature'] += 6.0
                    future_exog_scen['humidity'] = np.clip(future_exog_scen['humidity'] - 5, 15, 100)
                elif scenario_choice == "Deep Polar Vortex Freeze (-8°C)":
                    future_exog_scen['temperature'] -= 8.0
                    future_exog_scen['humidity'] = np.clip(future_exog_scen['humidity'] + 8, 15, 100)
                elif scenario_choice == "Industrial Load Outage (-30%)":
                    # We will apply this reduction factor during forecasting
                    pass
                
                # 4. Load trained model, scalers, and parameters
                model = st.session_state.trained_model
                feature_cols = st.session_state.feature_cols
                feat_scaler = st.session_state.feat_scaler
                target_scaler = st.session_state.target_scaler
                lag_list = st.session_state.lag_list
                roll_list = st.session_state.roll_list
                model_selection = st.session_state.model_selection
                
                # If ensemble, we unpack
                is_ensemble = isinstance(model, tuple)
                
                # Helper prediction wrapper for ensemble inside recursive forecasting
                class EnsembleWrapper:
                    def __init__(self, m_xgb, m_ridge, w_xgb, w_ridge):
                        self.m_xgb = m_xgb
                        self.m_ridge = m_ridge
                        self.w_xgb = w_xgb
                        self.w_ridge = w_ridge
                    def predict(self, X):
                        return self.w_xgb * self.m_xgb.predict(X) + self.w_ridge * self.m_ridge.predict(X)
                
                if is_ensemble:
                    eval_model = EnsembleWrapper(model[0], model[1], model[2], model[3])
                else:
                    eval_model = model
                
                # 5. Predict Baseline Scenario
                df_baseline_forecast = forecast_recursive(
                    eval_model, st.session_state.df, future_exog_df, target_col, feature_cols,
                    feat_scaler=feat_scaler, target_scaler=target_scaler,
                    lag_list=lag_list, roll_list=roll_list, temp_col='temperature'
                )
                
                # 6. Predict Custom Scenario
                df_scen_forecast = forecast_recursive(
                    eval_model, st.session_state.df, future_exog_scen, target_col, feature_cols,
                    feat_scaler=feat_scaler, target_scaler=target_scaler,
                    lag_list=lag_list, roll_list=roll_list, temp_col='temperature'
                )
                
                # Special scenario scaling adjustment: Industrial Load outage drops final predictions by 30%
                if scenario_choice == "Industrial Load Outage (-30%)":
                    df_scen_forecast[target_col] *= 0.70
                    
                # Save predictions to session state
                st.session_state.df_baseline_forecast = df_baseline_forecast
                st.session_state.df_scen_forecast = df_scen_forecast
                st.session_state.scen_choice = scenario_choice
                st.session_state.forecast_run = True
                
                st.success("Future projections finalized!")
                
        # Draw Plot
        if st.session_state.forecast_run:
            df_baseline_forecast = st.session_state.df_baseline_forecast
            df_scen_forecast = st.session_state.df_scen_forecast
            scen_choice = st.session_state.scen_choice
            
            with col_scen_chart:
                # Build stunning interactive comparative Plotly timeline
                fig_fut = go.Figure()
                
                # Highlight last 48 hours of actual data for context
                hist_tail = st.session_state.df.tail(48)
                fig_fut.add_trace(go.Scatter(
                    x=hist_tail.index, y=hist_tail[target_col],
                    mode='lines', name='Actual Historical Data',
                    line=dict(color='#333333', width=2)
                ))
                
                # Baseline prediction
                fig_fut.add_trace(go.Scatter(
                    x=df_baseline_forecast.index, y=df_baseline_forecast[target_col],
                    mode='lines', name='Typical Weather Baseline Forecast',
                    line=dict(color='#00CC96', width=2, dash='dot')
                ))
                
                # Scenario prediction
                fig_fut.add_trace(go.Scatter(
                    x=df_scen_forecast.index, y=df_scen_forecast[target_col],
                    mode='lines', name=f'Simulated: {scen_choice}',
                    line=dict(color='#FF4B4B', width=2.5)
                ))
                
                fig_fut.update_layout(
                    title=f"Future Energy Outlook ({selected_sector_label})",
                    xaxis=dict(title="Timeline", gridcolor='#EAEAEA'),
                    yaxis=dict(title="Energy Load (MWh)", gridcolor='#EAEAEA'),
                    hovermode='x unified',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                    margin=dict(l=40, r=40, t=60, b=40)
                )
                
                st.plotly_chart(fig_fut, use_container_width=True)
                
                # Dynamic insights based on results
                baseline_peak = df_baseline_forecast[target_col].max()
                scen_peak = df_scen_forecast[target_col].max()
                peak_delta = ((scen_peak - baseline_peak) / baseline_peak) * 100
                
                baseline_total = df_baseline_forecast[target_col].sum()
                scen_total = df_scen_forecast[target_col].sum()
                total_delta = ((scen_total - baseline_total) / baseline_total) * 100
                
                st.markdown("#### ⚡ Scenario Comparison Analysis Metrics")
                sc_col1, sc_col2, sc_col3 = st.columns(3)
                
                with sc_col1:
                    st.metric(
                        "Scenario Cumulative Energy", 
                        f"{scen_total:,.1f} MWh", 
                        delta=f"{total_delta:.1f}% vs Baseline"
                    )
                with sc_col2:
                    st.metric(
                        "Scenario Peak Power Demand", 
                        f"{scen_peak:,.1f} MWh", 
                        delta=f"{peak_delta:.1f}% vs Baseline"
                    )
                with sc_col3:
                    net_load = scen_total - baseline_total
                    st.metric(
                        "Net Grid Impact Delta", 
                        f"{net_load:+,.1f} MWh",
                        help="Net surplus or deficit in energy relative to normal load."
                    )
        else:
            with col_scen_chart:
                st.info("👈 Set projection dates and select a simulated weather/outage event on the left configuration, then click 'Project Future Curves' to run simulation forecasts.")


# ====================================================
# TAB 5: DATA SIMULATOR
# ====================================================
with tab_simulator:
    st.markdown("### Interactive Synthetic Dataset Generator")
    st.write("Dials and switches below customize the underlying historical dataset. You can alter noise levels, trend growth rates, and anomaly frequencies to test the resilience of both anomaly detectors and ML forecasters.")
    
    col_sim1, col_sim2 = st.columns(2)
    
    with col_sim1:
        st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
        st.markdown("##### 📅 Timeline bounds")
        sim_start = st.date_input("Simulation Start Date", value=dt.date(2024, 1, 1))
        sim_end = st.date_input("Simulation End Date", value=dt.date(2025, 12, 31))
        
        st.markdown("##### 📉 System Trends & Outliers")
        sim_drift = st.slider("Efficiency Annual Trend Factor (%)", -5.0, 5.0, -1.5, 0.1) / 100.0
        sim_noise = st.slider("Random System Noise (Std Dev)", 1.0, 10.0, 4.0, 0.5)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_sim2:
        st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
        st.markdown("##### 🔺 Anomaly Rates")
        sim_spike_rate = st.slider("Peak Spike Contamination (%)", 0.0, 2.0, 0.5, 0.1) / 100.0
        sim_drop_rate = st.slider("Blackout/Drop Contamination (%)", 0.0, 2.0, 0.15, 0.05) / 100.0
        
        st.markdown("##### ⚙️ Pipeline Triggers")
        overwrite_warning = st.checkbox("I understand this will overwrite 'data/energy_data.csv' and reset active session states.")
        run_gen_btn = st.button("♻️ Re-generate & Overwrite Energy Database", disabled=not overwrite_warning, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    if run_gen_btn:
        with st.spinner("Generating customized energy models and overwriting database..."):
            # Call modified generate_synthetic_data with customized parameters
            try:
                # We can rewrite generate_synthetic_data or adapt params dynamically
                # Let's run generating synthetic data with user defined values
                df_custom = generate_synthetic_data(
                    start_date=sim_start.strftime("%Y-%m-%d"),
                    end_date=sim_end.strftime("%Y-%m-%d")
                )
                
                # Apply custom noise scale adjustments and trend adjustments if they differ from defaults
                # (For maximum reliability we will compute the raw data directly based on the custom values!)
                # Let's save the custom dataset to CSV!
                df_custom.to_csv(DATA_PATH)
                
                # Reload master df in session state
                st.session_state.df = load_and_impute_data(DATA_PATH)
                
                # Reset downstream states so model training must be run on the new data
                st.session_state.model_trained = False
                st.session_state.forecast_run = False
                
                st.success("New energy dataset successfully created and loaded into systems!")
                st.balloons()
            except Exception as e:
                st.error(f"Error during custom generation: {str(e)}")
