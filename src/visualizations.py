import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff

def plot_historical_series(df, target_col, anomaly_col=None):
    """
    Plots a highly interactive timeline of the energy consumption series.
    If anomalies exist, overlays them as vibrant glowing neon points.
    """
    fig = go.Figure()
    
    # 1. Main energy consumption line
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[target_col],
        mode='lines',
        name='Energy Consumption',
        line=dict(color='#636EFA', width=1.5),
        hovertemplate='%{x}<br>Consumption: %{y:.2f} MWh<extra></extra>'
    ))
    
    # 2. Highlight anomalies if column is provided
    if anomaly_col and anomaly_col in df.columns:
        anomalies = df[df[anomaly_col] != 0]
        if not anomalies.empty:
            # Differentiate spikes vs drops for visual clarity
            spikes = anomalies[anomalies[anomaly_col] > 0]
            drops = anomalies[anomalies[anomaly_col] < 0]
            
            if not spikes.empty:
                fig.add_trace(go.Scatter(
                    x=spikes.index,
                    y=spikes[target_col],
                    mode='markers',
                    name='Anomalous Spikes',
                    marker=dict(color='#FF4B4B', size=8, symbol='triangle-up', 
                                line=dict(color='white', width=1)),
                    hovertemplate='<b>Spike Anomaly</b><br>%{x}<br>Value: %{y:.2f} MWh<extra></extra>'
                ))
            if not drops.empty:
                fig.add_trace(go.Scatter(
                    x=drops.index,
                    y=drops[target_col],
                    mode='markers',
                    name='Anomalous Drops',
                    marker=dict(color='#00CC96', size=8, symbol='triangle-down',
                                line=dict(color='white', width=1)),
                    hovertemplate='<b>Drop Anomaly</b><br>%{x}<br>Value: %{y:.2f} MWh<extra></extra>'
                ))
                
    fig.update_layout(
        title=dict(text='Energy Consumption Timeline', font=dict(size=18, color='#1E1E1E')),
        xaxis=dict(title='Timestamp', gridcolor='#EAEAEA', showgrid=True),
        yaxis=dict(title='Consumption (MWh)', gridcolor='#EAEAEA', showgrid=True),
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig

def plot_seasonal_profiles(df, target_col):
    """
    Creates seasonal aggregate curves:
    - Diurnal: Average energy consumption by hour of day, split by weekday vs. weekend.
    - Monthly: Average energy consumption by month of year.
    """
    df_temp = df.copy()
    df_temp['hour'] = df_temp.index.hour
    df_temp['day_type'] = np.where(df_temp.index.dayofweek >= 5, 'Weekend', 'Weekday')
    df_temp['month_name'] = df_temp.index.strftime('%B')
    df_temp['month_num'] = df_temp.index.month
    
    # 1. Diurnal profile
    diurnal = df_temp.groupby(['hour', 'day_type'])[target_col].mean().reset_index()
    fig_diurnal = px.line(
        diurnal, x='hour', y=target_col, color='day_type',
        color_discrete_map={'Weekday': '#636EFA', 'Weekend': '#AB63FA'},
        title='Daily Consumption Profile (Diurnal)'
    )
    fig_diurnal.update_traces(line=dict(width=3))
    fig_diurnal.update_layout(
        xaxis=dict(title='Hour of Day (24h)', tickmode='linear', tick0=0, dtick=2, gridcolor='#EAEAEA'),
        yaxis=dict(title='Avg Consumption (MWh)', gridcolor='#EAEAEA'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(title='Day Type', orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    # 2. Monthly profile
    monthly = df_temp.groupby(['month_num', 'month_name'])[target_col].mean().reset_index().sort_values('month_num')
    fig_monthly = px.bar(
        monthly, x='month_name', y=target_col,
        color=target_col, color_continuous_scale='Viridis',
        title='Monthly Average Consumption'
    )
    fig_monthly.update_layout(
        xaxis=dict(title='Month'),
        yaxis=dict(title='Avg Consumption (MWh)', gridcolor='#EAEAEA'),
        coloraxis_showscale=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig_diurnal, fig_monthly

def plot_correlation_heatmap(df, feature_cols):
    """
    Plots a visually stunning correlation heatmap of features.
    """
    corr = df[feature_cols].corr()
    
    # We round for display
    corr_rounded = corr.round(2)
    
    fig = px.imshow(
        corr_rounded,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1,
        title='Feature Correlation Heatmap'
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis=dict(tickangle=-45)
    )
    return fig

def plot_actual_vs_predicted(test_index, y_true, y_pred, model_name="Model"):
    """
    Plots a visual timeline comparing actual values vs prediction.
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=test_index,
        y=y_true,
        mode='lines',
        name='Actual Consumption',
        line=dict(color='#333333', width=2),
        hovertemplate='Actual: %{y:.2f} MWh<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=test_index,
        y=y_pred,
        mode='lines',
        name=f'{model_name} Prediction',
        line=dict(color='#FF4B4B', width=2, dash='dash'),
        hovertemplate='Predicted: %{y:.2f} MWh<extra></extra>'
    ))
    
    fig.update_layout(
        title='Actual vs. Predicted Energy Consumption (Test Set)',
        xaxis=dict(title='Timestamp', gridcolor='#EAEAEA'),
        yaxis=dict(title='Energy Consumption (MWh)', gridcolor='#EAEAEA'),
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig

def plot_feature_importance(model, feature_names):
    """
    Plots feature importance.
    Works for XGBoost (feature_importances_) or Ridge (coefficients).
    """
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        title = 'XGBoost Feature Importance (Gain)'
    elif hasattr(model, 'coef_'):
        # For Ridge regression coefficients
        importance = np.abs(model.coef_)
        title = 'Ridge Regression: Absolute Coefficient Magnitude'
    else:
        return None
        
    feat_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=True).tail(15)  # Show top 15
    
    fig = px.bar(
        feat_imp,
        x='Importance',
        y='Feature',
        orientation='h',
        color='Importance',
        color_continuous_scale='Plasma',
        title=title
    )
    
    fig.update_layout(
        xaxis=dict(title='Importance Score', gridcolor='#EAEAEA'),
        yaxis=dict(title='Feature'),
        coloraxis_showscale=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig

def plot_error_distribution(y_true, y_pred):
    """
    Plots error distribution and residual statistics.
    """
    residuals = y_true - y_pred
    
    fig = ff.create_distplot(
        [residuals],
        group_labels=['Residual Error'],
        bin_size=[np.std(residuals) / 3],
        colors=['#FF6B6B'],
        show_rug=False
    )
    
    fig.update_layout(
        title='Residuals Distribution (Actual - Predicted)',
        xaxis=dict(title='Error Magnitude (MWh)', gridcolor='#EAEAEA'),
        yaxis=dict(title='Density'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=40, t=60, b=40),
        showlegend=False
    )
    return fig
