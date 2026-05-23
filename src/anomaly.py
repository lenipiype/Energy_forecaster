import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

def detect_zscore_anomalies(df, target_col, threshold=3.0, rolling=True, window=24):
    """
    Detects anomalies using Z-score method.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Source dataframe.
    target_col : str
        Column to analyze.
    threshold : float
        Z-score threshold above which a point is labeled as an anomaly.
    rolling : bool
        If True, uses a rolling mean and std to compute local Z-score (better for non-stationary data).
        If False, uses global mean and std.
    window : int
        Rolling window size in hours (only used if rolling=True).
        
    Returns:
    --------
    pd.Series: Binary series where 1 = anomaly, 0 = normal.
    pd.Series: Calculated Z-scores.
    """
    if rolling:
        rolling_mean = df[target_col].rolling(window=window, min_periods=1).mean()
        rolling_std = df[target_col].rolling(window=window, min_periods=1).std()
        # Avoid division by zero
        rolling_std = rolling_std.replace(0, 1e-6).fillna(1.0)
        z_scores = (df[target_col] - rolling_mean) / rolling_std
    else:
        global_mean = df[target_col].mean()
        global_std = df[target_col].std()
        z_scores = (df[target_col] - global_mean) / (global_std if global_std > 0 else 1e-6)
        
    anomalies = (np.abs(z_scores) > threshold).astype(int)
    return anomalies, z_scores

def detect_iqr_anomalies(df, target_col, factor=1.5):
    """
    Detects anomalies using the Robust Interquartile Range (IQR) method.
    
    Parameters:
    -----------
    df : pd.DataFrame
    target_col : str
    factor : float
        Outlier multiplier factor (standard is 1.5, extreme is 3.0).
        
    Returns:
    --------
    pd.Series: Binary series where 1 = anomaly, 0 = normal.
    pd.Series: Calculated distance to nearest IQR boundary (negative if normal, positive if outlier).
    """
    q1 = df[target_col].quantile(0.25)
    q3 = df[target_col].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    anomalies = ((df[target_col] < lower_bound) | (df[target_col] > upper_bound)).astype(int)
    
    # Calculate score as normalized distance beyond the boundary
    dist_above = df[target_col] - upper_bound
    dist_below = lower_bound - df[target_col]
    max_dist = np.maximum(dist_above, dist_below)
    anomaly_scores = max_dist / (iqr if iqr > 0 else 1.0)
    
    return anomalies, anomaly_scores

def detect_isolation_forest_anomalies(df, feature_cols, contamination=0.01, seed=42):
    """
    Detects multidimensional anomalies using an Isolation Forest.
    
    Parameters:
    -----------
    df : pd.DataFrame
    feature_cols : list
        Columns to feed into Isolation Forest (e.g., ['total_consumption', 'temperature', 'hour', 'is_weekend']).
    contamination : float
        The expected proportion of anomalies in the dataset.
    seed : int
        Random seed.
        
    Returns:
    --------
    pd.Series: Binary series where 1 = anomaly, 0 = normal (converted from sklearn's -1/1 representation).
    pd.Series: Anomaly decision scores (more negative means more anomalous).
    """
    # Fit isolation forest
    clf = IsolationForest(contamination=contamination, random_state=seed, n_jobs=-1)
    
    # Extract only features, handle potential NaNs
    feat_df = df[feature_cols].copy().ffill().bfill()
    
    preds = clf.fit_predict(feat_df)
    scores = clf.decision_function(feat_df)
    
    # Convert predictions: sklearn uses -1 for anomaly, 1 for normal
    # We want 1 for anomaly, 0 for normal
    anomalies = pd.Series(np.where(preds == -1, 1, 0), index=df.index)
    anomaly_scores = pd.Series(-scores, index=df.index)  # invert score so higher is more anomalous
    
    return anomalies, anomaly_scores
