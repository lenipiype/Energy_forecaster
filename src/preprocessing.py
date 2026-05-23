import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def load_data(file_path):
    """
    Loads energy consumption dataset and sets the timestamp index.
    """
    df = pd.read_csv(file_path)
    # Convert timestamp column to datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
        
    df.sort_index(inplace=True)
    return df

def check_missing_values(df):
    """
    Checks for missing or null values in the dataframe.
    """
    missing_summary = df.isnull().sum()
    return missing_summary[missing_summary > 0]

def impute_missing_values(df, method='linear'):
    """
    Imputes missing values using seasonal-aware interpolation or forward-fill.
    """
    df_imputed = df.copy()
    if method == 'linear':
        df_imputed = df_imputed.interpolate(method='linear')
    elif method == 'ffill':
        df_imputed = df_imputed.ffill().bfill()
    else:
        # Season-aware / day-of-week hour-of-day mean imputation
        # Good for larger gaps
        for col in df_imputed.select_dtypes(include=[np.number]).columns:
            if df_imputed[col].isnull().any():
                # Compute mapping of (dayofweek, hour) to mean
                df_imputed['hour'] = df_imputed.index.hour
                df_imputed['dayofweek'] = df_imputed.index.dayofweek
                means = df_imputed.groupby(['dayofweek', 'hour'])[col].transform('mean')
                df_imputed[col] = df_imputed[col].fillna(means)
                df_imputed.drop(columns=['hour', 'dayofweek'], inplace=True, errors='ignore')
                
    # Fallback to ffill for any remaining NaNs
    if df_imputed.isnull().any().any():
        df_imputed = df_imputed.ffill().bfill()
        
    return df_imputed

def train_test_split_temporal(df, test_days=30):
    """
    Splits the dataframe into training and testing sets chronologically
    to avoid data leakage.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Source dataframe with DatetimeIndex.
    test_days : int
        Number of days at the end of the dataset to reserve for testing.
        
    Returns:
    --------
    train_df, test_df : pd.DataFrame, pd.DataFrame
    """
    split_date = df.index.max() - pd.Timedelta(days=test_days)
    train_df = df[df.index <= split_date].copy()
    test_df = df[df.index > split_date].copy()
    return train_df, test_df

def scale_features(train_df, test_df, feature_cols, target_col, scaler_type='minmax'):
    """
    Scales features and target variables using training set parameters
    to prevent data leakage.
    
    Parameters:
    -----------
    train_df, test_df : pd.DataFrame
        Training and testing sets.
    feature_cols : list
        List of feature column names to scale.
    target_col : str
        Target column name (e.g. 'total_consumption').
    scaler_type : str
        'minmax' or 'standard'.
        
    Returns:
    --------
    X_train_scaled, X_test_scaled : np.ndarray
    y_train_scaled, y_test_scaled : np.ndarray
    feat_scaler, target_scaler : Scaler objects
    """
    if scaler_type == 'standard':
        feat_scaler = StandardScaler()
        target_scaler = StandardScaler()
    else:
        feat_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()
        
    # Scale features
    X_train_scaled = feat_scaler.fit_transform(train_df[feature_cols])
    X_test_scaled = feat_scaler.transform(test_df[feature_cols])
    
    # Scale target
    y_train_scaled = target_scaler.fit_transform(train_df[[target_col]]).ravel()
    y_test_scaled = target_scaler.transform(test_df[[target_col]]).ravel()
    
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, feat_scaler, target_scaler
