import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.features import engineer_all_features

def predict_seasonal_naive(df, target_col, horizon=24, seasonal_period=24):
    """
    Seasonal Naive baseline forecaster.
    Repeats the value from exactly seasonal_period ago.
    
    E.g., if seasonal_period = 24 (1 day), the forecast for today at 3 PM 
    is the actual value from yesterday at 3 PM.
    If seasonal_period = 168 (1 week), it repeats the value from same hour last week.
    """
    predictions = []
    actuals = df[target_col].values
    n = len(actuals)
    
    # For testing on the end of a series
    for i in range(n - horizon, n):
        # Value from i - seasonal_period
        pred_idx = i - seasonal_period
        if pred_idx >= 0:
            predictions.append(actuals[pred_idx])
        else:
            predictions.append(np.mean(actuals[:24]))  # fallback
            
    return np.array(predictions)

def train_model(model_type, X_train, y_train, **kwargs):
    """
    Trains a forecasting model.
    
    Parameters:
    -----------
    model_type : str
        'xgboost' or 'ridge'.
    X_train : np.ndarray or pd.DataFrame
        Training features.
    y_train : np.ndarray or pd.Series
        Training target.
    kwargs : dict
        Hyperparameters.
        
    Returns:
    --------
    Trained model object.
    """
    if model_type == 'xgboost':
        params = {
            'n_estimators': kwargs.get('n_estimators', 100),
            'max_depth': kwargs.get('max_depth', 6),
            'learning_rate': kwargs.get('learning_rate', 0.1),
            'subsample': kwargs.get('subsample', 0.8),
            'colsample_bytree': kwargs.get('colsample_bytree', 0.8),
            'random_state': kwargs.get('random_state', 42),
            'n_jobs': -1
        }
        model = XGBRegressor(**params)
    elif model_type == 'ridge':
        params = {
            'alpha': kwargs.get('alpha', 1.0),
            'random_state': kwargs.get('random_state', 42)
        }
        model = Ridge(**params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    model.fit(X_train, y_train)
    return model

def evaluate_predictions(y_true, y_pred):
    """
    Computes standard regression evaluation metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Compute MAPE (handle zero values gracefully)
    y_true_safe = np.where(y_true == 0, 1e-5, y_true)
    mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100
    
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MAE': round(mae, 3),
        'RMSE': round(rmse, 3),
        'MAPE': round(mape, 3),
        'R2': round(r2, 4)
    }

def forecast_recursive(model, history_df, future_exog_df, target_col, feature_cols, 
                       feat_scaler=None, target_scaler=None, 
                       lag_list=[1, 2, 24, 168], roll_list=[6, 24, 168], 
                       temp_col='temperature'):
    """
    Performs recursive multi-step forecasting into the future, incorporating 
    exogenous predictions (like weather) and updating lag/rolling features dynamically.
    
    Parameters:
    -----------
    model : trained model object
    history_df : pd.DataFrame
        Historical dataset (including target and exogenous columns) up to last known point.
    future_exog_df : pd.DataFrame
        Future exogenous dataframe (index should cover forecast period, with weather & calendar columns).
    target_col : str
        Target column to predict (e.g. 'total_consumption').
    feature_cols : list
        List of feature column names in the exact order expected by the model.
    feat_scaler : Scaler object, optional
        Standard/MinMax scaler used for X features during training.
    target_scaler : Scaler object, optional
        Standard/MinMax scaler used for y target during training.
    lag_list, roll_list : list
        Parameters to match the feature engineering pipeline.
    temp_col : str
        Temperature column name.
        
    Returns:
    --------
    pd.DataFrame: Completed future forecast dataframe containing predicted values.
    """
    # Create working copy of history (we only need the last max(lags) days/hours to calculate features)
    max_history_needed = max(max(lag_list), max(roll_list)) + 5
    hist_subset = history_df.tail(max_history_needed).copy()
    
    # Prepare the future dataframe, initialized with NaNs for target
    future_df = future_exog_df.copy()
    future_df[target_col] = np.nan
    
    # Concatenate history and future
    full_df = pd.concat([hist_subset, future_df])
    
    # Get range of timestamps to predict
    prediction_timestamps = future_exog_df.index
    
    for t in prediction_timestamps:
        # 1. Compute features dynamically on the full dataframe up to timestamp t
        # Running feature engineering on the concatenated dataframe automatically
        # handles the lag and rolling computations for time t!
        # Note: We need a small buffer to avoid dropna deleting our active row.
        # We temporarily fill target NaNs with 0 in copy to avoid feature dropna issues,
        # but keep lag and rollings correct because we only shift/roll past values.
        temp_df = full_df.copy()
        
        # Fill NaN for the current step and future steps with 0 to prevent dropna 
        # from removing them. Lags look backwards, so as long as we fill past predicted 
        # steps with their predicted values, lag/rolling features are 100% correct!
        temp_df.loc[temp_df.index >= t, target_col] = 0.0
        
        # Run feature engineering
        temp_df_engineered = engineer_all_features(
            temp_df, 
            target_col=target_col, 
            lag_list=lag_list, 
            roll_list=roll_list, 
            temp_col=temp_col
        )
        
        # Extract features for the active timestamp t
        row_feat = temp_df_engineered.loc[[t], feature_cols].copy()
        
        # 2. Scale features if scaler is provided
        if feat_scaler is not None:
            X_scaled = feat_scaler.transform(row_feat)
        else:
            X_scaled = row_feat.values
            
        # 3. Predict scaled value
        pred_scaled = model.predict(X_scaled)
        
        # 4. Inverse-scale prediction if target scaler is provided
        if target_scaler is not None:
            pred_val = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()[0]
        else:
            pred_val = pred_scaled[0]
            
        # 5. Save unscaled prediction back into the master full_df so it is used in subsequent lag/rolling calculations
        full_df.loc[t, target_col] = pred_val
        
    # Return only the future predictions
    return full_df.loc[prediction_timestamps, [target_col] + list(future_exog_df.columns)]
