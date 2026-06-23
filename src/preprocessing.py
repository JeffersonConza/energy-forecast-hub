import pandas as pd
import os

def load_data(data_dir):
    """Loads train and test CSV files from the data directory."""
    train_path = os.path.join(data_dir, 'df_train.csv')
    test_path = os.path.join(data_dir, 'df_test.csv')
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"CSVs not found in {data_dir}")

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    return df_train, df_test

_rolling_mean_lookup = None

def get_rolling_mean_lookup():
    global _rolling_mean_lookup
    if _rolling_mean_lookup is not None:
        return _rolling_mean_lookup
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    train_path = os.path.join(root_dir, 'data', 'df_train.csv')
    test_path = os.path.join(root_dir, 'data', 'df_test.csv')
    
    if os.path.exists(train_path) and os.path.exists(test_path):
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
        df_combined = pd.concat([df_train, df_test], ignore_index=True)
        df_combined['parsed_date'] = pd.to_datetime(df_combined['date'])
        df_combined = df_combined.sort_values('parsed_date').reset_index(drop=True)
        
        df_combined['power_rolling_mean_7d'] = (
            df_combined['power_consumption']
            .shift(1)
            .rolling(window=7, min_periods=7)
            .mean()
        )
        
        df_combined['date_str'] = df_combined['parsed_date'].dt.strftime('%Y-%m-%d')
        _rolling_mean_lookup = df_combined.set_index('date_str')['power_rolling_mean_7d'].to_dict()
    else:
        _rolling_mean_lookup = {}
        
    return _rolling_mean_lookup

def preprocess_data(df):
    """
    Cleans data and AUTOMATICALLY adds Feature Engineering.
    This ensures that predict.py can work with just a 'date' input.
    """
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        
        if 'year' not in df.columns:
            df['year'] = df['date'].dt.year
        if 'month' not in df.columns:
            df['month'] = df['date'].dt.month
        if 'day_in_year' not in df.columns:
            df['day_in_year'] = df['date'].dt.dayofyear
        if 'quarter' not in df.columns:
            df['quarter'] = df['date'].dt.quarter
        if 'week_in_year' not in df.columns:
            df['week_in_year'] = df['date'].dt.isocalendar().week.astype(int)
        
        if 'semester' not in df.columns:
            df['semester'] = (df['date'].dt.month > 6).astype(int) + 1

        if 'day_in_week' not in df.columns:
            df['day_in_week'] = df['date'].dt.day_name()

        if 'power_rolling_mean_7d' not in df.columns:
            lookup = get_rolling_mean_lookup()
            date_strs = df['date'].dt.strftime('%Y-%m-%d')
            df['power_rolling_mean_7d'] = date_strs.map(lookup).fillna(1592.96)

    cast_cols = {
        'year': 'int64',
        'month': 'int64',
        'semester': 'int64',
        'quarter': 'int64',
        'week_in_year': 'int64',
        'day_in_year': 'int64',
        'power_rolling_mean_7d': 'float64'
    }
    for col, dtype in cast_cols.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)

    return df

def split_features_target(df, target_col='power_consumption', drop_cols=['date']):
    """Splits dataframe into X (features) and y (target)."""
    X = df.drop(columns=[target_col] + drop_cols, errors='ignore')
    y = df[target_col] if target_col in df.columns else None
    return X, y