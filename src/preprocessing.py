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

def preprocess_data(df):
    """
    Cleans data and AUTOMATICALLY adds Feature Engineering.
    This ensures that predict.py can work with just a 'date' input.
    """
    # 1. Convert date to datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        
        # 2. Feature Engineering: Create columns if they don't exist
        # This fixes the "columns are missing" error!
        if 'year' not in df.columns:
            df['year'] = df['date'].dt.year
        if 'month' not in df.columns:
            df['month'] = df['date'].dt.month
        if 'day_in_year' not in df.columns:
            df['day_in_year'] = df['date'].dt.dayofyear
        if 'quarter' not in df.columns:
            df['quarter'] = df['date'].dt.quarter
        if 'week_in_year' not in df.columns:
            # isocalendar().week returns a special type, cast to int
            df['week_in_year'] = df['date'].dt.isocalendar().week.astype(int)
        
        # Logic for semester (1 for Jan-Jun, 2 for Jul-Dec)
        if 'semester' not in df.columns:
            df['semester'] = (df['date'].dt.month <= 6).astype(int).replace({1: 1, 0: 2})

        # Ensure day_in_week exists (e.g., 'Monday', 'Tuesday')
        if 'day_in_week' not in df.columns:
            df['day_in_week'] = df['date'].dt.day_name()

    return df

def split_features_target(df, target_col='power_consumption', drop_cols=['date']):
    """Splits dataframe into X (features) and y (target)."""
    X = df.drop(columns=[target_col] + drop_cols, errors='ignore')
    y = df[target_col]
    return X, y