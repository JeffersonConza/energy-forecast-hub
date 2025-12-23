from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Saving and loading models
import joblib
import os

def get_preprocessor():
    """Defines the transformer to OneHotEncode the 'day_in_week' column."""
    categorical_features = ['day_in_week']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )
    return preprocessor

def train_models(X_train, y_train, n_estimators=1000, max_depth=None, random_state=42):
    """
    Trains three models with comparable parameters for fair comparison.
    
    Args:
        X_train: Training features
        y_train: Training target
        n_estimators: Number of trees for RF and XGBoost (default: 1000)
        max_depth: Maximum tree depth for RF and XGBoost (default: None for RF, 6 for XGBoost)
        random_state: Random seed for reproducibility (default: 42)
    """
    preprocessor = get_preprocessor()

    # Set default max_depth for XGBoost if not specified
    xgb_max_depth = max_depth if max_depth is not None else 6

    models = {
        "Linear Regression": make_pipeline(
            preprocessor, 
            LinearRegression()
        ),
        "Random Forest": make_pipeline(
            preprocessor, 
            RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1
            )
        ),
        "XGBoost": make_pipeline(
            preprocessor, 
            XGBRegressor(
                n_estimators=n_estimators,
                max_depth=xgb_max_depth,
                random_state=random_state,
                n_jobs=-1
            )
        )
    }

    trained_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        
    return trained_models

def save_model(model, filename, model_dir='models'):
    """
    Saves the trained model to disk using joblib.
    
    Args:
        model: The trained model to save
        filename: Name of the model (used for filename)
        model_dir: Directory to save the model (default: 'models')
    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    path = os.path.join(model_dir, f"{filename.replace(' ', '_').lower()}.joblib")
    joblib.dump(model, path, compress=3)
    print(f"Model saved to {path}")