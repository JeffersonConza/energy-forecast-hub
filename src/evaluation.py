from sklearn.metrics import mean_squared_error
import numpy as np

def evaluate_models(models, X_test, y_test):
    """Calculates RMSE for all trained models and returns results."""
    results = {}
    predictions = {}
    
    print("\n--- Model Evaluation (RMSE) ---")
    for name, model in models.items():
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        results[name] = rmse
        predictions[name] = y_pred
        
        print(f"{name}: {rmse:.2f} kW")
        
    return results, predictions