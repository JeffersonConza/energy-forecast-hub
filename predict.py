import joblib
import pandas as pd
import os
from src.preprocessing import preprocess_data

def load_model(model_name, model_dir='models'):
    path = os.path.join(model_dir, model_name)
    if not os.path.exists(path):
        # Fallback: check if the user forgot the .joblib extension
        if os.path.exists(path + ".joblib"):
            path += ".joblib"
        else:
            raise FileNotFoundError(f"Model {model_name} not found in {model_dir}")
    print(f"Loading from: {path}")
    return joblib.load(path)

def make_predictions(input_data, model_name='production_model.pkl.joblib'):
    # 1. Load Model
    # Note: Your logs showed the file ends in .pkl.joblib
    model = load_model(model_name)

    # 2. Preprocess Input Data
    df = pd.DataFrame(input_data)
    
    # This now magically fills in 'year', 'month', 'day_in_week', etc.
    df = preprocess_data(df) 

    # 3. Predict
    try:
        predictions = model.predict(df)
        return predictions
    except Exception as e:
        print(f"Prediction failed. Data columns: {df.columns}")
        raise e

if __name__ == "__main__":
    # MINIMAL INPUT: We only need the date now!
    new_data = [
        {'date': '2023-01-01'},
        {'date': '2023-01-02'},
        {'date': '2023-08-15'}
    ]

    try:
        # Based on your log: "Model saved to models\production_model.pkl.joblib"
        preds = make_predictions(new_data, model_name='production_model.pkl.joblib')

        print("\n--- Predictions ---")
        for i, p in enumerate(preds):
            print(f"Date: {new_data[i]['date']} | Predicted Consumption: {p:.2f} kW")
            
    except Exception as e:
        print(f"Error: {e}")