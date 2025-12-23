from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from src.preprocessing import preprocess_data

# 1. Initialize the API
app = FastAPI(title="Energy Forecast Hub API")

# 2. Define the expected input format
class DateInput(BaseModel):
    date: str  # Example: "2023-12-25"

# 3. Load the model once at startup
# we use the exact filename from logs
MODEL_PATH = os.path.join("models", "production_model.pkl.joblib")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run main.py first.")

print(f"Loading model from {MODEL_PATH}...")
model = joblib.load(MODEL_PATH)

@app.get("/")
def home():
    """Health check endpoint."""
    return {"message": "Energy Forecast Hub is running. Send POST requests to /predict"}

@app.post("/predict")
def predict_energy(data: DateInput):
    """
    Receives a date, generates features, and returns the predicted power consumption.
    """
    try:
        # A. Create a DataFrame from the input
        input_data = [{'date': data.date}]
        df = pd.DataFrame(input_data)
        
        # B. Run your robust Preprocessing pipeline
        # this auto-calculates Year, Month, Semester, etc.
        df_processed = preprocess_data(df)
        
        # C. Predict
        prediction = model.predict(df_processed)[0]
        
        return {
            "date": data.date,
            "predicted_consumption_kw": round(float(prediction), 2),
            "model_used": "Random Forest (Production)"
        }
    except Exception as e:
        # Return a clean error message if something breaks
        raise HTTPException(status_code=500, detail=str(e))