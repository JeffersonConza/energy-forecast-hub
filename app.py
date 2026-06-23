from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import sys
from loguru import logger
from src.preprocessing import preprocess_data

# Configure loguru logging
logger.remove()
logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", level="INFO")

# 1. Initialize the API
app = FastAPI(title="Energy Forecast Hub API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Define the expected input format
class DateInput(BaseModel):
    date: str  # Example: "2023-12-25"

# 3. Load the model once at startup
MODEL_PATH = os.path.join("models", "production_model.pkl.joblib")

if not os.path.exists(MODEL_PATH):
    logger.error(f"Model not found at {MODEL_PATH}. Run main.py first.")
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run main.py first.")

logger.info(f"Loading model from {MODEL_PATH}...")
model = joblib.load(MODEL_PATH)
logger.info("Model loaded successfully.")

@app.get("/")
def home():
    """Health check endpoint."""
    logger.info("Health check endpoint called.")
    return {"status": "online", "language": "Python"}

@app.post("/predict")
def predict_energy(data: DateInput):
    """
    Receives a date, generates features, and returns the predicted power consumption.
    """
    logger.info(f"Prediction requested for date: {data.date}")
    try:
        # A. Create a DataFrame from the input
        input_data = [{'date': data.date}]
        df = pd.DataFrame(input_data)
        
        # B. Run Preprocessing pipeline
        df_processed = preprocess_data(df)
        
        # C. Predict
        prediction = model.predict(df_processed)[0]
        pred_val = round(float(prediction), 2)
        
        logger.info(f"Prediction successful for {data.date}: {pred_val} kW")
        return {
            "date": data.date,
            "predicted_consumption_kw": pred_val,
            "model_used": "Random Forest (Production)"
        }
    except Exception as e:
        logger.error(f"Prediction failed for {data.date}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))