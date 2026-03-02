from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn

# 1. Load the trained model you downloaded from Colab
# Ensure 'airshield_model.pkl' is in the same folder as this script
model = joblib.load('airshield_model.pkl')

app = FastAPI(title="AirShield ML API")

# 2. Define the expected data structure from Node.js
class AirQualityInput(BaseModel):
    hour: int
    temp: float
    humidity: float
    mq135: float
    pm25: float

@app.get("/")
def home():
    return {"message": "AirShield ML Service is Running"}

@app.post("/predict")
def predict_aqi(data: AirQualityInput):
    # Convert incoming JSON to a DataFrame with EXACT column names from training
    input_data = pd.DataFrame([{
        'Hour': data.hour,
        'Temp (°C)': data.temp,
        'Humidity (%)': data.humidity,
        'MQ135_Value (PPM)': data.mq135,
        'PM2.5 (μg/m³)': data.pm25
    }])
    
    # Generate Prediction
    prediction = model.predict(input_data)[0]
    
    # Logic for Classification & Alerts
    status = "Safe"
    alert_triggered = False
    
    if prediction > 150:
        status = "Dangerous"
        alert_triggered = True
    elif prediction > 50:
        status = "Moderate"
    
    return {
        "predicted_aqi": round(prediction, 2),
        "status": status,
        "alert": alert_triggered
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)