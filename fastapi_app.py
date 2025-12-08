from fastapi import FastAPI
import uvicorn
import numpy as np
import joblib

app = FastAPI()

# Load model (when it is ready)
# model = joblib.load("model.pkl")

@app.post("/predict")
def predict(payload: dict):

    # Placeholder logic — replace with your model later
    # features = np.array([[payload['year'], payload['temperature'], ...]])
    # prediction = model.predict(features)[0]

    fake_prediction = 123456  # REMOVE when your model is ready

    return {"prediction": float(fake_prediction)}
