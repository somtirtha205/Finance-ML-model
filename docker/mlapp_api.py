import json

import joblib
import numpy as np
from fastapi import FastAPI, Request

# Load the model
model = joblib.load("ar.pkl")

# Initialize FastAPI
app = FastAPI()

# API endpoint for predictions
@app.post("/predict/")
async def predict(request: Request):
    data = await request.json()  # Get data posted as a JSON
    features = np.array(list(data.values())).reshape(1, -1)  # Convert input to NumPy array
    prediction = model.predict(features).tolist()  # Make prediction
    return {"prediction": prediction}

# API endpoint for predictions from a file
@app.post("/predict_file/")
async def predict_file(request: Request):
    data = await request.json()  # Get data posted as a JSON
    features = np.array(list(data.values())).reshape(1, -1)  # Convert input to NumPy array
    prediction = model.predict(features).tolist()  # Make prediction
    return {"prediction": prediction}

# Root endpoint
@app.get("/")
def root():
    return {"message": "ML Model API is running!"}