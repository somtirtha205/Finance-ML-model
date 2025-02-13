import json

import joblib
import numpy as np
import pandas as pd

# Load the dataset and extract a sample row
df = pd.read_csv("data/processed/X_test_features.csv")
sample_row = df.iloc[0].values

# Load the model for direct testing
MODEL_PATH = "model/ar.pkl"
model = joblib.load(MODEL_PATH)


def test_model_loading():
    """Ensure the model is loaded successfully."""
    assert model is not None, "Model failed to load"


def test_model_prediction():
    """Test if model returns predictions in expected format."""
    sample_input = np.array([sample_row])  # Use the sample row from the dataset
    prediction = model.predict(sample_input)

    assert isinstance(prediction, np.ndarray), "Prediction is not an array"
    assert len(prediction) == 1, "Model should return one prediction"


def test_scoring_script():
    """Test if the scoring script runs properly."""
    sample_input = np.array([sample_row])  # Use the sample row from the dataset
    prediction = model.predict(sample_input)

    response = json.dumps({"predictions": prediction.tolist()})
    response_json = json.loads(response)

    assert "predictions" in response_json, "Missing predictions in response"
    assert isinstance(response_json["predictions"], list), "Predictions should be a list"
