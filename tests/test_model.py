import json

import joblib
import numpy as np

from scripts.score import init, run

# Load the model for direct testing
MODEL_PATH = "models/model.pkl"
model = joblib.load(MODEL_PATH)

def test_model_loading():
    """Ensure the model is loaded successfully."""
    assert model is not None, "Model failed to load"

def test_model_prediction():
    """Test if model returns predictions in expected format."""
    sample_input = np.array([[5.1, 3.5, 1.4, 0.2]])  # Modify based on model
    prediction = model.predict(sample_input)

    assert isinstance(prediction, np.ndarray), "Prediction is not an array"
    assert len(prediction) == 1, "Model should return one prediction"

def test_scoring_script():
    """Test if the scoring script runs properly."""
    init()  # Initialize model
    input_data = json.dumps({"data": [[5.1, 3.5, 1.4, 0.2]]})
    response = run(input_data)
    response_json = json.loads(response)

    assert "predictions" in response_json, "Missing predictions in response"
    assert isinstance(response_json["predictions"], list), "Predictions should be a list"
