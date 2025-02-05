import json

import joblib
import numpy as np
from azureml.core.model import Model

# Global variable to store the model
model = None


def init():
    """Initialize the model by loading it into memory."""
    global model
    model_path = Model.get_model_path("AR_model")  # Ensure this matches your model name
    model = joblib.load(model_path)
    print("Model loaded successfully!")


def run(data):
    """Run the model on the input data."""
    try:
        # Convert input JSON into numpy array
        input_data = np.array(json.loads(data)["data"])

        # Make prediction
        result = model.predict(input_data)

        # Return results as JSON
        return json.dumps({"predictions": result.tolist()})

    except Exception as e:
        return json.dumps({"error": str(e)})
