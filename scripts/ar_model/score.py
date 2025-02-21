import json
import logging

import joblib
import numpy as np
from azureml.core.model import Model

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# Global variable to store the model
model = None


def init():
    """Initialize the model by loading it into memory."""
    global model
    try:
        model_path = Model.get_model_path("AR_model")  # Ensure this matches your model name
        model = joblib.load(model_path)
        logger.info(f"Model loaded successfully from {model_path}")
        print("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
    finally:
        pass


def run(data):
    """Run the model on the input data."""
    try:
        # Convert input JSON into array
        input_dict = json.loads(data)
        input_data = np.array(list(input_dict.values())).reshape(1, -1)

        # Make prediction
        result = model.predict(input_data)

        logger.info(f"Prediction successful. Input: {input_data}, Output: {result.tolist()}")

        # Return results as JSON
        return json.dumps({"predictions": result.tolist()})

    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return json.dumps({"error": str(e)})
