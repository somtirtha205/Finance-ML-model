import json
import os

import joblib
import numpy as np
import pytest
from ar_model import score

from ar_classification import data_load_and_process


@pytest.fixture()
def load_model():
    # Initialize the model before running tests
    score.init()
    current_dir = os.getcwd()
    print(f"Current directory: {current_dir}")
    model_path = os.path.join(current_dir, "model", "ar.pkl")
    model = joblib.load(model_path)
    print(f"Model loaded successfully from {model_path}")
    return model


@pytest.fixture
def sample_data():
    num = np.random.randint(0, 493)
    df = data_load_and_process.load_data("data/processed/X_test_features.csv")
    data = df.iloc[num]
    return data.to_json(orient="index")


def test_run(load_model, sample_data):
    model = load_model

    input_dict = json.loads(sample_data)
    input_data = np.array(list(input_dict.values())).reshape(1, -1)

    result_pred = score.run(sample_data)
    print("Result:", result_pred)

    result_dict = json.loads(result_pred)
    print("Result dict:", result_dict)

    assert model is not None, "Model should not be None"
    assert model.predict(input_data) is not None, "Model prediction should not be None"
    assert "predictions" in result_dict, "Result should contain 'predictions'"
    assert "error" not in result_dict, "Result should not contain 'error'"


if __name__ == "__main__":
    pytest.main()
