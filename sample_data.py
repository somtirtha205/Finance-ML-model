import os

import numpy as np

from ar_classification import data_load_and_process

num = np.random.randint(0, 493)
df = data_load_and_process.load_data("data/processed/X_test_features.csv")
data = df.iloc[num]

os.makedirs("data/testing", exist_ok=True)

data.to_json("data/testing/sample_data.json", orient="index")
