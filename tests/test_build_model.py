import pandas as pd
import pytest

from ar_classification import data_load_and_process, generate_feature, model_build_and_evaluate, split_data


def test_load_data():
    df = data_load_and_process.load_data("data/AR-dataset.csv")
    assert isinstance(df, pd.DataFrame), "load_data should return a DataFrame"
    assert not df.empty, "DataFrame should not be empty"

def test_preprocess_data():
    df = data_load_and_process.load_data("data/AR-dataset.csv")
    data_load_and_process.preprocess_data(df)
    assert isinstance(df, pd.DataFrame), "preprocess_data should return a DataFrame"
    assert not df.empty, "Processed DataFrame should not be empty"

def test_train_test_data_split():
    df = data_load_and_process.load_data("data/AR-dataset.csv")
    data_load_and_process.preprocess_data(df)
    X, X_train, X_test, y_train, y_test = split_data.train_test_data_split(df)
    assert len(X_train) > 0, "X_train should not be empty"
    assert len(X_test) > 0, "X_test should not be empty"
    assert len(y_train) > 0, "y_train should not be empty"
    assert len(y_test) > 0, "y_test should not be empty"

def test_get_features():
    df = data_load_and_process.load_data("data/AR-dataset.csv")
    data_load_and_process.preprocess_data(df)
    X, X_train, X_test, y_train, y_test = split_data.train_test_data_split(df)
    X_train_features = generate_feature.get_features(X_train, X_train)
    X_test_features = generate_feature.get_features(X_test, X_train)
    assert X_train_features.shape[0] == X_train.shape[0], "X_train_features should have the same number of rows as X_train"
    assert X_test_features.shape[0] == X_test.shape[0], "X_test_features should have the same number of rows as X_test"

def test_train_and_evaluate_model():
    df = data_load_and_process.load_data("data/AR-dataset.csv")
    data_load_and_process.preprocess_data(df)
    X, X_train, X_test, y_train, y_test = split_data.train_test_data_split(df)
    X_train_features = generate_feature.get_features(X_train, X_train)
    X_test_features = generate_feature.get_features(X_test, X_train)
    model = model_build_and_evaluate.train_and_evaluate_model(X_train_features, X_test_features, y_train, y_test)
    assert model is not None, "Model should not be None"
    assert hasattr(model, 'best_params_'), "Model should have best_params_ attribute"

if __name__ == "__main__":
    pytest.main()
