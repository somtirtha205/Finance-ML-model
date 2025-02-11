import mlflow

from ar_classification import data_load_and_process, generate_feature, model_build_and_evaluate, split_data


def build_model():
    mlflow.autolog()

    print("Hello from finance-ml-model!")

    df = data_load_and_process.load_data("data/AR-dataset.csv")

    data_load_and_process.preprocess_data(df)

    X, X_train, X_test, y_train, y_test = split_data.train_test_data_split(df)

    X_train_features = generate_feature.get_features(X_train, X_train)

    X_test_features = generate_feature.get_features(X_test, X_train)

    model1 = model_build_and_evaluate.train_and_evaluate_model(X_train_features, X_test_features, y_train, y_test)

    print(model1.best_params_)


build_model()
