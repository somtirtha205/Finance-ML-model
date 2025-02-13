import os

from src.ar_classification import data_load_and_process, generate_feature, model_build_and_evaluate, split_data

os.makedirs("data/processed", exist_ok=True)


def main():
    print("Hello from finance-ml-model!")

    df = data_load_and_process.load_data("data/AR-dataset.csv")

    data_load_and_process.preprocess_data(df)

    X, X_train, X_test, y_train, y_test = split_data.train_test_data_split(df)

    X_train_features = generate_feature.get_features(X_train, X_train)

    X_test_features = generate_feature.get_features(X_test, X_train)

    # Save the features to files
    X_train_features_path = "data/processed/X_train_features.csv"
    X_test_features_path = "data/processed/X_test_features.csv"
    X_train_features.to_csv(X_train_features_path, index=False)
    X_test_features.to_csv(X_test_features_path, index=False)

    model1 = model_build_and_evaluate.train_and_evaluate_model(X_train_features, X_test_features, y_train, y_test)

    print(model1.best_params_)


if __name__ == "__main__":
    main()
