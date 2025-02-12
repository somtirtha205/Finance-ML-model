import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    tscv = TimeSeriesSplit()

    pipeline1 = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", random_state=0))

    param_grid1 = {
        "standardscaler__with_mean": [True, False],
        "logisticregression__C": [0.1, 0.5, 1],
    }

    search1 = GridSearchCV(
        pipeline1, param_grid1, cv=tscv, n_jobs=5, scoring="f1_micro", return_train_score=True, verbose=3
    )

    model1 = search1.fit(X_train, y_train)

    y_pred = model1.predict(X_test)

    print("F1 Score   = {:.3f}".format(f1_score(y_test, y_pred, average="micro")))

    print("\nConfusion Matrix:")
    unique_label = np.unique([y_test, y_pred])
    cm = pd.DataFrame(
        confusion_matrix(y_test, y_pred, labels=unique_label),
        index=["true:{:}".format(x) for x in unique_label],
        columns=["pred:{:}".format(x) for x in unique_label],
    )
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    model_filename = "ar.pkl"
    joblib.dump(model1, model_filename)

    return model1
