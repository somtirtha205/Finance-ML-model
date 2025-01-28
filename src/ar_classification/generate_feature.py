import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def get_features(X, X_train):
    customer_features = (
        X_train.groupby("customerID")
        .agg(
            {
                "InvoiceDate": ["count"],
                "InvoiceAmount": ["sum", "mean", "max"],
                "Disputed": ["mean"],
                "PaperlessBill": ["sum", "mean"],
                "DaysToSettle": ["mean", "max"],
                "DaysLate": ["mean", "max"],
            }
        )
        .reset_index()
    )

    # Pandas group-by creates a MultiIndex, which we don't want. The following few lines
    # will rename the columns of the dataframe to something more reasonable.
    customer_features.columns = ["_".join(x) for x in customer_features.columns.ravel()]
    customer_features = customer_features.rename(columns={"customerID_": "customerID"})
    customer_features.info()

    cat_feature_names = ["countryCode", "InvoiceDate_DOW", "InvoiceDate_Month", "DueDate_DOW", "DueDate_Month"]

    # Fit the OneHotEncoder and save the `ohe` object for later.
    ohe = OneHotEncoder(categories="auto", handle_unknown="ignore", sparse_output=False)
    ohe = ohe.fit(X_train[cat_feature_names])
    ohe_feature_names = list(ohe.get_feature_names_out(cat_feature_names))

    # print("\n\nNames of {} OHE features:".format(len(ohe_feature_names)))
    # print(ohe_feature_names)

    # Use the OHE to get numerical features from categorical
    ohe_features = ohe.transform(X[cat_feature_names])
    ohe_features_df = pd.DataFrame(ohe_features, columns=ohe_feature_names)

    X_features = X.merge(customer_features, how="left", on="customerID").reset_index(drop=True)

    X_features = X_features.drop(
        [
            "customerID",
            "InvoiceDate",
            "countryCode",
            "InvoiceDate_DOW",
            "InvoiceDate_Month",
            "DueDate_DOW",
            "DueDate_Month",
            "Disputed",
            "SettledDate",
            "DaysToSettle",
            "DaysLate",
            "DueDate",
            "PaperlessDate",
        ],
        axis=1,
    )

    features_all = pd.concat([X_features, ohe_features_df], axis=1)

    return features_all
