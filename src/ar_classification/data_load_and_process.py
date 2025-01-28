import numpy as np
import pandas as pd


def load_data(filepath):
    """
    Load the data from the given filepath.
    """
    df = pd.read_csv(filepath)

    return df


def preprocess_data(df):
    df["PaperlessDate"] = pd.to_datetime(df["PaperlessDate"])
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["DueDate"] = pd.to_datetime(df["DueDate"])
    df["SettledDate"] = pd.to_datetime(df["SettledDate"])
    df["countryCode"] = df.countryCode.astype("category")
    df["customerID"] = df.customerID.astype("category")
    df["InvoiceDate_DOW"] = df["InvoiceDate"].dt.day_name()
    df["InvoiceDate_Week"] = df["InvoiceDate"].dt.strftime("%U")
    df["InvoiceDate_Month"] = df["InvoiceDate"].dt.month_name()
    df["InvoiceDate_IsWeekend"] = np.where(df["InvoiceDate"].dt.weekday < 5, 0, 1)
    df["DueDate_DOW"] = df["DueDate"].dt.day_name()
    df["DueDate_Week"] = df["DueDate"].dt.strftime("%U")
    df["DueDate_Month"] = df["DueDate"].dt.month_name()
    df["DueDate_IsWeekend"] = np.where(df["DueDate"].dt.weekday < 5, 0, 1)
    df["Disputed"] = np.where(df["Disputed"] == "Yes", 1, 0)
    df["PaperlessBill"] = np.where(df["PaperlessBill"] == "Electronic", 1, 0)

    df = df.sort_values(by="InvoiceDate")
    df = df.reset_index(drop=True)
    print(df)
