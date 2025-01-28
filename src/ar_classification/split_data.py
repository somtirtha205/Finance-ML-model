from sklearn.model_selection import train_test_split

target = "Late invoice"

def train_test_data_split(df):
    X = df.copy()
    y = df[target]
    X = X.drop(target, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25, 
                                                        shuffle=False, stratify=None)
    return X, X_train, X_test, y_train, y_test