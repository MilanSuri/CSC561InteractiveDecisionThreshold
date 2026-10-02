import polars as pl
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

def get_data():
    data = load_breast_cancer()

    X = pl.DataFrame (
        data.data,
        schema = data.feature_names.tolist()
    )

    y = pl.Series("target", data.target)

    return X, y

def train_log_reg(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X.to_numpy(),
        y.to_numpy(),
        test_size = 0.25,
        random_state = 42,
        stratify = y, # This sets the train and test to have relatively similar proportions of classes
    )

    model = LogisticRegression(max_iter=10_000)
    model.fit(X_train, y_train)

    return model, X_test, y_test

def predictions(model, X_test, threshold: float):
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    return probabilities, predictions

def generate_confusion_matrix (y_test, predictions):
    return confusion_matrix(y_test, predictions)