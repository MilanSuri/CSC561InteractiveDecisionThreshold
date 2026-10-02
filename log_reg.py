'''
Milan Suri
CSC 561
Interactive Decision Threshold

This file is for creating a modular setup for making a log regression for breast cancer. We use the classification
set from sklearn and then divide it into a train-test split and feed it to a log regression and allow user to pass
thresholds for how to classify.
'''

import polars as pl
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

def get_data():
    '''This is the method for loading in the dataset and defining features and responses'''
    data = load_breast_cancer()

    # get a df from data
    X = pl.DataFrame (
        data.data,
        schema = data.feature_names.tolist()
    )

    y = pl.Series("target", data.target)

    return X, y

def train_log_reg(X, y):
    '''This creates a 75-25 train test split. We use stratification to create relatively equal datasets
    and a random number is for consistency. '''
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
    ''' Get the probabilities of the x test set by running it through the model. Then you check predictions with the
     threshold to see how it performed and where to categorize it.'''
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    return probabilities, predictions

def generate_confusion_matrix (y_test, predictions):
    '''Generates a confusion matrix for predictions compared to the answers of the test set.'''
    return confusion_matrix(y_test, predictions)