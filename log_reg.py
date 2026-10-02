import polars as pl
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

data = load_breast_cancer()
X = pl.DataFrame (
    data.data,
    schema = data.feature_names.tolist()
)
y = pl.Series("target", data.target)