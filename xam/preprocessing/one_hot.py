import pandas as pd
from sklearn import preprocessing
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


class OneHotEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None, **fit_params):

        if not isinstance(X, pd.DataFrame):
            raise ValueError('X has to be a pandas.DataFrame')

        self.columns = X.columns if self.columns is None else self.columns
        self.labelers_ = {}
        self.encoders_ = {}
        self.most_common_ = {}

        for col in self.columns:

            # Label encode the values
            labeler = preprocessing.LabelEncoder().fit(X[col])
            labels = labeler.transform(X[col])
            self.labelers_[col] = labeler

            # One-hot encode the labels
            self.encoders_[col] = preprocessing.OneHotEncoder().fit(labels.reshape(-1, 1))

        return self


    def transform(self, X):

        if not isinstance(X, pd.DataFrame):
            raise ValueError('X has to be a pandas.DataFrame')

        for col in self.columns:
            # Label encode the values
            labels = self.labelers_[col].transform(X[col])
            # One-hot encode the labels
            one_hots = self.encoders_[col].transform(labels.reshape(-1, 1))
            # Create a DataFrame containing the one-hots
            columns = ['{}_{}'.format(col, klass) for klass in self.labelers_[col].classes_]
            frame = pd.DataFrame(one_hots.todense(), columns=columns).astype(bool)
            # Merge the input DataFrame with the one-hots
            X = pd.concat((X.drop(col, axis='columns'), frame), axis='columns')

        return X
