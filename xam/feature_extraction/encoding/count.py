import collections
import functools

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


class CountEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, columns=None, suffix='_count'):
        self.columns = columns
        self.suffix = suffix

    def fit(self, X, y=None):

        if not isinstance(X, pd.DataFrame):
            raise ValueError('X has to be a pandas.DataFrame')

        X = X.copy()

        # Default to using all the categorical columns
        columns = X.select_dtypes(['object', 'category']).columns\
            if self.columns is None\
            else self.columns

        names = []
        for cols in columns:
            if isinstance(cols, list):
                name = '_'.join(cols)
                names.append('_'.join(cols))
                X[name] = functools.reduce(
                    lambda a, b: a.astype(str) + '_' + b.astype(str),
                    [X[col] for col in cols]
                )
            else:
                names.append(cols)

        # Compute counts for each feature
        X = pd.concat((X[names], y.rename('y')), axis='columns')
        self.counts_ = {}

        for name in names:
            counts = X[name].value_counts()
            self.counts_[name] = collections.defaultdict(lambda: 0, counts.to_dict())

        return self

    def transform(self, X, y=None):

        if not isinstance(X, pd.DataFrame):
            raise ValueError('X has to be a pandas.DataFrame')

        for cols in self.columns:

            if isinstance(cols, list):
                name = '_'.join(cols)
                x = functools.reduce(
                    lambda a, b: a.astype(str) + '_' + b.astype(str),
                    [X[col] for col in cols]
                )
            else:
                name = cols
                x = X[name]

            X[name + self.suffix] = x.map(self.counts_[name])

        return X
