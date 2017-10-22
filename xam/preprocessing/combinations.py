import itertools

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


class FeatureCombiner(BaseEstimator, TransformerMixin):

    """
    Args:
        columns (list of strs): List of columns to be combined.
        orders (list ints): Orders to which columns should be combined.
        separator (str): Separator to use to combined the column names and
            values.
    """

    def __init__(self, columns=None, orders=[2, 3], separator='_'):
        self.columns = columns
        self.orders = orders
        self.separator = separator

    def fit(self, X, y=None, **fit_params):

        if not isinstance(X, pd.DataFrame):
            raise ValueError('X has to be a pandas.DataFrame')

        dtypes = X.dtypes
        if self.columns is None:
            self.columns = [col for col in X.columns if dtypes[col] in ('object', 'category')]

        return self

    def transform(self, X, y=None):

        if not isinstance(X, pd.DataFrame):
            raise ValueError('X has to be a pandas.DataFrame')

        for order in self.orders:
            for combo in itertools.combinations(self.columns, order):
                col_name = self.separator.join(combo)
                X[col_name] = X[combo[0]].str.cat([
                    X[col]
                    for col in combo[1:]
                ], sep=self.separator).astype('category')

        return X
