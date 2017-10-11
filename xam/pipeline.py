import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.utils import as_float_array
from sklearn.utils import check_array


class ColumnSelector(BaseEstimator, TransformerMixin):

    def __init__(self, columns=()):
        self.columns = columns

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, **transform_params):
        return X[self.columns]


class SeriesTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, func):
        self.func = func

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        return X.map(self.func)


class ToDataFrameTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, index=None, columns=None, dtype=None):
        self.index = index
        self.columns = columns
        self.dtype = dtype

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        if isinstance(X, pd.Series):
            return X.to_frame()
        X = as_float_array(X)
        X = check_array(X)
        return pd.DataFrame(X, index=self.index, columns=self.columns, dtype=self.dtype)


class LambdaTransfomer(BaseEstimator, TransformerMixin):

    def __init__(self, func=lambda x: x):
        self.func = func

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.func(X)
