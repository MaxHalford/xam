import numpy as np
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


class CellTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, func):
        self.func_vec = np.vectorize(func)

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, **transform_params):
        X = as_float_array(X)
        X = check_array(X)
        return self.func_vec(X)


class ToDataFrameTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, index, columns, dtype=None):
        self.index = index
        self.columns = columns
        self.dtype = dtype

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, **transform_params):
        X = as_float_array(X)
        X = check_array(X)
        return pd.DataFrame(X, index=self.index, columns=self.columns, dtype=self.dtype)


class LambdaExtractor(TransformerMixin):

    def __init__(self, func):
        self.func = func

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.func(X)
