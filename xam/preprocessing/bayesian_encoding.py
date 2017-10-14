import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


# class BayesianEncoding(BaseEstimator, TransformerMixin):

#     """
#     Args:
#         min_samples (int): List of columns to be combined.
#         smoothing (float): Orders to which columns should be combined.
#         separator (str)
#     """

#     def __init__(self, min_samples=1, smoothing=1):
#         self.min_samples = min_samples
#         self.smoothing = smoothing

#     def fit(self, X, y=None, **fit_params):

#         if not isinstance(X, pd.DataFrame):
#             raise ValueError('X has to be a pandas.DataFrame')

#         dtypes = X.dtypes
#         if self.columns is None:
#             self.columns = [col for col in X.columns if dtypes[col] in ('object', 'category')]

#         return self

#     def transform(self, X, y=None):

#         if not isinstance(X, pd.DataFrame):
#             raise ValueError('X has to be a pandas.DataFrame')

#         for order in self.orders:
#             for combo in itertools.combinations(self.columns, order):
#                 col_name = self.separator.join(combo)
#                 X[col_name] = X[combo[0]].str.cat([X[col] for col in combo[1:]], sep=self.separator)

#         return X

df = pd.DataFrame({
    'feature': ['a'] * 5 + ['b'] * 4 + ['c'] * 3,
    'target': [1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0]
})

agg = df.groupby('feature')['target'].agg(['count', 'mean'])
counts = agg['count']
means = agg['mean']
prior = means.mean()

print(prior)

print(1 / (1 + np.exp(-(counts - 4) / 10)))
