from collections import defaultdict

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import clone
from sklearn.base import TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import Imputer


def _get_mask(X, value_to_mask):
    """Compute the boolean mask X == missing_values."""
    if value_to_mask == "NaN" or np.isnan(value_to_mask):
        return np.isnan(X)
    else:
        return X == value_to_mask


class SupervisedImputer(BaseEstimator, TransformerMixin):

    def __init__(self, missing_values='NaN', strategy='mean'):
        self.missing_values = missing_values
        self.strategy = strategy

    def fit(self, X, y=None, **fit_params):

        self.classes_ = np.unique(y)
        self.imputers_ = defaultdict(dict)

        for i, x in enumerate(X.T):
            for c in self.classes_:
                group = x[y == c]
                self.imputers_[i][c] = Imputer(strategy=self.strategy).fit(
                    group[~_get_mask(group, self.missing_values)].reshape(-1, 1)
                )
        return self

    def transform(self, X, y=None):

        # Transform
        X_trans = np.copy(X)
        for i, x in enumerate(X.T):
            null_mask = _get_mask(x, self.missing_values)
            for c in self.classes_:
                nans = x[null_mask & (y == c)]
                if nans.size > 0:
                    X_trans[:, i][(null_mask) & (y == c)] = self.imputers_[i][c].transform(
                        nans.reshape(-1, 1)
                    )
        return X_trans

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params).transform(X, y)
