from collections import defaultdict

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.preprocessing import Imputer


class SupervisedImputer(BaseEstimator, TransformerMixin):

    def __init__(self, imputer):
        self.imputer = imputer

    def fit(self, X, y=None, **fit_params):
        self.classes_ = np.unique(y)
        self.imputers_ = defaultdict(dict)

        for i, x in enumerate(X.T):
            for c in self.classes_:
                non_nulls = x[y == c]
                self.imputers_[i][c] = Imputer().fit(
                    non_nulls[~np.isnan(non_nulls)].reshape(-1, 1)
                )
        return self

    def transform(self, X, y=None):
        X_trans = np.copy(X)
        for i, x in enumerate(X.T):
            for c in self.classes_:
                nans = x[(np.isnan(x)) & (y == c)]
                if nans.size > 0:
                    X_trans[:, i][(np.isnan(x)) & (y == c)] = self.imputers_[i][c].transform(
                        nans.reshape(-1, 1)
                    )
        return X_trans

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params).transform(X, y)
