from math import pi

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


class CycleTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        self.maximums_ = np.max(X, axis=0)
        return self

    def transform(self, X, **transform_params):
        return np.vstack((
            np.array([
                np.cos(2 * np.pi * x / (maximum + 1))
                for x, maximum in zip(X.T, self.maximums_)
            ]),
            np.array([
                np.sin(2 * np.pi * x / (maximum + 1))
                for x, maximum in zip(X.T, self.maximums_)
            ])
        )).T
