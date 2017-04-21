from math import pi

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


class CycleTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        self.maximums_ = np.max(X, axis=0)
        return self

    @staticmethod
    def cosine_sine_transform(x, maximum):
        radians = 2 * np.pi * x
        return np.cos(radians / (maximum + 1)) + np.sin(radians / (maximum + 1))

    def transform(self, X, **transform_params):
        return np.array([
            self.cosine_sine_transform(x, maximum)
            for x, maximum in zip(X.T, self.maximums_)
        ]).T
