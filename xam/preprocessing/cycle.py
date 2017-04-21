from math import pi

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


class CycleTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    @staticmethod
    def cosine_sine_transform(x):
        maximum = np.max(x)
        radians = 2 * np.pi * x
        return np.cos(radians / (maximum + 1)) + np.sin(radians / (maximum + 1))

    def transform(self, X, **transform_params):
        return np.apply_along_axis(
            func1d=lambda x: self.cosine_sine_transform(x),
            axis=0,
            arr=X
        )
