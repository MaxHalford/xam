from math import pi

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


class CycleTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, **transform_params):
        return np.apply_along_axis(
            func1d=lambda x: np.cos(2 * np.pi * x / (np.max(x) + 1)) +
                             np.sin(2 * np.pi * x / (np.max(x) + 1)),
            axis=0,
            arr=X
        )
