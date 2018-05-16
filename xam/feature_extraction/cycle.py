import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.utils import check_array


class CycleTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None, **fit_params):

        # scikit-learn checks
        X = check_array(X)

        self.maximums_ = np.max(X, axis=0)
        return self

    def transform(self, X, y=None):

        # scikit-learn checks
        X = check_array(X)

        if X.shape[1] != len(self.maximums_):
            raise ValueError("X has different shape than during fitting. "
                             "Expected %d, got %d." % (len(self.maximums_), X.shape[1]))

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
